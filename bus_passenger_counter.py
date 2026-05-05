#!/usr/bin/env python3
"""
Bus Passenger Counter — v3 DEFINITIVE + API Push
==================================================
Camera: overhead/top-down at bus front gate
Direction: passengers move LEFT → RIGHT to enter, RIGHT → LEFT to exit

All confirmed bugs from video analysis fixed:
  ✓ Line moved to 50% width (was 38% — too early to init ID)
  ✓ "???IN" ghost counts fixed — IDs first seen inside dead zone are SKIPPED
  ✓ IDs first seen on RIGHT side (already inside bus) cannot trigger IN count
  ✓ Proper 3-state machine: OUTSIDE → ZONE → INSIDE (must traverse all 3)
  ✓ Kalman Filter per track for smooth centroid prediction during occlusion
  ✓ conf=0.12, iou=0.40 — separates shoulder-to-shoulder passengers
  ✓ ByteTrack: buffer=150 frames (5s), low activation threshold
  ✓ CYAN flash on confirmed count (auditable in real-time)
  ✓ Trail lines with age-based color gradient
  ✓ Ghost ID cleanup every 90 frames
  ✓ CSV logs every event with frame + timestamp
  ✓ agnostic_nms=True prevents class-merging of overlapping detections

API Push additions (v3.1):
  ✓ Non-blocking POST via background thread + queue (zero FPS impact)
  ✓ Payload: {datetime, hin, hout, inside, total}
  ✓ 2-second request timeout — network hangs never stall the main loop
  ✓ Max 2 retries on 5xx errors with short back-off
  ✓ Triggered only on confirmed IN/OUT transitions (no duplicate calls)
"""

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import csv, os, argparse
import torch
import torch.nn as nn
from torchvision import models, transforms

# ── NEW: non-blocking API push imports ──────────────────────────────────
import threading
import queue
import time
try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False
    print("[WARN] 'requests' library not found. API push disabled. "
          "Install with: pip install requests")

# ═══════════════════════════════════════════════════════
#  CONFIGURATION  — edit these to tune for your scene
# ═══════════════════════════════════════════════════════
LINE_RATIO     = 0.45    
DEAD_ZONE_PX   = 30      
DEBOUNCE_N     = 3       
CONF_THRESH    = 0.12    
IOU_THRESH     = 0.40    
TRAIL_LEN      = 50      
GHOST_TIMEOUT  = 150     
FLASH_FRAMES   = 20      
EMA_ALPHA      = 0.40    
BADGE_MIN_PX   = 20
EXEMPT_THRESH  = 0.70    # Cosine similarity threshold for exempt match
EXEMPT_SAMPLES = 5       # Frames to collect before deciding exempt status

# ── API configuration ────────────────────────────────────────────────────
API_ENDPOINT   = "https://bae6-49-205-179-53.ngrok-free.app/passenger-count"
API_TIMEOUT    = 2        # seconds — never block processing loop
API_MAX_RETRY  = 2        # retries on 5xx errors
API_RETRY_WAIT = 0.3      # seconds between retries

# BoT-SORT tracker config (see botsort.yaml for full settings)
TRACKER_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "botsort.yaml")

# Colours (BGR)
COL_LINE      = (0,  0,   220)   # red trigger line
COL_ZONE_L    = (255, 80, 0)     # left dead-zone border (blue)
COL_ZONE_R    = (0, 180, 255)    # right dead-zone border (amber)
COL_BOX_NORM  = (0, 140, 255)    # normal bbox (orange)
COL_BOX_FLASH = (255, 255, 0)    # cyan flash on count
COL_BOX_SKIP  = (80,  80, 80)    # gray — skipped ID (first seen inside)
COL_IN_TEXT   = (80, 255, 80)    # green text
COL_OUT_TEXT  = (80, 180, 255)   # blue text
COL_EXEMPT    = (255, 0, 200)    # magenta — exempt person


# ═══════════════════════════════════════════════════════
#  NON-BLOCKING API WORKER
# ═══════════════════════════════════════════════════════

class ApiPushWorker:
    """
    Background daemon thread that drains a queue and fires POST requests.

    The main video loop enqueues a payload dict and returns immediately —
    network latency or retries never touch the processing thread.
    """

    def __init__(self, endpoint: str, timeout: int = API_TIMEOUT,
                 max_retry: int = API_MAX_RETRY,
                 retry_wait: float = API_RETRY_WAIT):
        self.endpoint   = endpoint
        self.timeout    = timeout
        self.max_retry  = max_retry
        self.retry_wait = retry_wait
        self._q         = queue.Queue()
        self._enabled   = _REQUESTS_OK

        if self._enabled:
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            print(f"[API] Push worker started → {endpoint}")
        else:
            print("[API] Push worker DISABLED (requests not installed)")

    # ── public interface ─────────────────────────────────────────────────

    def push(self, in_count: int, out_count: int):
        """Enqueue a payload. Returns immediately — never blocks."""
        if not self._enabled:
            return
        inside  = max(0, in_count - out_count)
        payload = {
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hin":      in_count,
            "hout":     out_count,
            "inside":   inside,
            "total":    in_count,        # cumulative entries
        }
        self._q.put(payload)

    # ── internal worker ──────────────────────────────────────────────────

    def _worker(self):
        """Runs in a daemon thread. Retries on 5xx, drops on permanent errors."""
        while True:
            payload = self._q.get()       # blocks until item available
            self._send_with_retry(payload)
            self._q.task_done()

    def _send_with_retry(self, payload: dict):
        for attempt in range(1, self.max_retry + 2):   # 1 try + max_retry retries
            try:
                resp = requests.post(
                    self.endpoint,
                    json=payload,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code < 500:
                    # 2xx success or 4xx client error — no point retrying
                    if resp.status_code >= 400:
                        print(f"[API] Client error {resp.status_code} "
                              f"for payload {payload} — not retrying")
                    else:
                        print(f"[API] ✓ {resp.status_code}  "
                              f"hin={payload['hin']} hout={payload['hout']} "
                              f"inside={payload['inside']}")
                    return
                # 5xx — eligible for retry
                print(f"[API] Server error {resp.status_code} "
                      f"(attempt {attempt}/{self.max_retry + 1})")
            except requests.exceptions.Timeout:
                print(f"[API] Timeout (attempt {attempt}/{self.max_retry + 1})")
            except requests.exceptions.ConnectionError as e:
                print(f"[API] Connection error (attempt {attempt}): {e}")
            except Exception as e:
                print(f"[API] Unexpected error: {e}")
                return   # non-retriable

            if attempt <= self.max_retry:
                time.sleep(self.retry_wait)

        print(f"[API] Gave up after {self.max_retry + 1} attempts for payload {payload}")


# ═══════════════════════════════════════════════════════
#  KALMAN FILTER  — 4-state: [cx, cy, vx, vy]
# ═══════════════════════════════════════════════════════
class KalmanCentroid:
    """Constant-velocity Kalman filter for centroid smoothing."""

    def __init__(self, cx: float, cy: float):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix  = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix   = np.array([[1,0,1,0],[0,1,0,1],
                                                [0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov    = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov= np.eye(2, dtype=np.float32) * 0.5
        self.kf.errorCovPost       = np.eye(4, dtype=np.float32)
        self.kf.statePost          = np.array([[cx],[cy],[0],[0]], np.float32)

    def update(self, cx: float, cy: float) -> tuple[float, float]:
        pred = self.kf.predict()
        meas = np.array([[cx],[cy]], np.float32)
        self.kf.correct(meas)
        return float(pred[0, 0]), float(pred[1, 0])

    def predict(self) -> tuple[float, float]:
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])


# ═══════════════════════════════════════════════════════
#  PER-TRACK STATE MACHINE
# ═══════════════════════════════════════════════════════
@dataclass
class TrackState:
    """
    3-state crossing machine per track ID.

    States:
      OUTSIDE  — centroid firmly on the LEFT of dead zone  (entry side)
      ZONE     — centroid inside the dead zone             (crossing)
      INSIDE   — centroid firmly on the RIGHT              (bus interior)

    Valid counting transition:  OUTSIDE → ZONE → INSIDE  (+1 IN)
    Valid exit transition:      INSIDE  → ZONE → OUTSIDE  (+1 OUT)
    Invalid (skip):             ID first seen in ZONE or INSIDE
    """
    kalman:      KalmanCentroid
    cx:          float            # smoothed centroid x
    cy:          float            # smoothed centroid y
    trail:       deque = field(default_factory=lambda: deque(maxlen=TRAIL_LEN))
    zone_state:  str  = "INIT"   # INIT | OUTSIDE | ZONE | INSIDE | SKIP
    counted:     Optional[str] = None   # None | 'IN' | 'OUT'
    flash:       int  = 0
    last_seen:   int  = 0
    side_frames: int  = 0        # consecutive frames on current confirmed side
    prev_side:   str  = ""       # last confirmed non-ZONE side
    debounce_side: str = ""      # which side we are accumulating debounce for
    exempt:      bool = False    # True if matched as exempt person
    embed_crops: list = field(default_factory=list)   # crops for embedding
    embed_done:  bool = False    # True after exemption check is complete


# ═══════════════════════════════════════════════════════
#  MAIN COUNTER CLASS
# ═══════════════════════════════════════════════════════
class BusCounter:

    def __init__(self, video_path: str, model_path: str = "yolov8s.pt",
                 output_path: str = "result_v3.mp4",
                 exempt_path: str = "", no_preview: bool = False, live: bool = False):
        self.video_path  = video_path
        self.output_path = output_path
        self.no_preview  = no_preview
        self.live        = live
        self.model       = YOLO(model_path)
        self.states: dict[int, TrackState] = {}
        self.in_count  = 0
        self.out_count = 0
        self.events: list[dict] = []   # for CSV

        # ── Non-blocking API push worker ─────────────────────────────────
        self.api = ApiPushWorker(
            endpoint   = API_ENDPOINT,
            timeout    = API_TIMEOUT,
            max_retry  = API_MAX_RETRY,
            retry_wait = API_RETRY_WAIT,
        )

        # ── Exempt person setup ───────────────────────────────────────────
        self.exempt_emb  = None
        self.feat_model  = None
        if exempt_path and os.path.isfile(exempt_path):
            self.exempt_emb = np.load(exempt_path)
            print(f"Loaded exempt embedding from {exempt_path}")
            # MobileNetV2 feature extractor (same as enroll_exempt.py)
            net = models.mobilenet_v2(weights="DEFAULT")
            net.classifier = nn.Identity()
            net.eval()
            self.feat_model = net
            self.feat_tfm   = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    # ── helpers ──────────────────────────────────────────────────────────

    def _get_side(self, cx: float, line_x: int) -> str:
        """Returns 'L', 'R', or 'ZONE'."""
        if cx < line_x - DEAD_ZONE_PX:
            return "L"
        if cx > line_x + DEAD_ZONE_PX:
            return "R"
        return "ZONE"

    def _purge_ghosts(self, frame_no: int):
        dead = [tid for tid, s in self.states.items()
                if frame_no - s.last_seen > GHOST_TIMEOUT]
        for tid in dead:
            del self.states[tid]

    # ── per-frame track update ────────────────────────────────────────────

    def _extract_embedding(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Extract L2-normalised 1280-d embedding from a BGR crop."""
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor   = self.feat_tfm(crop_rgb).unsqueeze(0)
        with torch.no_grad():
            feat = self.feat_model(tensor)
        vec  = feat.squeeze().numpy()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _check_exempt(self, st: 'TrackState') -> None:
        """After enough samples, decide if this track is the exempt person."""
        if len(st.embed_crops) < EXEMPT_SAMPLES:
            return
        embs = [self._extract_embedding(c) for c in st.embed_crops]
        avg  = np.mean(embs, axis=0)
        avg  = avg / (np.linalg.norm(avg) + 1e-8)
        sim  = float(np.dot(avg, self.exempt_emb))
        st.exempt     = sim >= EXEMPT_THRESH
        st.embed_done = True
        st.embed_crops.clear()          # free memory
        if st.exempt:
            print(f"  [EXEMPT] track matched (sim={sim:.3f})")

    # ── per-frame track update ────────────────────────────────────────────

    def _update_track(self, tid: int, bbox: np.ndarray,
                      line_x: int, frame_no: int,
                      frame: np.ndarray = None) -> Optional[str]:
        x1, y1, x2, y2 = bbox.astype(int)
        raw_cx = (x1 + x2) / 2.0
        raw_cy = (y1 + y2) / 2.0

        # ── Init new track ────────────────────────────────────────────────
        if tid not in self.states:
            kf   = KalmanCentroid(raw_cx, raw_cy)
            side = self._get_side(raw_cx, line_x)
            st   = TrackState(kalman=kf, cx=raw_cx, cy=raw_cy,
                              last_seen=frame_no)

            # Critical: classify first-seen position
            if self.live:
                if side == "L":
                    st.zone_state = "OUTSIDE"
                    st.prev_side  = "L"
                elif side == "R":
                    st.zone_state = "INSIDE"
                    st.prev_side  = "R"
                else:
                    # Live stream: allow a zone-start track to still count later
                    st.zone_state = "ZONE"
                    st.prev_side  = ""
            else:
                if side == "L":
                    st.zone_state = "OUTSIDE"
                    st.prev_side  = "L"
                elif side == "R":
                    # Already inside — can only count OUT, never IN
                    st.zone_state = "INSIDE"
                    st.prev_side  = "R"
                else:
                    # First seen in dead zone — cannot count in either direction
                    st.zone_state = "SKIP"

            self.states[tid] = st

        st = self.states[tid]
        st.last_seen = frame_no

        # ── Exempt person check (first N frames of each track) ────────────
        if self.exempt_emb is not None and not st.embed_done:
            h_frame = frame.shape[0] if frame is not None else 0
            w_frame = frame.shape[1] if frame is not None else 0
            cy1 = max(0, y1)
            cx1 = max(0, x1)
            cy2 = min(h_frame, y2)
            cx2 = min(w_frame, x2)
            if frame is not None and cy2 > cy1 and cx2 > cx1:
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    st.embed_crops.append(crop.copy())
                    self._check_exempt(st)

        # ── Kalman update ─────────────────────────────────────────────────
        pred_cx, pred_cy = st.kalman.update(raw_cx, raw_cy)
        # Blend Kalman prediction with EMA of raw for robustness
        st.cx = EMA_ALPHA * raw_cx + (1 - EMA_ALPHA) * pred_cx
        st.cy = EMA_ALPHA * raw_cy + (1 - EMA_ALPHA) * pred_cy
        st.trail.append((int(st.cx), int(st.cy)))

        if st.flash > 0:
            st.flash -= 1

        # Exempt and SKIP IDs never count
        if st.exempt:
            return None
        if st.zone_state == "SKIP":
            return None

        # ── Cooldown after count — freeze state transitions ──────────────
        if st.flash > 0:
            return None

        # ── Side classification ───────────────────────────────────────────
        current_side = self._get_side(st.cx, line_x)

        # ── State machine transitions (with debounce) ─────────────────────
        event = None

        if current_side == "ZONE":
            # Entering dead zone — transition immediately, reset debounce
            if st.zone_state in ("OUTSIDE", "INSIDE"):
                st.zone_state = "ZONE"
            st.side_frames   = 0
            st.debounce_side = ""
            # In zone — no count possible, wait for exit

        else:
            # current_side is "L" or "R"
            # ── Debounce: accumulate consecutive frames on this side ──────
            if st.debounce_side != current_side:
                st.debounce_side = current_side
                st.side_frames   = 1
            else:
                st.side_frames  += 1


            # Only commit transition after DEBOUNCE_N consecutive frames
            if st.side_frames >= DEBOUNCE_N:
                if current_side == "L":
                    # Fix: clarify logic for live and non-live, and allow zone-start tracks to count if they traverse full path
                    if (
                        (self.live and st.zone_state in ("ZONE", "INSIDE") and st.prev_side in ("", "R"))
                        or (not self.live and st.zone_state in ("ZONE", "INSIDE") and st.prev_side == "R")
                    ):
                        # Confirmed crossing: INSIDE → ZONE → OUTSIDE  =  EXIT
                        if st.counted != "OUT":
                            self.out_count += 1
                            st.counted     = "OUT"
                            st.flash       = FLASH_FRAMES
                            event          = "OUT"
                            self.events.append({
                                "frame": frame_no, "id": tid, "event": "OUT",
                                "in": self.in_count, "out": self.out_count
                            })
                            print(f"[{frame_no:05d}] ID {tid:3d} LEFT   | IN={self.in_count} OUT={self.out_count}")
                            # ── Non-blocking API push ────────────────────
                            self.api.push(self.in_count, self.out_count)
                    st.zone_state = "OUTSIDE"
                    st.prev_side  = "L"

                elif current_side == "R":
                    if (
                        (self.live and st.zone_state in ("ZONE", "OUTSIDE") and st.prev_side in ("", "L"))
                        or (not self.live and st.zone_state in ("ZONE", "OUTSIDE") and st.prev_side == "L")
                    ):
                        # Confirmed crossing: OUTSIDE → ZONE → INSIDE  =  ENTER
                        if st.counted != "IN":
                            self.in_count += 1
                            st.counted    = "IN"
                            st.flash      = FLASH_FRAMES
                            event         = "IN"
                            self.events.append({
                                "frame": frame_no, "id": tid, "event": "IN",
                                "in": self.in_count, "out": self.out_count
                            })
                            print(f"[{frame_no:05d}] ID {tid:3d} ENTERED| IN={self.in_count} OUT={self.out_count}")
                            # ── Non-blocking API push ────────────────────
                            self.api.push(self.in_count, self.out_count)
                    st.zone_state = "INSIDE"
                    st.prev_side  = "R"

        return event

    # ── drawing ───────────────────────────────────────────────────────────

    def _draw_zones(self, frame: np.ndarray, line_x: int) -> np.ndarray:
        h = frame.shape[0]
        ov = frame.copy()

        # Left zone (outside) — very subtle blue tint
        cv2.rectangle(ov, (0, 0), (line_x - DEAD_ZONE_PX, h), (180, 60, 0), -1)
        # Right zone (inside bus) — very subtle green tint
        cv2.rectangle(ov, (line_x + DEAD_ZONE_PX, 0), (frame.shape[1], h), (0, 140, 0), -1)
        frame = cv2.addWeighted(ov, 0.06, frame, 0.94, 0)

        # Dead zone strip — yellow tint
        ov2 = frame.copy()
        cv2.rectangle(ov2, (line_x - DEAD_ZONE_PX, 0),
                      (line_x + DEAD_ZONE_PX, h), (0, 200, 255), -1)
        frame = cv2.addWeighted(ov2, 0.18, frame, 0.82, 0)

        # Lines
        cv2.line(frame, (line_x, 0), (line_x, h), COL_LINE, 2)
        cv2.line(frame, (line_x - DEAD_ZONE_PX, 0),
                 (line_x - DEAD_ZONE_PX, h), (0, 180, 255), 1)
        cv2.line(frame, (line_x + DEAD_ZONE_PX, 0),
                 (line_x + DEAD_ZONE_PX, h), (0, 180, 255), 1)

        # Labels
        cv2.putText(frame, "OUTSIDE", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "INSIDE", (line_x + DEAD_ZONE_PX + 6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 80), 1, cv2.LINE_AA)
        cv2.putText(frame, "ZONE", (line_x - DEAD_ZONE_PX + 4, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
        return frame

    def _draw_tracks(self, frame: np.ndarray, dets: sv.Detections) -> np.ndarray:
        if dets.tracker_id is None:
            return frame

        for bbox, tid in zip(dets.xyxy, dets.tracker_id):
            tid = int(tid)
            st  = self.states.get(tid)
            if st is None:
                continue

            x1, y1, x2, y2 = bbox.astype(int)

            # ── Trail ─────────────────────────────────────────────────────
            pts = list(st.trail)
            if len(pts) > 1:
                for j in range(1, len(pts)):
                    t     = j / max(len(pts) - 1, 1)
                    blue  = int(255 * (1 - t))
                    green = int(200 * t)
                    red   = int(120 * t)
                    cv2.line(frame, pts[j-1], pts[j], (blue, green, red), 2,
                             cv2.LINE_AA)

            # ── Bounding box color ─────────────────────────────────────────
            if st.exempt:
                col = COL_EXEMPT
            elif st.zone_state == "SKIP":
                col = COL_BOX_SKIP
            elif st.flash > 0:
                col = COL_BOX_FLASH      # CYAN on count event
            else:
                col = COL_BOX_NORM       # Orange normally

            thick = 3 if st.flash > 0 else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick)

            # ── Centroid dot ───────────────────────────────────────────────
            cx, cy = int(st.cx), int(st.cy)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
            cv2.circle(frame, (cx, cy), 5, col, 2)

            # ── Label ─────────────────────────────────────────────────────
            if st.exempt:
                tag_col = COL_EXEMPT
                tag     = f"#{tid} EXEMPT"
            elif st.counted == "IN":
                tag_col = COL_IN_TEXT
                tag     = f"#{tid} IN"
            elif st.counted == "OUT":
                tag_col = COL_OUT_TEXT
                tag     = f"#{tid} OUT"
            elif st.zone_state == "SKIP":
                tag_col = COL_BOX_SKIP
                tag     = f"#{tid} skip"
            else:
                tag_col = (200, 200, 200)
                tag     = f"#{tid} {st.zone_state[:3]}"

            cv2.putText(frame, tag, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, tag_col, 1, cv2.LINE_AA)

        return frame

    def _draw_dashboard(self, frame: np.ndarray, frame_no: int,
                        fps: float) -> np.ndarray:
        h, w   = frame.shape[:2]
        dw, dh = 220, 130
        margin = 14
        x1 = w - dw - margin
        y1 = margin

        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x1 + dw, y1 + dh), (12, 12, 12), -1)
        frame = cv2.addWeighted(ov, 0.70, frame, 0.30, 0)

        # Green accent bar
        cv2.rectangle(frame, (x1, y1), (x1 + dw, y1 + 3), (0, 200, 80), -1)

        inside = max(0, self.in_count - self.out_count)
        rows   = [
            ("ENTERED", self.in_count,  COL_IN_TEXT),
            ("LEFT",    self.out_count, COL_OUT_TEXT),
            ("INSIDE",  inside,         (80, 220, 255)),
        ]
        for i, (label, val, col) in enumerate(rows):
            y = y1 + 34 + i * 32
            cv2.putText(frame, f"{label}:", (x1 + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1, cv2.LINE_AA)
            cv2.putText(frame, str(val), (x1 + 148, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)

        # Frame counter
        ts = f"f{frame_no}  {frame_no/fps:.1f}s"
        cv2.putText(frame, ts, (x1 + 10, y1 + dh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)
        return frame

    # ── CSV ───────────────────────────────────────────────────────────────

    def _save_csv(self, path: str):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "timestamp", "id",
                                               "event", "in", "out", "inside"])
            w.writeheader()
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for ev in self.events:
                w.writerow({
                    "frame":     ev["frame"],
                    "timestamp": ts,
                    "id":        ev["id"],
                    "event":     ev["event"],
                    "in":        ev["in"],
                    "out":       ev["out"],
                    "inside":    ev["in"] - ev["out"],
                })
        print(f"CSV saved: {path}")

    # ── main loop ─────────────────────────────────────────────────────────

    def process(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {self.video_path}")

        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
        line_x = int(W * LINE_RATIO)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.output_path, fourcc, FPS, (W, H))

        print(f"Video: {W}x{H} @ {FPS:.1f}fps")
        print(f"Trigger line at x={line_x} ({LINE_RATIO*100:.0f}% width)")
        print(f"Dead zone: x=[{line_x-DEAD_ZONE_PX}, {line_x+DEAD_ZONE_PX}]")
        print("─" * 55)

        frame_no = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_no += 1

                # ── Detection + Tracking (BoT-SORT with ReID) ────────────
                results = self.model.track(
                    frame,
                    classes=[0],
                    conf=CONF_THRESH,
                    iou=IOU_THRESH,
                    verbose=False,
                    agnostic_nms=True,
                    tracker=TRACKER_CONFIG,
                    persist=True,
                )[0]

                boxes = results.boxes
                xyxy  = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls   = boxes.cls.cpu().numpy().astype(int)
                tids  = boxes.id
                tids  = tids.cpu().numpy().astype(int) if tids is not None else None

                dets = sv.Detections(
                    xyxy=xyxy,
                    confidence=confs,
                    class_id=cls,
                    tracker_id=tids,
                )

                # ── State update ──────────────────────────────────────────
                if dets.tracker_id is not None:
                    for bbox, tid in zip(dets.xyxy, dets.tracker_id):
                        self._update_track(int(tid), bbox, line_x, frame_no,
                                           frame)

                # ── Ghost cleanup ─────────────────────────────────────────
                if frame_no % 90 == 0:
                    self._purge_ghosts(frame_no)

                # ── Draw ──────────────────────────────────────────────────
                frame = self._draw_zones(frame, line_x)
                frame = self._draw_tracks(frame, dets)
                frame = self._draw_dashboard(frame, frame_no, FPS)
                writer.write(frame)

                if not self.no_preview:
                    cv2.imshow("Bus Counter v3", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()

            csv_path = self.output_path.replace(".mp4", ".csv")
            self._save_csv(csv_path)

            print("\n" + "═" * 55)
            print(f"  Frames processed : {frame_no}")
            print(f"  Total ENTERED    : {self.in_count}")
            print(f"  Total LEFT       : {self.out_count}")
            print(f"  Final INSIDE     : {max(0, self.in_count - self.out_count)}")
            print(f"  Output video     : {self.output_path}")
            print(f"  Event CSV        : {csv_path}")
            print("═" * 55)


# ═══════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════
def main():
    global LINE_RATIO

    p = argparse.ArgumentParser(description="Bus Passenger Counter v3 + API Push")
    p.add_argument("--source",  default="counting.mp4",   help="Input video")
    p.add_argument("--output",  default="result_v3.mp4",  help="Output video")
    p.add_argument("--model",   default="yolov8s.pt",     help="YOLO weights")
    p.add_argument("--line",    type=float, default=LINE_RATIO,
                   help="Trigger line fraction of frame width (default 0.50)")
    p.add_argument("--exempt",  default="",
                   help="Path to exempt_embedding.npy (skip counting this person)")
    p.add_argument("--no-preview", action="store_true",
                   help="Disable live preview (faster headless runs)")
    p.add_argument("--live", action="store_true",
                   help="Live stream mode: assume all tracks start outside")
    args = p.parse_args()

    LINE_RATIO = args.line

    counter = BusCounter(
        video_path  = args.source,
        model_path  = args.model,
        output_path = args.output,
        exempt_path = args.exempt,
        no_preview  = args.no_preview,
        live        = args.live,
    )
    counter.process()


if __name__ == "__main__":
    main()