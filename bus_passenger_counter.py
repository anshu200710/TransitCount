#!/usr/bin/env python3
"""
Bus Passenger Counter — FINAL (Badge Conductor + API Push)
===========================================================
Camera : overhead/top-down at bus front gate
Direction: LEFT → RIGHT = ENTER  |  RIGHT → LEFT = EXIT

What's combined from both versions:
  ✓ Full 3-state machine: OUTSIDE → ZONE → INSIDE (file-1 logic)
  ✓ Kalman filter + EMA centroid smoothing              (file-1)
  ✓ Debounce-N consecutive frames before counting       (file-1)
  ✓ Live-stream mode (--live flag)                      (file-1)
  ✓ Non-blocking API push via background thread         (file-1)
  ✓ Ghost-ID cleanup, CSV logging, CYAN flash           (file-1)
  ✓ Badge detection: saffron + green HSV proximity      (file-2)
  ✓ Badge-exempt tracks are NEVER counted               (file-2)
  ✓ Optional .npy embedding exempt as fallback          (file-1)
  ✓ BoT-SORT YAML tracker config                        (both)
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
import threading, queue, time

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False
    print("[WARN] 'requests' not found — API push disabled. pip install requests")

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


# ═══════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════
LINE_RATIO      = 0.45   # trigger line as fraction of frame width
DEAD_ZONE_PX    = 30     # pixels either side of line = dead zone
DEBOUNCE_N      = 3      # consecutive frames required to confirm side
CONF_THRESH     = 0.12   # YOLO detection confidence
IOU_THRESH      = 0.40   # NMS IoU threshold
TRAIL_LEN       = 50     # centroid trail length
GHOST_TIMEOUT   = 150    # frames before removing unseen track
FLASH_FRAMES    = 20     # cyan-flash duration on count event
EMA_ALPHA       = 0.40   # blend factor: raw vs Kalman

# ── Badge (conductor) detection ───────────────────────────────────────────
BADGE_MIN_PX        = 20     # minimum saffron pixels to trigger badge check
BADGE_CONFIRM_N     = 3      # consecutive badge-positive frames to confirm exempt
# HSV ranges — tweak if lighting differs
BADGE_SAFFRON_LO    = np.array([ 0, 140,  80], np.uint8)
BADGE_SAFFRON_HI    = np.array([25, 255, 255], np.uint8)
BADGE_GREEN_LO      = np.array([60,  50,  40], np.uint8)
BADGE_GREEN_HI      = np.array([95, 255, 255], np.uint8)
BADGE_DEBUG         = True   # set False to silence per-frame HSV prints

# ── Embedding-based exempt (optional fallback) ────────────────────────────
EXEMPT_THRESH   = 0.70   # cosine similarity threshold
EXEMPT_SAMPLES  = 5      # frames before deciding embed-based exemption

# ── API ───────────────────────────────────────────────────────────────────
API_ENDPOINT    = "https://7d90-49-205-176-68.ngrok-free.app/api/passenger-count"
API_TIMEOUT     = 2
API_MAX_RETRY   = 2
API_RETRY_WAIT  = 0.3

# ── Tracker ───────────────────────────────────────────────────────────────
TRACKER_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "botsort.yaml")

# ── Colours (BGR) ─────────────────────────────────────────────────────────
COL_LINE       = (  0,   0, 220)
COL_BOX_NORM   = (  0, 140, 255)
COL_BOX_FLASH  = (255, 255,   0)
COL_BOX_SKIP   = ( 80,  80,  80)
COL_IN_TEXT    = ( 80, 255,  80)
COL_OUT_TEXT   = ( 80, 180, 255)
COL_EXEMPT     = (  0, 200, 255)   # yellow — conductor


# ═══════════════════════════════════════════════════════
#  NON-BLOCKING API WORKER
# ═══════════════════════════════════════════════════════
class ApiPushWorker:
    """Fires POST requests from a background daemon thread — zero FPS impact."""

    def __init__(self, endpoint, timeout=API_TIMEOUT,
                 max_retry=API_MAX_RETRY, retry_wait=API_RETRY_WAIT):
        self.endpoint   = endpoint
        self.timeout    = timeout
        self.max_retry  = max_retry
        self.retry_wait = retry_wait
        self._q         = queue.Queue()
        self._enabled   = _REQUESTS_OK
        if self._enabled:
            threading.Thread(target=self._worker, daemon=True).start()
            print(f"[API] Worker started → {endpoint}")
        else:
            print("[API] Worker DISABLED (requests not installed)")

    def push(self, in_count: int, out_count: int):
        """Enqueue payload — never blocks the caller."""
        if not self._enabled:
            return
        self._q.put({
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hin":      in_count,
            "hout":     out_count,
            "inside":   max(0, in_count - out_count),
            "total":    in_count,
        })

    def _worker(self):
        while True:
            payload = self._q.get()
            self._send_with_retry(payload)
            self._q.task_done()

    def _send_with_retry(self, payload):
        for attempt in range(1, self.max_retry + 2):
            try:
                r = requests.post(self.endpoint, json=payload,
                                  timeout=self.timeout,
                                  headers={"Content-Type": "application/json"})
                if r.status_code < 500:
                    if r.status_code >= 400:
                        print(f"[API] Client error {r.status_code} — not retrying")
                    else:
                        print(f"[API] ✓ {r.status_code}  "
                              f"hin={payload['hin']} hout={payload['hout']} "
                              f"inside={payload['inside']}")
                    return
                print(f"[API] Server {r.status_code} (attempt {attempt})")
            except requests.exceptions.Timeout:
                print(f"[API] Timeout (attempt {attempt})")
            except requests.exceptions.ConnectionError as e:
                print(f"[API] Connection error: {e}")
            except Exception as e:
                print(f"[API] Unexpected: {e}"); return
            if attempt <= self.max_retry:
                time.sleep(self.retry_wait)
        print(f"[API] Gave up after {self.max_retry + 1} attempts")


# ═══════════════════════════════════════════════════════
#  KALMAN FILTER  — [cx, cy, vx, vy]
# ═══════════════════════════════════════════════════════
class KalmanCentroid:
    def __init__(self, cx: float, cy: float):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],
                                                [0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self.kf.statePost           = np.array([[cx],[cy],[0],[0]], np.float32)

    def update(self, cx: float, cy: float):
        pred = self.kf.predict()
        self.kf.correct(np.array([[cx],[cy]], np.float32))
        return float(pred[0,0]), float(pred[1,0])


# ═══════════════════════════════════════════════════════
#  PER-TRACK STATE
# ═══════════════════════════════════════════════════════
@dataclass
class TrackState:
    """
    3-state crossing machine.

    OUTSIDE → ZONE → INSIDE  =  +1 IN
    INSIDE  → ZONE → OUTSIDE  =  +1 OUT
    SKIP / EXEMPT             =  ignored
    """
    kalman:        KalmanCentroid
    cx:            float
    cy:            float
    trail:         deque = field(default_factory=lambda: deque(maxlen=TRAIL_LEN))
    zone_state:    str   = "INIT"          # INIT|OUTSIDE|ZONE|INSIDE|SKIP
    counted:       Optional[str] = None    # None | 'IN' | 'OUT'
    flash:         int   = 0
    last_seen:     int   = 0
    side_frames:   int   = 0
    prev_side:     str   = ""
    debounce_side: str   = ""
    # ── Conductor badge ───────────────────────────────────────────────────
    exempt:        bool  = False
    badge_hits:    int   = 0               # consecutive frames with badge visible
    # ── Embedding fallback ────────────────────────────────────────────────
    embed_crops:   list  = field(default_factory=list)
    embed_done:    bool  = False


# ═══════════════════════════════════════════════════════
#  MAIN COUNTER
# ═══════════════════════════════════════════════════════
class BusCounter:

    def __init__(self, video_path: str,
                 model_path:   str  = "yolov8s.pt",
                 output_path:  str  = "result_final.mp4",
                 exempt_path:  str  = "",
                 no_preview:   bool = False,
                 live:         bool = False):
        self.video_path  = video_path
        self.output_path = output_path
        self.no_preview  = no_preview
        self.live        = live
        self.model       = YOLO(model_path)
        self.states: dict[int, TrackState] = {}
        self.in_count    = 0
        self.out_count   = 0
        self.events:     list[dict] = []

        self.api = ApiPushWorker(API_ENDPOINT, API_TIMEOUT,
                                 API_MAX_RETRY, API_RETRY_WAIT)

        # ── Optional embedding-based exemption ────────────────────────────
        self.exempt_emb  = None
        self.feat_model  = None
        if exempt_path and os.path.isfile(exempt_path) and _TORCH_OK:
            self.exempt_emb = np.load(exempt_path)
            print(f"[EXEMPT] Embedding loaded from {exempt_path}")
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
        elif exempt_path and not _TORCH_OK:
            print("[WARN] torch/torchvision not available — embedding exempt disabled")

    # ─────────────────────────────────────────────────────────────────────
    #  BADGE DETECTION  (saffron + green proximity check)
    # ─────────────────────────────────────────────────────────────────────

    def _has_badge(self, crop_bgr: np.ndarray, tid: int) -> bool:
        """
        Returns True when a conductor badge (saffron + green) is present.

        Strategy:
          1. Convert crop to HSV.
          2. Threshold saffron (orange-red) and green separately.
          3. Dilate the saffron mask to bridge small gaps.
          4. Check spatial proximity: dilated-saffron overlaps with green.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return False

        # Reject crops too small to contain a badge reliably
        h_crop, w_crop = crop_bgr.shape[:2]
        if h_crop < 20 or w_crop < 10:
            return False

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

        # ── Optional calibration debug ─────────────────────────────────────
        if BADGE_DEBUG:
            m_s_all = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([25, 255, 255]))
            m_g_all = cv2.inRange(hsv, np.array([60, 50, 40]), np.array([95, 255, 255]))
            max_s = int(np.max(hsv[:,:,1], where=m_s_all > 0, initial=0))
            max_g = int(np.max(hsv[:,:,1], where=m_g_all > 0, initial=0))
            if max_s > 100 or max_g > 50:
                print(f"  [BADGE DEBUG ID {tid:2d}] Peak Sat → Saffron:{max_s}  Green:{max_g}")

        # ── Strict masks ───────────────────────────────────────────────────
        mask_s = cv2.inRange(hsv, BADGE_SAFFRON_LO, BADGE_SAFFRON_HI)   # saffron
        mask_g = cv2.inRange(hsv, BADGE_GREEN_LO,   BADGE_GREEN_HI)     # green

        saffron_pixels = int(np.sum(mask_s > 0))
        green_pixels   = int(np.sum(mask_g > 0))

        if saffron_pixels < BADGE_MIN_PX or green_pixels < 3:
            return False

        # ── Proximity check: dilated saffron must touch green ──────────────
        kernel    = np.ones((7, 7), np.uint8)
        dilated_s = cv2.dilate(mask_s, kernel, iterations=2)
        overlap   = int(np.sum(cv2.bitwise_and(dilated_s, mask_g) > 0))

        return overlap > 1

    # ─────────────────────────────────────────────────────────────────────
    #  EMBEDDING-BASED EXEMPT CHECK  (optional)
    # ─────────────────────────────────────────────────────────────────────

    def _extract_embedding(self, crop_bgr: np.ndarray) -> np.ndarray:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor   = self.feat_tfm(crop_rgb).unsqueeze(0)
        with torch.no_grad():
            feat = self.feat_model(tensor)
        vec  = feat.squeeze().numpy()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _check_embed_exempt(self, st: TrackState) -> None:
        if len(st.embed_crops) < EXEMPT_SAMPLES:
            return
        embs = [self._extract_embedding(c) for c in st.embed_crops]
        avg  = np.mean(embs, axis=0)
        avg  = avg / (np.linalg.norm(avg) + 1e-8)
        sim  = float(np.dot(avg, self.exempt_emb))
        st.exempt     = sim >= EXEMPT_THRESH
        st.embed_done = True
        st.embed_crops.clear()
        if st.exempt:
            print(f"  [EXEMPT-EMB] track matched (cosine={sim:.3f})")

    # ─────────────────────────────────────────────────────────────────────
    #  SIDE HELPER
    # ─────────────────────────────────────────────────────────────────────

    def _get_side(self, cx: float, line_x: int) -> str:
        if cx < line_x - DEAD_ZONE_PX: return "L"
        if cx > line_x + DEAD_ZONE_PX: return "R"
        return "ZONE"

    # ─────────────────────────────────────────────────────────────────────
    #  PER-TRACK UPDATE  (counting state machine)
    # ─────────────────────────────────────────────────────────────────────

    def _update_track(self, tid: int, bbox: np.ndarray,
                      line_x: int, frame_no: int,
                      frame: np.ndarray) -> Optional[str]:

        x1, y1, x2, y2 = bbox.astype(int)
        raw_cx = (x1 + x2) / 2.0
        raw_cy = (y1 + y2) / 2.0

        # ── Init new track ────────────────────────────────────────────────
        if tid not in self.states:
            side = self._get_side(raw_cx, line_x)
            st   = TrackState(kalman=KalmanCentroid(raw_cx, raw_cy),
                              cx=raw_cx, cy=raw_cy, last_seen=frame_no)
            if self.live:
                if   side == "L":    st.zone_state, st.prev_side = "OUTSIDE", "L"
                elif side == "R":    st.zone_state, st.prev_side = "INSIDE",  "R"
                else:                st.zone_state = "ZONE"
            else:
                if   side == "L":    st.zone_state, st.prev_side = "OUTSIDE", "L"
                elif side == "R":    st.zone_state, st.prev_side = "INSIDE",  "R"
                else:                st.zone_state = "SKIP"
            self.states[tid] = st

        st           = self.states[tid]
        st.last_seen = frame_no

        # ── Crop for badge / embedding checks ─────────────────────────────
        crop = None
        if frame is not None:
            fy1 = max(0, y1); fy2 = min(frame.shape[0], y2)
            fx1 = max(0, x1); fx2 = min(frame.shape[1], x2)
            if fy2 > fy1 and fx2 > fx1:
                crop = frame[fy1:fy2, fx1:fx2]

        # ── Badge-based exemption (primary conductor detection) ────────────
        if not st.exempt:
            if crop is not None and self._has_badge(crop, tid):
                st.badge_hits += 1
                if st.badge_hits >= BADGE_CONFIRM_N:
                    st.exempt = True
                    print(f"  [CONDUCTOR] ID {tid} — badge confirmed "
                          f"(hit streak={st.badge_hits})")
            else:
                # reset streak on a miss (avoids false positives from 1 bad frame)
                st.badge_hits = max(0, st.badge_hits - 1)

        # ── Embedding-based exemption (optional fallback) ─────────────────
        if (not st.exempt and self.exempt_emb is not None
                and _TORCH_OK and not st.embed_done):
            if crop is not None and crop.size > 0:
                st.embed_crops.append(crop.copy())
                self._check_embed_exempt(st)

        # ── Kalman + EMA smoothing ─────────────────────────────────────────
        pred_cx, pred_cy = st.kalman.update(raw_cx, raw_cy)
        st.cx = EMA_ALPHA * raw_cx + (1 - EMA_ALPHA) * pred_cx
        st.cy = EMA_ALPHA * raw_cy + (1 - EMA_ALPHA) * pred_cy
        st.trail.append((int(st.cx), int(st.cy)))
        if st.flash > 0:
            st.flash -= 1

        # ── Exempt and SKIP tracks never count ────────────────────────────
        if st.exempt or st.zone_state == "SKIP":
            return None

        # ── Cooldown: freeze state machine right after a count ────────────
        if st.flash > 0:
            return None

        # ── State machine ─────────────────────────────────────────────────
        current_side = self._get_side(st.cx, line_x)
        event        = None

        if current_side == "ZONE":
            if st.zone_state in ("OUTSIDE", "INSIDE"):
                st.zone_state = "ZONE"
            st.side_frames    = 0
            st.debounce_side  = ""

        else:
            # Accumulate debounce
            if st.debounce_side != current_side:
                st.debounce_side = current_side
                st.side_frames   = 1
            else:
                st.side_frames  += 1

            if st.side_frames >= DEBOUNCE_N:

                if current_side == "L":
                    # ── EXIT: INSIDE → ZONE → OUTSIDE ─────────────────────
                    valid_exit = (
                        (    self.live and st.zone_state in ("ZONE","INSIDE")
                                      and st.prev_side  in ("","R"))
                        or (not self.live and st.zone_state in ("ZONE","INSIDE")
                                         and st.prev_side == "R")
                    )
                    if valid_exit and st.counted != "OUT":
                        self.out_count += 1
                        st.counted, st.flash = "OUT", FLASH_FRAMES
                        event = "OUT"
                        self.events.append({"frame": frame_no, "id": tid,
                                            "event": "OUT",
                                            "in": self.in_count,
                                            "out": self.out_count})
                        print(f"[{frame_no:05d}] ID {tid:3d} LEFT    | "
                              f"IN={self.in_count} OUT={self.out_count}")
                        self.api.push(self.in_count, self.out_count)
                    st.zone_state, st.prev_side = "OUTSIDE", "L"

                elif current_side == "R":
                    # ── ENTER: OUTSIDE → ZONE → INSIDE ────────────────────
                    valid_enter = (
                        (    self.live and st.zone_state in ("ZONE","OUTSIDE")
                                      and st.prev_side  in ("","L"))
                        or (not self.live and st.zone_state in ("ZONE","OUTSIDE")
                                         and st.prev_side == "L")
                    )
                    if valid_enter and st.counted != "IN":
                        self.in_count += 1
                        st.counted, st.flash = "IN", FLASH_FRAMES
                        event = "IN"
                        self.events.append({"frame": frame_no, "id": tid,
                                            "event": "IN",
                                            "in": self.in_count,
                                            "out": self.out_count})
                        print(f"[{frame_no:05d}] ID {tid:3d} ENTERED | "
                              f"IN={self.in_count} OUT={self.out_count}")
                        self.api.push(self.in_count, self.out_count)
                    st.zone_state, st.prev_side = "INSIDE", "R"

        return event

    # ─────────────────────────────────────────────────────────────────────
    #  DRAWING HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def _draw_zones(self, frame: np.ndarray, line_x: int) -> np.ndarray:
        h, w = frame.shape[:2]
        ov   = frame.copy()
        # Subtle tinted overlays
        cv2.rectangle(ov, (0, 0), (line_x - DEAD_ZONE_PX, h), (180, 60, 0), -1)
        cv2.rectangle(ov, (line_x + DEAD_ZONE_PX, 0), (w, h), (0, 140, 0), -1)
        frame = cv2.addWeighted(ov, 0.06, frame, 0.94, 0)
        ov2   = frame.copy()
        cv2.rectangle(ov2, (line_x - DEAD_ZONE_PX, 0),
                      (line_x + DEAD_ZONE_PX, h), (0, 200, 255), -1)
        frame = cv2.addWeighted(ov2, 0.18, frame, 0.82, 0)
        # Lines + labels
        cv2.line(frame, (line_x, 0), (line_x, h), COL_LINE, 2)
        cv2.line(frame, (line_x - DEAD_ZONE_PX, 0),
                 (line_x - DEAD_ZONE_PX, h), (0, 180, 255), 1)
        cv2.line(frame, (line_x + DEAD_ZONE_PX, 0),
                 (line_x + DEAD_ZONE_PX, h), (0, 180, 255), 1)
        cv2.putText(frame, "OUTSIDE", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,255), 1, cv2.LINE_AA)
        cv2.putText(frame, "ZONE",    (line_x - DEAD_ZONE_PX + 4, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,220,255), 1, cv2.LINE_AA)
        cv2.putText(frame, "INSIDE",  (line_x + DEAD_ZONE_PX + 6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,255,80), 1, cv2.LINE_AA)
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
            # Trail
            pts = list(st.trail)
            for j in range(1, len(pts)):
                t = j / max(len(pts) - 1, 1)
                cv2.line(frame, pts[j-1], pts[j],
                         (int(255*(1-t)), int(200*t), int(120*t)), 2, cv2.LINE_AA)
            # Box colour
            if st.exempt:
                col = COL_EXEMPT
            elif st.zone_state == "SKIP":
                col = COL_BOX_SKIP
            elif st.flash > 0:
                col = COL_BOX_FLASH
            else:
                col = COL_BOX_NORM
            thick = 3 if st.flash > 0 else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick)
            # Centroid dot
            cv2.circle(frame, (int(st.cx), int(st.cy)), 5, (255,255,255), -1)
            cv2.circle(frame, (int(st.cx), int(st.cy)), 5, col, 2)
            # Label
            if st.exempt:
                tag, tag_col = f"#{tid} CONDUCTOR", COL_EXEMPT
            elif st.counted == "IN":
                tag, tag_col = f"#{tid} IN",   COL_IN_TEXT
            elif st.counted == "OUT":
                tag, tag_col = f"#{tid} OUT",  COL_OUT_TEXT
            elif st.zone_state == "SKIP":
                tag, tag_col = f"#{tid} skip", COL_BOX_SKIP
            else:
                tag, tag_col = f"#{tid} {st.zone_state[:3]}", (200,200,200)
            cv2.putText(frame, tag, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, tag_col, 1, cv2.LINE_AA)
        return frame

    def _draw_dashboard(self, frame: np.ndarray,
                        frame_no: int, fps: float) -> np.ndarray:
        h, w   = frame.shape[:2]
        dw, dh = 220, 130
        x1 = w - dw - 14;  y1 = 14
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x1 + dw, y1 + dh), (12, 12, 12), -1)
        frame = cv2.addWeighted(ov, 0.70, frame, 0.30, 0)
        cv2.rectangle(frame, (x1, y1), (x1 + dw, y1 + 3), (0, 200, 80), -1)
        inside = max(0, self.in_count - self.out_count)
        rows   = [
            ("ENTERED",   self.in_count,  COL_IN_TEXT),
            ("LEFT",      self.out_count, COL_OUT_TEXT),
            ("INSIDE",    inside,         (80, 220, 255)),
        ]
        for i, (label, val, col) in enumerate(rows):
            y = y1 + 34 + i * 32
            cv2.putText(frame, f"{label}:", (x1 + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160,160,160), 1, cv2.LINE_AA)
            cv2.putText(frame, str(val),    (x1 + 148, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)
        cv2.putText(frame,
                    f"f{frame_no}  {frame_no/fps:.1f}s",
                    (x1 + 10, y1 + dh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,80,80), 1, cv2.LINE_AA)
        return frame

    # ─────────────────────────────────────────────────────────────────────
    #  CSV
    # ─────────────────────────────────────────────────────────────────────

    def _save_csv(self, path: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame","timestamp","id",
                                               "event","in","out","inside"])
            w.writeheader()
            for ev in self.events:
                w.writerow({"frame": ev["frame"], "timestamp": ts,
                             "id": ev["id"], "event": ev["event"],
                             "in": ev["in"], "out": ev["out"],
                             "inside": ev["in"] - ev["out"]})
        print(f"CSV saved: {path}")

    # ─────────────────────────────────────────────────────────────────────
    #  MAIN LOOP
    # ─────────────────────────────────────────────────────────────────────

    def process(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {self.video_path}")

        W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        line_x = int(W * LINE_RATIO)

        writer = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS, (W, H)
        )

        print(f"Video  : {W}x{H} @ {FPS:.1f}fps")
        print(f"Line   : x={line_x}  ({LINE_RATIO*100:.0f}% width)")
        print(f"Zone   : x=[{line_x - DEAD_ZONE_PX}, {line_x + DEAD_ZONE_PX}]")
        print(f"Badge  : saffron+green proximity | confirm={BADGE_CONFIRM_N} frames")
        print("─" * 58)

        frame_no = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_no += 1

                # ── Detect + Track ─────────────────────────────────────────
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
                tids  = boxes.id
                tids_np = tids.cpu().numpy().astype(int) if tids is not None else None

                dets = sv.Detections(
                    xyxy=boxes.xyxy.cpu().numpy(),
                    confidence=boxes.conf.cpu().numpy(),
                    class_id=boxes.cls.cpu().numpy().astype(int),
                    tracker_id=tids_np,
                )

                # ── State machine update ───────────────────────────────────
                if dets.tracker_id is not None:
                    for bbox, tid in zip(dets.xyxy, dets.tracker_id):
                        self._update_track(int(tid), bbox, line_x,
                                           frame_no, frame)

                # ── Ghost cleanup ──────────────────────────────────────────
                if frame_no % 90 == 0:
                    self.states = {
                        tid: s for tid, s in self.states.items()
                        if frame_no - s.last_seen < GHOST_TIMEOUT
                    }

                # ── Render ────────────────────────────────────────────────
                frame = self._draw_zones(frame, line_x)
                frame = self._draw_tracks(frame, dets)
                frame = self._draw_dashboard(frame, frame_no, FPS)
                writer.write(frame)

                if not self.no_preview:
                    cv2.imshow("Bus Counter — FINAL", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()

            csv_path = self.output_path.replace(".mp4", ".csv")
            self._save_csv(csv_path)

            print("\n" + "═" * 58)
            print(f"  Frames processed : {frame_no}")
            print(f"  Total ENTERED    : {self.in_count}")
            print(f"  Total LEFT       : {self.out_count}")
            print(f"  Final INSIDE     : {max(0, self.in_count - self.out_count)}")
            print(f"  Output video     : {self.output_path}")
            print(f"  Event CSV        : {csv_path}")
            print("═" * 58)


# ═══════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════
def main():
    global LINE_RATIO

    p = argparse.ArgumentParser(description="Bus Passenger Counter — FINAL")
    p.add_argument("--source",     default="counting.mp4",
                   help="Input video file or camera index")
    p.add_argument("--output",     default="result_final.mp4",
                   help="Output annotated video")
    p.add_argument("--model",      default="yolov8s.pt",
                   help="YOLO weights (.pt)")
    p.add_argument("--line",       type=float, default=LINE_RATIO,
                   help="Trigger-line position as fraction of frame width "
                        "(default 0.45)")
    p.add_argument("--exempt",     default="",
                   help="Path to exempt_embedding.npy — skip counting a "
                        "specific person by appearance (optional, requires torch)")
    p.add_argument("--no-preview", action="store_true",
                   help="Disable live preview window (faster headless processing)")
    p.add_argument("--live",       action="store_true",
                   help="Live-stream mode: assume all first-seen tracks start OUTSIDE")
    p.add_argument("--no-badge-debug", action="store_true",
                   help="Suppress per-frame HSV badge calibration output")
    args = p.parse_args()

    LINE_RATIO = args.line
    if args.no_badge_debug:
        global BADGE_DEBUG
        BADGE_DEBUG = False

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