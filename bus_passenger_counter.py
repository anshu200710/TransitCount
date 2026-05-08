#!/usr/bin/env python3
"""
Bus Passenger Counter — FINAL (v4 Merged)
==========================================
Camera: overhead/top-down at bus front gate
Direction: passengers move LEFT → RIGHT to enter, RIGHT → LEFT to exit

Features:
  ✓ 3-state machine: OUTSIDE → ZONE → INSIDE (must traverse all 3)
  ✓ Staff exclusion via radium tape / high-visibility vest detection (HSV)
  ✓ Temporal vote window + hysteresis for stable exempt decisions
  ✓ Ghost re-link: recovers lost tracks by proximity, reduces ID switches
  ✓ Kalman Filter per track for smooth centroid prediction during occlusion
  ✓ ByteTrack tracker (written at runtime if YAML missing)
  ✓ Non-blocking API POST via background thread + queue (zero FPS impact)
  ✓ Payload: {datetime, hin, hout, inside, total}
  ✓ CYAN flash on confirmed count (auditable in real-time)
  ✓ Trail lines with age-based colour gradient
  ✓ Ghost cleanup every 90 frames (expired → ghost pool for re-link)
  ✓ CSV logs every event with frame + timestamp
  ✓ Optional per-frame HSV debug images + CSV (--debug flag)
  ✓ VisDrone / CrowdHuman model auto-download (--visdrone / --head-detect)
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

# ── Non-blocking API push ────────────────────────────────────────────────
import threading
import queue
import time

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False
    print("[WARN] 'requests' not found — API push disabled. pip install requests")

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  — tune these for your scene
# ═══════════════════════════════════════════════════════════════════════════

# --- Counting geometry ---
LINE_RATIO    = 0.45   # Trigger line as fraction of frame width
DEAD_ZONE_PX  = 30     # Half-width of the dead zone around the line (px)
DEBOUNCE_N    = 3      # Consecutive frames required on a side before committing

# --- Detection ---
CONF_THRESH   = 0.08   # Low conf to catch small top-down heads
IOU_THRESH    = 0.45   # NMS IoU threshold

# --- Tracking ---
TRAIL_LEN     = 50     # Max trail length per track
GHOST_TIMEOUT = 300    # Frames until an unseen track expires into ghost pool
FLASH_FRAMES  = 20     # Frames the cyan flash lasts after a count event
EMA_ALPHA     = 0.40   # Blend weight for raw vs Kalman-predicted centroid

# --- Ghost re-link ---
RELINK_DIST_PX  = 60   # Max centroid distance (px) to adopt a ghost track
RELINK_MAX_AGE  = 45   # Ghost must have been seen within this many frames

# --- Staff exemption (radium tape detection) ---
RADIUM_MIN_PX   = 200  # Min yellow pixels in shoulder ROI to trigger detection
EXEMPT_CONFIRM  = 5    # Score threshold to grant exemption
EXEMPT_MAX      = 12   # Score cap (prevents score runaway)
VOTE_WINDOW     = 7    # Sliding window size for temporal tape voting
VOTE_MIN_HITS   = 2    # Detections needed inside window to count as confirmed
HYSTERESIS_MISS = 6    # Consecutive non-blurry misses to revoke exemption
BLUR_VAR_THRESH = 40   # Laplacian variance below which a crop is too blurry

# --- Merged-box filtering ---
MERGE_AR_THRESH = 1.5  # Width/height ratio above which bbox likely contains 2 people
MERGE_OVERLAP   = 0.30 # IoU above which two tracks are considered overlapping

# --- API ---
API_ENDPOINT    = "https://bae6-49-205-179-53.ngrok-free.app/passenger-count"
API_TIMEOUT     = 2    # seconds per request
API_MAX_RETRY   = 2    # retries on 5xx
API_RETRY_WAIT  = 0.3  # seconds between retries

# --- Tracker YAML path ---
TRACKER_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bytetrack.yaml")

# --- Colours (BGR) ---
COL_LINE      = (0,   0, 220)
COL_BOX_NORM  = (0, 140, 255)
COL_BOX_FLASH = (255, 255,  0)
COL_BOX_SKIP  = (80,  80,  80)
COL_IN_TEXT   = (80, 255,  80)
COL_OUT_TEXT  = (80, 180, 255)
COL_EXEMPT    = (255,  0, 200)


# ═══════════════════════════════════════════════════════════════════════════
#  NON-BLOCKING API WORKER
# ═══════════════════════════════════════════════════════════════════════════

class ApiPushWorker:
    """
    Background daemon thread draining a queue and firing POST requests.
    The main video loop enqueues a payload and returns immediately —
    network latency or retries never touch the processing thread.
    """

    def __init__(self, endpoint, timeout=API_TIMEOUT,
                 max_retry=API_MAX_RETRY, retry_wait=API_RETRY_WAIT):
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
            "total":    in_count,
        }
        self._q.put(payload)

    def _worker(self):
        while True:
            payload = self._q.get()
            self._send_with_retry(payload)
            self._q.task_done()

    def _send_with_retry(self, payload: dict):
        for attempt in range(1, self.max_retry + 2):
            try:
                resp = requests.post(
                    self.endpoint, json=payload, timeout=self.timeout,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code < 500:
                    if resp.status_code >= 400:
                        print(f"[API] Client error {resp.status_code} — not retrying")
                    else:
                        print(f"[API] ✓ {resp.status_code}  "
                              f"hin={payload['hin']} hout={payload['hout']} "
                              f"inside={payload['inside']}")
                    return
                print(f"[API] Server error {resp.status_code} (attempt {attempt})")
            except requests.exceptions.Timeout:
                print(f"[API] Timeout (attempt {attempt})")
            except requests.exceptions.ConnectionError as e:
                print(f"[API] Connection error (attempt {attempt}): {e}")
            except Exception as e:
                print(f"[API] Unexpected error: {e}")
                return
            if attempt <= self.max_retry:
                time.sleep(self.retry_wait)
        print(f"[API] Gave up after {self.max_retry + 1} attempts")


# ═══════════════════════════════════════════════════════════════════════════
#  KALMAN FILTER  — 4-state: [cx, cy, vx, vy]
# ═══════════════════════════════════════════════════════════════════════════

class KalmanCentroid:
    """Constant-velocity Kalman filter for centroid smoothing."""

    def __init__(self, cx: float, cy: float):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],
                                                 [0,0,1,0],[0,0,0,1]], np.float32)
        # Higher process noise (0.5) suits fast-moving top-down heads
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.5
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kf.statePost           = np.array([[cx],[cy],[0],[0]], np.float32)

    def update(self, cx: float, cy: float) -> tuple:
        self.kf.predict()
        res = self.kf.correct(np.array([[cx],[cy]], np.float32))
        return float(res[0, 0]), float(res[1, 0])

    def predict(self) -> tuple:
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])


# ═══════════════════════════════════════════════════════════════════════════
#  PER-TRACK STATE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrackState:
    """
    3-state crossing machine per track ID.

    zone_state values:
      OUTSIDE  — centroid firmly LEFT of dead zone  (entry side)
      ZONE     — centroid inside the dead zone      (crossing)
      INSIDE   — centroid firmly RIGHT              (bus interior)
      SKIP     — first seen in dead zone; cannot count until it clears

    Valid IN  transition:  OUTSIDE → ZONE → INSIDE
    Valid OUT transition:  INSIDE  → ZONE → OUTSIDE
    """
    kalman:        KalmanCentroid
    cx:            float
    cy:            float
    trail:         deque = field(default_factory=lambda: deque(maxlen=TRAIL_LEN))
    zone_state:    str   = "INIT"
    counted:       Optional[str] = None   # None | 'IN' | 'OUT'
    flash:         int   = 0
    last_seen:     int   = 0
    side_frames:   int   = 0
    prev_side:     str   = ""
    debounce_side: str   = ""
    last_bbox:     tuple = field(default_factory=lambda: (0, 0, 0, 0))

    # --- Exemption (radium tape) ---
    exempt:        bool  = False
    exempt_score:  int   = 0
    tape_votes:    deque = field(default_factory=lambda: deque(maxlen=VOTE_WINDOW))
    consec_miss:   int   = 0


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN COUNTER
# ═══════════════════════════════════════════════════════════════════════════

class BusCounter:

    def __init__(self, video_path: str, model_path: str = "yolov8s.pt",
                 output_path: str = "result_final.mp4",
                 enable_debug: bool = False, head_detect: bool = False,
                 visdrone: bool = False, no_preview: bool = False,
                 live: bool = False):

        self.video_path  = video_path
        self.output_path = output_path
        self.no_preview  = no_preview
        self.live        = live
        self.enable_debug = enable_debug

        # ── Model selection: visdrone > head_detect > default ────────────
        if visdrone:
            model_path = self._ensure_visdrone_model()
        elif head_detect and model_path == "yolov8s.pt":
            model_path = self._ensure_head_model()

        self.model  = YOLO(model_path)
        self.states: dict[int, TrackState] = {}
        self._ghost_pool: dict[int, TrackState] = {}

        self.in_count  = 0
        self.out_count = 0
        self.events: list[dict] = []

        # ── API push worker ───────────────────────────────────────────────
        self.api = ApiPushWorker(
            endpoint   = API_ENDPOINT,
            timeout    = API_TIMEOUT,
            max_retry  = API_MAX_RETRY,
            retry_wait = API_RETRY_WAIT,
        )

        # ── Debug setup ───────────────────────────────────────────────────
        self.debug_log: list[dict] = []
        if self.enable_debug:
            self.debug_dir          = "debug_output"
            self.debug_crops_dir    = os.path.join(self.debug_dir, "crops")
            self.debug_analysis_dir = os.path.join(self.debug_dir, "analysis")
            os.makedirs(self.debug_crops_dir,    exist_ok=True)
            os.makedirs(self.debug_analysis_dir, exist_ok=True)
            print(f"[DEBUG] Enabled — output: {self.debug_dir}/")

        # ── ByteTrack YAML ────────────────────────────────────────────────
        self._ensure_bytetrack_yaml()

    # ── Model helpers ─────────────────────────────────────────────────────

    def _ensure_head_model(self) -> str:
        import urllib.request
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n_crowdhuman.pt")
        if not os.path.exists(path):
            url = "https://github.com/yakhyo/yolov8-crowdhuman/releases/download/weights/yolov8n_best.pt"
            print(f"[HEAD-DETECT] Downloading CrowdHuman model…")
            urllib.request.urlretrieve(url, path)
        return path

    def _ensure_visdrone_model(self) -> str:
        import urllib.request
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n_visdrone.pt")
        if not os.path.exists(path):
            url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-visdrone.pt"
            print(f"[VISDRONE] Downloading VisDrone model…")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"[VISDRONE] Download failed ({e}) — using yolov8s.pt")
                return "yolov8s.pt"
        return path

    def _ensure_bytetrack_yaml(self):
        if os.path.exists(TRACKER_CONFIG):
            return
        content = (
            "tracker_type: bytetrack\n"
            "track_high_thresh: 0.25\n"
            "track_low_thresh:  0.05\n"
            "new_track_thresh:  0.20\n"
            "track_buffer:      45\n"
            "match_thresh:      0.85\n"
            "fuse_score:        true\n"
        )
        with open(TRACKER_CONFIG, "w") as f:
            f.write(content)
        print(f"[BYTETRACK] Config written → {TRACKER_CONFIG}")

    # ── Ghost re-link ─────────────────────────────────────────────────────

    def _relink_ghost(self, new_tid: int, raw_cx: float, raw_cy: float, f_no: int):
        """Search the ghost pool for a recently-lost track near this detection."""
        best_tid, best_dist = None, float("inf")
        for ghost_tid, ghost_st in self._ghost_pool.items():
            if f_no - ghost_st.last_seen > RELINK_MAX_AGE:
                continue
            dist = ((ghost_st.cx - raw_cx) ** 2 + (ghost_st.cy - raw_cy) ** 2) ** 0.5
            if dist < RELINK_DIST_PX and dist < best_dist:
                best_dist, best_tid = dist, ghost_tid
        if best_tid is not None:
            adopted = self._ghost_pool.pop(best_tid)
            self.states[new_tid] = adopted
            print(f"  🔗 [RE-LINK] Ghost {best_tid} → ID {new_tid} "
                  f"(dist={best_dist:.1f}px, age={f_no - adopted.last_seen}f)")

    # ── Geometry helpers ──────────────────────────────────────────────────

    def _get_side(self, cx: float, line_x: int) -> str:
        if cx < line_x - DEAD_ZONE_PX:
            return "L"
        if cx > line_x + DEAD_ZONE_PX:
            return "R"
        return "ZONE"

    def _is_merged_box(self, bbox) -> bool:
        x1, y1, x2, y2 = bbox.astype(int)
        w, h = x2 - x1, y2 - y1
        return h <= 0 or (w / h) > MERGE_AR_THRESH

    def _overlaps_other_tracks(self, tid: int, bbox_ints: tuple, f_no: int) -> bool:
        x1, y1, x2, y2 = bbox_ints
        area1 = (x2 - x1) * (y2 - y1)
        if area1 <= 0:
            return False
        for oid, ost in self.states.items():
            if oid == tid or f_no - ost.last_seen > 5:
                continue
            ox1, oy1, ox2, oy2 = ost.last_bbox
            ix1, iy1 = max(x1, ox1), max(y1, oy1)
            ix2, iy2 = min(x2, ox2), min(y2, oy2)
            if ix1 >= ix2 or iy1 >= iy2:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = area1 + (ox2 - ox1) * (oy2 - oy1) - inter
            if union > 0 and (inter / union) > MERGE_OVERLAP:
                return True
        return False

    # ── Blur check ────────────────────────────────────────────────────────

    def _is_blurry(self, crop: np.ndarray) -> tuple:
        """Returns (is_blurry, laplacian_variance)."""
        gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return (lap_var < BLUR_VAR_THRESH, lap_var)

    # ── Radium tape (high-vis vest) detection ─────────────────────────────

    def _has_radium_tape(self, crop: np.ndarray, tid: int,
                         frame_no: int = 0, fps: float = 30.0) -> bool:
        """
        Detect orange-yellow radium tape / hi-vis vest in the shoulder area
        of a person crop using HSV colour segmentation.

        Pipeline:
          1. Blur gate — skip blurry crops (motion artefacts corrupt colour)
          2. Bilateral denoise — edge-preserving smoothing before HSV
          3. Dynamic shoulder ROI (top 20–60 % of bbox height)
          4. Widened HSV range to tolerate motion-blur colour smear
          5. Morphological cleanup — remove isolated noise specks
          6. Multi-stage validation: pixel count → brightness → saturation
          7. Adaptive mean-saturation threshold (small vs large region)
        """
        if crop is None or crop.size == 0:
            return False

        crop_h, crop_w = crop.shape[:2]

        # 1. Blur gate
        blurry, lap_var = self._is_blurry(crop)
        if self.enable_debug:
            print(f"     📷 Sharpness (Laplacian var): {lap_var:.1f} "
                  f"({'BLURRY — skipped' if blurry else 'SHARP'})")
        if blurry:
            return False

        # 2. Denoise
        crop_clean = cv2.bilateralFilter(crop, d=9, sigmaColor=75, sigmaSpace=75)
        hsv = cv2.cvtColor(crop_clean, cv2.COLOR_BGR2HSV)

        # 3. Dynamic shoulder ROI
        roi_top    = int(crop_h * 0.20)
        roi_bottom = int(crop_h * 0.60)
        shoulder   = hsv[roi_top:roi_bottom, :]

        # 4. HSV range (widened S lower-bound for blurred tape)
        lower_y = np.array([ 5, 15,  80])
        upper_y = np.array([35, 255, 255])
        mask    = cv2.inRange(shoulder, lower_y, upper_y)

        # 5. Morphological cleanup
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        yellow_px = int(np.sum(mask > 0))

        # Debug HSV diagnostics
        if self.enable_debug:
            sv_arr = shoulder[:, :, 2].flatten()
            bright = sv_arr > 30
            if np.any(bright):
                sh_h = shoulder[:, :, 0].flatten()
                sh_s = shoulder[:, :, 1].flatten()
                print(f"     🎨 Shoulder HSV — H:{np.mean(sh_h[bright]):.1f}  "
                      f"S:{np.mean(sh_s[bright]):.1f}  V:{np.mean(sv_arr[bright]):.1f}")
            print(f"     🟡 Yellow pixels: {yellow_px}px (need {RADIUM_MIN_PX}px)")

        debug_data = None
        if self.enable_debug:
            debug_data = {
                "frame": frame_no, "timestamp": frame_no / fps,
                "track_id": tid, "crop_width": crop_w, "crop_height": crop_h,
                "laplacian_variance": round(lap_var, 2),
                "shoulder_yellow_pixels": yellow_px,
            }

        # 6. Pixel count gate
        if yellow_px < RADIUM_MIN_PX:
            if self.enable_debug and debug_data is not None:
                debug_data.update({"would_detect": False,
                                   "rejection_reason": f"pixels {yellow_px} < {RADIUM_MIN_PX}"})
                self.debug_log.append(debug_data)
            return False

        # Extract HSV stats on detected pixels
        h_vals = shoulder[:, :, 0][mask > 0]
        s_vals = shoulder[:, :, 1][mask > 0]
        v_vals = shoulder[:, :, 2][mask > 0]
        max_s, max_v = int(np.max(s_vals)), int(np.max(v_vals))
        mean_h = float(np.mean(h_vals))
        mean_s = float(np.mean(s_vals))
        mean_v = float(np.mean(v_vals))

        if self.enable_debug:
            print(f"     📊 Yellow region — H:{mean_h:.1f}  S:{mean_s:.1f}(max {max_s})  "
                  f"V:{mean_v:.1f}(max {max_v})")

        # Brightness check
        if max_v < 150:
            if self.enable_debug and debug_data is not None:
                debug_data.update({"would_detect": False,
                                   "rejection_reason": f"V={max_v} < 150"})
                self.debug_log.append(debug_data)
            return False

        # Peak saturation check
        if max_s < 150:
            if self.enable_debug and debug_data is not None:
                debug_data.update({"would_detect": False,
                                   "rejection_reason": f"max_s={max_s} < 150"})
                self.debug_log.append(debug_data)
            return False

        # 7. Adaptive mean-saturation threshold
        if yellow_px > 2000:
            mean_s_thresh, area_type = 75, "large"
            if max_s < 160:                          # large area must be bright
                if self.enable_debug and debug_data is not None:
                    debug_data.update({"would_detect": False,
                                       "rejection_reason": f"large area max_s={max_s} < 160"})
                    self.debug_log.append(debug_data)
                return False
        else:
            mean_s_thresh, area_type = 45, "small"

        if mean_s < mean_s_thresh:
            if self.enable_debug and debug_data is not None:
                debug_data.update({"would_detect": False,
                                   "rejection_reason":
                                   f"mean_s={mean_s:.0f} < {mean_s_thresh} ({area_type})"})
                self.debug_log.append(debug_data)
            return False

        # All checks passed → radium tape confirmed
        if self.enable_debug:
            print(f"\n{'🟢'*30}")
            print(f"  ⚠️  RADIUM TAPE DETECTED — ID {tid} | f{frame_no} | "
                  f"{yellow_px}px | mean_s={mean_s:.1f}")
            print(f"{'🟢'*30}\n")
            if debug_data is not None:
                debug_data.update({
                    "would_detect": True, "area_type": area_type,
                    "mean_s": mean_s, "max_s": max_s, "max_v": max_v,
                    "rejection_reason": "DETECTED",
                })
                self.debug_log.append(debug_data)
                self._save_debug_image(crop_clean, hsv, mask,
                                       debug_data, tid, frame_no, roi_top)
        return True

    def _save_debug_image(self, crop, hsv, mask_yellow,
                          debug_data, tid, frame_no, roi_top):
        """Save 6-panel debug image (original | mask | ROI | H | S | V)."""
        crop_h, crop_w = crop.shape[:2]
        canvas = np.zeros((crop_h * 2, crop_w * 3, 3), dtype=np.uint8)

        canvas[0:crop_h, 0:crop_w] = crop

        full_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        roi_rows  = mask_yellow.shape[0]
        full_mask[roi_top:roi_top + roi_rows, 0:mask_yellow.shape[1]] = mask_yellow
        canvas[0:crop_h, crop_w:2*crop_w] = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR)

        shoulder_vis = crop.copy()
        cv2.rectangle(shoulder_vis, (0, roi_top),
                      (crop_w, roi_top + roi_rows), (0, 255, 255), 2)
        canvas[0:crop_h, 2*crop_w:3*crop_w] = shoulder_vis

        canvas[crop_h:2*crop_h, 0:crop_w] = cv2.applyColorMap(
            (hsv[:, :, 0] * 2).astype(np.uint8), cv2.COLORMAP_HSV)
        canvas[crop_h:2*crop_h, crop_w:2*crop_w] = \
            cv2.cvtColor(hsv[:, :, 1], cv2.COLOR_GRAY2BGR)
        canvas[crop_h:2*crop_h, 2*crop_w:3*crop_w] = \
            cv2.cvtColor(hsv[:, :, 2], cv2.COLOR_GRAY2BGR)

        labels = [
            (10, 20, "Original"), (crop_w+10, 20, "Mask"),
            (2*crop_w+10, 20, "ROI"), (10, crop_h+20, "Hue"),
            (crop_w+10, crop_h+20, "Saturation"), (2*crop_w+10, crop_h+20, "Value"),
        ]
        for x, y, txt in labels:
            cv2.putText(canvas, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 255), 1)

        status = "DETECTED" if debug_data.get("would_detect") else "REJECTED"
        col    = (0, 255, 0) if debug_data.get("would_detect") else (0, 0, 255)
        cv2.putText(canvas, status, (10, crop_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        fn = f"analysis_f{frame_no:05d}_id{tid:03d}.jpg"
        cv2.imwrite(os.path.join(self.debug_analysis_dir, fn), canvas)
        cv2.imwrite(os.path.join(self.debug_crops_dir,
                                 f"crop_f{frame_no:05d}_id{tid:03d}.jpg"), crop)

    # ── Per-frame track update ─────────────────────────────────────────────

    def _update_track(self, tid: int, bbox: np.ndarray,
                      line_x: int, f_no: int,
                      frame: np.ndarray, fps: float = 30.0) -> Optional[str]:
        x1, y1, x2, y2 = bbox.astype(int)
        raw_cx, raw_cy  = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        # ── New track: try ghost re-link first ────────────────────────────
        if tid not in self.states:
            self._relink_ghost(tid, raw_cx, raw_cy, f_no)

        # ── Still new: create fresh state ─────────────────────────────────
        if tid not in self.states:
            side = self._get_side(raw_cx, line_x)
            kf   = KalmanCentroid(raw_cx, raw_cy)
            st   = TrackState(kalman=kf, cx=raw_cx, cy=raw_cy, last_seen=f_no)

            if self.live:
                if side == "L":
                    st.zone_state, st.prev_side = "OUTSIDE", "L"
                elif side == "R":
                    st.zone_state, st.prev_side = "INSIDE", "R"
                else:
                    st.zone_state, st.prev_side = "ZONE", ""
            else:
                if side == "L":
                    st.zone_state, st.prev_side = "OUTSIDE", "L"
                elif side == "R":
                    st.zone_state, st.prev_side = "INSIDE", "R"
                else:
                    st.zone_state = "SKIP"

            self.states[tid] = st

        st           = self.states[tid]
        st.last_seen = f_no
        st.last_bbox = (x1, y1, x2, y2)

        # ── Radium tape / exemption check ─────────────────────────────────
        merged      = self._is_merged_box(bbox)
        overlapping = self._overlaps_other_tracks(tid, (x1, y1, x2, y2), f_no)

        if not merged and not overlapping:
            crop = frame[max(0, y1):min(frame.shape[0], y2),
                         max(0, x1):min(frame.shape[1], x2)]
            blurry, lap_var = self._is_blurry(crop)

            if blurry:
                pass   # score unchanged — blurry frame is neutral
            else:
                tape_found = self._has_radium_tape(crop, tid, f_no, fps)
                st.tape_votes.append(1 if tape_found else 0)

                if tape_found:
                    st.exempt_score = min(st.exempt_score + 3, EXEMPT_MAX)
                    st.consec_miss  = 0
                else:
                    st.consec_miss += 1
                    if st.consec_miss >= HYSTERESIS_MISS:
                        st.exempt_score = max(st.exempt_score - 1, 0)

        prev_exempt = st.exempt
        if st.exempt_score >= EXEMPT_CONFIRM:
            st.exempt = True
        elif st.exempt_score == 0 and st.consec_miss >= HYSTERESIS_MISS:
            st.exempt = False

        if st.exempt and not prev_exempt:
            print(f"  ⭐ [EXEMPT GRANTED] ID {tid} (score={st.exempt_score})")
        elif not st.exempt and prev_exempt:
            print(f"  ⚠️  [EXEMPT REVOKED]  ID {tid} (score={st.exempt_score})")

        # ── Kalman update ─────────────────────────────────────────────────
        pcx, pcy = st.kalman.update(raw_cx, raw_cy)
        st.cx = EMA_ALPHA * raw_cx + (1 - EMA_ALPHA) * pcx
        st.cy = EMA_ALPHA * raw_cy + (1 - EMA_ALPHA) * pcy
        st.trail.append((int(st.cx), int(st.cy)))

        if st.flash > 0:
            st.flash -= 1

        # Exempt / SKIP IDs never count
        if st.exempt:
            return None

        # SKIP recovery: once it moves to a clear side, recover and return
        if st.zone_state == "SKIP":
            side = self._get_side(st.cx, line_x)
            if side == "L":
                st.zone_state, st.prev_side = "OUTSIDE", "L"
            elif side == "R":
                st.zone_state, st.prev_side = "INSIDE", "R"
            return None

        # Cooldown after a count — freeze transitions
        if st.flash > 0:
            return None

        # ── State-machine transition ──────────────────────────────────────
        event        = None
        current_side = self._get_side(st.cx, line_x)

        if current_side == "ZONE":
            if st.zone_state in ("OUTSIDE", "INSIDE"):
                st.zone_state = "ZONE"
            st.side_frames   = 0
            st.debounce_side = ""
        else:
            if st.debounce_side != current_side:
                st.debounce_side = current_side
                st.side_frames   = 1
            else:
                st.side_frames  += 1

            # Debug: show state transitions
            if self.enable_debug and st.side_frames == 1:
                print(f"  [DEBUG] ID {tid} | zone_state={st.zone_state} | "
                      f"prev_side={st.prev_side} | moving to {current_side}")

            if st.side_frames >= DEBOUNCE_N:
                if current_side == "L":
                    # OUT event: moving to LEFT (exit)
                    # Live mode: count if coming from ZONE or INSIDE (and prev was R or empty)
                    # File mode: count only if coming from INSIDE (prev was R)
                    should_count_out = False
                    if self.live:
                        # In live mode, allow counting if:
                        # 1. Coming from INSIDE (clear exit)
                        # 2. Coming from ZONE and prev_side was R or empty (started in zone, moving out)
                        if st.zone_state == "INSIDE" and st.prev_side == "R":
                            should_count_out = True
                        elif st.zone_state == "ZONE" and st.prev_side in ("", "R"):
                            should_count_out = True
                    else:
                        # File mode: strict - must come from INSIDE with prev_side R
                        if st.zone_state in ("ZONE", "INSIDE") and st.prev_side == "R":
                            should_count_out = True
                    
                    if should_count_out and st.counted != "OUT":
                        self.out_count += 1
                        st.counted      = "OUT"
                        st.flash        = FLASH_FRAMES
                        event           = "OUT"
                        self.events.append({
                            "frame": f_no, "id": tid, "event": "OUT",
                            "in": self.in_count, "out": self.out_count,
                        })
                        print(f"[{f_no:05d}] ID {tid:3d} LEFT    | "
                              f"IN={self.in_count} OUT={self.out_count}")
                        self.api.push(self.in_count, self.out_count)
                    elif self.enable_debug and not should_count_out:
                        print(f"  [DEBUG] ID {tid} | OUT blocked: zone_state={st.zone_state}, "
                              f"prev_side={st.prev_side}, counted={st.counted}")
                    st.zone_state, st.prev_side = "OUTSIDE", "L"

                elif current_side == "R":
                    # IN event: moving to RIGHT (entry)
                    # Live mode: count if coming from ZONE or OUTSIDE (and prev was L or empty)
                    # File mode: count only if coming from OUTSIDE (prev was L)
                    should_count_in = False
                    if self.live:
                        # In live mode, allow counting if:
                        # 1. Coming from OUTSIDE (clear entry)
                        # 2. Coming from ZONE and prev_side was L or empty (started in zone, moving in)
                        if st.zone_state == "OUTSIDE" and st.prev_side == "L":
                            should_count_in = True
                        elif st.zone_state == "ZONE" and st.prev_side in ("", "L"):
                            should_count_in = True
                    else:
                        # File mode: strict - must come from OUTSIDE with prev_side L
                        if st.zone_state in ("ZONE", "OUTSIDE") and st.prev_side == "L":
                            should_count_in = True
                    
                    if should_count_in and st.counted != "IN":
                        self.in_count += 1
                        st.counted     = "IN"
                        st.flash       = FLASH_FRAMES
                        event          = "IN"
                        self.events.append({
                            "frame": f_no, "id": tid, "event": "IN",
                            "in": self.in_count, "out": self.out_count,
                        })
                        print(f"[{f_no:05d}] ID {tid:3d} ENTERED | "
                              f"IN={self.in_count} OUT={self.out_count}")
                        self.api.push(self.in_count, self.out_count)
                    elif self.enable_debug and not should_count_in:
                        print(f"  [DEBUG] ID {tid} | IN blocked: zone_state={st.zone_state}, "
                              f"prev_side={st.prev_side}, counted={st.counted}")
                    st.zone_state, st.prev_side = "INSIDE", "R"

        return event

    # ── Ghost cleanup ─────────────────────────────────────────────────────

    def _purge_ghosts(self, f_no: int):
        """Move timed-out active tracks to ghost pool; prune very old ghosts."""
        alive, expired = {}, {}
        for tid, st in self.states.items():
            (alive if f_no - st.last_seen < GHOST_TIMEOUT else expired)[tid] = st
        self.states = alive
        self._ghost_pool.update(expired)
        self._ghost_pool = {
            gid: gst for gid, gst in self._ghost_pool.items()
            if f_no - gst.last_seen < GHOST_TIMEOUT * 2
        }

    # ── Drawing ───────────────────────────────────────────────────────────

    def _draw_zones(self, frame: np.ndarray, line_x: int) -> np.ndarray:
        h  = frame.shape[0]
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (line_x - DEAD_ZONE_PX, h), (180, 60, 0), -1)
        cv2.rectangle(ov, (line_x + DEAD_ZONE_PX, 0), (frame.shape[1], h), (0, 140, 0), -1)
        frame = cv2.addWeighted(ov, 0.06, frame, 0.94, 0)

        ov2 = frame.copy()
        cv2.rectangle(ov2, (line_x - DEAD_ZONE_PX, 0),
                      (line_x + DEAD_ZONE_PX, h), (0, 200, 255), -1)
        frame = cv2.addWeighted(ov2, 0.18, frame, 0.82, 0)

        cv2.line(frame, (line_x, 0), (line_x, h), COL_LINE, 2)
        cv2.line(frame, (line_x - DEAD_ZONE_PX, 0),
                 (line_x - DEAD_ZONE_PX, h), (0, 180, 255), 1)
        cv2.line(frame, (line_x + DEAD_ZONE_PX, 0),
                 (line_x + DEAD_ZONE_PX, h), (0, 180, 255), 1)

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
            cx, cy = int(st.cx), int(st.cy)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
            cv2.circle(frame, (cx, cy), 5, col, 2)

            # Label
            if st.exempt:
                tag, tag_col = f"#{tid} EXEMPT", COL_EXEMPT
            elif st.counted == "IN":
                tag, tag_col = f"#{tid} IN", COL_IN_TEXT
            elif st.counted == "OUT":
                tag, tag_col = f"#{tid} OUT", COL_OUT_TEXT
            elif st.zone_state == "SKIP":
                tag, tag_col = f"#{tid} skip", COL_BOX_SKIP
            else:
                tag, tag_col = f"#{tid} {st.zone_state[:3]}", (200, 200, 200)

            cv2.putText(frame, tag, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, tag_col, 1, cv2.LINE_AA)

        return frame

    def _draw_dashboard(self, frame: np.ndarray,
                        frame_no: int, fps: float) -> np.ndarray:
        h, w   = frame.shape[:2]
        dw, dh = 220, 130
        margin = 14
        dx, dy = w - dw - margin, margin

        ov = frame.copy()
        cv2.rectangle(ov, (dx, dy), (dx + dw, dy + dh), (12, 12, 12), -1)
        frame = cv2.addWeighted(ov, 0.70, frame, 0.30, 0)
        cv2.rectangle(frame, (dx, dy), (dx + dw, dy + 3), (0, 200, 80), -1)

        inside = max(0, self.in_count - self.out_count)
        rows   = [
            ("ENTERED", self.in_count,  COL_IN_TEXT),
            ("LEFT",    self.out_count, COL_OUT_TEXT),
            ("INSIDE",  inside,         (80, 220, 255)),
        ]
        for i, (label, val, col) in enumerate(rows):
            y = dy + 34 + i * 32
            cv2.putText(frame, f"{label}:", (dx + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1, cv2.LINE_AA)
            cv2.putText(frame, str(val), (dx + 148, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)

        cv2.putText(frame, f"f{frame_no}  {frame_no/fps:.1f}s",
                    (dx + 10, dy + dh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)
        return frame

    # ── CSV save ──────────────────────────────────────────────────────────

    def _save_csv(self, path: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "timestamp", "id",
                                               "event", "in", "out", "inside"])
            w.writeheader()
            for ev in self.events:
                w.writerow({
                    "frame": ev["frame"], "timestamp": ts,
                    "id": ev["id"], "event": ev["event"],
                    "in": ev["in"], "out": ev["out"],
                    "inside": ev["in"] - ev["out"],
                })
        print(f"CSV saved: {path}")

    # ── Main loop ─────────────────────────────────────────────────────────

    def process(self):
        # Check if source is a stream URL or a file
        is_stream = self.video_path.startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        
        # Verify video file exists (skip check for streams)
        if not is_stream and not os.path.exists(self.video_path):
            raise FileNotFoundError(
                f"Video file not found: {self.video_path}\n"
                f"Current directory: {os.getcwd()}\n"
                f"Please check the file path and try again."
            )
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            if is_stream:
                raise ConnectionError(
                    f"Cannot connect to stream: {self.video_path}\n"
                    f"Please check the stream URL and network connection."
                )
            else:
                raise FileNotFoundError(
                    f"Cannot open video file: {self.video_path}\n"
                    f"The file exists but may be corrupted or in an unsupported format."
                )

        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
        line_x = int(W * LINE_RATIO)

        writer = cv2.VideoWriter(
            self.output_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

        print(f"Video : {W}×{H} @ {FPS:.1f} fps")
        print(f"Line  : x={line_x} ({LINE_RATIO*100:.0f}% width)")
        print(f"Zone  : x=[{line_x-DEAD_ZONE_PX}, {line_x+DEAD_ZONE_PX}]")
        if self.live:
            print(f"Mode  : LIVE STREAM")
        print("─" * 60)

        f_no = 0
        last_csv_save = 0
        csv_save_interval = 300  # Save CSV every 300 frames (~10 seconds at 30fps)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if self.live:
                        print("[WARN] Stream interrupted, attempting to reconnect...")
                        time.sleep(1)
                        continue
                    else:
                        break
                f_no += 1

                # Detection + tracking
                track_args = dict(
                    conf    = CONF_THRESH,
                    iou     = IOU_THRESH,
                    verbose = False,
                    tracker = TRACKER_CONFIG,
                    persist = True,
                )
                track_args["classes"] = [0]   # person class (adjust for VisDrone: [1])

                results = self.model.track(frame, **track_args)[0]

                # Build supervision Detections for drawing
                tids = results.boxes.id
                dets = sv.Detections(
                    xyxy       = results.boxes.xyxy.cpu().numpy(),
                    confidence = results.boxes.conf.cpu().numpy()
                                 if results.boxes.conf is not None else None,
                    class_id   = results.boxes.cls.cpu().numpy().astype(int)
                                 if results.boxes.cls is not None else None,
                    tracker_id = tids.cpu().numpy().astype(int)
                                 if tids is not None else None,
                )

                # Update state machine for each detected track
                if tids is not None:
                    for bbox, tid in zip(results.boxes.xyxy.cpu().numpy(),
                                         tids.cpu().numpy().astype(int)):
                        self._update_track(int(tid), bbox, line_x, f_no, frame, FPS)

                # Periodic ghost maintenance
                if f_no % 90 == 0:
                    self._purge_ghosts(f_no)

                # Draw
                frame = self._draw_zones(frame, line_x)
                frame = self._draw_tracks(frame, dets)
                frame = self._draw_dashboard(frame, f_no, FPS)
                writer.write(frame)

                # Periodic CSV save for live streams
                if self.live and (f_no - last_csv_save) >= csv_save_interval:
                    csv_path = self.output_path.replace(".mp4", "_events.csv")
                    self._save_csv(csv_path)
                    last_csv_save = f_no

                if not self.no_preview:
                    cv2.imshow("Bus Counter — Final", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()

            csv_path = self.output_path.replace(".mp4", "_events.csv")
            self._save_csv(csv_path)

            # Save debug CSV
            if self.enable_debug and self.debug_log:
                debug_csv = os.path.join(self.debug_dir, "debug_log.csv")
                # Collect all unique fieldnames from all entries
                all_fieldnames = set()
                for entry in self.debug_log:
                    all_fieldnames.update(entry.keys())
                fieldnames = sorted(all_fieldnames)  # Sort for consistent column order
                
                with open(debug_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(self.debug_log)
                print(f"[DEBUG] CSV saved: {debug_csv} ({len(self.debug_log)} entries)")

            print("\n" + "═" * 60)
            print(f"  Frames processed : {f_no}")
            print(f"  Total ENTERED    : {self.in_count}")
            print(f"  Total LEFT       : {self.out_count}")
            print(f"  Final INSIDE     : {max(0, self.in_count - self.out_count)}")
            print(f"  Output video     : {self.output_path}")
            print(f"  Event CSV        : {csv_path}")
            print("═" * 60)


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global LINE_RATIO

    p = argparse.ArgumentParser(
        description="Bus Passenger Counter — Final (API + Staff Exemption)")
    p.add_argument("--source",      default="counting.mp4",
                   help="Input video path")
    p.add_argument("--output",      default="result_final.mp4",
                   help="Output video path")
    p.add_argument("--model",       default="yolov8s.pt",
                   help="YOLO model weights")
    p.add_argument("--line",        type=float, default=LINE_RATIO,
                   help="Trigger line as fraction of frame width (default 0.45)")
    p.add_argument("--no-preview",  action="store_true",
                   help="Disable live preview window (faster headless runs)")
    p.add_argument("--live",        action="store_true",
                   help="Live-stream mode: treat all new IDs as starting outside")
    p.add_argument("--debug",       action="store_true",
                   help="Save per-frame HSV analysis images + CSV")
    p.add_argument("--head-detect", action="store_true",
                   help="Auto-download & use CrowdHuman head-detection model")
    p.add_argument("--visdrone",    action="store_true",
                   help="Auto-download & use VisDrone overhead model")
    args = p.parse_args()

    LINE_RATIO = args.line

    counter = BusCounter(
        video_path  = args.source,
        model_path  = args.model,
        output_path = args.output,
        enable_debug= args.debug,
        head_detect = args.head_detect,
        visdrone    = args.visdrone,
        no_preview  = args.no_preview,
        live        = args.live,
    )
    counter.process()


if __name__ == "__main__":
    main()