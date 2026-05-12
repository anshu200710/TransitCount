#!/usr/bin/env python3
"""
Bus Passenger Counter — v6.0 UNIFIED
======================================
Camera : overhead / top-down at bus front gate
Direction: passengers move LEFT → RIGHT to enter, RIGHT → LEFT to exit

Best-of-both merged from v4 (Merged) and v5 (BLOB-FIRST):

  ✓ YOLOv8x (best accuracy) with ByteTrack — tuned for top-down pedestrian tracking
  ✓ 3-state machine per track: OUTSIDE → ZONE → INSIDE (valid crossing only)
  ✓ Ultra-fast crossing detection: catches single-frame line crossings
  ✓ Ghost re-link: recovers lost track IDs by proximity (reduces ID switches)
  ✓ Kalman filter per track: smooth centroid prediction during occlusion
  ✓ Ghost box extrapolation: Kalman-predicted box drawn while detection is absent
  ✓ Bounding-box EMA smoothing: removes per-frame YOLO jitter visually
  ✓ Staff exemption via BLOB-FIRST radium tape / hi-vis vest detection:
      - Temporal vote window + hysteresis for stable exempt decisions
      - Blob-first HSV: immune to both blur (far) AND dilution (close)
      - Area ratio replaces raw pixel count → distance-invariant
  ✓ Non-blocking API POST via background thread + queue (zero FPS impact)
  ✓ Payload: {datetime, hin, hout, inside, total}
  ✓ CYAN flash on confirmed count (auditable in real-time)
  ✓ Trail lines with age-based colour gradient
  ✓ Ghost cleanup every 90 frames
  ✓ CSV logs every event with frame + timestamp
  ✓ Optional per-frame HSV debug images + CSV (--debug flag)
  ✓ Live-stream mode + optional processing delay
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
    print("[WARN] 'requests' not found — API push disabled.  pip install requests")


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  — tune these for your scene
# ═══════════════════════════════════════════════════════════════════════════

# --- Counting geometry ---
LINE_RATIO    = 0.45   # Trigger line as fraction of frame width (DO NOT CHANGE)
DEAD_ZONE_PX  = 30     # Half-width of dead zone around the line (px)
DEBOUNCE_N    = 3      # Consecutive frames required on a side before committing

# --- Detection ---
CONF_THRESH   = 0.20   # Balanced: catches partial heads, avoids excessive noise
IOU_THRESH    = 0.45   # NMS IoU threshold

# --- Tracking ---
TRAIL_LEN      = 50    # Max trail length per track
GHOST_TIMEOUT  = 300   # Frames until an unseen track expires into ghost pool
FLASH_FRAMES   = 20    # Frames the cyan flash lasts after a count event
EMA_ALPHA      = 0.50  # Centroid EMA blend: raw vs Kalman-predicted

# --- Bounding-box smoothing ---
BOX_EMA_ALPHA  = 0.45  # Higher = follows detection closely; lower = smoother

# --- Ghost box extrapolation ---
GHOST_BOX_FRAMES = 8   # Max frames to show Kalman-predicted ghost box

# --- Ghost re-link ---
RELINK_DIST_PX = 100   # Max centroid distance (px) to adopt a ghost track
RELINK_MAX_AGE  = 60   # Ghost must have been seen within this many frames

# --- Staff exemption: radium tape / hi-vis vest (BLOB-FIRST) ---
# Blob-first approach: find compact yellow contours, validate each blob
# independently — immune to both blur (far) AND dilution (close range).
BLOB_MIN_RATIO   = 0.003  # Min blob area as fraction of crop area
BLOB_MAX_RATIO   = 0.25   # Max blob area as fraction of crop area
BLOB_MIN_SAT     = 100    # Min peak saturation within blob pixels
BLOB_MIN_VAL     = 120    # Min peak value within blob pixels
BLOB_COMPACT_MAX = 25.0   # Max (perimeter²/area) — tape is compact, not scattered

EXEMPT_CONFIRM = 5     # Score threshold to grant exemption
EXEMPT_MAX     = 12    # Score cap
VOTE_WINDOW    = 7     # Sliding window size for temporal tape voting
VOTE_MIN_HITS  = 2     # Detections needed inside window to confirm
HYSTERESIS_MISS = 6    # Consecutive non-blurry misses to revoke exemption
BLUR_VAR_THRESH = 40   # Laplacian variance below which a crop is too blurry

# --- Merged-box filtering ---
MERGE_AR_THRESH = 1.5  # Width/height ratio above which bbox likely contains 2 people
MERGE_OVERLAP   = 0.30 # IoU above which two tracks are considered overlapping

# --- API ---
API_ENDPOINT   = "https://9424-49-205-176-68.ngrok-free.app/api/passenger-count"
API_TIMEOUT    = 2     # seconds per request
API_MAX_RETRY  = 2     # retries on 5xx
API_RETRY_WAIT = 0.3   # seconds between retries

# --- Tracker YAML path ---
TRACKER_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bytetrack.yaml")

# --- Colours (BGR) ---
COL_LINE       = (0,   0,  220)
COL_BOX_NORM   = (0, 140,  255)
COL_BOX_FLASH  = (255, 255,  0)
COL_BOX_SKIP   = (80,  80,   80)
COL_IN_TEXT    = (80, 255,   80)
COL_OUT_TEXT   = (80, 180,  255)
COL_EXEMPT     = (255,  0,  200)
COL_GHOST_BOX  = (180, 180,  60)


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
            "total":    inside,
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
    """Constant-velocity Kalman filter for centroid smoothing & ghost prediction."""

    def __init__(self, cx: float, cy: float):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],
                                                 [0,0,1,0],[0,0,0,1]], np.float32)
        # Higher process noise suits fast-moving top-down heads
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.5
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kf.statePost           = np.array([[cx],[cy],[0],[0]], np.float32)

    def update(self, cx: float, cy: float) -> tuple:
        self.kf.predict()
        res = self.kf.correct(np.array([[cx],[cy]], np.float32))
        return float(res[0, 0]), float(res[1, 0])

    def predict(self) -> tuple:
        """Return predicted position without consuming a measurement update."""
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
      SKIP     — first seen in dead zone; wait until it clears

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
    smooth_bbox:   tuple = field(default_factory=lambda: (0, 0, 0, 0))

    # --- Exemption (radium tape / hi-vis vest) ---
    exempt:        bool  = False
    exempt_score:  int   = 0
    tape_votes:    deque = field(default_factory=lambda: deque(maxlen=VOTE_WINDOW))
    consec_miss:   int   = 0

    # --- Ultra-fast crossing tracking ---
    prev_cx:       float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN COUNTER
# ═══════════════════════════════════════════════════════════════════════════

class BusCounter:

    def __init__(self, video_path: str, model_path: str = "yolov8x.pt",
                 output_path: str = "result_v6.mp4",
                 enable_debug: bool = False, head_detect: bool = False,
                 visdrone: bool = False, no_preview: bool = False,
                 live: bool = False, delay_seconds: int = 0):

        self.video_path    = video_path
        self.output_path   = output_path
        self.no_preview    = no_preview
        self.live          = live
        self.enable_debug  = enable_debug
        self.delay_seconds = delay_seconds
        self.frame_buffer  = deque()
        self.head_detect   = head_detect

        # ── Model selection: visdrone > head_detect > default ────────────
        if visdrone:
            model_path = self._ensure_visdrone_model()
        elif head_detect and model_path == "yolov8x.pt":
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

    # ─────────────────────────────────────────────────────────────────────
    #  Model helpers
    # ─────────────────────────────────────────────────────────────────────

    def _ensure_head_model(self) -> str:
        import urllib.request
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "yolov8n_crowdhuman.pt")
        if not os.path.exists(path):
            url = ("https://github.com/yakhyo/yolov8-crowdhuman/releases/"
                   "download/weights/yolov8n_best.pt")
            print(f"[HEAD-DETECT] Downloading CrowdHuman model…")
            urllib.request.urlretrieve(url, path)
        return path

    def _ensure_visdrone_model(self) -> str:
        import urllib.request
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "yolov8n_visdrone.pt")
        if not os.path.exists(path):
            url = ("https://github.com/ultralytics/assets/releases/"
                   "download/v0.0.0/yolov8n-visdrone.pt")
            print(f"[VISDRONE] Downloading VisDrone model…")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"[VISDRONE] Download failed ({e}) — using yolov8x.pt")
                return "yolov8x.pt"
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

    # ─────────────────────────────────────────────────────────────────────
    #  Ghost re-link
    # ─────────────────────────────────────────────────────────────────────

    def _relink_ghost(self, new_tid: int, raw_cx: float, raw_cy: float, f_no: int):
        """
        Search the ghost pool for a recently-lost track near this detection.
        If found, migrate that ghost's state to new_tid — preserving zone_state,
        counted flag, and exemption score across the ID switch.
        """
        best_tid, best_dist = None, float("inf")
        for ghost_tid, ghost_st in self._ghost_pool.items():
            if f_no - ghost_st.last_seen > RELINK_MAX_AGE:
                continue
            dist = ((ghost_st.cx - raw_cx) ** 2 +
                    (ghost_st.cy - raw_cy) ** 2) ** 0.5
            if dist < RELINK_DIST_PX and dist < best_dist:
                best_dist, best_tid = dist, ghost_tid

        if best_tid is not None:
            adopted = self._ghost_pool.pop(best_tid)
            self.states[new_tid] = adopted
            print(f"  🔗 [RE-LINK] Ghost {best_tid} → ID {new_tid} "
                  f"(dist={best_dist:.1f}px, age={f_no - adopted.last_seen}f)")

    # ─────────────────────────────────────────────────────────────────────
    #  Geometry helpers
    # ─────────────────────────────────────────────────────────────────────

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

    # ─────────────────────────────────────────────────────────────────────
    #  Blur check
    # ─────────────────────────────────────────────────────────────────────

    def _is_blurry(self, crop: np.ndarray) -> tuple:
        gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return (lap_var < BLUR_VAR_THRESH, lap_var)

    # ─────────────────────────────────────────────────────────────────────
    #  BLOB-FIRST radium tape / hi-vis vest detection  (v5 algorithm)
    # ─────────────────────────────────────────────────────────────────────

    def _has_radium_tape(self, crop: np.ndarray, tid: int,
                         frame_no: int = 0, fps: float = 30.0) -> bool:
        """
        Detect radium tape (orange-yellow hi-vis) using a BLOB-FIRST approach.

        Pipeline:
          1. Blur gate       — skip blurry crops entirely (score unchanged)
          2. Denoise         — bilateral filter to preserve edges
          3. Shoulder ROI    — top 20-60% of crop height
          4. HSV mask        — wide S lower-bound for blur-smear tolerance
          5. Morphology      — remove isolated noise specks
          6. Blob contours   — find individual yellow regions
          7. Per-blob tests  — area ratio, compactness, peak S, peak V
          8. Decision        — any single blob passing all tests → tape found

        Why blob-first beats whole-ROI pixel counting:
          • Far / blurry : blob area ratio stays stable; we measure the blob
            core pixels only, not the diluted background.
          • Close range  : background yellow forms large blobs that fail the
            compactness or ratio filter; real tape forms a tight compact strip.
        """
        if crop is None or crop.size == 0:
            return False

        crop_h, crop_w = crop.shape[:2]
        crop_area      = crop_h * crop_w

        # Step 1 — Blur gate
        blurry, lap_var = self._is_blurry(crop)
        if self.enable_debug:
            print(f"     📷 Sharpness (Laplacian var): {lap_var:.1f} "
                  f"({'BLURRY — skipped' if blurry else 'SHARP'})")
        if blurry:
            return False

        # Step 2 — Edge-preserving denoise
        crop_clean = cv2.bilateralFilter(crop, d=9, sigmaColor=75, sigmaSpace=75)
        hsv        = cv2.cvtColor(crop_clean, cv2.COLOR_BGR2HSV)

        # Step 3 — Shoulder ROI (top 20–60%)
        roi_top    = int(crop_h * 0.20)
        roi_bottom = int(crop_h * 0.60)
        shoulder   = hsv[roi_top:roi_bottom, :]
        roi_area   = shoulder.shape[0] * shoulder.shape[1]

        # Step 4 — HSV mask (wide S lower-bound for blur tolerance)
        lower_y = np.array([ 5, 15,  80])
        upper_y = np.array([35, 255, 255])
        mask    = cv2.inRange(shoulder, lower_y, upper_y)

        # Step 5 — Morphological cleanup
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        yellow_px = int(np.sum(mask > 0))
        if self.enable_debug:
            print(f"     🟡 Yellow pixels: {yellow_px}px "
                  f"({100*yellow_px/max(roi_area,1):.1f}% of shoulder ROI)")

        if yellow_px == 0:
            return False

        # Step 6 — Find individual yellow blobs
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if self.enable_debug:
            print(f"     🔍 Found {len(contours)} yellow blob(s)")

        # Step 7 — Per-blob validation
        tape_blob_found = False
        for i, cnt in enumerate(contours):
            blob_area = cv2.contourArea(cnt)
            if blob_area < 4:
                continue

            ratio       = blob_area / crop_area
            perimeter   = cv2.arcLength(cnt, True)
            compactness = (perimeter ** 2) / blob_area if blob_area > 0 else 9999.0

            blob_mask  = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
            blob_s = shoulder[:, :, 1][blob_mask > 0]
            blob_v = shoulder[:, :, 2][blob_mask > 0]

            peak_s = int(np.max(blob_s)) if len(blob_s) > 0 else 0
            peak_v = int(np.max(blob_v)) if len(blob_v) > 0 else 0

            if self.enable_debug:
                print(f"        Blob {i+1}: area={blob_area:.0f} ratio={ratio:.4f} "
                      f"compact={compactness:.1f} peak_S={peak_s} peak_V={peak_v}")

            if ratio < BLOB_MIN_RATIO:
                continue
            if ratio > BLOB_MAX_RATIO:
                continue
            if compactness > BLOB_COMPACT_MAX:
                continue
            if peak_s < BLOB_MIN_SAT:
                continue
            if peak_v < BLOB_MIN_VAL:
                continue

            # All checks passed
            if self.enable_debug:
                print(f"           → ✅ TAPE BLOB CONFIRMED "
                      f"(ratio={ratio:.4f}, compact={compactness:.1f}, "
                      f"peak_S={peak_s}, peak_V={peak_v})")
            tape_blob_found = True
            break

        # Step 8 — Debug image
        if self.enable_debug:
            debug_data = {
                "frame": frame_no, "timestamp": frame_no / fps,
                "track_id": tid, "crop_width": crop_w, "crop_height": crop_h,
                "laplacian_variance": round(lap_var, 2),
                "yellow_pixels": yellow_px, "num_blobs": len(contours),
                "would_detect": tape_blob_found,
                "rejection_reason": ("DETECTED" if tape_blob_found
                                     else "No valid tape blob"),
            }
            self._save_debug_image(crop_clean, hsv, mask,
                                   debug_data, tid, frame_no, roi_top)
            self.debug_log.append(debug_data)

        if tape_blob_found and self.enable_debug:
            print(f"\n{'🟢'*30}")
            print(f"  ⚠️  RADIUM TAPE DETECTED — ID {tid} | f{frame_no} | "
                  f"{yellow_px}px")
            print(f"{'🟢'*30}\n")

        return tape_blob_found

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

        for x, y, txt in [
            (10, 20, "Original"), (crop_w+10, 20, "Mask"),
            (2*crop_w+10, 20, "ROI"), (10, crop_h+20, "Hue"),
            (crop_w+10, crop_h+20, "Saturation"), (2*crop_w+10, crop_h+20, "Value"),
        ]:
            cv2.putText(canvas, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 255), 1)

        status = "DETECTED" if debug_data.get("would_detect") else "REJECTED"
        col    = (0, 255, 0) if debug_data.get("would_detect") else (0, 0, 255)
        cv2.putText(canvas, status, (10, crop_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        cv2.imwrite(os.path.join(self.debug_analysis_dir,
                                 f"analysis_f{frame_no:05d}_id{tid:03d}.jpg"), canvas)
        cv2.imwrite(os.path.join(self.debug_crops_dir,
                                 f"crop_f{frame_no:05d}_id{tid:03d}.jpg"), crop)

    # ─────────────────────────────────────────────────────────────────────
    #  Per-frame track update  (state machine + counting logic)
    # ─────────────────────────────────────────────────────────────────────

    def _update_track(self, tid: int, bbox: np.ndarray,
                      line_x: int, f_no: int,
                      frame: np.ndarray, fps: float = 30.0) -> Optional[str]:

        x1, y1, x2, y2 = bbox.astype(int)
        raw_cx, raw_cy  = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        # ── Ghost re-link attempt ─────────────────────────────────────────
        if tid not in self.states:
            self._relink_ghost(tid, raw_cx, raw_cy, f_no)

        # ── Create fresh state if still not linked ────────────────────────
        if tid not in self.states:
            side = self._get_side(raw_cx, line_x)
            kf   = KalmanCentroid(raw_cx, raw_cy)
            st   = TrackState(kalman=kf, cx=raw_cx, cy=raw_cy, last_seen=f_no)

            if self.live:
                if side == "L":   st.zone_state, st.prev_side = "OUTSIDE", "L"
                elif side == "R": st.zone_state, st.prev_side = "INSIDE",  "R"
                else:             st.zone_state = "ZONE"
            else:
                if side == "L":   st.zone_state, st.prev_side = "OUTSIDE", "L"
                elif side == "R": st.zone_state, st.prev_side = "INSIDE",  "R"
                else:             st.zone_state = "SKIP"

            self.states[tid] = st

        st           = self.states[tid]
        st.last_seen = f_no
        st.last_bbox = (x1, y1, x2, y2)

        # ── Bounding-box EMA smoothing ────────────────────────────────────
        sx1, sy1, sx2, sy2 = st.smooth_bbox
        if sx1 == 0 and sy1 == 0 and sx2 == 0 and sy2 == 0:
            st.smooth_bbox = (x1, y1, x2, y2)
        else:
            a = BOX_EMA_ALPHA
            st.smooth_bbox = (
                int(a * x1 + (1 - a) * sx1), int(a * y1 + (1 - a) * sy1),
                int(a * x2 + (1 - a) * sx2), int(a * y2 + (1 - a) * sy2),
            )

        # ── Radium tape / exemption check (BLOB-FIRST) ────────────────────
        merged      = self._is_merged_box(bbox)
        overlapping = self._overlaps_other_tracks(tid, (x1, y1, x2, y2), f_no)

        if not merged and not overlapping:
            crop = frame[max(0, y1):min(frame.shape[0], y2),
                         max(0, x1):min(frame.shape[1], x2)]
            blurry, _ = self._is_blurry(crop)

            if not blurry:
                tape_found = self._has_radium_tape(crop, tid, f_no, fps)
                st.tape_votes.append(1 if tape_found else 0)

                if tape_found:
                    st.exempt_score = min(st.exempt_score + 3, EXEMPT_MAX)
                    st.consec_miss  = 0
                else:
                    st.consec_miss += 1
                    if st.consec_miss >= HYSTERESIS_MISS:
                        st.exempt_score = max(st.exempt_score - 1, 0)
            # Blurry frame → score/miss count unchanged (neutral)

        prev_exempt = st.exempt
        if st.exempt_score >= EXEMPT_CONFIRM:
            st.exempt = True
        elif st.exempt_score == 0 and st.consec_miss >= HYSTERESIS_MISS:
            st.exempt = False

        if st.exempt and not prev_exempt:
            print(f"  ⭐ [EXEMPT GRANTED] ID {tid} (score={st.exempt_score})")
        elif not st.exempt and prev_exempt:
            print(f"  ⚠️  [EXEMPT REVOKED]  ID {tid} (score={st.exempt_score})")

        # ── Kalman update — blended centroid ──────────────────────────────
        pcx, pcy = st.kalman.update(raw_cx, raw_cy)
        st.cx    = EMA_ALPHA * raw_cx + (1 - EMA_ALPHA) * pcx
        st.cy    = EMA_ALPHA * raw_cy + (1 - EMA_ALPHA) * pcy
        st.trail.append((int(st.cx), int(st.cy)))

        if st.flash > 0:
            st.flash -= 1

        # Exempt tracks are never counted
        if st.exempt:
            return None

        # ── SKIP recovery ─────────────────────────────────────────────────
        if st.zone_state == "SKIP":
            side = self._get_side(st.cx, line_x)
            if side == "L":
                st.zone_state, st.prev_side = "OUTSIDE", "L"
            elif side == "R":
                st.zone_state, st.prev_side = "INSIDE",  "R"
            return None

        # Cooldown after a count
        if st.flash > 0:
            return None

        # ── ULTRA-FAST crossing: single-frame line jump ───────────────────
        #   Catches people who cross the line in one frame (high speed),
        #   bypassing the normal ZONE dwell. We check if the centroid
        #   straddled the line between the previous and current frame.
        if st.prev_cx != 0.0:
            if st.prev_cx < line_x <= st.cx:
                # Jumped L → R in one frame  (ENTRY)
                if st.counted != "IN":
                    self.in_count += 1
                    st.counted, st.flash = "IN", FLASH_FRAMES
                    st.zone_state, st.prev_side = "INSIDE", "R"
                    self.events.append({
                        "frame": f_no, "id": tid, "event": "IN",
                        "in": self.in_count, "out": self.out_count,
                    })
                    print(f"[{f_no:05d}] ID {tid:3d} ENTERED (FAST) | "
                          f"IN={self.in_count} OUT={self.out_count}")
                    self.api.push(self.in_count, self.out_count)
                    st.prev_cx = st.cx
                    return "IN"

            elif st.prev_cx > line_x >= st.cx:
                # Jumped R → L in one frame  (EXIT)
                if st.counted != "OUT":
                    self.out_count += 1
                    st.counted, st.flash = "OUT", FLASH_FRAMES
                    st.zone_state, st.prev_side = "OUTSIDE", "L"
                    self.events.append({
                        "frame": f_no, "id": tid, "event": "OUT",
                        "in": self.in_count, "out": self.out_count,
                    })
                    print(f"[{f_no:05d}] ID {tid:3d} LEFT    (FAST) | "
                          f"IN={self.in_count} OUT={self.out_count}")
                    self.api.push(self.in_count, self.out_count)
                    st.prev_cx = st.cx
                    return "OUT"

        st.prev_cx = st.cx

        # ── Normal 3-state machine transition ─────────────────────────────
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

            if st.side_frames >= DEBOUNCE_N:
                if current_side == "L":
                    # Potential EXIT
                    should_out = False
                    if self.live:
                        if st.zone_state == "INSIDE" and st.prev_side == "R":
                            should_out = True
                        elif st.zone_state == "ZONE" and st.prev_side in ("", "R"):
                            should_out = True
                    else:
                        if st.zone_state in ("ZONE", "INSIDE") and st.prev_side == "R":
                            should_out = True

                    if should_out and st.counted != "OUT":
                        self.out_count += 1
                        st.counted, st.flash = "OUT", FLASH_FRAMES
                        event = "OUT"
                        self.events.append({
                            "frame": f_no, "id": tid, "event": "OUT",
                            "in": self.in_count, "out": self.out_count,
                        })
                        print(f"[{f_no:05d}] ID {tid:3d} LEFT    | "
                              f"IN={self.in_count} OUT={self.out_count}")
                        self.api.push(self.in_count, self.out_count)

                    st.zone_state, st.prev_side = "OUTSIDE", "L"

                elif current_side == "R":
                    # Potential ENTRY
                    should_in = False
                    if self.live:
                        if st.zone_state == "OUTSIDE" and st.prev_side == "L":
                            should_in = True
                        elif st.zone_state == "ZONE" and st.prev_side in ("", "L"):
                            should_in = True
                    else:
                        if st.zone_state in ("ZONE", "OUTSIDE") and st.prev_side == "L":
                            should_in = True

                    if should_in and st.counted != "IN":
                        self.in_count += 1
                        st.counted, st.flash = "IN", FLASH_FRAMES
                        event = "IN"
                        self.events.append({
                            "frame": f_no, "id": tid, "event": "IN",
                            "in": self.in_count, "out": self.out_count,
                        })
                        print(f"[{f_no:05d}] ID {tid:3d} ENTERED | "
                              f"IN={self.in_count} OUT={self.out_count}")
                        self.api.push(self.in_count, self.out_count)

                    st.zone_state, st.prev_side = "INSIDE", "R"

        return event

    # ─────────────────────────────────────────────────────────────────────
    #  Ghost cleanup
    # ─────────────────────────────────────────────────────────────────────

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

    # ─────────────────────────────────────────────────────────────────────
    #  Drawing
    # ─────────────────────────────────────────────────────────────────────

    def _draw_zones(self, frame: np.ndarray, line_x: int) -> np.ndarray:
        h  = frame.shape[0]
        W  = frame.shape[1]
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (line_x - DEAD_ZONE_PX, h), (180, 60, 0), -1)
        cv2.rectangle(ov, (line_x + DEAD_ZONE_PX, 0), (W, h), (0, 140, 0), -1)
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

        cv2.putText(frame, "OUTSIDE",  (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "INSIDE",   (line_x + DEAD_ZONE_PX + 6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 80), 1, cv2.LINE_AA)
        cv2.putText(frame, "ZONE",     (line_x - DEAD_ZONE_PX + 4, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
        return frame

    def _draw_tracks(self, frame: np.ndarray,
                     detected_ids: set, f_no: int, W: int, H: int) -> np.ndarray:
        """Draw boxes, trails, ghost boxes, and labels for all active tracks."""

        for tid, st in self.states.items():
            frames_lost = f_no - st.last_seen

            # ── Active detection box (EMA-smoothed) ──────────────────────
            if tid in detected_ids:
                bx1, by1, bx2, by2 = st.smooth_bbox

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
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, thick)

                # Centroid dot
                cx, cy = int(st.cx), int(st.cy)
                cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
                cv2.circle(frame, (cx, cy), 5, col, 2)

                # Label
                if st.exempt:
                    tag, tag_col = f"#{tid} EXEMPT", COL_EXEMPT
                elif st.counted == "IN":
                    tag, tag_col = f"#{tid} IN",     COL_IN_TEXT
                elif st.counted == "OUT":
                    tag, tag_col = f"#{tid} OUT",    COL_OUT_TEXT
                elif st.zone_state == "SKIP":
                    tag, tag_col = f"#{tid} skip",   COL_BOX_SKIP
                else:
                    tag, tag_col = (f"#{tid} {st.zone_state[:3]}",
                                    (200, 200, 200))

                cv2.putText(frame, tag, (bx1, by1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, tag_col, 1, cv2.LINE_AA)

            # ── Ghost box extrapolation (v5 feature) ─────────────────────
            elif 1 <= frames_lost <= GHOST_BOX_FRAMES:
                bx1, by1, bx2, by2 = st.smooth_bbox
                bw = max(bx2 - bx1, 1)
                bh = max(by2 - by1, 1)

                pred_cx, pred_cy = st.kalman.predict()
                pred_cx = int(np.clip(pred_cx, bw // 2, W - bw // 2))
                pred_cy = int(np.clip(pred_cy, bh // 2, H - bh // 2))

                gx1, gy1 = pred_cx - bw // 2, pred_cy - bh // 2
                gx2, gy2 = pred_cx + bw // 2, pred_cy + bh // 2

                alpha  = max(0.15, 0.6 - frames_lost * 0.07)
                ghost  = frame.copy()
                g_col  = COL_EXEMPT if st.exempt else COL_GHOST_BOX
                cv2.rectangle(ghost, (gx1, gy1), (gx2, gy2), g_col, 2)
                cv2.putText(ghost, f"#{tid} ?", (gx1, gy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, g_col, 1, cv2.LINE_AA)
                frame = cv2.addWeighted(ghost, alpha, frame, 1 - alpha, 0)

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

    # ─────────────────────────────────────────────────────────────────────
    #  CSV save
    # ─────────────────────────────────────────────────────────────────────

    def _save_csv(self, path: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "timestamp", "id",
                                               "event", "in", "out", "inside"])
            w.writeheader()
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

    # ─────────────────────────────────────────────────────────────────────
    #  Single-frame processing helper
    # ─────────────────────────────────────────────────────────────────────

    def _process_frame(self, frame_data: dict, line_x: int,
                       fps: float, writer: cv2.VideoWriter) -> np.ndarray:
        frame = frame_data["frame"]
        f_no  = frame_data["frame_no"]
        W, H  = frame.shape[1], frame.shape[0]

        # Detection + tracking
        track_args = dict(
            conf    = CONF_THRESH,
            iou     = IOU_THRESH,
            verbose = False,
            tracker = TRACKER_CONFIG,
            persist = True,
        )
        # person class = 0 for COCO / CrowdHuman; VisDrone uses class 1
        if not self.head_detect:
            track_args["classes"] = [0]

        results = self.model.track(frame, **track_args)[0]

        # Collect detected IDs this frame
        detected_ids: set = set()
        tids = results.boxes.id
        if tids is not None:
            detected_ids = set(tids.cpu().numpy().astype(int).tolist())
            for bbox, tid in zip(results.boxes.xyxy.cpu().numpy(),
                                 tids.cpu().numpy().astype(int)):
                self._update_track(int(tid), bbox, line_x, f_no, frame, fps)

        # Periodic ghost maintenance
        if f_no % 90 == 0:
            self._purge_ghosts(f_no)

        # Draw
        frame = self._draw_zones(frame, line_x)
        frame = self._draw_tracks(frame, detected_ids, f_no, W, H)
        frame = self._draw_dashboard(frame, f_no, fps)
        writer.write(frame)
        return frame

    # ─────────────────────────────────────────────────────────────────────
    #  Main loop
    # ─────────────────────────────────────────────────────────────────────

    def process(self):
        is_stream = self.video_path.startswith(
            ("rtsp://", "rtmp://", "http://", "https://"))

        if not is_stream and not os.path.exists(self.video_path):
            raise FileNotFoundError(
                f"Video file not found: {self.video_path}\n"
                f"Current directory: {os.getcwd()}")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise (ConnectionError if is_stream else FileNotFoundError)(
                f"Cannot open: {self.video_path}")

        W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        line_x = int(W * LINE_RATIO)

        writer = cv2.VideoWriter(
            self.output_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

        print(f"Video  : {W}×{H} @ {FPS:.1f} fps")
        print(f"Model  : {self.model.model_name if hasattr(self.model, 'model_name') else 'yolov8x.pt'}")
        print(f"Line   : x={line_x} ({LINE_RATIO*100:.0f}% width)  "
              f"Zone=[{line_x-DEAD_ZONE_PX}, {line_x+DEAD_ZONE_PX}]")
        print(f"Mode   : {'LIVE STREAM' if self.live else 'FILE'}"
              + (f" + {self.delay_seconds}s DELAY" if self.delay_seconds else ""))
        print("─" * 60)

        f_no            = 0
        last_csv_save   = 0
        csv_interval    = 300
        use_delay       = (not self.live) and (self.delay_seconds > 0)
        win_title       = "Bus Counter v6"

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if self.live:
                        print("[WARN] Stream interrupted — retrying…")
                        time.sleep(1)
                        continue
                    else:
                        # Drain remaining buffered frames
                        if use_delay:
                            while self.frame_buffer:
                                self._process_frame(
                                    self.frame_buffer.popleft(), line_x, FPS, writer)
                        break

                f_no += 1
                current_time = time.time()

                if use_delay:
                    self.frame_buffer.append({
                        "frame": frame.copy(),
                        "frame_no": f_no,
                        "timestamp": current_time,
                    })
                    while self.frame_buffer:
                        oldest = self.frame_buffer[0]
                        if current_time - oldest["timestamp"] >= self.delay_seconds:
                            bd = self.frame_buffer.popleft()
                            pf = self._process_frame(bd, line_x, FPS, writer)
                            if bd["frame_no"] - last_csv_save >= csv_interval:
                                self._save_csv(
                                    self.output_path.replace(".mp4", "_events.csv"))
                                last_csv_save = bd["frame_no"]
                            if not self.no_preview:
                                cv2.putText(pf,
                                    f"DELAY:{self.delay_seconds}s "
                                    f"buf:{len(self.frame_buffer)}",
                                    (10, H - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 255), 2, cv2.LINE_AA)
                                cv2.imshow(win_title, pf)
                                if cv2.waitKey(1) & 0xFF == ord("q"):
                                    return
                        else:
                            break
                    time.sleep(1.0 / FPS)

                else:
                    pf = self._process_frame({
                        "frame": frame, "frame_no": f_no,
                        "timestamp": current_time,
                    }, line_x, FPS, writer)

                    if self.live and f_no - last_csv_save >= csv_interval:
                        self._save_csv(
                            self.output_path.replace(".mp4", "_events.csv"))
                        last_csv_save = f_no

                    if not self.no_preview:
                        cv2.imshow(win_title, pf)
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
                all_keys = sorted({k for row in self.debug_log for k in row})
                debug_csv = os.path.join(self.debug_dir, "debug_log.csv")
                with open(debug_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=all_keys)
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
        description="Bus Passenger Counter v6 — Unified (counting + staff exempt + fast movers + API)")
    p.add_argument("--source",     default="counting.mp4",
                   help="Input video path or stream URL (rtsp/rtmp/http)")
    p.add_argument("--output",     default="result_v6.mp4",
                   help="Output annotated video path")
    p.add_argument("--model",      default="yolov8x.pt",
                   help="YOLO model weights (default: yolov8x.pt — best accuracy)")
    p.add_argument("--line",       type=float, default=LINE_RATIO,
                   help="Trigger line as fraction of frame width (default 0.45 — do not change)")
    p.add_argument("--no-preview", action="store_true",
                   help="Disable live preview window (faster headless runs)")
    p.add_argument("--live",       action="store_true",
                   help="Live-stream mode (all new IDs treated as starting outside)")
    p.add_argument("--delay",      type=int, default=0,
                   help="Processing delay in seconds for recorded video (default 0)")
    p.add_argument("--debug",      action="store_true",
                   help="Save per-frame HSV blob analysis images + CSV")
    p.add_argument("--head-detect",action="store_true",
                   help="Auto-download & use CrowdHuman head-detection model")
    p.add_argument("--visdrone",   action="store_true",
                   help="Auto-download & use VisDrone overhead model")
    args = p.parse_args()

    LINE_RATIO = args.line

    counter = BusCounter(
        video_path    = args.source,
        model_path    = args.model,
        output_path   = args.output,
        enable_debug  = args.debug,
        head_detect   = args.head_detect,
        visdrone      = args.visdrone,
        no_preview    = args.no_preview,
        live          = args.live,
        delay_seconds = args.delay,
    )
    counter.process()


if __name__ == "__main__":
    main()