#!/usr/bin/env python3
"""
Bus Passenger Counter — v5.0 BLOB-FIRST TAPE + GHOST BOX + BOX EMA + API PUSH
=======================================================================
Production-ready merge of v5.0 detection/counting logic with:
  • Non-blocking API POST via background daemon thread + queue (zero FPS impact)
  • Flask MJPEG stream server (/video_feed) for remote monitoring
  • Async frame reader (prevents RTMP socket timeout)
  • Frame-skip (process every Nth frame, buffer the rest)
  • BLOB-FIRST tape detection: immune to blur/dilution simultaneously
  • Ghost box extrapolation: Kalman-predicted box when detection absent
  • Centroid-distance fallback ghost re-link (fewer ID switches)
  • EMA smoothing on all 4 bounding-box corners (jitter-free visuals)
  • High-speed zone crossing override (counts even mid-dead-zone)
  • Hysteresis-based exemption scoring with temporal vote window
"""

import cv2
import numpy as np
import supervision as sv  # noqa: F401  (kept for compatibility)
from ultralytics import YOLO
import threading
import queue
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import csv, os, argparse
from flask import Flask, Response

# ── Shared frame for MJPEG stream ────────────────────────────────────────────
latest_frame = None
latest_frame_lock = threading.Lock()

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False
    print("[WARN] 'requests' not installed — API push disabled.  pip install requests")


# ═══════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════
LINE_RATIO      = 0.45
DEAD_ZONE_PX    = 30
DEBOUNCE_N      = 1
CONF_THRESH     = 0.25    # High enough to avoid flickering, low enough to catch persons
IOU_THRESH      = 0.45    # Allows tracker to match faster-moving heads
TRAIL_LEN       = 50
GHOST_TIMEOUT   = 300     # Keeps lost tracks alive through occlusions (~10 s @ 30 fps)
FLASH_FRAMES    = 20
EMA_ALPHA       = 0.40
RADIUM_MIN_PX   = 200     # Minimum yellow pixels for radium tape detection (legacy ref)
EXEMPT_CONFIRM  = 5       # Score needed to confirm exemption
EXEMPT_MAX      = 12      # Cap on exemption score
MERGE_AR_THRESH = 1.5     # W/H ratio above which bbox likely contains 2 people
MERGE_OVERLAP   = 0.30    # IoU above which two tracks are considered overlapping

# Motion-blur robustness
BLUR_VAR_THRESH = 40      # Laplacian variance below which a crop is too blurry
VOTE_WINDOW     = 7       # Sliding-window size for temporal tape voting
VOTE_MIN_HITS   = 2       # Detections needed within window to confirm tape
HYSTERESIS_MISS = 6       # Consecutive non-blurry misses to revoke exemption

# Ghost re-link
RELINK_DIST_PX  = 60      # Max centroid distance (px) to consider a ghost match
RELINK_MAX_AGE  = 45      # Ghost must have been seen within this many frames

# Blob-first tape detection
BLOB_MIN_RATIO   = 0.003  # Min blob area as fraction of crop area
BLOB_MAX_RATIO   = 0.25   # Max blob area as fraction of crop area
BLOB_MIN_SAT     = 100    # Min peak saturation within blob pixels
BLOB_MIN_VAL     = 120    # Min peak value within blob pixels
BLOB_COMPACT_MAX = 25.0   # Max (perimeter² / area); tape is compact, not scattered

# Ghost box extrapolation
GHOST_BOX_FRAMES = 45     # Frames to show ghost box (synced with track_buffer)

# Bounding-box EMA smoothing
BOX_EMA_ALPHA    = 0.45   # Higher = follows detection more; lower = smoother

# Frame processing
FRAME_SKIP       = 3      # Process every Nth frame (1 = process all)

# API push
API_ENDPOINT   = "https://9424-49-205-176-68.ngrok-free.app/api/passenger-count"
API_TIMEOUT    = 2        # seconds per request
API_MAX_RETRY  = 2        # retries on 5xx
API_RETRY_WAIT = 0.3      # seconds between retries

# ByteTrack tracker config
TRACKER_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bytetrack.yaml")

# Colours (BGR)
COL_LINE      = (0,   0,  220)
COL_BOX_NORM  = (0, 140,  255)
COL_BOX_FLASH = (255, 255,  0)
COL_BOX_SKIP  = (80,  80,  80)
COL_IN_TEXT   = (80, 255,  80)
COL_OUT_TEXT  = (80, 180, 255)
COL_EXEMPT    = (255,  0,  200)


# ═══════════════════════════════════════════════════════
#  NON-BLOCKING API PUSH WORKER
# ═══════════════════════════════════════════════════════

class ApiPushWorker:
    """
    Background daemon thread draining a queue and firing POST requests.
    The main video loop enqueues a payload and returns immediately —
    network latency or retries never touch the processing thread.
    """

    def __init__(self, endpoint: str,
                 timeout: float = API_TIMEOUT,
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


# ═══════════════════════════════════════════════════════
#  TRACK STATE & KALMAN FILTER
# ═══════════════════════════════════════════════════════

@dataclass
class TrackState:
    kalman:        'KalmanCentroid'
    cx:            float
    cy:            float
    trail:         deque = field(default_factory=lambda: deque(maxlen=TRAIL_LEN))
    zone_state:    str   = "INIT"
    counted:       Optional[str] = None
    flash:         int   = 0
    last_seen:     int   = 0
    side_frames:   int   = 0
    prev_side:     str   = ""
    debounce_side: str   = ""
    exempt:        bool  = False
    exempt_score:  int   = 0
    last_bbox:     tuple = field(default_factory=lambda: (0, 0, 0, 0))
    smooth_bbox:   tuple = field(default_factory=lambda: (0, 0, 0, 0))
    tape_votes:    deque = field(default_factory=lambda: deque(maxlen=VOTE_WINDOW))
    consec_miss:   int   = 0


class KalmanCentroid:
    def __init__(self, cx: float, cy: float):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix   = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix    = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                                [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        # Raised process noise: top-down heads move faster and less predictably
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.5
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kf.statePost           = np.array([[cx], [cy], [0], [0]], np.float32)

    def update(self, cx: float, cy: float):
        self.kf.predict()
        meas = np.array([[cx], [cy]], np.float32)
        res  = self.kf.correct(meas)
        return float(res[0, 0]), float(res[1, 0])

    def predict(self) -> tuple:
        """Return predicted position without a measurement update (for ghost display)."""
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])


# ═══════════════════════════════════════════════════════
#  MAIN COUNTER CLASS
# ═══════════════════════════════════════════════════════

class BusCounter:
    def __init__(self, video_path: str, model_path: str = "yolov8x.pt",
                 output_path: str = "result_v5_prod.mp4",
                 enable_debug: bool = False, head_detect: bool = False,
                 visdrone: bool = False):
        self.video_path  = video_path
        self.output_path = output_path
        self.head_detect = head_detect

        # Model selection priority: visdrone > head_detect > default
        if visdrone:
            model_path = self._ensure_visdrone_model()
        elif head_detect and model_path == "yolov8x.pt":
            model_path = self._ensure_head_model()

        self.model      = YOLO(model_path)
        self.states: dict[int, TrackState] = {}
        self.in_count   = 0
        self.out_count  = 0
        self.events: list[dict] = []

        # Ghost re-link table: maps lost track_id → last known TrackState
        self._ghost_pool: dict[int, TrackState] = {}

        # Non-blocking API push worker
        self.api = ApiPushWorker(API_ENDPOINT, API_TIMEOUT, API_MAX_RETRY, API_RETRY_WAIT)

        self._ensure_bytetrack_yaml()

        # Debug setup
        self.enable_debug = enable_debug
        self.debug_log: list[dict] = []
        if self.enable_debug:
            self.debug_dir          = "debug_integrated"
            self.debug_crops_dir    = os.path.join(self.debug_dir, "crops")
            self.debug_analysis_dir = os.path.join(self.debug_dir, "analysis")
            os.makedirs(self.debug_dir,          exist_ok=True)
            os.makedirs(self.debug_crops_dir,    exist_ok=True)
            os.makedirs(self.debug_analysis_dir, exist_ok=True)
            print(f"[DEBUG MODE] Enabled - Output: {self.debug_dir}/")

    # ── Model helpers ──────────────────────────────────────────────────────────

    def _ensure_head_model(self) -> str:
        import urllib.request
        head_model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "yolov8n_crowdhuman.pt"
        )
        if not os.path.exists(head_model_path):
            url = "https://github.com/yakhyo/yolov8-crowdhuman/releases/download/weights/yolov8n_best.pt"
            print(f"[HEAD-DETECT] Downloading CrowdHuman model from {url}...")
            urllib.request.urlretrieve(url, head_model_path)
            print(f"[HEAD-DETECT] Saved to {head_model_path}")
        else:
            print(f"[HEAD-DETECT] Using existing model: {head_model_path}")
        return head_model_path

    def _ensure_visdrone_model(self) -> str:
        return "yolov8x.pt"

    def _ensure_bytetrack_yaml(self):
        if os.path.exists(TRACKER_CONFIG):
            return
        yaml_content = (
            "tracker_type: bytetrack\n"
            "track_high_thresh: 0.25\n"
            "track_low_thresh:  0.05\n"
            "new_track_thresh:  0.20\n"
            "track_buffer:      45\n"
            "match_thresh:      0.85\n"
            "fuse_score:        true\n"
        )
        with open(TRACKER_CONFIG, "w") as f:
            f.write(yaml_content)
        print(f"[BYTETRACK] Config written to {TRACKER_CONFIG}")

    # ── Ghost re-link ──────────────────────────────────────────────────────────

    def _relink_ghost(self, new_tid: int, raw_cx: float, raw_cy: float, f_no: int) -> int:
        """
        Before creating a brand-new TrackState for new_tid, search the ghost pool
        for a recently-lost track whose last centroid is within RELINK_DIST_PX.
        Migrates the ghost state to new_tid and resets counting flags.
        """
        best_tid  = None
        best_dist = float("inf")

        for ghost_tid, ghost_st in self._ghost_pool.items():
            age = f_no - ghost_st.last_seen
            if age > RELINK_MAX_AGE:
                continue
            dist = ((ghost_st.cx - raw_cx) ** 2 + (ghost_st.cy - raw_cy) ** 2) ** 0.5
            if dist < RELINK_DIST_PX and dist < best_dist:
                best_dist = dist
                best_tid  = ghost_tid

        if best_tid is not None:
            adopted = self._ghost_pool.pop(best_tid)
            adopted.counted       = None
            adopted.zone_state    = "INIT"
            adopted.flash         = 0
            adopted.side_frames   = 0
            adopted.prev_side     = ""
            adopted.debounce_side = ""
            adopted.trail.clear()
            self.states[new_tid] = adopted
            print(f"  🔗 [RE-LINK] Ghost ID {best_tid} re-adopted as ID {new_tid} "
                  f"(dist={best_dist:.1f}px, age={f_no - adopted.last_seen}f) "
                  f"— counting state RESET")
            return new_tid

        return new_tid

    # ── Utility helpers ────────────────────────────────────────────────────────

    def _get_side(self, cx: float, line_x: int) -> str:
        if cx < line_x - DEAD_ZONE_PX: return "L"
        if cx > line_x + DEAD_ZONE_PX: return "R"
        return "ZONE"

    def _is_blurry(self, crop: np.ndarray) -> tuple:
        gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return (lap_var < BLUR_VAR_THRESH, lap_var)

    def _is_merged_box(self, bbox) -> bool:
        x1, y1, x2, y2 = bbox.astype(int)
        w, h = x2 - x1, y2 - y1
        if h <= 0: return True
        return (w / h) > MERGE_AR_THRESH

    def _overlaps_other_tracks(self, tid, bbox_ints, f_no) -> bool:
        x1, y1, x2, y2 = bbox_ints
        area1 = (x2 - x1) * (y2 - y1)
        if area1 <= 0: return False
        for oid, ost in self.states.items():
            if oid == tid or f_no - ost.last_seen > 5: continue
            ox1, oy1, ox2, oy2 = ost.last_bbox
            ix1, iy1 = max(x1, ox1), max(y1, oy1)
            ix2, iy2 = min(x2, ox2), min(y2, oy2)
            if ix1 >= ix2 or iy1 >= iy2: continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            area2 = (ox2 - ox1) * (oy2 - oy1)
            union = area1 + area2 - inter
            if union > 0 and (inter / union) > MERGE_OVERLAP:
                return True
        return False

    # ── Tape detection ─────────────────────────────────────────────────────────

    def _has_radium_tape(self, crop: np.ndarray, tid: int,
                         frame_no: int = 0, fps: float = 30.0) -> bool:
        """
        Detect radium tape (orange-yellow) using blob-first analysis.

        Pipeline:
          1. Blur gate  — skip blurry crops (score unchanged)
          2. Denoise    — bilateral filter to preserve edges
          3. ROI        — top 20–60% of crop (shoulder band)
          4. HSV mask   — wide S lower-bound to tolerate blur smear
          5. Morphology — clean noise from mask
          6. Contours   — find individual yellow blobs
          7. Per-blob   — filter by area ratio, compactness, peak S and V
          8. Decision   — any blob passing all filters → tape detected
        """
        if crop is None or crop.size == 0:
            return False

        crop_h, crop_w = crop.shape[:2]
        crop_area      = crop_h * crop_w

        # Step 1: Blur gate
        blurry, lap_var = self._is_blurry(crop)
        print(f"     📷 Sharpness (Laplacian var): {lap_var:.1f} "
              f"({'BLURRY — detection skipped' if blurry else 'SHARP — proceeding'})")
        if blurry:
            return False

        # Step 2: Denoise
        crop_clean = cv2.bilateralFilter(crop, d=9, sigmaColor=75, sigmaSpace=75)
        hsv        = cv2.cvtColor(crop_clean, cv2.COLOR_BGR2HSV)

        # Step 3: Shoulder ROI (top 20–60%)
        roi_top      = int(crop_h * 0.20)
        roi_bottom   = int(crop_h * 0.60)
        shoulder_roi = hsv[roi_top:roi_bottom, :]
        roi_area     = shoulder_roi.shape[0] * shoulder_roi.shape[1]

        # Step 4: HSV mask
        lower_yellow = np.array([ 5,  15,  80])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow  = cv2.inRange(shoulder_roi, lower_yellow, upper_yellow)

        # Step 5: Morphological cleanup
        kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN,  kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

        yellow_px = int(np.sum(mask_yellow > 0))
        print(f"     🟡 Yellow Pixels in ROI: {yellow_px}px "
              f"({100*yellow_px/max(roi_area,1):.1f}% of shoulder ROI)")

        if yellow_px == 0:
            print(f"     ❌ No yellow pixels found in shoulder ROI")
            return False

        # Step 6: Find blobs
        contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        print(f"     🔍 Found {len(contours)} yellow blob(s) — analysing each...")

        # Step 7: Per-blob validation
        tape_blob_found = False
        for i, cnt in enumerate(contours):
            blob_area = cv2.contourArea(cnt)
            if blob_area < 4:
                continue

            ratio       = blob_area / crop_area
            perimeter   = cv2.arcLength(cnt, True)
            compactness = (perimeter ** 2) / blob_area if blob_area > 0 else 9999.0

            blob_mask   = np.zeros(mask_yellow.shape, dtype=np.uint8)
            cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
            blob_s_vals = shoulder_roi[:, :, 1][blob_mask > 0]
            blob_v_vals = shoulder_roi[:, :, 2][blob_mask > 0]

            peak_s = int(np.max(blob_s_vals)) if len(blob_s_vals) > 0 else 0
            peak_v = int(np.max(blob_v_vals)) if len(blob_v_vals) > 0 else 0
            mean_s = float(np.mean(blob_s_vals)) if len(blob_s_vals) > 0 else 0.0

            print(f"        Blob {i+1}: area={blob_area:.0f}px "
                  f"ratio={ratio:.4f} compact={compactness:.1f} "
                  f"peak_S={peak_s} peak_V={peak_v} mean_S={mean_s:.1f}")

            if ratio < BLOB_MIN_RATIO:
                print(f"           → ❌ Too small (ratio {ratio:.4f} < {BLOB_MIN_RATIO})"); continue
            if ratio > BLOB_MAX_RATIO:
                print(f"           → ❌ Too large (ratio {ratio:.4f} > {BLOB_MAX_RATIO})"); continue
            if compactness > BLOB_COMPACT_MAX:
                print(f"           → ❌ Not compact ({compactness:.1f} > {BLOB_COMPACT_MAX})"); continue
            if peak_s < BLOB_MIN_SAT:
                print(f"           → ❌ Peak saturation too low ({peak_s} < {BLOB_MIN_SAT})"); continue
            if peak_v < BLOB_MIN_VAL:
                print(f"           → ❌ Peak value too low ({peak_v} < {BLOB_MIN_VAL})"); continue

            print(f"           → ✅ TAPE BLOB CONFIRMED "
                  f"(ratio={ratio:.4f}, compact={compactness:.1f}, "
                  f"peak_S={peak_s}, peak_V={peak_v})")
            tape_blob_found = True
            break

        # Step 8: Debug image
        if self.enable_debug:
            debug_data = {
                'frame':              frame_no,
                'timestamp':          frame_no / fps,
                'track_id':           tid,
                'crop_width':         crop_w,
                'crop_height':        crop_h,
                'laplacian_variance': round(lap_var, 2),
                'yellow_pixels':      yellow_px,
                'num_blobs':          len(contours),
                'would_detect':       tape_blob_found,
                'rejection_reason':   'DETECTED' if tape_blob_found else 'No valid tape blob',
            }
            self._save_debug_image(crop_clean, hsv, mask_yellow,
                                   debug_data, tid, frame_no, roi_top)
            self.debug_log.append(debug_data)

        if tape_blob_found:
            print(f"\n{'🟢'*40}")
            print(f"  ⚠️  EXEMPT MATCH DETECTED — PLEASE VERIFY!")
            print(f"  ID: {tid} | Frame: {frame_no:05d} | Yellow: {yellow_px}px")
            print(f"{'🟢'*40}\n")
        else:
            print(f"     ❌ No blob passed all tape filters")

        return tape_blob_found

    def _save_debug_image(self, crop: np.ndarray, hsv: np.ndarray, mask_yellow: np.ndarray,
                          debug_data: dict, tid: int, frame_no: int, roi_top: int = 0):
        if not self.enable_debug:
            return

        crop_h, crop_w = crop.shape[:2]
        canvas_w = crop_w * 3
        canvas_h = crop_h * 2
        canvas   = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        canvas[0:crop_h, 0:crop_w] = crop

        full_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        roi_rows  = mask_yellow.shape[0]
        full_mask[roi_top:roi_top + roi_rows, 0:mask_yellow.shape[1]] = mask_yellow
        canvas[0:crop_h, crop_w:2*crop_w] = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR)

        shoulder_vis = crop.copy()
        roi_bottom   = roi_top + roi_rows
        cv2.rectangle(shoulder_vis, (0, roi_top), (crop_w, roi_bottom), (0, 255, 255), 2)
        canvas[0:crop_h, 2*crop_w:3*crop_w] = shoulder_vis

        h_vis = cv2.applyColorMap((hsv[:, :, 0] * 2).astype(np.uint8), cv2.COLORMAP_HSV)
        s_vis = cv2.cvtColor(hsv[:, :, 1], cv2.COLOR_GRAY2BGR)
        v_vis = cv2.cvtColor(hsv[:, :, 2], cv2.COLOR_GRAY2BGR)

        canvas[crop_h:2*crop_h, 0:crop_w]              = h_vis
        canvas[crop_h:2*crop_h, crop_w:2*crop_w]       = s_vis
        canvas[crop_h:2*crop_h, 2*crop_w:3*crop_w]     = v_vis

        for x, y, txt in [
            (10,              20,          "Original"),
            (crop_w+10,       20,          "Yellow Mask"),
            (2*crop_w+10,     20,          "Shoulder ROI"),
            (10,              crop_h+20,   "Hue"),
            (crop_w+10,       crop_h+20,   "Saturation"),
            (2*crop_w+10,     crop_h+20,   "Value"),
        ]:
            cv2.putText(canvas, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        status = "DETECTED" if debug_data.get('would_detect') else "REJECTED"
        color  = (0, 255, 0) if debug_data.get('would_detect') else (0, 0, 255)
        cv2.putText(canvas, status, (10, crop_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if not debug_data.get('would_detect'):
            cv2.putText(canvas, debug_data.get('rejection_reason', ''),
                        (10, crop_h-110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.imwrite(os.path.join(self.debug_analysis_dir,
                                 f"analysis_f{frame_no:05d}_id{tid:03d}.jpg"), canvas)
        cv2.imwrite(os.path.join(self.debug_crops_dir,
                                 f"crop_f{frame_no:05d}_id{tid:03d}.jpg"), crop)

    # ── Track update ───────────────────────────────────────────────────────────

    def _update_track(self, tid: int, bbox, line_x: int, f_no: int,
                      frame: np.ndarray, fps: float = 30.0):
        x1, y1, x2, y2 = bbox.astype(int)
        raw_cx, raw_cy = (x1+x2)/2.0, (y1+y2)/2.0

        print(f"\n{'='*80}")
        print(f"  FRAME {f_no:05d} | ID {tid:2d} | Timestamp: {f_no/fps:.2f}s")
        print(f"{'='*80}")

        # Ghost re-link on new track
        if tid not in self.states:
            self._relink_ghost(tid, raw_cx, raw_cy, f_no)

        # Create fresh state if still not present
        if tid not in self.states:
            side = self._get_side(raw_cx, line_x)
            st   = TrackState(kalman=KalmanCentroid(raw_cx, raw_cy),
                              cx=raw_cx, cy=raw_cy, last_seen=f_no)
            if   side == "L": st.zone_state, st.prev_side = "OUTSIDE", "L"
            elif side == "R": st.zone_state, st.prev_side = "INSIDE",  "R"
            else:             st.zone_state = "SKIP"
            self.states[tid] = st

        st           = self.states[tid]
        st.last_seen = f_no

        # EMA smoothing on bounding-box corners
        sx1, sy1, sx2, sy2 = st.smooth_bbox
        if sx1 == 0 and sy1 == 0 and sx2 == 0 and sy2 == 0:
            st.smooth_bbox = (x1, y1, x2, y2)
        else:
            a = BOX_EMA_ALPHA
            st.smooth_bbox = (
                int(a*x1 + (1-a)*sx1), int(a*y1 + (1-a)*sy1),
                int(a*x2 + (1-a)*sx2), int(a*y2 + (1-a)*sy2),
            )
        st.last_bbox = (x1, y1, x2, y2)

        # Tape / exemption detection
        w_box, h_box = x2-x1, y2-y1
        ar = (w_box/h_box) if h_box > 0 else 999.0
        merged      = self._is_merged_box(bbox)
        overlapping = self._overlaps_other_tracks(tid, (x1, y1, x2, y2), f_no)

        print(f"  📦 Bounding Box: {w_box:3d}x{h_box:3d} | Aspect Ratio: {ar:.2f}")
        print(f"  📍 Position: ({x1}, {y1}) → ({x2}, {y2})")

        tape_found = False
        if merged:
            print(f"  ⚠️  Status: MERGED BOX (AR > {MERGE_AR_THRESH}) - Skipping tape detection")
        elif overlapping:
            print(f"  ⚠️  Status: OVERLAPPING - Skipping tape detection")
        else:
            print(f"  ✓ Status: Clean single-person box - Running tape detection...")
            print(f"  {'-'*76}")
            crop = frame[max(0,y1):min(frame.shape[0],y2),
                         max(0,x1):min(frame.shape[1],x2)]

            blurry, lap_var = self._is_blurry(crop)
            if blurry:
                print(f"  ⚠️  Crop too blurry (Lap var={lap_var:.1f} < {BLUR_VAR_THRESH}) "
                      f"— score unchanged")
            else:
                tape_found = self._has_radium_tape(crop, tid, f_no, fps)
                print(f"  {'-'*76}")

                # Temporal vote window
                st.tape_votes.append(1 if tape_found else 0)
                vote_hits = sum(st.tape_votes)

                if tape_found:
                    st.exempt_score = min(st.exempt_score + 3, EXEMPT_MAX)
                    st.consec_miss  = 0
                    print(f"  ✅ RADIUM TAPE DETECTED → Score +3 | "
                          f"Votes: {vote_hits}/{len(st.tape_votes)}")
                else:
                    st.consec_miss += 1
                    if st.consec_miss >= HYSTERESIS_MISS:
                        st.exempt_score = max(st.exempt_score - 1, 0)
                        print(f"  ❌ NO TAPE — {st.consec_miss} consec misses → Score -1 | "
                              f"Votes: {vote_hits}/{len(st.tape_votes)}")
                    else:
                        print(f"  ❌ NO TAPE — consec miss {st.consec_miss}/{HYSTERESIS_MISS} "
                              f"(score held) | Votes: {vote_hits}/{len(st.tape_votes)}")

        prev_exempt = st.exempt
        if st.exempt_score >= EXEMPT_CONFIRM:
            st.exempt = True
        elif st.exempt_score == 0 and st.consec_miss >= HYSTERESIS_MISS:
            st.exempt = False

        print(f"\n  📊 EXEMPTION STATUS:")
        print(f"     Current Score: {st.exempt_score:2d}/{EXEMPT_CONFIRM} (max: {EXEMPT_MAX})")
        print(f"     Exempt: {'YES ✓' if st.exempt else 'NO ✗'}")

        if st.exempt and not prev_exempt:
            print(f"\n{'⭐'*40}")
            print(f"  🎉 [EXEMPT GRANTED] ID {tid} | Score: {st.exempt_score}/{EXEMPT_CONFIRM}")
            print(f"  ⚠️  VERIFY: Is this person wearing radium tape?")
            print(f"{'⭐'*40}\n")
        elif not st.exempt and prev_exempt:
            print(f"\n{'⚠️ '*20}")
            print(f"  ⚠️  [EXEMPT REVOKED] ID {tid} (score={st.exempt_score})")
            print(f"{'⚠️ '*20}\n")

        print(f"{'='*80}\n")

        # Kalman + EMA centroid update
        pcx, pcy = st.kalman.update(raw_cx, raw_cy)
        st.cx = EMA_ALPHA * raw_cx + (1 - EMA_ALPHA) * pcx
        st.cy = EMA_ALPHA * raw_cy + (1 - EMA_ALPHA) * pcy
        st.trail.append((int(st.cx), int(st.cy)))
        if st.flash > 0:
            st.flash -= 1

        # Exempt tracks are never counted
        if st.exempt:
            return

        # ── Counting logic (from v5.0 file 2) ────────────────────────
        side = self._get_side(st.cx, line_x)

        if side == "ZONE":
            # High-speed override: count if centroid jumps the line mid-zone
            curr_strict = "L" if st.cx < line_x else "R"
            if curr_strict != st.prev_side:
                self._perform_counting(tid, st, line_x, f_no, fps)
                st.prev_side = curr_strict
            st.zone_state    = "ZONE"
            st.side_frames   = 0
            st.debounce_side = ""
        else:
            if st.debounce_side != side:
                st.debounce_side, st.side_frames = side, 1
            else:
                st.side_frames += 1

            if st.side_frames >= DEBOUNCE_N:
                self._perform_counting(tid, st, line_x, f_no, fps)
                st.zone_state = "OUTSIDE" if side == "L" else "INSIDE"
                st.prev_side  = side

    def _perform_counting(self, tid: int, st: TrackState,
                          line_x: int, f_no: int, fps: float):
        """Core crossing detection and IN/OUT count trigger with API push."""
        curr_strict = "L" if st.cx < line_x else "R"
        crossed_in  = (curr_strict == "R" and st.prev_side == "L")
        crossed_out = (curr_strict == "L" and st.prev_side == "R")

        if crossed_out and st.counted != "OUT":
            self.out_count += 1
            st.counted, st.flash = "OUT", FLASH_FRAMES
            self.events.append({"frame": f_no, "id": tid, "event": "OUT",
                                 "in": self.in_count, "out": self.out_count})
            self.api.push(self.in_count, self.out_count)
            print(f"\n{'🚪'*40}")
            print(f"  🚶 PERSON LEFT THE BUS")
            print(f"     ID: {tid} | Frame: {f_no:05d} | Time: {f_no/fps:.2f}s")
            print(f"     📊 Total Count: IN={self.in_count} | OUT={self.out_count}")
            print(f"{'🚪'*40}\n")

        elif crossed_in and st.counted != "IN":
            self.in_count += 1
            st.counted, st.flash = "IN", FLASH_FRAMES
            self.events.append({"frame": f_no, "id": tid, "event": "IN",
                                 "in": self.in_count, "out": self.out_count})
            self.api.push(self.in_count, self.out_count)
            print(f"\n{'🚪'*40}")
            print(f"  🚶 PERSON ENTERED THE BUS")
            print(f"     ID: {tid} | Frame: {f_no:05d} | Time: {f_no/fps:.2f}s")
            print(f"     📊 Total Count: IN={self.in_count} | OUT={self.out_count}")
            print(f"{'🚪'*40}\n")

    # ── Main processing loop ───────────────────────────────────────────────────

    def process(self):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        if not cap.isOpened():
            print(f"ERROR: Cannot open source: {self.video_path}")
            return

        W, H    = int(cap.get(3)), int(cap.get(4))
        FPS     = cap.get(5) or 30.0
        line_x  = int(W * LINE_RATIO)
        writer  = cv2.VideoWriter(self.output_path,
                                  cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

        # Async frame reader: always drains the network socket so RTMP
        # connections never time out regardless of AI processing speed.
        frame_queue = queue.Queue(maxsize=1500)
        stop_event  = threading.Event()

        def reader_thread():
            f_count = 0
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    frame_queue.put(None)
                    break
                f_count += 1
                frame_queue.put((f_count, frame))

        reader = threading.Thread(target=reader_thread, daemon=True)
        reader.start()
        print(f"[STREAM] Started async reader for {self.video_path}")

        try:
            while True:
                try:
                    item = frame_queue.get(timeout=10.0)
                except queue.Empty:
                    print("[WARN] Stream data delayed - waiting...")
                    continue

                if item is None:
                    break

                f_no, frame = item

                # Frame skip: buffer frames are read (preventing socket timeout)
                # but only every FRAME_SKIP-th frame is processed by the AI.
                if f_no % FRAME_SKIP != 0:
                    continue

                if f_no % 30 == 0:
                    q_size = frame_queue.qsize()
                    print(f"[PROCESSING] Frame {f_no} | "
                          f"Buffer: {q_size} frames ({q_size/FPS:.1f}s delay)")

                track_args = dict(
                    conf    = CONF_THRESH,
                    iou     = IOU_THRESH,
                    verbose = False,
                    tracker = TRACKER_CONFIG,
                    persist = True,
                )
                if not self.head_detect:
                    track_args["classes"] = [0]

                res = self.model.track(frame, **track_args)[0]
                if res.boxes.id is not None:
                    for b, tid in zip(res.boxes.xyxy.cpu().numpy(),
                                      res.boxes.id.cpu().numpy().astype(int)):
                        self._update_track(tid, b, line_x, f_no, frame, FPS)

                # Prune timed-out active tracks → ghost pool every 90 processed frames
                if f_no % 90 == 0:
                    alive, expired = {}, {}
                    for t_id, s in self.states.items():
                        if f_no - s.last_seen < GHOST_TIMEOUT:
                            alive[t_id] = s
                        else:
                            expired[t_id] = s
                    self.states = alive
                    self._ghost_pool.update(expired)
                    self._ghost_pool = {
                        g_id: g_st for g_id, g_st in self._ghost_pool.items()
                        if f_no - g_st.last_seen < GHOST_TIMEOUT * 2
                    }

                frame = self._draw_ui(frame, line_x, res, f_no)
                writer.write(frame)

                # Publish latest frame for MJPEG stream (thread-safe)
                global latest_frame
                with latest_frame_lock:
                    latest_frame = frame.copy()

        finally:
            stop_event.set()
            cap.release()
            writer.release()
            cv2.destroyAllWindows()

        # Save event CSV
        with open(self.output_path.replace(".mp4", ".csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "id", "event", "in", "out"])
            w.writeheader()
            w.writerows(self.events)

        # Save debug CSV
        if self.enable_debug and self.debug_log:
            debug_csv_path = os.path.join(self.debug_dir, "debug_log.csv")
            fieldnames     = list(self.debug_log[0].keys())
            with open(debug_csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(self.debug_log)
            print(f"\n[DEBUG] CSV log saved: {debug_csv_path} ({len(self.debug_log)} entries)")

    # ── UI rendering ───────────────────────────────────────────────────────────

    def _draw_ui(self, frame, line_x, res, f_no: int = 0):
        h, w = frame.shape[:2]

        # Zone overlays
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (line_x - DEAD_ZONE_PX, h), (180, 60, 0), -1)
        cv2.rectangle(ov, (line_x + DEAD_ZONE_PX, 0), (w, h), (0, 140, 0), -1)
        frame = cv2.addWeighted(ov, 0.06, frame, 0.94, 0)
        cv2.line(frame, (line_x, 0), (line_x, h), COL_LINE, 2)

        # Dashboard panel
        dx1, dy1, dw, dh = w - 220, 20, 200, 120
        ov2 = frame.copy()
        cv2.rectangle(ov2, (dx1, dy1), (dx1 + dw, dy1 + dh), (15, 15, 15), -1)
        frame = cv2.addWeighted(ov2, 0.7, frame, 0.3, 0)
        cv2.rectangle(frame, (dx1, dy1), (dx1 + dw, dy1 + 3), (0, 200, 80), -1)
        for i, (lbl, val, col) in enumerate([
            ("ENTERED", self.in_count,                          COL_IN_TEXT),
            ("LEFT",    self.out_count,                         COL_OUT_TEXT),
            ("INSIDE",  max(0, self.in_count - self.out_count), (80, 220, 255)),
        ]):
            cv2.putText(frame, f"{lbl}: {val}", (dx1 + 15, dy1 + 35 + i * 30),
                        0, 0.55, col, 2)

        # Currently detected IDs
        detected_ids: set = set()
        if res.boxes.id is not None:
            detected_ids = set(res.boxes.id.cpu().numpy().astype(int).tolist())

        # Draw EMA-smoothed detection boxes + trails
        if res.boxes.id is not None:
            for b, tid in zip(res.boxes.xyxy.cpu().numpy(),
                              res.boxes.id.cpu().numpy().astype(int)):
                st = self.states.get(tid)
                if not st:
                    continue
                bx1, by1, bx2, by2 = st.smooth_bbox
                col = (COL_EXEMPT     if st.exempt
                       else COL_BOX_SKIP  if st.zone_state == "SKIP"
                       else COL_BOX_FLASH if st.flash > 0
                       else COL_BOX_NORM)
                thickness = 3 if st.flash > 0 else 2
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, thickness)
                label = f"#{tid}" + (" EXEMPT" if st.exempt else "")
                cv2.putText(frame, label, (bx1, by1 - 10), 0, 0.5, col, 1)

                pts = list(st.trail)
                for j in range(1, len(pts)):
                    t = j / max(len(pts) - 1, 1)
                    cv2.line(frame, pts[j-1], pts[j],
                             (int(255*(1-t)), int(200*t), int(120*t)), 2)

        # Ghost box extrapolation for recently-lost tracks
        for tid, st in self.states.items():
            if tid in detected_ids:
                continue
            frames_lost = f_no - st.last_seen
            if frames_lost < 1 or frames_lost > GHOST_BOX_FRAMES:
                continue

            bx1, by1, bx2, by2 = st.smooth_bbox
            bw = max(bx2 - bx1, 1)
            bh = max(by2 - by1, 1)

            pred_cx, pred_cy = st.kalman.predict()
            pred_cx = int(np.clip(pred_cx, bw//2, w - bw//2))
            pred_cy = int(np.clip(pred_cy, bh//2, h - bh//2))

            gx1, gy1 = pred_cx - bw//2, pred_cy - bh//2
            gx2, gy2 = pred_cx + bw//2, pred_cy + bh//2

            alpha  = max(0.15, 0.6 - frames_lost * 0.07)
            ghost  = frame.copy()
            g_col  = COL_EXEMPT if st.exempt else (180, 180, 60)
            cv2.rectangle(ghost, (gx1, gy1), (gx2, gy2), g_col, 2)
            cv2.putText(ghost, f"#{tid} ?", (gx1, gy1-10), 0, 0.45, g_col, 1)
            frame = cv2.addWeighted(ghost, alpha, frame, 1-alpha, 0)

        return frame


# ═══════════════════════════════════════════════════════
#  FLASK MJPEG STREAM SERVER
# ═══════════════════════════════════════════════════════

app = Flask(__name__)


def _generate_mjpeg():
    """Generator that yields MJPEG frames from the shared latest_frame."""
    global latest_frame
    while True:
        with latest_frame_lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.01)
            continue

        ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if not ok:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + buf.tobytes()
            + b'\r\n'
        )


@app.route('/video_feed')
def video_feed():
    return Response(_generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/health')
def health():
    """Lightweight health-check endpoint."""
    from flask import jsonify
    inside = max(0, 0)  # populated at runtime via counter reference
    return jsonify({"status": "ok"})


# ═══════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Bus Passenger Counter v5.0 Production")
    p.add_argument("--source",      default="rtmp://54.163.94.60/live/stream",
                   help="Video source: file path or RTMP URL")
    p.add_argument("--output",      default="result_v5_prod.mp4")
    p.add_argument("--model",       default="yolov8x.pt",
                   help="Path to YOLO model weights")
    p.add_argument("--debug",       action="store_true",
                   help="Enable detailed HSV debugging (saves CSV + images)")
    p.add_argument("--head-detect", action="store_true",
                   help="Use CrowdHuman-trained model for head/partial body detection")
    p.add_argument("--visdrone",    action="store_true",
                   help="Use YOLOv8x (best for top-down bus-camera views)")
    p.add_argument("--port",        type=int, default=5000,
                   help="Flask MJPEG stream port (default: 5000)")
    p.add_argument("--api-endpoint", default=API_ENDPOINT,
                   help="Override the API push endpoint URL")
    args = p.parse_args()

    # Allow CLI override of the API endpoint
    if args.api_endpoint != API_ENDPOINT:
        API_ENDPOINT = args.api_endpoint

    counter = BusCounter(
        video_path   = args.source,
        model_path   = args.model,
        output_path  = args.output,
        enable_debug = args.debug,
        head_detect  = args.head_detect,
        visdrone     = args.visdrone,
    )

    # Start the counter in a background daemon thread
    processing_thread = threading.Thread(target=counter.process, daemon=True)
    processing_thread.start()
    print(f"[MAIN] Counter thread started")
    print(f"[MAIN] MJPEG stream available at  http://0.0.0.0:{args.port}/video_feed")
    print(f"[MAIN] Health check available at   http://0.0.0.0:{args.port}/health")

    # Run Flask in the main thread (threaded=True handles concurrent viewers)
    app.run(host="0.0.0.0", port=args.port, threaded=True)
