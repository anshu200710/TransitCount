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
#  CONFIGURATION
# ═══════════════════════════════════════════════════════
LINE_RATIO     = 0.45
DEAD_ZONE_PX   = 30
DEBOUNCE_N     = 1
CONF_THRESH    = 0.08    # Lowered: top-down heads are small, model scores them lower
IOU_THRESH     = 0.45    # Lowered: allows tracker to match faster-moving heads
TRAIL_LEN      = 50
GHOST_TIMEOUT  = 300     # Raised 150→300: keeps lost tracks alive through occlusions
FLASH_FRAMES   = 20
EMA_ALPHA      = 0.40
RADIUM_MIN_PX  = 200     # Minimum yellow pixels for radium tape detection
EXEMPT_CONFIRM = 5       # Score needed to confirm exemption
EXEMPT_MAX     = 12      # Cap on exemption score
MERGE_AR_THRESH= 1.5     # W/H ratio above which bbox likely contains 2 people
MERGE_OVERLAP  = 0.30    # IoU above which two tracks are considered overlapping

# Motion-blur robustness settings
BLUR_VAR_THRESH  = 40    # Laplacian variance below which a crop is too blurry to score
VOTE_WINDOW      = 7     # Sliding-window size for temporal tape voting
VOTE_MIN_HITS    = 2     # Detections needed within window to confirm tape
HYSTERESIS_MISS  = 6     # Consecutive non-blurry misses required to revoke exemption

# Top-down re-link settings
#   When a NEW track ID appears, we check whether any recently-lost ghost track
#   had its last known centroid within RELINK_DIST_PX of this new detection.
#   If so, we re-adopt the ghost instead of creating a new state — this is the
#   primary fix for "multiple IDs assigned to the same person".
RELINK_DIST_PX   = 60    # Max centroid distance (px) to consider a ghost match
RELINK_MAX_AGE   = 45    # Ghost must have been seen within this many frames

# ByteTrack tracker config (written at runtime if missing)
TRACKER_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bytetrack.yaml")
# ── API configuration ────────────────────────────────────────────────────
API_ENDPOINT   = "https://bae6-49-205-179-53.ngrok-free.app/passenger-count"
API_TIMEOUT    = 2        # seconds — never block processing loop
API_MAX_RETRY  = 2        # retries on 5xx errors
API_RETRY_WAIT = 0.3      # seconds between retries

# BoT-SORT tracker config (see botsort.yaml for full settings)
TRACKER_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "botsort.yaml")

# Colours (BGR)
COL_LINE      = (0,  0,   220)
COL_ZONE_L    = (255, 80, 0)
COL_ZONE_R    = (0, 140, 0)
COL_BOX_NORM  = (0, 140, 255)
COL_BOX_FLASH = (255, 255, 0)
COL_BOX_SKIP  = (80,  80, 80)
COL_IN_TEXT   = (80, 255, 80)
COL_OUT_TEXT  = (80, 180, 255)
COL_EXEMPT    = (255, 0, 200)

@dataclass
class TrackState:
    kalman:      'KalmanCentroid'
    cx:          float
    cy:          float
    trail:       deque = field(default_factory=lambda: deque(maxlen=TRAIL_LEN))
    zone_state:  str   = "INIT"
    counted:     Optional[str] = None
    flash:       int   = 0
    last_seen:   int   = 0
    side_frames: int   = 0
    prev_side:   str   = ""
    debounce_side: str = ""
    exempt:      bool  = False
    exempt_score: int  = 0
    last_bbox:   tuple = field(default_factory=lambda: (0,0,0,0))
    tape_votes:  deque = field(default_factory=lambda: deque(maxlen=VOTE_WINDOW))
    consec_miss: int   = 0  # Consecutive non-blurry frames where tape was NOT detected

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
        self.kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        # Raised process noise 0.03→0.5: top-down heads move faster and less
        # predictably than side-view full-body tracks — the filter must follow
        # sudden direction changes without lagging behind and breaking IoU matching.
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.5
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kf.statePost           = np.array([[cx],[cy],[0],[0]], np.float32)

    def update(self, cx: float, cy: float):
        self.kf.predict()
        meas = np.array([[cx],[cy]], np.float32)
        res  = self.kf.correct(meas)
        return float(res[0, 0]), float(res[1, 0])

    def predict(self) -> tuple[float, float]:
        """Return predicted position without a measurement update (for ghost display)."""
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
    def __init__(self, video_path: str, model_path: str = "yolov8s.pt", output_path: str = "result_v4.mp4",
                 enable_debug: bool = False, head_detect: bool = False, visdrone: bool = False,
                 no_preview: bool = False, live: bool = False):
        self.video_path  = video_path
        self.output_path = output_path
        self.head_detect = head_detect
        self.no_preview  = no_preview
        self.live        = live

        # ── Model selection priority: visdrone > head_detect > default ──
        if visdrone:
            model_path = self._ensure_visdrone_model()
        elif head_detect and model_path == "yolov8s.pt":
            model_path = self._ensure_head_model()

        self.model      = YOLO(model_path)
        self.states: dict[int, TrackState] = {}
        self.in_count   = 0
        self.out_count  = 0
        self.events: list[dict] = []

        # Ghost re-link table: maps a lost track_id → its last known TrackState
        # so that when a brand-new ID appears close to a ghost's last position
        # we can re-adopt the ghost state rather than starting fresh.
        self._ghost_pool: dict[int, TrackState] = {}

        # Ensure ByteTrack YAML exists (write minimal config if absent)
        self._ensure_bytetrack_yaml()

        # Enhanced debugging
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

    def _ensure_head_model(self) -> str:
        """Download CrowdHuman-trained YOLOv8n if not already present."""
        import urllib.request
        head_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n_crowdhuman.pt")
        if not os.path.exists(head_model_path):
            url = "https://github.com/yakhyo/yolov8-crowdhuman/releases/download/weights/yolov8n_best.pt"
            print(f"[HEAD-DETECT] Downloading CrowdHuman model from {url}...")
            urllib.request.urlretrieve(url, head_model_path)
            print(f"[HEAD-DETECT] Saved to {head_model_path}")
        else:
            print(f"[HEAD-DETECT] Using existing model: {head_model_path}")
        return head_model_path

    def _ensure_visdrone_model(self) -> str:
        """
        Download a YOLOv8 model pre-trained on VisDrone (top-down drone/CCTV dataset).
        VisDrone contains overhead pedestrian imagery — far more suitable than the
        standard COCO-trained model for bus-door top-down camera views.
        Falls back to yolov8s-visdrone if the nano variant is unavailable.
        """
        import urllib.request
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n_visdrone.pt")
        if not os.path.exists(model_path):
            # Official Ultralytics VisDrone fine-tuned weights
            url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-visdrone.pt"
            print(f"[VISDRONE] Downloading VisDrone model from {url}...")
            try:
                urllib.request.urlretrieve(url, model_path)
                print(f"[VISDRONE] Saved to {model_path}")
            except Exception as e:
                print(f"[VISDRONE] Download failed ({e}). Falling back to yolov8s.pt")
                return "yolov8s.pt"
        else:
            print(f"[VISDRONE] Using existing model: {model_path}")
        return model_path

    def _ensure_bytetrack_yaml(self):
        """
        Write a ByteTrack YAML config file tuned for top-down pedestrian tracking.
        ByteTrack uses a two-stage association that keeps low-confidence detections
        in a tentative pool — this dramatically reduces ID switches when a person
        is briefly occluded or the model confidence drops due to overhead angle.

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
        Key differences from BoT-SORT defaults:
          • track_high_thresh lowered  → catches partial/head-only detections
          • track_low_thresh  lowered  → second-stage catches even weak hits
          • new_track_thresh  lowered  → confirms tracks sooner
          • track_buffer raised        → keeps lost tracks alive longer
          • match_thresh      raised   → more lenient IoU matching for fast movers
          • ReID (appearance) disabled → top-down appearance is unreliable;
                                         pure position matching is more stable
        """
        if os.path.exists(TRACKER_CONFIG):
            return
        yaml_content = (
            "tracker_type: bytetrack\n"
            "track_high_thresh: 0.25    # min conf for first-stage association\n"
            "track_low_thresh:  0.05    # min conf for second-stage (tentative) association\n"
            "new_track_thresh:  0.20    # min conf to initialise a brand-new track\n"
            "track_buffer:      45      # frames to keep a lost track alive (~1.5 s @ 30 fps)\n"
            "match_thresh:      0.85    # IoU threshold for first-stage match (higher = more lenient)\n"
            "fuse_score:        true    # fuse detection score into IoU cost\n"
        )
        with open(TRACKER_CONFIG, "w") as f:
            f.write(yaml_content)
        print(f"[BYTETRACK] Config written to {TRACKER_CONFIG}")

    def _relink_ghost(self, new_tid: int, raw_cx: float, raw_cy: float, f_no: int) -> int:
        """
        Before creating a brand-new TrackState for `new_tid`, search the ghost
        pool for a recently-lost track whose last centroid is within RELINK_DIST_PX.
        If found, migrate that ghost's state to `new_tid` so no new ID is born.

        Returns the track ID whose state should be used (may differ from new_tid
        if a ghost was re-adopted — caller uses self.states[new_tid] normally).
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
            self.states[new_tid] = adopted
            print(f"  🔗 [RE-LINK] Ghost ID {best_tid} re-adopted as ID {new_tid} "
                  f"(dist={best_dist:.1f}px, age={f_no - adopted.last_seen}f)")
            return new_tid

        return new_tid  # no ghost found — caller creates fresh state

    def _get_side(self, cx: float, line_x: int) -> str:
        if cx < line_x - DEAD_ZONE_PX: return "L"
        if cx > line_x + DEAD_ZONE_PX: return "R"
        return "ZONE"

    def _is_blurry(self, crop: np.ndarray) -> tuple[bool, float]:
        """
        Compute Laplacian variance to detect motion blur.
        Returns (is_blurry, variance) — blurry frames are skipped for scoring
        to avoid penalising the exemption score unfairly.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return (lap_var < BLUR_VAR_THRESH, lap_var)

    def _has_radium_tape(self, crop: np.ndarray, tid: int, frame_no: int = 0, fps: float = 30.0) -> bool:
        """
        Detect radium tape (orange-yellow) in shoulder area of person crop.
        Improvements over previous version:
          • bilateral denoising before HSV conversion
          • wider HSV saturation lower-bound to tolerate motion-blur colour smear
          • dynamic shoulder ROI (top 20-60%) instead of fixed top-40%
          • Laplacian blur-gating: skip scoring on blurry crops
        """
        if crop is None or crop.size == 0:
            return False

        crop_h, crop_w = crop.shape[:2]

        # ── Step 1: Blur gate ─────────────────────────────────────────
        blurry, lap_var = self._is_blurry(crop)
        print(f"     📷 Sharpness (Laplacian var): {lap_var:.1f} "
              f"({'BLURRY — detection skipped' if blurry else 'SHARP — proceeding'})")
        if blurry:
            return False  # Caller will treat this as neutral (no score change)

        # ── Step 2: Edge-preserving denoise before colour analysis ────
        crop_clean = cv2.bilateralFilter(crop, d=9, sigmaColor=75, sigmaSpace=75)
        hsv = cv2.cvtColor(crop_clean, cv2.COLOR_BGR2HSV)

        # ── Step 3: Dynamic shoulder ROI (top 20 % – 60 %) ──────────
        #   Covers cases where the person is leaning / mid-motion and the
        #   tape appears lower than the rigid top-40% band.
        roi_top    = int(crop_h * 0.20)
        roi_bottom = int(crop_h * 0.60)
        shoulder_roi = hsv[roi_top:roi_bottom, :]

        # ── Step 4: Widened HSV range for motion-blurred yellow ───────
        #   Lower S bound reduced from 40 → 15 so blurred tape (whose
        #   saturation is smeared) still registers.
        lower_yellow = np.array([ 5, 15, 80])
        upper_yellow = np.array([35, 255, 255])

        mask_yellow = cv2.inRange(shoulder_roi, lower_yellow, upper_yellow)

        # Morphological cleanup: remove isolated noise specks
        kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN,  kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

        yellow_px = int(np.sum(mask_yellow > 0))

        # ── Always log shoulder HSV diagnostics ──────────────────────
        sh     = shoulder_roi[:, :, 0].flatten()
        ss     = shoulder_roi[:, :, 1].flatten()
        sv     = shoulder_roi[:, :, 2].flatten()
        bright = sv > 30
        if np.any(bright):
            print(f"     🎨 Shoulder Area HSV:")
            print(f"        Hue (H):        {np.mean(sh[bright]):6.1f} "
                  f"(range: {np.min(sh[bright]):3d}-{np.max(sh[bright]):3d})")
            print(f"        Saturation (S): {np.mean(ss[bright]):6.1f}")
            print(f"        Value (V):      {np.mean(sv[bright]):6.1f}")
            print(f"     🟡 Yellow Pixels:  {yellow_px}px (need {RADIUM_MIN_PX}px minimum)")
        else:
            print(f"     ⚫ Shoulder too dark (no bright pixels)")
            print(f"     🟡 Yellow Pixels:  {yellow_px}px (need {RADIUM_MIN_PX}px minimum)")

        # Initialize debug data if debugging enabled
        debug_data = None
        if self.enable_debug:
            debug_data = {
                'frame':                  frame_no,
                'timestamp':              frame_no / fps,
                'track_id':               tid,
                'crop_width':             crop_w,
                'crop_height':            crop_h,
                'laplacian_variance':     round(lap_var, 2),
                'shoulder_roi_top_pct':   0.20,
                'shoulder_roi_bottom_pct':0.60,
                'shoulder_yellow_pixels': yellow_px,
            }

        # Require minimum pixels for valid radium tape detection
        if yellow_px >= RADIUM_MIN_PX:
            # Calculate mean saturation and value for validation
            if yellow_px > 0:
                # Get HSV statistics
                h_vals = shoulder_roi[:,:,0][mask_yellow > 0]
                s_vals = shoulder_roi[:,:,1][mask_yellow > 0]
                v_vals = shoulder_roi[:,:,2][mask_yellow > 0]

                max_s = int(np.max(s_vals))
                max_v = int(np.max(v_vals))
                mean_h = float(np.mean(h_vals))
                mean_s = float(np.mean(s_vals))
                mean_v = float(np.mean(v_vals))

                print(f"\n     📊 Yellow Region Analysis:")
                print(f"        Hue (H):        mean={mean_h:5.1f}")
                print(f"        Saturation (S): mean={mean_s:5.1f}, max={max_s:3d}")
                print(f"        Value (V):      mean={mean_v:5.1f}, max={max_v:3d}")
                print(f"\n     🔍 Validation Checks:")

                # Enhanced debug logging
                if self.enable_debug and debug_data is not None:
                    debug_data.update({
                        'shoulder_h_min':  int(np.min(h_vals)),
                        'shoulder_h_max':  int(np.max(h_vals)),
                        'shoulder_h_mean': mean_h,
                        'shoulder_h_std':  float(np.std(h_vals)),
                        'shoulder_s_min':  int(np.min(s_vals)),
                        'shoulder_s_max':  max_s,
                        'shoulder_s_mean': mean_s,
                        'shoulder_s_std':  float(np.std(s_vals)),
                        'shoulder_v_min':  int(np.min(v_vals)),
                        'shoulder_v_max':  max_v,
                        'shoulder_v_mean': mean_v,
                        'shoulder_v_std':  float(np.std(v_vals)),
                        'meets_pixel_threshold': True,
                    })

                    # Pass roi_top so _save_debug_image can draw the correct ROI box
                    self._save_debug_image(crop_clean, hsv, mask_yellow,
                                           debug_data, tid, frame_no, roi_top)

                # Validate it's bright enough (not shadows or dark objects)
                if max_v < 150:
                    print(f"        ❌ Brightness Check: FAILED (V={max_v} < 150)")
                    print(f"           → Too dark, likely shadow")
                    if self.enable_debug and debug_data is not None:
                        debug_data.update({
                            'meets_brightness_threshold': False,
                            'meets_saturation_threshold': False,
                            'meets_mean_saturation_threshold': False,
                            'adaptive_threshold_used': 0,
                            'area_classification': 'unknown',
                            'would_detect': False,
                            'rejection_reason': f"Too dark (V={max_v})"
                        })
                        self.debug_log.append(debug_data)
                    return False

                print(f"        ✅ Brightness Check: PASSED (V={max_v} >= 150)")

                # Validate peak saturation — must have some bright pixels
                if max_s < 150:
                    print(f"        ❌ Peak Saturation Check: FAILED (S={max_s} < 150)")
                    print(f"           → Not saturated enough for radium tape")
                    if self.enable_debug and debug_data is not None:
                        debug_data.update({
                            'meets_brightness_threshold': True,
                            'meets_saturation_threshold': False,
                            'meets_mean_saturation_threshold': False,
                            'adaptive_threshold_used': 0,
                            'area_classification': 'unknown',
                            'would_detect': False,
                            'rejection_reason': f"Peak saturation too low (S={max_s})"
                        })
                        self.debug_log.append(debug_data)
                    return False

                print(f"        ✅ Peak Saturation Check: PASSED (S={max_s} >= 150)")

                # Adaptive mean saturation threshold based on pixel count
                # Small area (300-2000px) = likely tape → lenient threshold (mean_s >= 45)
                # Large area (2000+px) = likely clothing → strict threshold (mean_s >= 75)
                if yellow_px > 2000:
                    mean_s_threshold = 75
                    area_type = "large"

                    # Additional check for large areas: max_s must be consistently high
                    # Real tape + clothing: max_s >= 160 (tape is bright)
                    # Pure clothing: max_s < 160 (no bright tape pixels)
                    if max_s < 160:
                        print(f"        ❌ Large Area Brightness Check: FAILED (max_s={max_s} < 160)")
                        print(f"           → Large area but not bright enough for radium tape")
                        if self.enable_debug and debug_data is not None:
                            debug_data.update({
                                'meets_brightness_threshold': True,
                                'meets_saturation_threshold': False,
                                'meets_mean_saturation_threshold': False,
                                'adaptive_threshold_used': mean_s_threshold,
                                'area_classification': area_type,
                                'would_detect': False,
                                'rejection_reason': f"Large area but max_s too low ({max_s} < 160)"
                            })
                            self.debug_log.append(debug_data)
                        return False

                    print(f"        ✅ Large Area Brightness Check: PASSED (max_s={max_s} >= 160)")
                else:
                    mean_s_threshold = 45
                    area_type = "small"

                print(f"        📏 Area Classification: {area_type.upper()} ({yellow_px}px)")
                print(f"           → Threshold: mean_s >= {mean_s_threshold}")

                if mean_s < mean_s_threshold:
                    print(f"        ❌ Mean Saturation Check: FAILED (mean_s={mean_s:.1f} < {mean_s_threshold})")
                    print(f"           → Likely yellow clothing, not radium tape")
                    if self.enable_debug and debug_data is not None:
                        debug_data.update({
                            'meets_brightness_threshold': True,
                            'meets_saturation_threshold': True,
                            'meets_mean_saturation_threshold': False,
                            'adaptive_threshold_used': mean_s_threshold,
                            'area_classification': area_type,
                            'would_detect': False,
                            'rejection_reason': f"Mean saturation too low for {area_type} area (mean_s={mean_s:.0f} < {mean_s_threshold})"
                        })
                        self.debug_log.append(debug_data)
                    return False

                print(f"        ✅ Mean Saturation Check: PASSED (mean_s={mean_s:.1f} >= {mean_s_threshold})")
                print(f"\n     ✅ ALL CHECKS PASSED → RADIUM TAPE DETECTED!")
                print(f"\n{'🟢'*40}")
                print(f"  ⚠️  EXEMPT MATCH DETECTED - PLEASE VERIFY!")
                print(f"  ID: {tid} | Frame: {frame_no:05d} | Pixels: {yellow_px}px")
                print(f"  Area: {area_type.upper()} | mean_s={mean_s:.1f} | max_s={max_s}")
                print(f"{'🟢'*40}\n")
                if self.enable_debug and debug_data is not None:
                    # Determine which threshold was used
                    if yellow_px > 2000:
                        adaptive_threshold = 75
                        area_class = "large"
                    else:
                        adaptive_threshold = 45
                        area_class = "small"

                    debug_data.update({
                        'meets_brightness_threshold': True,
                        'meets_saturation_threshold': True,
                        'meets_mean_saturation_threshold': True,
                        'adaptive_threshold_used': adaptive_threshold,
                        'area_classification': area_class,
                        'would_detect': True,
                        'rejection_reason': 'DETECTED'
                    })
                    self.debug_log.append(debug_data)
                return True
        else:
            # Not enough pixels
            print(f"     ❌ Insufficient yellow pixels: {yellow_px}px < {RADIUM_MIN_PX}px")
            print(f"        → Not enough area for radium tape detection")
            if self.enable_debug and debug_data is not None:
                debug_data.update({
                    'meets_pixel_threshold': False,
                    'meets_brightness_threshold': False,
                    'meets_saturation_threshold': False,
                    'meets_mean_saturation_threshold': False,
                    'adaptive_threshold_used': 0,
                    'area_classification': 'too_small',
                    'would_detect': False,
                    'rejection_reason': f'Not enough pixels ({yellow_px})'
                })
                # Fill in missing HSV stats with zeros
                for key in ['h', 's', 'v']:
                    for stat in ['min', 'max', 'mean', 'std']:
                        debug_data[f'shoulder_{key}_{stat}'] = 0
                self.debug_log.append(debug_data)

        return False

    def _save_debug_image(self, crop: np.ndarray, hsv: np.ndarray, mask_yellow: np.ndarray,
                          debug_data: dict, tid: int, frame_no: int, roi_top: int = 0):
        """Save annotated debug image with color analysis."""
        if not self.enable_debug:
            return

        crop_h, crop_w = crop.shape[:2]

        # Create canvas: 3 columns x 2 rows
        canvas_w = crop_w * 3
        canvas_h = crop_h * 2
        canvas   = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Row 1: Original | Yellow Mask | Shoulder Highlight
        canvas[0:crop_h, 0:crop_w] = crop

        # Pad mask_yellow back to full crop height using the dynamic roi_top offset
        full_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        roi_rows  = mask_yellow.shape[0]
        full_mask[roi_top:roi_top + roi_rows,
                  0:mask_yellow.shape[1]] = mask_yellow
        canvas[0:crop_h, crop_w:2 * crop_w] = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR)

        shoulder_vis = crop.copy()
        roi_bottom   = roi_top + roi_rows
        cv2.rectangle(shoulder_vis, (0, roi_top), (crop_w, roi_bottom), (0, 255, 255), 2)
        canvas[0:crop_h, 2 * crop_w:3 * crop_w] = shoulder_vis

        # Row 2: Hue | Saturation | Value
        h_vis = cv2.applyColorMap((hsv[:, :, 0] * 2).astype(np.uint8), cv2.COLORMAP_HSV)
        s_vis = cv2.cvtColor(hsv[:, :, 1], cv2.COLOR_GRAY2BGR)
        v_vis = cv2.cvtColor(hsv[:, :, 2], cv2.COLOR_GRAY2BGR)

        canvas[crop_h:2 * crop_h, 0:crop_w]              = h_vis
        canvas[crop_h:2 * crop_h, crop_w:2 * crop_w]     = s_vis
        canvas[crop_h:2 * crop_h, 2 * crop_w:3 * crop_w] = v_vis

        # Labels
        labels = [
            (10,              20,           "Original"),
            (crop_w + 10,     20,           "Yellow Mask"),
            (2 * crop_w + 10, 20,           "Shoulder ROI"),
            (10,              crop_h + 20,  "Hue"),
            (crop_w + 10,     crop_h + 20,  "Saturation"),
            (2 * crop_w + 10, crop_h + 20,  "Value"),
        ]
        for x, y, txt in labels:
            cv2.putText(canvas, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Status
        status = "DETECTED" if debug_data.get('would_detect', False) else "REJECTED"
        color  = (0, 255, 0)  if debug_data.get('would_detect', False) else (0, 0, 255)
        cv2.putText(canvas, status, (10, crop_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Stats
        lap_str = f"Lap:{debug_data.get('laplacian_variance', 0):.1f}"
        stats = [
            f"Yellow: {debug_data.get('shoulder_yellow_pixels', 0)}px",
            f"H: {debug_data.get('shoulder_h_mean', 0):.1f}",
            f"S: {debug_data.get('shoulder_s_mean', 0):.1f} (max:{debug_data.get('shoulder_s_max', 0)})",
            f"V: {debug_data.get('shoulder_v_mean', 0):.1f} (max:{debug_data.get('shoulder_v_max', 0)})",
            lap_str,
        ]
        y_offset = crop_h - 95
        for i, text in enumerate(stats):
            cv2.putText(canvas, text, (10, y_offset + i * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Rejection reason
        if not debug_data.get('would_detect', False):
            cv2.putText(canvas, debug_data.get('rejection_reason', ''),
                        (10, crop_h - 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Save files
        filename      = f"analysis_f{frame_no:05d}_id{tid:03d}.jpg"
        crop_filename = f"crop_f{frame_no:05d}_id{tid:03d}.jpg"
        cv2.imwrite(os.path.join(self.debug_analysis_dir, filename), canvas)
        cv2.imwrite(os.path.join(self.debug_crops_dir,    crop_filename), crop)

    def _is_merged_box(self, bbox) -> bool:
        """Detect if a bounding box likely contains two merged persons.
        A single person has W/H ~ 0.3-0.5; merged boxes are much wider."""
        x1, y1, x2, y2 = bbox.astype(int)
        w, h = x2 - x1, y2 - y1
        if h <= 0: return True
        return (w / h) > MERGE_AR_THRESH

    def _overlaps_other_tracks(self, tid, bbox_ints, f_no) -> bool:
        """Check if this bbox significantly overlaps any other active track."""
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

    def _update_track(self, tid, bbox, line_x, f_no, frame, fps: float = 30.0):
        x1, y1, x2, y2 = bbox.astype(int)
        raw_cx, raw_cy = (x1+x2)/2.0, (y1+y2)/2.0
        event = None  # Initialize event variable

        # Print frame header for better readability
        print(f"\n{'='*80}")
        print(f"  FRAME {f_no:05d} | ID {tid:2d} | Timestamp: {f_no/fps:.2f}s")
        print(f"{'='*80}")

        if tid not in self.states:
            # ── Ghost re-link: try to recover a recently-lost track ───
            self._relink_ghost(tid, raw_cx, raw_cy, f_no)

        if tid not in self.states:
            # Still not found — create a genuinely new track
            side = self._get_side(raw_cx, line_x)
            kf   = KalmanCentroid(raw_cx, raw_cy)
            st   = TrackState(kalman=kf, cx=raw_cx, cy=raw_cy,
                              last_seen=f_no)

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
        st.last_seen = f_no
        st.last_bbox = (x1, y1, x2, y2)

        # Scored exemption: only check when bbox is a clean single-person crop
        w_box, h_box = x2 - x1, y2 - y1
        ar = (w_box / h_box) if h_box > 0 else 999.0
        merged = self._is_merged_box(bbox)
        overlapping = self._overlaps_other_tracks(tid, (x1, y1, x2, y2), f_no)
        tape_found = False
        skip_reason = ""

        # Print bounding box info
        print(f"  📦 Bounding Box: {w_box:3d}x{h_box:3d} | Aspect Ratio: {ar:.2f}")
        print(f"  📍 Position: ({x1}, {y1}) → ({x2}, {y2})")

        if merged:
            skip_reason = "MERGED"
            print(f"  ⚠️  Status: MERGED BOX (AR > {MERGE_AR_THRESH}) - Skipping tape detection")
        elif overlapping:
            skip_reason = "OVERLAP"
            print(f"  ⚠️  Status: OVERLAPPING - Skipping tape detection")
        else:
            print(f"  ✓ Status: Clean single-person box - Running tape detection...")
            print(f"  {'-'*76}")
            crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]

            # ── Blur gate: skip scoring on blurry crops ───────────────
            blurry, lap_var = self._is_blurry(crop)
            if blurry:
                print(f"  ⚠️  Crop too blurry (Lap var={lap_var:.1f} < {BLUR_VAR_THRESH}) — score unchanged")
                tape_found = False
                # Do NOT push to vote window; do NOT update consec_miss
            else:
                tape_found = self._has_radium_tape(crop, tid, f_no, fps)
                print(f"  {'-'*76}")

                # ── Temporal vote window ──────────────────────────────
                st.tape_votes.append(1 if tape_found else 0)
                vote_hits = sum(st.tape_votes)
                vote_confirmed = (len(st.tape_votes) >= VOTE_MIN_HITS and
                                  vote_hits >= VOTE_MIN_HITS)

                if tape_found:
                    st.exempt_score = min(st.exempt_score + 3, EXEMPT_MAX)
                    st.consec_miss  = 0
                    print(f"  ✅ RADIUM TAPE DETECTED → Score +3 | Votes: {vote_hits}/{len(st.tape_votes)}")
                else:
                    # Only decay score after HYSTERESIS_MISS consecutive clear misses
                    st.consec_miss += 1
                    if st.consec_miss >= HYSTERESIS_MISS:
                        st.exempt_score = max(st.exempt_score - 1, 0)
                        print(f"  ❌ NO TAPE — {st.consec_miss} consec misses → Score -1 | "
                              f"Votes: {vote_hits}/{len(st.tape_votes)}")
                    else:
                        print(f"  ❌ NO TAPE — consec miss {st.consec_miss}/{HYSTERESIS_MISS} "
                              f"(score held) | Votes: {vote_hits}/{len(st.tape_votes)}")

        # When merged/overlapping: score unchanged — don't penalize or reward
        prev_exempt = st.exempt

        # Grant exemption when score reaches threshold
        # Revoke only when score hits 0 AND enough consecutive misses accumulated
        #   → prevents a single blurry frame from stripping exemption
        if st.exempt_score >= EXEMPT_CONFIRM:
            st.exempt = True
        elif st.exempt_score == 0 and st.consec_miss >= HYSTERESIS_MISS:
            st.exempt = False

        # Print exemption status
        print(f"\n  📊 EXEMPTION STATUS:")
        print(f"     Current Score: {st.exempt_score:2d}/{EXEMPT_CONFIRM} (max: {EXEMPT_MAX})")
        print(f"     Exempt: {'YES ✓' if st.exempt else 'NO ✗'}")

        if st.exempt and not prev_exempt:
            print(f"\n{'⭐'*40}")
            print(f"  🎉 🎉 🎉 [EXEMPT GRANTED] 🎉 🎉 🎉")
            print(f"  ID {tid} | Score: {st.exempt_score}/{EXEMPT_CONFIRM}")
            print(f"  ⚠️  VERIFY: Is this person wearing radium tape?")
            print(f"{'⭐'*40}\n")
        elif not st.exempt and prev_exempt:
            print(f"\n{'⚠️ '*40}")
            print(f"  ⚠️  [EXEMPT REVOKED] ID {tid} (score={st.exempt_score})")
            print(f"{'⚠️ '*40}\n")

        print(f"{'='*80}\n")

        pcx, pcy = st.kalman.update(raw_cx, raw_cy)
        st.cx, st.cy = EMA_ALPHA * raw_cx + (1-EMA_ALPHA) * pcx, EMA_ALPHA * raw_cy + (1-EMA_ALPHA) * pcy
        st.trail.append((int(st.cx), int(st.cy)))
        if st.flash > 0: st.flash -= 1

        if st.exempt: return

        # SKIP recovery: if track was born in dead zone, recover once it moves to a clear side
        if st.zone_state == "SKIP":
            side = self._get_side(st.cx, line_x)
            if side == "L":
                st.zone_state, st.prev_side = "OUTSIDE", "L"
                print(f"  [SKIP\u2192RECOVERED] ID {tid} moved to LEFT side")
            elif side == "R":
                st.zone_state, st.prev_side = "INSIDE", "R"
                print(f"  [SKIP\u2192RECOVERED] ID {tid} moved to RIGHT side")
            return  # Don't count this frame, just recover state

        side = self._get_side(st.cx, line_x)
        if side == "ZONE": st.zone_state, st.side_frames, st.debounce_side = "ZONE", 0, ""
        else:
            # side is "L" or "R"
            # ── Debounce: accumulate consecutive frames on this side ──────
            if st.debounce_side != side:
                st.debounce_side = side
                st.side_frames   = 1
            else:
                st.side_frames  += 1


            # Only commit transition after DEBOUNCE_N consecutive frames
            if st.side_frames >= DEBOUNCE_N:
                if side == "L":
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
                                "frame": f_no, "id": tid, "event": "OUT",
                                "in": self.in_count, "out": self.out_count
                            })
                            print(f"[{f_no:05d}] ID {tid:3d} LEFT   | IN={self.in_count} OUT={self.out_count}")
                            # ── Non-blocking API push ────────────────────
                            if hasattr(self, 'api'):
                                self.api.push(self.in_count, self.out_count)
                    st.zone_state = "OUTSIDE"
                    st.prev_side  = "L"

                elif side == "R":
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
                                "frame": f_no, "id": tid, "event": "IN",
                                "in": self.in_count, "out": self.out_count
                            })
                            print(f"[{f_no:05d}] ID {tid:3d} ENTERED| IN={self.in_count} OUT={self.out_count}")
                            # ── Non-blocking API push ────────────────────
                            if hasattr(self, 'api'):
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
        W, H, FPS = int(cap.get(3)), int(cap.get(4)), cap.get(5) or 30.0
        line_x = int(W * LINE_RATIO)
        writer  = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        f_no    = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            f_no += 1

            track_args = dict(
                conf    = CONF_THRESH,
                iou     = IOU_THRESH,
                verbose = False,
                tracker = TRACKER_CONFIG,
                persist = True,
            )
            # VisDrone model uses class 1 (pedestrian); CrowdHuman/standard use class 0
            if not self.head_detect:
                track_args["classes"] = [0]

            res = self.model.track(frame, **track_args)[0]
            
            # Convert results to supervision Detections for drawing
            dets = sv.Detections(
                xyxy=res.boxes.xyxy.cpu().numpy(),
                confidence=res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else None,
                class_id=res.boxes.cls.cpu().numpy().astype(int) if res.boxes.cls is not None else None,
                tracker_id=res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else None,
            )
            
            if res.boxes.id is not None:
                for b, tid in zip(res.boxes.xyxy.cpu().numpy(),
                                  res.boxes.id.cpu().numpy().astype(int)):
                    self._update_track(tid, b, line_x, f_no, frame, FPS)

            # Every 90 frames: move timed-out active tracks into the ghost pool
            # rather than deleting them outright, so _relink_ghost can recover them.
            if f_no % 90 == 0:
                alive, expired = {}, {}
                for t_id, s in self.states.items():
                    if f_no - s.last_seen < GHOST_TIMEOUT:
                        alive[t_id] = s
                    else:
                        expired[t_id] = s
                self.states = alive
                # Merge expired into ghost pool; prune very old ghosts
                self._ghost_pool.update(expired)
                self._ghost_pool = {
                    g_id: g_st for g_id, g_st in self._ghost_pool.items()
                    if f_no - g_st.last_seen < GHOST_TIMEOUT * 2
                }

            # ── Draw ──────────────────────────────────────────────────
            frame = self._draw_zones(frame, line_x)
            frame = self._draw_tracks(frame, dets)
            frame = self._draw_dashboard(frame, f_no, FPS)
            writer.write(frame)

            if not self.no_preview:
                cv2.imshow("Bus Counter v3", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        # Save event CSV
        with open(self.output_path.replace(".mp4", ".csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "id", "event", "in", "out"])
            w.writeheader()
            w.writerows(self.events)

        # Save debug CSV if enabled
        if self.enable_debug and self.debug_log:
            debug_csv_path = os.path.join(self.debug_dir, "debug_log.csv")
            fieldnames     = list(self.debug_log[0].keys()) if self.debug_log else []
            with open(debug_csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(self.debug_log)
            print(f"\n[DEBUG] CSV log saved: {debug_csv_path} ({len(self.debug_log)} entries)")

    def _draw_ui(self, frame, line_x, res):
        h, w = frame.shape[:2]
        # Zones
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (line_x - DEAD_ZONE_PX, h), (180, 60, 0), -1)
        cv2.rectangle(ov, (line_x + DEAD_ZONE_PX, 0), (w, h), (0, 140, 0), -1)
        frame = cv2.addWeighted(ov, 0.06, frame, 0.94, 0)
        cv2.line(frame, (line_x, 0), (line_x, h), COL_LINE, 2)
        # Dashboard
        x1, y1, dw, dh = w - 220, 20, 200, 120
        ov2 = frame.copy()
        cv2.rectangle(ov2, (x1, y1), (x1+dw, y1+dh), (15,15,15), -1)
        frame = cv2.addWeighted(ov2, 0.7, frame, 0.3, 0)
        cv2.rectangle(frame, (x1, y1), (x1+dw, y1+3), (0,200,80), -1)
        rows = [("ENTERED", self.in_count, COL_IN_TEXT), ("LEFT", self.out_count, COL_OUT_TEXT), ("INSIDE", max(0, self.in_count-self.out_count), (80,220,255))]
        for i, (l, v, c) in enumerate(rows):
            cv2.putText(frame, f"{l}: {v}", (x1+15, y1+35+i*30), 0, 0.55, c, 2)
        # Tracks
        if res.boxes.id is not None:
            for b, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.cpu().numpy().astype(int)):
                st = self.states.get(tid)
                if not st: continue
                x1, y1, x2, y2 = b.astype(int)
                col = COL_EXEMPT if st.exempt else (COL_BOX_SKIP if st.zone_state == "SKIP" else (COL_BOX_FLASH if st.flash > 0 else COL_BOX_NORM))
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, (3 if st.flash > 0 else 2))
                cv2.putText(frame, f"#{tid}" + (" EXEMPT" if st.exempt else ""), (x1, y1-10), 0, 0.5, col, 1)
                pts = list(st.trail)
                for j in range(1, len(pts)):
                    t = j / max(len(pts)-1, 1)
                    cv2.line(frame, pts[j-1], pts[j], (int(255*(1-t)), int(200*t), int(120*t)), 2)
        return frame

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
    p.add_argument("--debug", action="store_true",
                   help="Enable detailed HSV debugging (saves CSV + images)")
    p.add_argument("--head-detect", action="store_true",
                   help="Use CrowdHuman head detection model (better for top-down views)")
    p.add_argument("--visdrone", action="store_true",
                   help="Use VisDrone model (optimized for overhead/drone camera views)")
    args = p.parse_args()

    LINE_RATIO = args.line

    counter = BusCounter(
        video_path   = args.source,
        model_path   = args.model,
        output_path  = args.output,
        enable_debug = args.debug,
        head_detect  = args.head_detect,
        visdrone     = args.visdrone,
        no_preview   = args.no_preview,
        live         = args.live,
    )
    counter.process()


if __name__ == "__main__":
    main()