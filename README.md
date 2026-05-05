# Bus Passenger Counter

A Python-based video analytics project for counting bus passengers using YOLO detection and BoT-SORT tracking.

## Overview

This project processes overhead/top-down video of a bus front gate to count passengers entering and leaving. It uses:

- `ultralytics` YOLO model for people detection
- `supervision` for detection/tracking data structures
- BoT-SORT for multi-object tracking with ReID
- OpenCV for visualization, video I/O, and Kalman smoothing

The counting logic is based on a 3-state track machine:

- `OUTSIDE` — centroid left of the dead zone
- `ZONE` — centroid inside the dead zone around a trigger line
- `INSIDE` — centroid right of the dead zone

Valid count transitions:

- `OUTSIDE -> ZONE -> INSIDE` = `IN`
- `INSIDE -> ZONE -> OUTSIDE` = `OUT`

IDs first seen in `ZONE` or `INSIDE` are skipped to avoid ghost or false counts.

## Key Files

- `bus_passenger_counter.py` — main script
- `botsort.yaml` — BoT-SORT tracker configuration
- `requirements.txt` — Python dependencies
- `yolov8s.pt`, `yolov8n.pt` — YOLO model weights
- sample videos: `count.mp4`, `counting.mp4`, `new_3.mp4`, `output_result.mp4`, `result.mp4`, `result_v3.mp4`, `pcs.mp4`, `testing.mp4`
- log files: `bus_log.csv`, `bus_log_fixed.csv`, `result.csv`, `result1.csv`, `result1.mp4`, `result_v3.csv`

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Run the counter with:

```bash
python bus_passenger_counter.py --source counting.mp4 --output new_3.mp4
```

Optional arguments:

- `--source` — input video path (default: `counting.mp4`)
- `--output` — output video path (default: `result_v3.mp4`)
- `--model` — YOLO weights file (default: `yolov8s.pt`)
- `--line` — trigger line position as fraction of frame width (default: `0.45`)
- `--no-preview` — disable live preview window

> Note: The script currently always shows the preview window unless `--no-preview` is implemented in the processing loop.

## Configuration

The following tuning parameters are defined in `bus_passenger_counter.py`:

- `LINE_RATIO` — trigger line position relative to frame width
- `DEAD_ZONE_PX` — pixel buffer around the line where tracks are considered in the zone
- `DEBOUNCE_N` — required consecutive frames on one side before confirming a transition
- `CONF_THRESH` — YOLO confidence threshold
- `IOU_THRESH` — NMS/intersection threshold for detection filtering
- `TRAIL_LEN` — number of centroid positions stored for trail drawing
- `GHOST_TIMEOUT` — frames to keep a track before purging missing IDs
- `FLASH_FRAMES` — frames to highlight a count event
- `EMA_ALPHA` — exponential smoothing weight for centroid smoothing

## Tracker Config (`botsort.yaml`)

BoT-SORT parameters include:

- `track_high_thresh`: high score threshold for track activation
- `track_low_thresh`: low detection threshold
- `new_track_thresh`: threshold to create a new track
- `track_buffer`: frames to retain disappeared tracks
- `match_thresh`: association matching threshold
- `gmc_method`: motion compensation method
- `proximity_thresh` and `appearance_thresh`: matching criteria
- `with_reid`: enable appearance-based ReID
- `model`: auto-select appearance model
- `fuse_score`: fuse detection and track scores

## Output

The script writes:

- annotated video file at the chosen `--output` path
- CSV event log next to the output video with columns: `frame`, `timestamp`, `id`, `event`, `in`, `out`, `inside`

It also prints count updates and final totals to the console.

## Algorithm Notes

- A per-track Kalman filter smooths centroid motion.
- Track IDs first observed inside the bus or inside the dead zone are marked as `SKIP` and do not count.
- After a confirmed count, that track enters a short flash cooldown to avoid duplicate counts.
- Ghost tracks are purged every 90 frames based on `GHOST_TIMEOUT`.

## Improvements & Observations

The project already includes several robustness improvements:

- `agnostic_nms=True` to prevent merging overlapping people boxes
- low-confidence threshold for partial/occluded detections
- a dedicated dead zone around the trigger line
- color-coded visualization for states, trails, and count flashes

## Notes

- The workspace uses a Python virtual environment in `venv/`.
- If you want a headless run, the preview window should be disabled or hidden.
- If you want consistent line placement, tune `LINE_RATIO` based on camera alignment.
