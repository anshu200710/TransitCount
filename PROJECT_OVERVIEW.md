# 🚌 Bus Passenger Counter System — Project Documentation

## 📋 Overview

This project is an advanced AI-powered bus passenger counting system using YOLOv8 and ByteTrack for accurate real-time tracking. It counts passengers entering and exiting through the bus door, detects staff exemptions, and integrates with backend APIs.

### ✨ Key Features

- Accurate 3-state machine counting (OUTSIDE → ZONE → INSIDE)
- Staff exemption via high-visibility vest/radium tape detection
- Real-time and delayed (simulated) processing
- Live stream (RTMP, RTSP, HTTP) and recorded video support
- Ghost re-linking and Kalman filtering for robust tracking
- Non-blocking API POST integration
- CSV event logging and real-time visual feedback
- Debug mode with detailed logs and crop images

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone <your-repo-url>
cd bus-passenger-counter
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Basic Usage

- **Recorded Video:**  
  `python bus_passenger_counter.py --source counting.mp4`
- **Live RTMP Stream:**  
  `python bus_passenger_counter.py --source rtmp://your-stream-url --live`
- **Simulated Real-Time:**  
  `python bus_passenger_counter.py --source counting.mp4 --delay 60`

---

## ⚙️ Command Reference

- `--source`: Input video path or stream URL
- `--output`: Output video path (default: result_final.mp4)
- `--model`: YOLO model weights (default: yolov8s.pt)
- `--line`: Counting line position (0.0-1.0, default: 0.45)
- `--delay`: Processing delay in seconds (default: 0)
- `--live`: Enable live stream mode
- `--no-preview`: Disable preview window
- `--debug`: Enable debug output
- `--head-detect`: Use CrowdHuman head detection model
- `--visdrone`: Use VisDrone overhead model

---

## 🏗️ How It Works

- **Detection:** Uses YOLOv8 for person detection.
- **Tracking:** ByteTrack with custom configuration for fast/occluded motion.
- **Counting:** 3-state logic and line crossing detection (handles ultra-fast crossings).
- **Staff Exemption:** Detects yellow/reflective tape on staff using HSV and IR masks.
- **Ghost Re-linking:** Recovers lost tracks, resets age for accurate counting.
- **API Integration:** Sends non-blocking POST requests for each event.
- **Debugging:** Optional debug mode outputs crops, analysis images, and logs.

---

## 🛠️ Configuration

Edit constants in `bus_passenger_counter.py` for fine-tuning:
- `LINE_RATIO`, `DEAD_ZONE_PX`, `DEBOUNCE_N`
- `CONF_THRESH`, `IOU_THRESH`
- `TRAIL_LEN`, `GHOST_TIMEOUT`, `RELINK_DIST_PX`
- `RADIUM_MIN_PX`, `EXEMPT_CONFIRM`, `VOTE_WINDOW`
- `API_ENDPOINT`, `API_TIMEOUT`, `API_MAX_RETRY`

---

## 📝 Output Files

- **Video:** `result_final.mp4` (with overlays)
- **CSV Log:** `result_final_events.csv` (frame, timestamp, id, event, in, out, inside)
- **Debug:** `debug_output/` (crops, analysis, debug_log.csv)

---

## 🧪 Testing

- `python test_delayed_processing.py` — Quick test for delayed mode
- `python test_fast_motion.py` — Test fast motion handling
- `python test_ultra_fast_crossing.py` — Test ultra-fast line crossing
- `python test_counting_logic.py` — Validate counting logic

---

## 🩹 Fixes & Improvements

- **Normal People Counting:** Fixed duplicate field, ghost re-linking age, and added debug logs.
- **Ultra-Fast Line Crossing:** Fixed undefined variables, added trajectory-based detection.
- **Staff Identification:** Added state locking, detection delay, adaptive lighting, ROI refinement, and increased ghost re-linking age.

---

## 🧑‍💻 Dependencies

- `ultralytics>=8.4.0`
- `supervision>=0.27.0`
- `opencv-python>=4.13.0`
- `numpy>=1.24.0`

---

## 🆘 Troubleshooting

- **Stream Connection Failed:** Check URL, network, and stream status.
- **Low Accuracy:** Use head detection, adjust line position.
- **High Memory Usage:** Reduce delay, use `--no-preview`.
- **Fast-Moving Persons Not Tracked:** Lower `CONF_THRESH`, increase `RELINK_DIST_PX`, adjust tracker config.
- **Fast Entry/Exit Not Counted:** Line crossing detection is enabled by default.

---

## 📚 Further Reading

- See `COUNTING_ISSUE_FIXES.md`, `LINE_CROSSING_FIX.md`, and `STAFF_IDENTIFICATION_IMPROVEMENTS.md` for technical deep-dives and code snippets.

---
