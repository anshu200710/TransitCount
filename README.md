# 🚌 Bus Passenger Counter System

## 📋 Overview

An advanced AI-powered bus passenger counting system using YOLOv8 and ByteTrack for accurate real-time tracking. The system counts passengers entering and exiting through the bus door, with staff exemption detection and API integration.

### ✨ Key Features

- ✅ **Accurate Counting** - 3-state machine (OUTSIDE → ZONE → INSIDE) for reliable counting
- ✅ **Staff Exemption** - Automatic detection of high-visibility vests/radium tape
- ✅ **Real-Time Processing** - Works with live streams (RTMP, RTSP, HTTP)
- ✅ **Delayed Processing** - Simulate real-time with recorded videos
- ✅ **Ghost Re-linking** - Recovers lost tracks to reduce ID switches
- ✅ **Kalman Filtering** - Smooth tracking during occlusions
- ✅ **API Integration** - Non-blocking POST requests to backend
- ✅ **CSV Logging** - Detailed event logs with timestamps
- ✅ **Visual Feedback** - Real-time overlay with tracking trails

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd bus-passenger-counter

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Process recorded video
python bus_passenger_counter.py --source counting.mp4

# Process live RTMP stream
python bus_passenger_counter.py --source rtmp://13.217.114.127:1935/live/stream --live

# Process with 30-second delay (simulates real-time)
python bus_passenger_counter.py --source rtmp://13.217.114.127:1935/live/stream --delay 30
```

---

## 📖 Complete Command Reference

### 1. Live Stream Processing

#### RTMP Stream
```bash
python bus_passenger_counter.py \
    --source rtmp://13.217.114.127:1935/live/stream \
    --live
```

#### RTSP Stream
```bash
python bus_passenger_counter.py \
    --source rtsp://192.168.1.100:554/stream \
    --live
```

#### HTTP Stream
```bash
python bus_passenger_counter.py \
    --source http://camera-ip:port/stream \
    --live
```

#### With Custom Output
```bash
python bus_passenger_counter.py \
    --source rtmp://13.217.114.127:1935/live/stream \
    --live \
    --output live_result.mp4
```

---

### 2. Recorded Video Processing

#### Basic Processing
```bash
python bus_passenger_counter.py --source counting.mp4
```

#### With Custom Output
```bash
python bus_passenger_counter.py \
    --source counting.mp4 \
    --output result.mp4
```

#### Headless Mode (No Preview)
```bash
python bus_passenger_counter.py \
    --source counting.mp4 \
    --no-preview
```

---

### 3. Delayed Processing (Real-Time Simulation)

#### Standard Delay (60 seconds)
```bash
python bus_passenger_counter.py \
    --source counting.mp4 \
    --delay 60
```

#### Short Delay (30 seconds)
```bash
python bus_passenger_counter.py \
    --source rtmp://13.217.114.127:1935/live/stream \
    --delay 30
```

#### Extended Delay (2 minutes)
```bash
python bus_passenger_counter.py \
    --source counting.mp4 \
    --delay 120
```

#### Delayed + Headless
```bash
python bus_passenger_counter.py \
    --source counting.mp4 \
    --delay 60 \
    --no-preview
```

---

### 4. Model Selection

#### Standard Model (Default)
```bash
python bus_passenger_counter.py \
    --source counting.mp4 \
    --model yolov8s.pt
```

#### CrowdHuman Model (Better for Head Detection)
```bash
python bus_passenger_counter.py \
    --source counting.mp4 \
    --head-detect
```

#### VisDrone Model (Optimized for Overhead Views)
```bash
python bus_passenger_counter.py \
    --source counting.mp4 \
    --visdrone
```

#### Custom Model
```bash
python bus_passenger_counter.py \
    --source counting.mp4 \
    --model path/to/your/model.pt
```

---

### 5. Debug Mode

#### Enable Debug Output
```bash
python bus_passenger_counter.py \
    --source counting.mp4 \
    --debug
```

This creates:
- `debug_output/crops/` - Person crop images
- `debug_output/analysis/` - HSV analysis images
- `debug_output/debug_log.csv` - Detailed debug log

#### Debug with Live Stream
```bash
python bus_passenger_counter.py \
    --source rtmp://13.217.114.127:1935/live/stream \
    --live \
    --debug
```

---

### 6. Advanced Configuration

#### Custom Counting Line Position
```bash
# Line at 40% of frame width
python bus_passenger_counter.py \
    --source counting.mp4 \
    --line 0.40

# Line at 50% of frame width
python bus_passenger_counter.py \
    --source counting.mp4 \
    --line 0.50
```

#### Full Configuration Example
```bash
python bus_passenger_counter.py \
    --source rtmp://13.217.114.127:1935/live/stream \
    --output live_output.mp4 \
    --model yolov8s.pt \
    --line 0.45 \
    --live \
    --delay 30 \
    --debug
```

---

## 🎯 Common Use Cases

### Use Case 1: Production Live Stream
```bash
python bus_passenger_counter.py \
    --source rtmp://13.217.114.127:1935/live/stream \
    --live \
    --no-preview
```
**When to use:** Actual bus deployment, production environment

---

### Use Case 2: Testing with Recorded Video
```bash
python bus_passenger_counter.py \
    --source counting.mp4 \
    --delay 60
```
**When to use:** Testing real-time scenarios without live camera

---

### Use Case 3: Quick Analysis
```bash
python bus_passenger_counter.py \
    --source counting.mp4 \
    --no-preview
```
**When to use:** Fast batch processing, getting results quickly

---

### Use Case 4: Development & Debugging
```bash
python bus_passenger_counter.py \
    --source counting.mp4 \
    --delay 30 \
    --debug
```
**When to use:** Troubleshooting, understanding system behavior

---

### Use Case 5: API Integration Testing
```bash
python bus_passenger_counter.py \
    --source counting.mp4 \
    --delay 60 \
    --no-preview
```
**When to use:** Testing backend API with realistic timing

---

## 📊 Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source` | string | `counting.mp4` | Input video path or stream URL |
| `--output` | string | `result_final.mp4` | Output video path |
| `--model` | string | `yolov8s.pt` | YOLO model weights path |
| `--line` | float | `0.45` | Counting line position (0.0-1.0) |
| `--delay` | int | `0` | Processing delay in seconds (0=no delay) |
| `--live` | flag | `False` | Enable live stream mode |
| `--no-preview` | flag | `False` | Disable preview window |
| `--debug` | flag | `False` | Enable debug output |
| `--head-detect` | flag | `False` | Use CrowdHuman head detection model |
| `--visdrone` | flag | `False` | Use VisDrone overhead model |

---

## 🎨 Output Files

### Video Output
- **File:** `result_final.mp4` (or custom via `--output`)
- **Content:** Processed video with tracking overlays
- **Features:**
  - Bounding boxes around detected persons
  - Track IDs and status labels
  - Trail lines showing movement
  - Zone visualization (OUTSIDE, ZONE, INSIDE)
  - Real-time dashboard with counts

### CSV Event Log
- **File:** `result_final_events.csv`
- **Columns:**
  - `frame` - Frame number
  - `timestamp` - Event timestamp
  - `id` - Track ID
  - `event` - Event type (IN/OUT)
  - `in` - Total entered count
  - `out` - Total exited count
  - `inside` - Current inside count

### Debug Output (if `--debug` enabled)
- **Folder:** `debug_output/`
- **Contents:**
  - `crops/` - Person crop images
  - `analysis/` - 6-panel HSV analysis images
  - `debug_log.csv` - Detailed frame-by-frame log

---

## 🔧 Configuration

### Counting Geometry
Edit these constants in `bus_passenger_counter.py`:

```python
LINE_RATIO    = 0.45   # Trigger line position (45% of frame width)
DEAD_ZONE_PX  = 30     # Dead zone half-width (pixels)
DEBOUNCE_N    = 3      # Frames required on a side before counting
```

### Detection Parameters
```python
CONF_THRESH   = 0.08   # Detection confidence threshold
IOU_THRESH    = 0.45   # NMS IoU threshold
```

### Tracking Parameters
```python
TRAIL_LEN     = 50     # Max trail length per track
GHOST_TIMEOUT = 300    # Frames until track expires
RELINK_DIST_PX = 60    # Max distance for ghost re-linking
```

### Staff Exemption
```python
RADIUM_MIN_PX   = 200  # Min yellow pixels for detection
EXEMPT_CONFIRM  = 5    # Score threshold for exemption
VOTE_WINDOW     = 7    # Temporal voting window size
```

### API Configuration
```python
API_ENDPOINT    = "https://your-api-endpoint.com/passenger-count"
API_TIMEOUT     = 2    # Request timeout (seconds)
API_MAX_RETRY   = 2    # Max retry attempts
```

---

## 🎯 Processing Modes

### Mode 1: Normal (Fast Processing)
```bash
python bus_passenger_counter.py --source counting.mp4
```
- ⚡ Processes as fast as CPU allows
- 💾 Minimal memory usage
- 🎯 Best for: Quick analysis, batch processing

### Mode 2: Delayed (Real-Time Simulation)
```bash
python bus_passenger_counter.py --source counting.mp4 --delay 60
```
- ⏱️ Processes at real-time speed
- 💾 Buffers frames in memory
- 🎯 Best for: Testing, API validation

### Mode 3: Live (Production)
```bash
python bus_passenger_counter.py --source rtmp://stream-url --live
```
- 📡 Processes live stream
- 💾 No buffering
- 🎯 Best for: Production deployment

---

## 📈 Performance Optimization

### For Fast Processing
```bash
python bus_passenger_counter.py \
    --source counting.mp4 \
    --no-preview
```

### For Better Accuracy
```bash
python bus_passenger_counter.py \
    --source counting.mp4 \
    --head-detect \
    --line 0.45
```

### For Low Memory
```bash
python bus_passenger_counter.py \
    --source counting.mp4 \
    --delay 30 \
    --no-preview
```

### For GPU Acceleration
Ensure PyTorch with CUDA is installed:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🧪 Testing

### Quick Test
```bash
python test_delayed_processing.py
```

### Test Different Videos
```bash
# Test video
python bus_passenger_counter.py --source test.mp4

# Conductor video
python bus_passenger_counter.py --source conductor.mp4

# Count video
python bus_passenger_counter.py --source count.mp4
```

### Test Live Stream
```bash
python bus_passenger_counter.py \
    --source rtmp://13.217.114.127:1935/live/stream \
    --live
```

---

## 🔍 Troubleshooting

### Issue: Stream Connection Failed
```bash
# Check stream URL
python bus_passenger_counter.py \
    --source rtmp://13.217.114.127:1935/live/stream \
    --live
```
**Solution:** Verify stream URL, check network connection, ensure stream is active

### Issue: Low Accuracy
```bash
# Try head detection model
python bus_passenger_counter.py \
    --source counting.mp4 \
    --head-detect
```
**Solution:** Use specialized models, adjust counting line position

### Issue: High Memory Usage
```bash
# Reduce delay or disable preview
python bus_passenger_counter.py \
    --source counting.mp4 \
    --delay 30 \
    --no-preview
```
**Solution:** Reduce delay duration, use `--no-preview`, close other applications

### Issue: Slow Processing
```bash
# Disable preview
python bus_passenger_counter.py \
    --source counting.mp4 \
    --no-preview
```
**Solution:** Use GPU acceleration, disable preview, reduce video resolution

### Issue: Fast-Moving Persons Not Tracked
```bash
# Adjust tracking parameters in code
CONF_THRESH = 0.05  # Lower threshold
GHOST_TIMEOUT = 150  # Shorter timeout
RELINK_DIST_PX = 100  # Larger re-link distance
```
**Solution:** See "Fast-Moving Person Fix" section below

### Issue: Fast Entry/Exit Not Counted
```bash
# ✅ FIXED with line crossing detection!
# Detects crossings by comparing positions between frames
# Works for ANY crossing speed (even 1 frame crossings)
```
**Solution:** Line crossing detection automatically enabled
**Look for:** "(FAST)" label in console output

---

## 🚀 Fast-Moving Person Fix

### Problem
Fast-moving persons may not be tracked properly due to:
- High detection threshold
- Insufficient frame rate
- Large motion between frames
- Track loss during rapid movement

### Solutions

#### 1. Lower Detection Threshold
Edit `bus_passenger_counter.py`:
```python
CONF_THRESH = 0.05  # Lower from 0.08 to 0.05
```

#### 2. Increase Ghost Re-link Distance
```python
RELINK_DIST_PX = 100  # Increase from 60 to 100
```

#### 3. Adjust Tracker Configuration
Edit `bytetrack.yaml`:
```yaml
tracker_type: bytetrack
track_high_thresh: 0.20  # Lower from 0.25
track_low_thresh: 0.03   # Lower from 0.05
new_track_thresh: 0.15   # Lower from 0.20
track_buffer: 60         # Increase from 45
match_thresh: 0.80       # Lower from 0.85
fuse_score: true
```

#### 4. Use Faster Model
```bash
# Use YOLOv8n (nano) for faster inference
python bus_passenger_counter.py \
    --source rtmp://13.217.114.127:1935/live/stream \
    --model yolov8n.pt \
    --live
```

#### 5. Reduce Debounce Frames
```python
DEBOUNCE_N = 2  # Reduce from 3 to 2
```

---

## 📚 Additional Documentation

- **[Delayed Processing Guide](DELAYED_PROCESSING_GUIDE.md)** - Complete guide for delayed processing
- **[Quick Start](QUICK_START_DELAYED.md)** - Quick reference for delayed mode
- **[Example Commands](EXAMPLE_COMMANDS.md)** - Copy-paste ready commands
- **[Feature Summary](DELAY_FEATURE_SUMMARY.md)** - Visual feature overview
- **[Implementation Details](IMPLEMENTATION_SUMMARY.md)** - Technical documentation

---

## 🎓 Examples

### Example 1: Basic Live Stream
```bash
python bus_passenger_counter.py \
    --source rtmp://13.217.114.127:1935/live/stream \
    --live
```

### Example 2: Live Stream with Delay
```bash
python bus_passenger_counter.py \
    --source rtmp://13.217.114.127:1935/live/stream \
    --delay 30
```

### Example 3: Recorded Video Analysis
```bash
python bus_passenger_counter.py \
    --source counting.mp4 \
    --output analysis_result.mp4
```

### Example 4: Production Deployment
```bash
python bus_passenger_counter.py \
    --source rtmp://13.217.114.127:1935/live/stream \
    --live \
    --no-preview \
    --output /var/log/bus_counter/output.mp4
```

### Example 5: Debug Mode
```bash
python bus_passenger_counter.py \
    --source counting.mp4 \
    --debug \
    --delay 30
```

---

## 🔐 API Integration

### Endpoint Configuration
Edit in `bus_passenger_counter.py`:
```python
API_ENDPOINT = "https://your-api-endpoint.com/passenger-count"
```

### Payload Format
```json
{
    "datetime": "2024-05-08 14:30:45",
<<<<<<< HEAD
    "hin": 15,
    "hout": 8,
    "inside": 7,
    "total": 15
}
```

=======
    "hin": 15,      // Total people entered (cumulative)
    "hout": 8,      // Total people exited (cumulative)
    "inside": 7,    // People currently inside the bus
    "total": 7      // People currently inside (same as "inside")
}
```

**Note:** The `total` field sends the number of people **currently inside the bus**, not the total entered.

### Calculation
```python
inside = max(0, hin - hout)
total = inside  // Same value
```

>>>>>>> 28e708955590cac18e7b687acfb6342199eb457c
### Testing API
```bash
# Run with delay to test API timing
python bus_passenger_counter.py \
    --source counting.mp4 \
    --delay 60
```

<<<<<<< HEAD
=======
**See:** `API_PAYLOAD_FIX.md` for detailed API documentation

>>>>>>> 28e708955590cac18e7b687acfb6342199eb457c
---

## 📊 System Requirements

### Minimum Requirements
- **CPU:** Intel i5 or equivalent
- **RAM:** 8 GB
- **GPU:** Optional (NVIDIA with CUDA support)
- **Storage:** 2 GB free space
- **OS:** Windows 10/11, Linux, macOS

### Recommended Requirements
- **CPU:** Intel i7 or equivalent
- **RAM:** 16 GB
- **GPU:** NVIDIA RTX series with 4GB+ VRAM
- **Storage:** 10 GB free space
- **Network:** Stable connection for live streams

---

## 🛠️ Dependencies

```txt
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
supervision>=0.16.0
requests>=2.31.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 📝 License

[Add your license information here]

---

## 👥 Contributors

[Add contributor information here]

---

## 📧 Support

For issues and questions:
- Create an issue on GitHub
- Contact: [your-email@example.com]
- Documentation: See `docs/` folder

---

## 🎉 Quick Command Reference

```bash
# Live stream
python bus_passenger_counter.py --source rtmp://13.217.114.127:1935/live/stream --live

# Live stream with delay
python bus_passenger_counter.py --source rtmp://13.217.114.127:1935/live/stream --delay 30

# Recorded video
python bus_passenger_counter.py --source counting.mp4

# Recorded video with delay
python bus_passenger_counter.py --source counting.mp4 --delay 60

# Headless mode
python bus_passenger_counter.py --source counting.mp4 --no-preview

# Debug mode
python bus_passenger_counter.py --source counting.mp4 --debug

# Custom model
python bus_passenger_counter.py --source counting.mp4 --head-detect

# Full configuration
python bus_passenger_counter.py --source rtmp://stream --live --delay 30 --output result.mp4
```

---

**Version:** 4.0  
**Last Updated:** 2026-05-08  
**Status:** ✅ Production Ready
