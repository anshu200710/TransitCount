# Bus Passenger Counter - Staff Identification Improvements

## Overview
Updated the `bus_passenger_counter.py` script with robust staff identification features for real-world scenarios. The system now uses advanced computer vision techniques to reliably detect radium tape on staff uniforms.

## Key Improvements Implemented

### 1. STATE LOCKING ✅
- **Feature**: Once a Track ID is marked as 'exempt' (Staff), it remains exempt permanently
- **Implementation**: Added `exempt_locked` field to `TrackState` dataclass
- **Logic**: When `exempt_score >= EXEMPT_CONFIRM`, both `exempt` and `exempt_locked` are set to `True`
- **Benefit**: Prevents false negatives when tape is temporarily hidden or obscured

### 2. DETECTION DELAY ✅
- **Feature**: Minimum Track Age of 10 frames before counting anyone
- **Implementation**: Added `MIN_TRACK_AGE = 10` constant and `first_seen` field to track creation time
- **Logic**: `track_age = frame_no - st.first_seen` must be >= `MIN_TRACK_AGE` before counting
- **Benefit**: Gives radium tape detector time to work before person crosses the counting line

### 3. ADAPTIVE LIGHTING ✅
- **Feature**: Dual detection for both yellow tape and IR reflective glow
- **Implementation**: 
  - Added reflective mask: `cv2.inRange(shoulder_roi, [0,0,240], [180,30,255])`
  - Detection triggers on either: `(yellow_px >= RADIUM_MIN_PX) or (reflective_px >= RADIUM_MIN_PX)`
- **Logic**: In IR/Night mode, retroreflective tape appears as pure white/glow (Value > 240)
- **Benefit**: Works in both daylight and night vision camera modes

### 4. ROI REFINEMENT ✅
- **Feature**: Enhanced shoulder zone detection
- **Implementation**: 
  - Creates left shoulder (0-25% width), right shoulder (75-100% width), and center zones
  - Focuses on top 20-60% of bounding box height where vest stripes are most visible
- **Logic**: Analyzes shoulder areas where reflective stripes are typically positioned
- **Benefit**: More accurate detection by focusing on high-probability areas

### 5. GHOST RE-LINKING ✅
- **Feature**: Increased RELINK_MAX_AGE to 90 frames
- **Implementation**: `RELINK_MAX_AGE = 90` (increased from 45)
- **Logic**: Handles cases where staff member is blocked by crowds for several seconds
- **Benefit**: Maintains consistent tracking even during heavy occlusion

## Technical Details

### Enhanced Detection Algorithm
```python
# Dual detection approach
lower_yellow = np.array([5, 15, 80])      # Yellow tape detection
upper_yellow = np.array([35, 255, 255])

lower_reflective = np.array([0, 0, 240])   # IR reflective glow
upper_reflective = np.array([180, 30, 255])

# Staff detected if EITHER condition is met
staff_detected = (yellow_px >= RADIUM_MIN_PX) or (reflective_px >= RADIUM_MIN_PX)
```

### State Locking Logic
```python
# Once exempt is granted, it's permanent
if not st.exempt_locked and st.exempt_score >= EXEMPT_CONFIRM:
    st.exempt = True
    st.exempt_locked = True  # Lock permanently
    print(f"🔒 EXEMPT STATUS LOCKED PERMANENTLY for ID {tid}")
```

### Minimum Age Check
```python
# Only count if track is old enough
track_age = f_no - st.first_seen
if track_age < MIN_TRACK_AGE:
    print(f"⏳ Track too young (age={track_age}) - Delaying count")
    return  # Skip counting
```

## Configuration Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `MIN_TRACK_AGE` | 10 | Minimum frames before counting |
| `RELINK_MAX_AGE` | 90 | Max frames to keep ghost tracks |
| `EXEMPT_CONFIRM` | 5 | Score threshold for exemption |
| `RADIUM_MIN_PX` | 200 | Minimum pixels for detection |

## Visual Indicators

- **🔒 Lock Icon**: Shows permanently exempt tracks in UI
- **⏳ Age Warning**: Displays when tracks are too young to count
- **🟢 Detection Alert**: Highlights staff detection events
- **📊 Status Display**: Shows exemption score and lock status

## Benefits for Real-World Deployment

1. **Reliability**: State locking prevents false negatives
2. **Accuracy**: Detection delay reduces false positives
3. **Robustness**: Dual detection works in all lighting conditions
4. **Persistence**: Enhanced ghost re-linking maintains tracking continuity
5. **Precision**: ROI refinement focuses on optimal detection areas

## Usage

Run with the same command-line arguments as before:
```bash
python bus_passenger_counter.py --source video.mp4 --debug --visdrone
```

The system will automatically apply all improvements while maintaining backward compatibility with existing configurations.