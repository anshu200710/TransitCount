# Line Crossing Detection Fix

## Problem Identified

The bus passenger counter was not detecting people crossing the line because the **ultra-fast line crossing detection** code was referencing undefined variables:

1. `st.crossed_line` - was never set to a value before being checked
2. `crossing_direction` - was never calculated before being used

This caused the fast crossing detection logic (lines 700-740) to never execute, meaning people who crossed the line quickly were not counted.

## Root Cause

The code had:
- **Duplicate field definitions** in the `TrackState` dataclass (fields were defined twice)
- **Missing trajectory calculation** - the `prev_cx` field existed but was never used to detect line crossings
- **Undefined variables** - `crossing_direction` was used without being calculated

## Solution Applied

### 1. Fixed TrackState Dataclass (lines 150-180)
- Removed duplicate field definitions
- Kept only one set of fields including `prev_cx` and `crossed_line`

### 2. Added Line Crossing Detection Logic (lines 690-710)
Added the missing trajectory-based crossing detection:

```python
# Detect if person crossed the line in a single frame (skipping ZONE)
crossing_direction = None
st.crossed_line = False

if st.prev_cx != 0.0:  # Not first frame for this track
    # Check if line was crossed between prev_cx and current cx
    if st.prev_cx < line_x <= st.cx:
        # Crossed from LEFT to RIGHT (ENTRY)
        st.crossed_line = True
        crossing_direction = "IN"
    elif st.prev_cx > line_x >= st.cx:
        # Crossed from RIGHT to LEFT (EXIT)
        st.crossed_line = True
        crossing_direction = "OUT"

# Update prev_cx for next frame
st.prev_cx = st.cx
```

## How It Works Now

The system now has **two detection mechanisms**:

1. **Fast Crossing Detection** (NEW - now working):
   - Compares previous centroid position (`prev_cx`) with current position (`cx`)
   - If the line was crossed between frames, counts immediately
   - Catches people who move so fast they skip the ZONE entirely

2. **State Machine Detection** (existing):
   - Tracks progression: OUTSIDE → ZONE → INSIDE (or reverse)
   - Requires debouncing (3 consecutive frames on a side)
   - Handles normal-speed crossings

## Testing Recommendations

Run the counter with debug mode to verify:
```bash
python bus_passenger_counter.py --source counting.mp4 --debug
```

Watch for console output:
- `ENTERED (FAST)` - fast crossing detected
- `ENTERED` - normal state machine detected
- Debug logs will show state transitions

## Expected Behavior

- **Fast crossings**: Counted immediately when line is crossed
- **Normal crossings**: Counted after 3 frames of debouncing
- **Staff with radium tape**: Excluded from counting (exempt system)
- **Duplicate prevention**: Each person counted only once per direction
