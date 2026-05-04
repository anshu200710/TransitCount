#!/usr/bin/env python3
"""
Bus Passenger Counter — v3.6 BACK TO NORMAL
===========================================
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

# ═══════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════
LINE_RATIO     = 0.45    
DEAD_ZONE_PX   = 30      
DEBOUNCE_N     = 3       
CONF_THRESH    = 0.12    
IOU_THRESH     = 0.40    
TRAIL_LEN      = 50      
GHOST_TIMEOUT  = 150     
FLASH_FRAMES   = 20      
EMA_ALPHA      = 0.40    
BADGE_MIN_PX   = 20      

# BoT-SORT tracker config
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

class KalmanCentroid:
    def __init__(self, cx: float, cy: float):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix  = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix   = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov    = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov= np.eye(2, dtype=np.float32) * 0.5
        self.kf.statePost          = np.array([[cx],[cy],[0],[0]], np.float32)
    def update(self, cx: float, cy: float):
        self.kf.predict()
        meas = np.array([[cx],[cy]], np.float32)
        res = self.kf.correct(meas)
        return float(res[0,0]), float(res[1,0])

class BusCounter:
    def __init__(self, video_path: str, model_path: str = "yolov8s.pt", output_path: str = "result_v3.mp4"):
        self.video_path  = video_path
        self.output_path = output_path
        self.model       = YOLO(model_path)
        self.states: dict[int, TrackState] = {}
        self.in_count  = 0
        self.out_count = 0
        self.events: list[dict] = []

    def _get_side(self, cx: float, line_x: int) -> str:
        if cx < line_x - DEAD_ZONE_PX: return "L"
        if cx > line_x + DEAD_ZONE_PX: return "R"
        return "ZONE"

    def _has_badge(self, crop: np.ndarray, tid: int) -> bool:
        if crop is None or crop.size == 0: return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Calibration Logger
        m_s_all = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([25, 255, 255]))
        m_g_all = cv2.inRange(hsv, np.array([60, 50, 40]), np.array([95, 255, 255]))
        max_s = int(np.max(hsv[:,:,1], where=m_s_all>0, initial=0))
        max_g = int(np.max(hsv[:,:,1], where=m_g_all>0, initial=0))
        if max_s > 100 or max_g > 50:
            print(f"  [DEBUG ID {tid:2d}] Peak Sat -> Saffron: {max_s}, Green: {max_g}")

        # Calibrated Math
        mask_s = cv2.inRange(hsv, np.array([0, 160, 80]), np.array([25, 255, 255]))
        mask_g = cv2.inRange(hsv, np.array([60, 60, 40]), np.array([95, 255, 255]))
        
        if np.sum(mask_s > 0) > BADGE_MIN_PX and np.sum(mask_g > 0) > 3:
            dilated_s = cv2.dilate(mask_s, np.ones((7,7), np.uint8), iterations=2)
            if np.sum(cv2.bitwise_and(dilated_s, mask_g) > 0) > 1:
                return True
        return False

    def _update_track(self, tid, bbox, line_x, f_no, frame):
        x1, y1, x2, y2 = bbox.astype(int)
        raw_cx, raw_cy = (x1+x2)/2.0, (y1+y2)/2.0
        if tid not in self.states:
            side = self._get_side(raw_cx, line_x)
            st = TrackState(kalman=KalmanCentroid(raw_cx, raw_cy), cx=raw_cx, cy=raw_cy, last_seen=f_no)
            if side == "L": st.zone_state, st.prev_side = "OUTSIDE", "L"
            elif side == "R": st.zone_state, st.prev_side = "INSIDE", "R"
            else: st.zone_state = "SKIP"
            self.states[tid] = st
        st = self.states[tid]
        st.last_seen = f_no

        if not st.exempt:
            crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
            if self._has_badge(crop, tid):
                st.exempt = True
                print(f"  [EXEMPT] ID {tid} Badge verified.")

        pcx, pcy = st.kalman.update(raw_cx, raw_cy)
        st.cx, st.cy = EMA_ALPHA * raw_cx + (1-EMA_ALPHA) * pcx, EMA_ALPHA * raw_cy + (1-EMA_ALPHA) * pcy
        st.trail.append((int(st.cx), int(st.cy)))
        if st.flash > 0: st.flash -= 1
        
        if st.exempt or st.zone_state == "SKIP": return

        side = self._get_side(st.cx, line_x)
        if side == "ZONE": st.zone_state, st.side_frames, st.debounce_side = "ZONE", 0, ""
        else:
            if st.debounce_side != side: st.debounce_side, st.side_frames = side, 1
            else: st.side_frames += 1
            if st.side_frames >= DEBOUNCE_N:
                if side == "L" and st.zone_state == "ZONE" and st.prev_side == "R":
                    if st.counted != "OUT":
                        self.out_count += 1
                        st.counted, st.flash = "OUT", FLASH_FRAMES
                        self.events.append({"frame": f_no, "id": tid, "event": "OUT", "in": self.in_count, "out": self.out_count})
                        print(f"[{f_no:05d}] ID {tid:3d} LEFT   | IN={self.in_count} OUT={self.out_count}")
                elif side == "R" and st.zone_state == "ZONE" and st.prev_side == "L":
                    if st.counted != "IN":
                        self.in_count += 1
                        st.counted, st.flash = "IN", FLASH_FRAMES
                        self.events.append({"frame": f_no, "id": tid, "event": "IN", "in": self.in_count, "out": self.out_count})
                        print(f"[{f_no:05d}] ID {tid:3d} ENTERED| IN={self.in_count} OUT={self.out_count}")
                st.zone_state, st.prev_side = ("OUTSIDE" if side=="L" else "INSIDE"), side

    def process(self):
        cap = cv2.VideoCapture(self.video_path)
        W, H, FPS = int(cap.get(3)), int(cap.get(4)), cap.get(5) or 30.0
        line_x, writer = int(W * LINE_RATIO), cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        f_no = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            f_no += 1
            res = self.model.track(frame, classes=[0], conf=CONF_THRESH, iou=IOU_THRESH, verbose=False, tracker=TRACKER_CONFIG, persist=True)[0]
            if res.boxes.id is not None:
                for b, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.cpu().numpy().astype(int)):
                    self._update_track(tid, b, line_x, f_no, frame)
            if f_no % 90 == 0:
                self.states = {tid: s for tid, s in self.states.items() if f_no - s.last_seen < GHOST_TIMEOUT}
            
            # Premium Visuals
            frame = self._draw_ui(frame, line_x, res)
            writer.write(frame); cv2.imshow("Bus Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release(); writer.release(); cv2.destroyAllWindows()
        with open(self.output_path.replace(".mp4", ".csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "id", "event", "in", "out"])
            w.writeheader(); w.writerows(self.events)

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

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="testing.mp4"); p.add_argument("--output", default="result_v3.mp4")
    args = p.parse_args(); BusCounter(args.source, output_path=args.output).process()