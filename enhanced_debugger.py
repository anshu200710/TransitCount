#!/usr/bin/env python3
"""
Enhanced Debugger for Bus Passenger Counter
============================================
Logs detailed HSV color information for every person in every frame.
Helps diagnose radium tape detection issues.

Output:
- Detailed CSV log with HSV values per person per frame
- Debug images with color analysis overlays
- Console output with real-time statistics
"""

import cv2
import numpy as np
from ultralytics import YOLO
import csv
import os
import argparse
from datetime import datetime

# ═══════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════
MODEL_PATH = "yolov8s.pt"
CONF_THRESH = 0.12
IOU_THRESH = 0.65

# Radium tape threshold (from analysis)
LOWER_YELLOW = np.array([10, 72, 92])
UPPER_YELLOW = np.array([29, 255, 255])
RADIUM_MIN_PX = 300

# Output directories
DEBUG_DIR = "debug_enhanced"
CROPS_DIR = os.path.join(DEBUG_DIR, "crops")
ANALYSIS_DIR = os.path.join(DEBUG_DIR, "analysis")


class EnhancedDebugger:
    def __init__(self, video_path: str, output_csv: str = "debug_log.csv"):
        self.video_path = video_path
        self.output_csv = os.path.join(DEBUG_DIR, output_csv)
        self.model = YOLO(MODEL_PATH)
        self.log_data = []
        
        # Create output directories
        os.makedirs(DEBUG_DIR, exist_ok=True)
        os.makedirs(CROPS_DIR, exist_ok=True)
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
    
    def analyze_person_crop(self, crop: np.ndarray, person_id: int, frame_no: int):
        """
        Perform detailed HSV analysis on a person crop.
        Returns dictionary with all color statistics.
        """
        if crop is None or crop.size == 0:
            return None
        
        crop_h, crop_w = crop.shape[:2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Focus on shoulder area (top 40%)
        shoulder_h = int(crop_h * 0.4)
        shoulder_roi = hsv[0:shoulder_h, :]
        full_roi = hsv
        
        analysis = {
            'frame': frame_no,
            'person_id': person_id,
            'crop_width': crop_w,
            'crop_height': crop_h,
        }
        
        # ── Shoulder Area Analysis ──────────────────────────────────
        mask_shoulder = cv2.inRange(shoulder_roi, LOWER_YELLOW, UPPER_YELLOW)
        shoulder_yellow_px = int(np.sum(mask_shoulder > 0))
        
        analysis['shoulder_yellow_pixels'] = shoulder_yellow_px
        
        if shoulder_yellow_px > 0:
            # Get HSV statistics for yellow pixels in shoulder
            h_shoulder = shoulder_roi[:,:,0][mask_shoulder > 0]
            s_shoulder = shoulder_roi[:,:,1][mask_shoulder > 0]
            v_shoulder = shoulder_roi[:,:,2][mask_shoulder > 0]
            
            analysis['shoulder_h_min'] = int(np.min(h_shoulder))
            analysis['shoulder_h_max'] = int(np.max(h_shoulder))
            analysis['shoulder_h_mean'] = float(np.mean(h_shoulder))
            analysis['shoulder_h_std'] = float(np.std(h_shoulder))
            
            analysis['shoulder_s_min'] = int(np.min(s_shoulder))
            analysis['shoulder_s_max'] = int(np.max(s_shoulder))
            analysis['shoulder_s_mean'] = float(np.mean(s_shoulder))
            analysis['shoulder_s_std'] = float(np.std(s_shoulder))
            
            analysis['shoulder_v_min'] = int(np.min(v_shoulder))
            analysis['shoulder_v_max'] = int(np.max(v_shoulder))
            analysis['shoulder_v_mean'] = float(np.mean(v_shoulder))
            analysis['shoulder_v_std'] = float(np.std(v_shoulder))
        else:
            # No yellow in shoulder
            for key in ['h', 's', 'v']:
                for stat in ['min', 'max', 'mean', 'std']:
                    analysis[f'shoulder_{key}_{stat}'] = 0
        
        # ── Full Body Analysis ──────────────────────────────────────
        mask_full = cv2.inRange(full_roi, LOWER_YELLOW, UPPER_YELLOW)
        full_yellow_px = int(np.sum(mask_full > 0))
        
        analysis['full_yellow_pixels'] = full_yellow_px
        
        if full_yellow_px > 0:
            h_full = full_roi[:,:,0][mask_full > 0]
            s_full = full_roi[:,:,1][mask_full > 0]
            v_full = full_roi[:,:,2][mask_full > 0]
            
            analysis['full_h_mean'] = float(np.mean(h_full))
            analysis['full_s_mean'] = float(np.mean(s_full))
            analysis['full_v_mean'] = float(np.mean(v_full))
        else:
            analysis['full_h_mean'] = 0
            analysis['full_s_mean'] = 0
            analysis['full_v_mean'] = 0
        
        # ── Overall Crop Statistics ─────────────────────────────────
        # Get dominant colors in entire crop
        h_all = hsv[:,:,0].flatten()
        s_all = hsv[:,:,1].flatten()
        v_all = hsv[:,:,2].flatten()
        
        analysis['crop_h_mean'] = float(np.mean(h_all))
        analysis['crop_s_mean'] = float(np.mean(s_all))
        analysis['crop_v_mean'] = float(np.mean(v_all))
        
        # ── Detection Decision ──────────────────────────────────────
        analysis['meets_pixel_threshold'] = shoulder_yellow_px >= RADIUM_MIN_PX
        
        if shoulder_yellow_px >= RADIUM_MIN_PX:
            max_s = analysis['shoulder_s_max']
            max_v = analysis['shoulder_v_max']
            
            analysis['meets_brightness_threshold'] = max_v >= 150
            analysis['meets_saturation_threshold'] = max_s >= 185
            analysis['would_detect'] = (max_v >= 150 and max_s >= 185)
            
            if not analysis['would_detect']:
                if max_v < 150:
                    analysis['rejection_reason'] = f"Too dark (V={max_v})"
                elif max_s < 185:
                    analysis['rejection_reason'] = f"Not saturated (S={max_s})"
                else:
                    analysis['rejection_reason'] = "Unknown"
            else:
                analysis['rejection_reason'] = "DETECTED"
        else:
            analysis['meets_brightness_threshold'] = False
            analysis['meets_saturation_threshold'] = False
            analysis['would_detect'] = False
            analysis['rejection_reason'] = f"Not enough pixels ({shoulder_yellow_px})"
        
        return analysis
    
    def save_debug_image(self, crop: np.ndarray, analysis: dict, frame_no: int, person_id: int):
        """Save annotated debug image with color analysis."""
        if crop is None or crop.size == 0:
            return
        
        crop_h, crop_w = crop.shape[:2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Create canvas: 3 columns x 2 rows
        canvas_w = crop_w * 3
        canvas_h = crop_h * 2
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # ── Row 1: Original | Yellow Mask | Shoulder Highlight ──────
        canvas[0:crop_h, 0:crop_w] = crop
        
        # Yellow mask (full body)
        mask_full = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
        canvas[0:crop_h, crop_w:2*crop_w] = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2BGR)
        
        # Shoulder area highlight
        shoulder_vis = crop.copy()
        shoulder_h = int(crop_h * 0.4)
        cv2.rectangle(shoulder_vis, (0, 0), (crop_w, shoulder_h), (0, 255, 255), 2)
        cv2.putText(shoulder_vis, "SHOULDER", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        canvas[0:crop_h, 2*crop_w:3*crop_w] = shoulder_vis
        
        # ── Row 2: Hue | Saturation | Value ─────────────────────────
        h_vis = cv2.applyColorMap((hsv[:,:,0] * 2).astype(np.uint8), cv2.COLORMAP_HSV)
        s_vis = cv2.cvtColor(hsv[:,:,1], cv2.COLOR_GRAY2BGR)
        v_vis = cv2.cvtColor(hsv[:,:,2], cv2.COLOR_GRAY2BGR)
        
        canvas[crop_h:2*crop_h, 0:crop_w] = h_vis
        canvas[crop_h:2*crop_h, crop_w:2*crop_w] = s_vis
        canvas[crop_h:2*crop_h, 2*crop_w:3*crop_w] = v_vis
        
        # ── Labels ──────────────────────────────────────────────────
        labels = [
            (10, 20, "Original"),
            (crop_w+10, 20, "Yellow Mask"),
            (2*crop_w+10, 20, "Shoulder Area"),
            (10, crop_h+20, "Hue"),
            (crop_w+10, crop_h+20, "Saturation"),
            (2*crop_w+10, crop_h+20, "Value")
        ]
        for x, y, txt in labels:
            cv2.putText(canvas, txt, (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # ── Statistics Overlay ──────────────────────────────────────
        status = "DETECTED" if analysis['would_detect'] else "REJECTED"
        color = (0, 255, 0) if analysis['would_detect'] else (0, 0, 255)
        
        cv2.putText(canvas, f"Status: {status}", (10, crop_h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        stats_text = [
            f"Shoulder Yellow: {analysis['shoulder_yellow_pixels']}px",
            f"H: {analysis.get('shoulder_h_mean', 0):.1f}",
            f"S: {analysis.get('shoulder_s_mean', 0):.1f} (max:{analysis.get('shoulder_s_max', 0)})",
            f"V: {analysis.get('shoulder_v_mean', 0):.1f} (max:{analysis.get('shoulder_v_max', 0)})",
        ]
        
        y_offset = crop_h - 80
        for i, text in enumerate(stats_text):
            cv2.putText(canvas, text, (10, y_offset + i*15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Rejection reason
        if not analysis['would_detect']:
            cv2.putText(canvas, analysis['rejection_reason'], (10, crop_h - 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Save
        filename = f"analysis_f{frame_no:05d}_p{person_id:02d}.jpg"
        cv2.imwrite(os.path.join(ANALYSIS_DIR, filename), canvas)
        
        # Also save clean crop
        crop_filename = f"crop_f{frame_no:05d}_p{person_id:02d}.jpg"
        cv2.imwrite(os.path.join(CROPS_DIR, crop_filename), crop)
    
    def process_video(self, start_frame: int = 0, end_frame: int = None, 
                     sample_every: int = 1):
        """Process video and generate debug logs."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open {self.video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if end_frame is None:
            end_frame = total_frames
        
        print("=" * 80)
        print("ENHANCED DEBUGGER - Radium Tape Detection Analysis")
        print("=" * 80)
        print(f"Video: {self.video_path}")
        print(f"Resolution: {W}x{H} @ {fps:.1f}fps")
        print(f"Total frames: {total_frames}")
        print(f"Processing: frames {start_frame} to {end_frame} (every {sample_every} frames)")
        print(f"Output directory: {DEBUG_DIR}/")
        print("=" * 80)
        
        frame_no = 0
        processed = 0
        detected_count = 0
        rejected_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_no += 1
            
            if frame_no < start_frame:
                continue
            if frame_no > end_frame:
                break
            if (frame_no - start_frame) % sample_every != 0:
                continue
            
            # Detect persons
            results = self.model(frame, classes=[0], conf=CONF_THRESH, 
                               iou=IOU_THRESH, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            
            if len(boxes) == 0:
                print(f"[Frame {frame_no:05d}] No persons detected")
                continue
            
            # Process each detected person
            timestamp = frame_no / fps
            print(f"\n[Frame {frame_no:05d} @ {timestamp:.2f}s] Detected {len(boxes)} person(s)")
            
            for person_id, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                crop = frame[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
                
                if crop.size == 0:
                    continue
                
                # Analyze crop
                analysis = self.analyze_person_crop(crop, person_id, frame_no)
                if analysis is None:
                    continue
                
                # Add timestamp
                analysis['timestamp'] = timestamp
                
                # Save to log
                self.log_data.append(analysis)
                
                # Save debug image
                self.save_debug_image(crop, analysis, frame_no, person_id)
                
                # Console output
                status = "✓ DETECTED" if analysis['would_detect'] else "✗ REJECTED"
                print(f"  Person {person_id}: {status}")
                print(f"    Shoulder Yellow: {analysis['shoulder_yellow_pixels']}px")
                if analysis['shoulder_yellow_pixels'] > 0:
                    print(f"    HSV: H={analysis.get('shoulder_h_mean', 0):.1f} "
                          f"S={analysis.get('shoulder_s_mean', 0):.1f} (max:{analysis.get('shoulder_s_max', 0)}) "
                          f"V={analysis.get('shoulder_v_mean', 0):.1f} (max:{analysis.get('shoulder_v_max', 0)})")
                if not analysis['would_detect']:
                    print(f"    Reason: {analysis['rejection_reason']}")
                
                if analysis['would_detect']:
                    detected_count += 1
                else:
                    rejected_count += 1
                
                processed += 1
        
        cap.release()
        
        # Save CSV log
        self.save_csv_log()
        
        # Print summary
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        print(f"Total persons analyzed: {processed}")
        print(f"Detected (radium tape): {detected_count} ({100*detected_count/processed if processed > 0 else 0:.1f}%)")
        print(f"Rejected: {rejected_count} ({100*rejected_count/processed if processed > 0 else 0:.1f}%)")
        print(f"\nOutput files:")
        print(f"  CSV log: {self.output_csv}")
        print(f"  Debug images: {ANALYSIS_DIR}/")
        print(f"  Crops: {CROPS_DIR}/")
        print("=" * 80)
    
    def save_csv_log(self):
        """Save detailed CSV log."""
        if not self.log_data:
            print("No data to save")
            return
        
        # Get all keys from first entry
        fieldnames = list(self.log_data[0].keys())
        
        with open(self.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.log_data)
        
        print(f"\nCSV log saved: {self.output_csv}")
        print(f"Total entries: {len(self.log_data)}")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced debugger for radium tape detection"
    )
    parser.add_argument("--source", required=True, help="Input video file")
    parser.add_argument("--start", type=int, default=0, help="Start frame")
    parser.add_argument("--end", type=int, default=None, help="End frame")
    parser.add_argument("--sample", type=int, default=1, 
                       help="Sample every N frames (1=all frames)")
    parser.add_argument("--output", default="debug_log.csv", 
                       help="Output CSV filename")
    
    args = parser.parse_args()
    
    debugger = EnhancedDebugger(args.source, args.output)
    debugger.process_video(args.start, args.end, args.sample)


if __name__ == "__main__":
    main()
