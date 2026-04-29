#!/usr/bin/env python3
"""
Bus Passenger Counting System
Uses YOLO11 and Supervision library for person detection and tracking.
Camera mounted at bus front gate - counts passengers entering and leaving.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from datetime import datetime
import csv
import os
import argparse


class BusPassengerCounter:
    def __init__(self, model_path: str = "yolov8n.pt", video_path: str = "bus_entry.mp4"):
        """
        Initialize the Bus Passenger Counter.
        
        Args:
            model_path: Path to YOLO11 model weights (default: yolov8n.pt)
            video_path: Path to input video file
        """
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.tracker = sv.ByteTrack()
        
        # Counters
        self.entered_count = 0
        self.left_count = 0
        
        # Zone definition - central rectangular zone (red box)
        # Define as polygon points: top-left, top-right, bottom-right, bottom-left
        self.zone_polygon = np.array([
            [320, 200],  # top-left
            [960, 200],  # top-right
            [960, 600],  # bottom-right
            [320, 600],  # bottom-left
        ])
        
        self.zone = sv.PolygonZone(polygon=self.zone_polygon)
        
        # Store previous positions for direction detection
        self.tracker_history = {}
        
        # Output video writer
        self.out = None
        
        # CSV logging
        self.log_file = "bus_log.csv"
        self._init_csv()
    
    def parse_args(self):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description='Bus Passenger Counting System')
        parser.add_argument('--source', type=str, default='counting.mp4', 
                          help='Path to input video file')
        parser.add_argument('--output', type=str, default='result.mp4', 
                          help='Path to save output video')
        parser.add_argument('--model', type=str, default='yolov8n.pt', 
                          help='Path to YOLO11 model weights')
        return parser.parse_args()
    
    def _init_csv(self):
        """Initialize CSV log file with headers."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['frame_number', 'timestamp', 'entered', 'left', 'inside'])
    
    def log_data(self, frame_number: int):
        """Log current counts to CSV file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        inside = self.entered_count - self.left_count
        
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([frame_number, timestamp, self.entered_count, self.left_count, inside])
    
    def detect_direction(self, track_id: int, bbox: np.ndarray) -> str:
        """
        Detect movement direction based on position change.
        
        Args:
            track_id: Unique identifier for the tracked person
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            'ENTER', 'LEFT', or 'UNKNOWN'
        """
        # Get center of bounding box
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Get previous position if exists
        if track_id in self.tracker_history:
            prev_center_x = self.tracker_history[track_id]['center_x']
            
            # Calculate movement direction
            if center_x > prev_center_x + 10:  # Moved right (ENTERING)
                self.tracker_history[track_id]['direction'] = 'ENTER'
                self.tracker_history[track_id]['confirmed'] = True
                return 'ENTER'
            elif center_x < prev_center_x - 10:  # Moved left (LEAVING)
                self.tracker_history[track_id]['direction'] = 'LEFT'
                self.tracker_history[track_id]['confirmed'] = True
                return 'LEFT'
        
        # Store current position
        self.tracker_history[track_id] = {
            'center_x': center_x,
            'center_y': center_y,
            'direction': None,
            'confirmed': False
        }
        
        return 'UNKNOWN'
    
    def update_counters(self, track_id: int, direction: str, in_zone: bool):
        """
        Update counters based on direction and zone entry/exit.
        
        Args:
            track_id: Unique identifier for the tracked person
            direction: Movement direction ('ENTER' or 'LEFT')
            in_zone: Whether person is currently in the zone
        """
        if track_id not in self.tracker_history:
            return
        
        tracker_data = self.tracker_history[track_id]
        
        # Only count when person enters or exits the zone
        if direction == 'ENTER' and not tracker_data.get('counted_enter', False):
            if in_zone:
                self.entered_count += 1
                tracker_data['counted_enter'] = True
                print(f"Person entered. Total entered: {self.entered_count}")
        
        elif direction == 'LEFT' and not tracker_data.get('counted_left', False):
            if not in_zone and tracker_data.get('was_in_zone', False):
                self.left_count += 1
                tracker_data['counted_left'] = True
                print(f"Person left. Total left: {self.left_count}")
        
        # Track zone state
        tracker_data['was_in_zone'] = in_zone
    
    def draw_dashboard(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw semi-transparent dashboard in top-right corner showing counts.
        
        Args:
            frame: Input video frame
            
        Returns:
            Frame with dashboard overlay
        """
        # Create overlay for semi-transparent background
        overlay = frame.copy()
        
        # Dashboard dimensions
        dashboard_width = 250
        dashboard_height = 120
        margin = 20
        
        # Top-right corner coordinates
        x1 = frame.shape[1] - dashboard_width - margin
        y1 = margin
        x2 = frame.shape[1] - margin
        y2 = margin + dashboard_height
        
        # Draw semi-transparent black rectangle using addWeighted
        alpha = 0.6  # 60% opacity (40% transparent)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw text
        inside = self.entered_count - self.left_count
        
        text_color = (255, 255, 255)  # White
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        
        # ENTERED
        cv2.putText(frame, f"ENTERED: {self.entered_count}", 
                    (x1 + 20, y1 + 40), font, font_scale, text_color, font_thickness)
        
        # LEFT
        cv2.putText(frame, f"LEFT: {self.left_count}", 
                    (x1 + 20, y1 + 70), font, font_scale, text_color, font_thickness)
        
        # INSIDE (calculated as ENTERED - LEFT)
        cv2.putText(frame, f"INSIDE: {inside}", 
                    (x1 + 20, y1 + 100), font, font_scale, text_color, font_thickness)
        
        return frame
    
    def draw_zone(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the trigger zone as a semi-transparent red rectangle.
        
        Args:
            frame: Input video frame
            
        Returns:
            Frame with zone overlay
        """
        # Create overlay for semi-transparent red fill
        overlay = frame.copy()
        
        # Draw filled polygon with red color and transparency
        alpha = 0.3  # 30% opacity (70% transparent)
        cv2.fillPoly(overlay, [self.zone_polygon], color=(0, 0, 255))  # Red
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw red border around zone
        cv2.polylines(frame, [self.zone_polygon], isClosed=True, color=(0, 0, 255), 
                      thickness=3)
        
        return frame
    
    def draw_tracker(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        Draw green bounding boxes and head markers for each tracked person.
        
        Args:
            frame: Input video frame
            detections: Supervision Detections object
            
        Returns:
            Frame with tracker visualizations
        """
        # Get bounding boxes and track IDs
        xyxy = detections.xyxy
        track_ids = detections.tracker_id if detections.tracker_id is not None else []
        
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw green bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw head marker - solid black circle at top-center of bounding box
            center_x = (x1 + x2) // 2
            head_y = y1  # Top of bounding box (head area)
            cv2.circle(frame, (center_x, head_y), 20, (0, 0, 0), -1)  # Black circle
            
            # Optional: Add track ID text
            if i < len(track_ids):
                cv2.putText(frame, f"ID: {track_ids[i]}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def process_video(self, output_path: str = "result.mp4"):
        """
        Process video file and count passengers.
        
        Args:
            output_path: Path to save output video
        """
        # Open video capture
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_number += 1
                
                # Run YOLO detection
                results = self.model(frame, classes=[0], verbose=False)  # class 0 = person
                
                # Convert to supervision detections
                result = results[0]
                detections = sv.Detections(
                    xyxy=result.boxes.xyxy.cpu().numpy(),
                    confidence=result.boxes.conf.cpu().numpy(),
                    class_id=result.boxes.cls.cpu().numpy().astype(int)
                )
                
                # Update tracker
                detections = self.tracker.update_with_detections(detections)
                
                # Get zone mask
                in_zone_mask = self.zone.trigger(detections=detections)
                
                # Process each detection
                for i, (bbox, track_id) in enumerate(zip(detections.xyxy, detections.tracker_id)):
                    # Detect direction
                    direction = self.detect_direction(track_id, bbox)
                    
                    # Check if in zone
                    in_zone = bool(in_zone_mask[i]) if i < len(in_zone_mask) else False
                    
                    # Update counters
                    self.update_counters(track_id, direction, in_zone)
                
                # Draw visualizations
                frame = self.draw_zone(frame)
                frame = self.draw_tracker(frame, detections)
                frame = self.draw_dashboard(frame)
                
                # Write frame
                self.out.write(frame)
                
                # Log data every 30 frames (about 1 second at 30fps)
                if frame_number % 30 == 0:
                    self.log_data(frame_number)
                
                # Display frame (optional, can be disabled for faster processing)
                cv2.imshow('Bus Passenger Counter', frame)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            # Cleanup
            cap.release()
            self.out.release()
            cv2.destroyAllWindows()
            
            # Final log entry
            self.log_data(frame_number)
            print(f"\nProcessing complete!")
            print(f"Total entered: {self.entered_count}")
            print(f"Total left: {self.left_count}")
            print(f"Final inside count: {self.entered_count - self.left_count}")
            print(f"Output saved to: {output_path}")
            print(f"Log saved to: {self.log_file}")


def main():
    """Main entry point."""
    # Initialize counter
    counter = BusPassengerCounter()
    args = counter.parse_args()
    
    # Update video path from args
    counter.video_path = args.source
    
    # Process video
    counter.process_video(output_path=args.output)


if __name__ == "__main__":
    main()
