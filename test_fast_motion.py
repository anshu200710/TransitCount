#!/usr/bin/env python3
"""
Quick test script for fast-moving person tracking
Tests the optimized configuration with your live stream
"""

import subprocess
import sys

def print_header(text):
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")

def test_live_stream():
    """Test with live RTMP stream"""
    print_header("🏃 TESTING FAST-MOVING PERSON TRACKING")
    
    print("Configuration Applied:")
    print("  ✓ Detection threshold: 0.05 (lower for fast motion)")
    print("  ✓ Re-link distance: 100px (larger for fast displacement)")
    print("  ✓ Debounce frames: 2 (faster counting)")
    print("  ✓ Ghost timeout: 150 frames (shorter for fast motion)")
    print("  ✓ EMA alpha: 0.50 (more responsive)")
    print("  ✓ Track buffer: 60 frames (longer persistence)")
    print("  ✓ Match threshold: 0.75 (more lenient)")
    print()
    
    stream_url = "rtmp://13.217.114.127:1935/live/stream"
    
    print("Testing with live stream:")
    print(f"  Stream: {stream_url}")
    print()
    print("What to watch for:")
    print("  ✓ Continuous tracking trails (no gaps)")
    print("  ✓ Minimal ID switches")
    print("  ✓ Accurate counts for fast-moving persons")
    print("  ✓ Smooth bounding boxes")
    print()
    print("Press 'q' in the preview window to stop")
    print("-"*70)
    
    cmd = [
        sys.executable,
        "bus_passenger_counter.py",
        "--source", stream_url,
        "--live",
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print_header("✅ TEST COMPLETED")
        print("Review the results:")
        print("  - Check result_final.mp4 for tracking quality")
        print("  - Check result_final_events.csv for count accuracy")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error: Test failed with exit code {e.returncode}")
        print("\nTroubleshooting:")
        print("  1. Check stream URL is correct")
        print("  2. Verify network connection")
        print("  3. Ensure stream is active")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)

def test_with_fast_model():
    """Test with YOLOv8n (fastest model)"""
    print_header("🚀 TESTING WITH FAST MODEL (YOLOv8n)")
    
    print("Using YOLOv8n for maximum speed:")
    print("  ✓ Fastest inference")
    print("  ✓ More frames processed per second")
    print("  ✓ Better tracking of fast motion")
    print()
    
    stream_url = "rtmp://13.217.114.127:1935/live/stream"
    
    cmd = [
        sys.executable,
        "bus_passenger_counter.py",
        "--source", stream_url,
        "--model", "yolov8n.pt",
        "--live",
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    print("Press 'q' in the preview window to stop")
    print("-"*70)
    
    try:
        subprocess.run(cmd, check=True)
        print_header("✅ FAST MODEL TEST COMPLETED")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error: Test failed with exit code {e.returncode}")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)

def show_menu():
    """Show test menu"""
    print_header("🏃 FAST-MOVING PERSON TRACKING TEST")
    
    print("Select test to run:\n")
    print("  1. Test with standard model (YOLOv8s)")
    print("  2. Test with fast model (YOLOv8n) - Recommended")
    print("  3. Run both tests")
    print("  4. Exit")
    print()
    
    choice = input("Enter choice (1-4): ").strip()
    return choice

def main():
    """Main test function"""
    
    # Check if optimizations are applied
    print_header("🔍 CHECKING CONFIGURATION")
    
    import os
    if not os.path.exists("bytetrack.yaml"):
        print("✗ bytetrack.yaml not found!")
        print("  Run: python apply_fast_motion_fix.py")
        sys.exit(1)
    
    # Read and check config
    with open("bytetrack.yaml", "r") as f:
        config = f.read()
        if "0.75" in config and "60" in config:
            print("✓ Optimized ByteTrack configuration detected")
        else:
            print("⚠️  ByteTrack configuration may not be optimized")
            print("  Run: python apply_fast_motion_fix.py")
    
    # Check code parameters
    with open("bus_passenger_counter.py", "r") as f:
        code = f.read()
        if "CONF_THRESH   = 0.05" in code:
            print("✓ Optimized code parameters detected")
        else:
            print("⚠️  Code parameters may not be optimized")
            print("  Run: python apply_fast_motion_fix.py")
    
    print()
    
    # Show menu
    choice = show_menu()
    
    if choice == "1":
        test_live_stream()
    elif choice == "2":
        test_with_fast_model()
    elif choice == "3":
        test_live_stream()
        print("\n" + "─"*70)
        input("Press Enter to continue to fast model test...")
        test_with_fast_model()
    elif choice == "4":
        print("\nExiting...")
        sys.exit(0)
    else:
        print("\n✗ Invalid choice!")
        sys.exit(1)

if __name__ == "__main__":
    main()
