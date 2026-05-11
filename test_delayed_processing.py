#!/usr/bin/env python3
"""
Test script for delayed video processing
Demonstrates how to use the --delay parameter to simulate real-time processing
"""

import subprocess
import sys

def test_delayed_processing():
    """
    Test the bus passenger counter with 1-minute delay
    This simulates real-time processing on recorded video
    """
    
    print("=" * 70)
    print("Testing Bus Passenger Counter with 1-Minute Delay")
    print("=" * 70)
    print()
    print("This will process a recorded video with a 60-second delay,")
    print("simulating real-time processing conditions.")
    print()
    print("Features:")
    print("  ✓ Frames are buffered for 60 seconds before processing")
    print("  ✓ Processing happens at real-time speed (matching video FPS)")
    print("  ✓ Perfect for testing real-time scenarios with recorded footage")
    print("  ✓ Buffer size displayed in preview window")
    print()
    print("-" * 70)
    
    # Example command with 60-second delay
    cmd = [
        sys.executable,
        "bus_passenger_counter.py",
        "--source", "counting.mp4",
        "--output", "result_delayed.mp4",
        "--delay", "60",  # 1 minute delay
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    print("Press 'q' in the preview window to stop processing")
    print("-" * 70)
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print()
        print("=" * 70)
        print("Processing completed successfully!")
        print("Output saved to: result_delayed.mp4")
        print("=" * 70)
    except subprocess.CalledProcessError as e:
        print(f"Error: Processing failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    test_delayed_processing()
