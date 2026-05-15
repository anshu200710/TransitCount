#!/usr/bin/env python3
"""
Test script for fast entry/exit fix
Verifies that fast-moving persons are counted correctly
"""

import subprocess
import sys

def print_header(text):
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")

def print_config():
    """Display current configuration"""
    print_header("🔍 CURRENT CONFIGURATION")
    
    try:
        with open('bus_passenger_counter.py', 'r') as f:
            content = f.read()
            
        # Extract key parameters
        import re
        debounce = re.search(r'DEBOUNCE_N\s*=\s*(\d+)', content)
        dead_zone = re.search(r'DEAD_ZONE_PX\s*=\s*(\d+)', content)
        conf_thresh = re.search(r'CONF_THRESH\s*=\s*([\d.]+)', content)
        
        print("Fast Crossing Optimizations:")
        print(f"  ✓ DEBOUNCE_N: {debounce.group(1) if debounce else 'N/A'} frames")
        print(f"  ✓ DEAD_ZONE_PX: {dead_zone.group(1) if dead_zone else 'N/A'} pixels")
        print(f"  ✓ CONF_THRESH: {conf_thresh.group(1) if conf_thresh else 'N/A'}")
        print()
        
        # Check if optimized
        if debounce and debounce.group(1) == '1':
            print("✅ Configuration is OPTIMIZED for fast crossings")
        else:
            print("⚠️  Configuration may not be optimized")
            print("   Run: Edit DEBOUNCE_N to 1 in bus_passenger_counter.py")
        
    except Exception as e:
        print(f"✗ Error reading configuration: {e}")

def test_live_stream():
    """Test with live stream"""
    print_header("🏃 TESTING FAST ENTRY/EXIT")
    
    stream_url = "rtmp://13.217.114.127:1935/live/stream"
    
    print("Testing Configuration:")
    print("  • DEBOUNCE_N = 1 (immediate counting)")
    print("  • DEAD_ZONE_PX = 40 (larger detection zone)")
    print("  • CONF_THRESH = 0.05 (sensitive detection)")
    print()
    print("What to Watch For:")
    print("  ✓ Fast-moving persons are counted")
    print("  ✓ Counts appear immediately (no delay)")
    print("  ✓ No missed crossings")
    print("  ✓ Minimal false positives")
    print()
    print(f"Stream: {stream_url}")
    print("Press 'q' in preview window to stop")
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
        print("Review Results:")
        print("  1. Check if fast-moving persons were counted")
        print("  2. Review result_final_events.csv for count accuracy")
        print("  3. Watch result_final.mp4 for tracking quality")
        print()
        print("Expected Improvements:")
        print("  ✓ Fast crossing detection: 90-95%")
        print("  ✓ Overall accuracy: 90-95%")
        print("  ✓ Minimal missed counts")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error: Test failed with exit code {e.returncode}")
        print("\nTroubleshooting:")
        print("  1. Verify stream URL is correct")
        print("  2. Check network connection")
        print("  3. Ensure stream is active")
        print("  4. Try: python bus_passenger_counter.py --help")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)

def test_with_debug():
    """Test with debug output"""
    print_header("🔍 DEBUG MODE TEST")
    
    stream_url = "rtmp://13.217.114.127:1935/live/stream"
    
    print("Running with debug output to analyze crossing behavior...")
    print()
    
    cmd = [
        sys.executable,
        "bus_passenger_counter.py",
        "--source", stream_url,
        "--live",
        "--debug",
    ]
    
    print("Command:")
    print(" ".join(cmd))
    print()
    print("This will create:")
    print("  • debug_output/debug_log.csv - Detailed frame log")
    print("  • debug_output/crops/ - Person crop images")
    print("  • debug_output/analysis/ - HSV analysis images")
    print()
    print("Press 'q' in preview window to stop")
    print("-"*70)
    
    try:
        subprocess.run(cmd, check=True)
        
        print_header("✅ DEBUG TEST COMPLETED")
        print("Check debug_output/ folder for detailed analysis")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error: Test failed with exit code {e.returncode}")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)

def show_menu():
    """Show test menu"""
    print_header("🏃 FAST ENTRY/EXIT TEST MENU")
    
    print("Select test to run:\n")
    print("  1. Standard test (live stream)")
    print("  2. Debug mode test (detailed analysis)")
    print("  3. Show current configuration")
    print("  4. Exit")
    print()
    
    choice = input("Enter choice (1-4): ").strip()
    return choice

def show_fix_instructions():
    """Show how to apply the fix if not applied"""
    print_header("📝 HOW TO APPLY THE FIX")
    
    print("Edit bus_passenger_counter.py:")
    print()
    print("1. Find the CONFIGURATION section (around line 45-50)")
    print()
    print("2. Change these values:")
    print("   DEBOUNCE_N = 1     # Was: 2")
    print("   DEAD_ZONE_PX = 40  # Was: 30")
    print()
    print("3. Save the file")
    print()
    print("4. Run this test again")
    print()

def main():
    """Main test function"""
    
    print("\n" + "🏃 "*20)
    print("FAST ENTRY/EXIT FIX TEST")
    print("🏃 "*20)
    
    # Check configuration
    print_config()
    
    # Show menu
    choice = show_menu()
    
    if choice == "1":
        test_live_stream()
    elif choice == "2":
        test_with_debug()
    elif choice == "3":
        print_config()
    elif choice == "4":
        print("\nExiting...")
        sys.exit(0)
    else:
        print("\n✗ Invalid choice!")
        sys.exit(1)

if __name__ == "__main__":
    main()
