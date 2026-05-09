#!/usr/bin/env python3
"""
Automated script to apply fast-moving person tracking fixes
This script modifies the configuration to optimize for fast motion tracking
"""

import os
import shutil
from datetime import datetime

def backup_file(filepath):
    """Create a backup of the file with timestamp"""
    if os.path.exists(filepath):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{filepath}.backup_{timestamp}"
        shutil.copy2(filepath, backup_path)
        print(f"✓ Backed up {filepath} to {backup_path}")
        return backup_path
    return None

def apply_tracker_fix():
    """Apply optimized ByteTrack configuration"""
    print("\n" + "="*70)
    print("APPLYING FAST MOTION TRACKER FIX")
    print("="*70)
    
    # Backup original
    backup_file("bytetrack.yaml")
    
    # Copy optimized config
    if os.path.exists("bytetrack_fast_motion.yaml"):
        shutil.copy2("bytetrack_fast_motion.yaml", "bytetrack.yaml")
        print("✓ Applied optimized ByteTrack configuration")
        print("  - Lower detection thresholds")
        print("  - Longer track buffer (60 frames)")
        print("  - More lenient matching (0.75 IoU)")
    else:
        print("✗ bytetrack_fast_motion.yaml not found!")
        return False
    
    return True

def show_code_modifications():
    """Show the code modifications that should be made"""
    print("\n" + "="*70)
    print("RECOMMENDED CODE MODIFICATIONS")
    print("="*70)
    print("\nEdit 'bus_passenger_counter.py' and change these values:\n")
    
    modifications = [
        ("CONF_THRESH", "0.08", "0.05", "Lower detection threshold"),
        ("DEBOUNCE_N", "3", "2", "Reduce debounce frames"),
        ("RELINK_DIST_PX", "60", "100", "Increase re-link distance"),
        ("GHOST_TIMEOUT", "300", "150", "Shorter ghost timeout"),
        ("EMA_ALPHA", "0.40", "0.50", "More responsive tracking"),
        ("RELINK_MAX_AGE", "45", "60", "Longer re-link age"),
    ]
    
    print("┌─────────────────┬──────────┬──────────┬─────────────────────────┐")
    print("│ Parameter       │ Old      │ New      │ Purpose                 │")
    print("├─────────────────┼──────────┼──────────┼─────────────────────────┤")
    for param, old, new, purpose in modifications:
        print(f"│ {param:15} │ {old:8} │ {new:8} │ {purpose:23} │")
    print("└─────────────────┴──────────┴──────────┴─────────────────────────┘")
    
    print("\n" + "─"*70)
    print("MANUAL STEPS:")
    print("─"*70)
    print("1. Open 'bus_passenger_counter.py' in your editor")
    print("2. Find the CONFIGURATION section (around line 40-70)")
    print("3. Change the values as shown in the table above")
    print("4. Save the file")
    print("─"*70)

def show_test_commands():
    """Show test commands to verify the fix"""
    print("\n" + "="*70)
    print("TESTING THE FIX")
    print("="*70)
    print("\nTest with your live stream:\n")
    
    print("# Basic test")
    print("python bus_passenger_counter.py \\")
    print("    --source rtmp://13.217.114.127:1935/live/stream \\")
    print("    --live\n")
    
    print("# Test with fast model (recommended)")
    print("python bus_passenger_counter.py \\")
    print("    --source rtmp://13.217.114.127:1935/live/stream \\")
    print("    --model yolov8n.pt \\")
    print("    --live\n")
    
    print("# Test with debug output")
    print("python bus_passenger_counter.py \\")
    print("    --source rtmp://13.217.114.127:1935/live/stream \\")
    print("    --live \\")
    print("    --debug\n")

def show_summary():
    """Show summary of changes"""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n✓ Applied optimized ByteTrack configuration")
    print("✓ Created backup of original configuration")
    print("\n⚠ MANUAL STEP REQUIRED:")
    print("  Edit 'bus_passenger_counter.py' to change code parameters")
    print("  (See 'RECOMMENDED CODE MODIFICATIONS' section above)")
    print("\n📚 Documentation:")
    print("  - Full guide: FAST_MOTION_FIX.md")
    print("  - Project README: README.md")
    print("\n🧪 Next Steps:")
    print("  1. Apply code modifications (see above)")
    print("  2. Test with live stream")
    print("  3. Fine-tune if needed")
    print("\n" + "="*70)

def main():
    print("\n" + "🏃 "*20)
    print("FAST-MOVING PERSON TRACKING FIX")
    print("🏃 "*20)
    
    # Apply tracker fix
    success = apply_tracker_fix()
    
    if not success:
        print("\n✗ Fix application failed!")
        return
    
    # Show code modifications
    show_code_modifications()
    
    # Show test commands
    show_test_commands()
    
    # Show summary
    show_summary()
    
    print("\n✅ Tracker configuration updated successfully!")
    print("⚠️  Remember to apply code modifications manually\n")

if __name__ == "__main__":
    main()
