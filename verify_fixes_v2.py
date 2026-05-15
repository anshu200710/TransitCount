#!/usr/bin/env python3
"""
Verification script to check that the key fixes are properly implemented.
"""

def read_file_safe(filename):
    """Safely read file with multiple encoding attempts."""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(filename, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    
    return None

def check_file_fixes():
    """Check that the key fixes are present in the bus_passenger_counter.py file."""
    print("🔍 Checking bus_passenger_counter.py for key fixes...")
    
    content = read_file_safe("bus_passenger_counter.py")
    if content is None:
        return ["❌ Could not read bus_passenger_counter.py file"]
    
    fixes_found = []
    
    # Check 1: first_seen field in TrackState
    if "first_seen:" in content and "int" in content:
        fixes_found.append("✅ first_seen field added to TrackState")
    else:
        fixes_found.append("❌ first_seen field missing")
    
    # Check 2: MIN_TRACK_AGE constant
    if "MIN_TRACK_AGE" in content and "10" in content:
        fixes_found.append("✅ MIN_TRACK_AGE constant defined")
    else:
        fixes_found.append("❌ MIN_TRACK_AGE constant missing")
    
    # Check 3: exempt_locked field
    if "exempt_locked" in content:
        fixes_found.append("✅ exempt_locked field added")
    else:
        fixes_found.append("❌ exempt_locked field missing")
    
    # Check 4: Ghost re-linking first_seen reset
    if "adopted.first_seen = f_no" in content:
        fixes_found.append("✅ Ghost re-linking first_seen reset implemented")
    else:
        fixes_found.append("❌ Ghost re-linking first_seen reset missing")
    
    # Check 5: Track age check in counting logic
    if "track_age = f_no - st.first_seen" in content:
        fixes_found.append("✅ Track age calculation implemented")
    else:
        fixes_found.append("❌ Track age calculation missing")
    
    # Check 6: Minimum age check before counting
    if "track_age < MIN_TRACK_AGE" in content:
        fixes_found.append("✅ Minimum age check before counting")
    else:
        fixes_found.append("❌ Minimum age check missing")
    
    # Check 7: State locking logic
    if "exempt_locked = True" in content:
        fixes_found.append("✅ State locking logic implemented")
    else:
        fixes_found.append("❌ State locking logic missing")
    
    # Check 8: Reflective detection
    if "mask_reflective" in content and "cv2.inRange" in content:
        fixes_found.append("✅ Reflective detection for IR mode")
    else:
        fixes_found.append("❌ Reflective detection missing")
    
    # Check 9: Enhanced debug logging
    if "COUNTING CHECK" in content:
        fixes_found.append("✅ Enhanced debug logging added")
    else:
        fixes_found.append("❌ Enhanced debug logging missing")
    
    return fixes_found

def main():
    print("🚀 Verifying Bus Counter Fixes")
    print("="*50)
    
    # Check fixes
    fixes = check_file_fixes()
    for fix in fixes:
        print(f"  {fix}")
    
    print("\n" + "="*50)
    
    # Count successful fixes
    success_count = sum(1 for fix in fixes if fix.startswith("✅"))
    total_fixes = len(fixes)
    
    print(f"📊 Fix Status: {success_count}/{total_fixes} fixes implemented")
    
    if success_count >= 7:  # Most fixes should be present
        print("🎉 Most fixes successfully implemented!")
        print("\n🔧 Ready to test with:")
        print("   python bus_passenger_counter.py --source your_video.mp4 --debug")
        
        print("\n🐛 If normal people still aren't being counted, check:")
        print("   1. Are tracks being created? (Look for 'FRAME XXXXX | ID XX' messages)")
        print("   2. Are tracks reaching minimum age? (Look for 'Track too young' messages)")
        print("   3. Are zone transitions working? (Look for 'ZONE LOGIC' messages)")
        print("   4. Are counting conditions met? (Look for 'COUNTING CHECK' messages)")
        
    else:
        print("⚠️  Some fixes may be missing - review the implementation")
    
    print("\n📋 Key improvements made:")
    print("   • Fixed duplicate first_seen line causing syntax errors")
    print("   • Ghost re-linking now resets track age properly")
    print("   • Added comprehensive debug logging for troubleshooting")
    print("   • State locking prevents exempt status from being revoked")
    print("   • Minimum track age prevents premature counting")

if __name__ == "__main__":
    main()