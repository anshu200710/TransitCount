#!/usr/bin/env python3
"""
Verification script to check that the key fixes are properly implemented.
"""

def check_file_fixes():
    """Check that the key fixes are present in the bus_passenger_counter.py file."""
    print("🔍 Checking bus_passenger_counter.py for key fixes...")
    
    try:
        with open("bus_passenger_counter.py", "r") as f:
            content = f.read()
        
        fixes_found = []
        
        # Check 1: first_seen field in TrackState
        if "first_seen:  int   = 0" in content:
            fixes_found.append("✅ first_seen field added to TrackState")
        else:
            fixes_found.append("❌ first_seen field missing")
        
        # Check 2: MIN_TRACK_AGE constant
        if "MIN_TRACK_AGE  = 10" in content:
            fixes_found.append("✅ MIN_TRACK_AGE constant defined")
        else:
            fixes_found.append("❌ MIN_TRACK_AGE constant missing")
        
        # Check 3: exempt_locked field
        if "exempt_locked: bool = False" in content:
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
        if "if track_age < MIN_TRACK_AGE:" in content:
            fixes_found.append("✅ Minimum age check before counting")
        else:
            fixes_found.append("❌ Minimum age check missing")
        
        # Check 7: State locking logic
        if "st.exempt_locked = True" in content:
            fixes_found.append("✅ State locking logic implemented")
        else:
            fixes_found.append("❌ State locking logic missing")
        
        # Check 8: Reflective detection
        if "mask_reflective = cv2.inRange" in content:
            fixes_found.append("✅ Reflective detection for IR mode")
        else:
            fixes_found.append("❌ Reflective detection missing")
        
        # Check 9: Enhanced debug logging
        if "🚦 COUNTING CHECK:" in content:
            fixes_found.append("✅ Enhanced debug logging added")
        else:
            fixes_found.append("❌ Enhanced debug logging missing")
        
        return fixes_found
        
    except FileNotFoundError:
        return ["❌ bus_passenger_counter.py file not found"]
    except Exception as e:
        return [f"❌ Error reading file: {e}"]

def check_syntax():
    """Check for basic syntax issues."""
    print("\n🔍 Checking for syntax issues...")
    
    try:
        with open("bus_passenger_counter.py", "r") as f:
            content = f.read()
        
        # Check for duplicate lines that might cause issues
        lines = content.split('\n')
        issues = []
        
        # Look for duplicate first_seen assignments
        first_seen_lines = [i for i, line in enumerate(lines) if "first_seen=f_no" in line]
        if len(first_seen_lines) > 2:  # Should appear twice: in creation and ghost relink
            issues.append(f"⚠️  Multiple first_seen assignments found at lines: {first_seen_lines}")
        
        # Check for proper indentation in key sections
        in_update_track = False
        for i, line in enumerate(lines):
            if "def _update_track" in line:
                in_update_track = True
            elif in_update_track and line.strip().startswith("def "):
                in_update_track = False
            
            # Look for common indentation issues
            if in_update_track and line.strip() and not line.startswith(' ') and not line.startswith('\t') and i > 0:
                if not any(keyword in line for keyword in ['def ', 'class ', '#', '"""', "'''"]):
                    issues.append(f"⚠️  Possible indentation issue at line {i+1}: {line.strip()[:50]}")
        
        if not issues:
            issues.append("✅ No obvious syntax issues found")
        
        return issues
        
    except Exception as e:
        return [f"❌ Error checking syntax: {e}"]

def main():
    print("🚀 Verifying Bus Counter Fixes")
    print("="*50)
    
    # Check fixes
    fixes = check_file_fixes()
    for fix in fixes:
        print(f"  {fix}")
    
    # Check syntax
    syntax_issues = check_syntax()
    for issue in syntax_issues:
        print(f"  {issue}")
    
    print("\n" + "="*50)
    
    # Count successful fixes
    success_count = sum(1 for fix in fixes if fix.startswith("✅"))
    total_fixes = len(fixes)
    
    print(f"📊 Fix Status: {success_count}/{total_fixes} fixes implemented")
    
    if success_count == total_fixes:
        print("🎉 All fixes successfully implemented!")
        print("\n🔧 Ready to test with:")
        print("   python bus_passenger_counter.py --source your_video.mp4 --debug")
    else:
        print("⚠️  Some fixes may be missing - review the implementation")
    
    print("\n📋 Key improvements:")
    print("   • State locking prevents exempt status from being revoked")
    print("   • Minimum track age prevents premature counting")
    print("   • Ghost re-linking properly resets track age")
    print("   • Enhanced debug logging shows counting decisions")
    print("   • Dual detection works in both daylight and IR modes")

if __name__ == "__main__":
    main()