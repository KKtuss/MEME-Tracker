#!/usr/bin/env python3
"""
Test script to verify that all dependencies are installed correctly
and the camera tracker can be initialized.
"""

import sys
import importlib

def test_imports():
    """Test if all required modules can be imported"""
    required_modules = [
        'cv2',
        'mediapipe', 
        'numpy',
        'PIL',
        'tkinter'
    ]
    
    print("Testing imports...")
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"[OK] {module}")
        except ImportError as e:
            print(f"[FAIL] {module}: {e}")
            return False
    return True

def test_camera_access():
    """Test if camera can be accessed"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("[OK] Camera access successful")
            cap.release()
            return True
        else:
            print("[FAIL] Camera access failed")
            return False
    except Exception as e:
        print(f"[FAIL] Camera test failed: {e}")
        return False

def test_tracker_initialization():
    """Test if the camera tracker can be initialized"""
    try:
        from camera_tracker import CameraTracker
        tracker = CameraTracker()
        print("[OK] Camera tracker initialization successful")
        return True
    except Exception as e:
        print(f"[FAIL] Camera tracker initialization failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("MEME TRACKER INSTALLATION TEST")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Camera Access Test", test_camera_access), 
        ("Tracker Initialization Test", test_tracker_initialization)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        if not test_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("[SUCCESS] ALL TESTS PASSED!")
        print("You can now run: py main_app.py")
    else:
        print("[ERROR] SOME TESTS FAILED!")
        print("Please check the error messages above and fix any issues.")
    print("=" * 50)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
