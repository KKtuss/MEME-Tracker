#!/usr/bin/env python3
"""
Test script to verify that OpenCV dependencies are installed correctly
and the camera tracker can be initialized.
"""

import sys
import importlib

def test_imports():
    """Test if all required modules can be imported"""
    required_modules = [
        'cv2',
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

def test_opencv_cascades():
    """Test if OpenCV cascades are available"""
    try:
        import cv2
        # Test if haarcascade files are available
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        smile_cascade_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        
        cascades = [
            ('Face cascade', face_cascade_path),
            ('Smile cascade', smile_cascade_path),
            ('Eye cascade', eye_cascade_path)
        ]
        
        for name, path in cascades:
            cascade = cv2.CascadeClassifier(path)
            if cascade.empty():
                print(f"[FAIL] {name} not available at {path}")
                return False
            else:
                print(f"[OK] {name} loaded successfully")
        
        return True
    except Exception as e:
        print(f"[FAIL] Cascade test failed: {e}")
        return False

def test_tracker_initialization():
    """Test if the OpenCV camera tracker can be initialized"""
    try:
        from camera_tracker_opencv import CameraTrackerOpenCV
        tracker = CameraTrackerOpenCV()
        print("[OK] OpenCV camera tracker initialization successful")
        return True
    except Exception as e:
        print(f"[FAIL] OpenCV camera tracker initialization failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("MEME TRACKER OPENCV INSTALLATION TEST")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("OpenCV Cascades Test", test_opencv_cascades),
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
        print("You can now run: py main_app_opencv.py")
    else:
        print("[ERROR] SOME TESTS FAILED!")
        print("Please check the error messages above and fix any issues.")
    print("=" * 50)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

