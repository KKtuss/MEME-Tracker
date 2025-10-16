#!/usr/bin/env python3
"""
Demo script to test individual modules separately.
Run this to test each component individually.
"""

import cv2
import sys
import time

def test_landmarks():
    """Test facial landmarks detection"""
    print("Testing Facial Landmarks Detection...")
    print("Press 'q' to quit")
    
    try:
        from facial_landmarks import FacialLandmarks
        detector = FacialLandmarks()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            landmark_data = detector.get_landmark_data(frame)
            frame = detector.draw_landmarks(frame, landmark_data)
            
            cv2.imshow('Facial Landmarks Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Landmarks test completed.")
        
    except ImportError as e:
        print(f"Error importing landmarks module: {e}")
    except Exception as e:
        print(f"Error in landmarks test: {e}")

def test_emotion():
    """Test emotion detection"""
    print("Testing Emotion Detection...")
    print("Press 'q' to quit")
    
    try:
        from emotion_detector import EmotionDetector
        detector = EmotionDetector()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            emotion_result = detector.analyze_emotion(frame)
            frame = detector.draw_emotion_overlay(frame, emotion_result)
            
            cv2.imshow('Emotion Detection Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Emotion test completed.")
        
    except ImportError as e:
        print(f"Error importing emotion module: {e}")
    except Exception as e:
        print(f"Error in emotion test: {e}")

def test_gaze():
    """Test gaze tracking"""
    print("Testing Gaze Tracking...")
    print("Press 'q' to quit")
    
    try:
        from gaze_tracker import GazeTracker
        tracker = GazeTracker()
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                gaze_data = tracker.track_gaze(frame, face_roi, (x, y, w, h))
                frame = tracker.draw_gaze_overlay(frame, gaze_data["gaze"], gaze_data["eyes"])
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            cv2.imshow('Gaze Tracking Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Gaze test completed.")
        
    except ImportError as e:
        print(f"Error importing gaze module: {e}")
    except Exception as e:
        print(f"Error in gaze test: {e}")

def main():
    """Main demo function"""
    print("Facial Expression Detection - Individual Module Tests")
    print("=" * 50)
    print("1. Test Facial Landmarks")
    print("2. Test Emotion Detection")
    print("3. Test Gaze Tracking")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                test_landmarks()
            elif choice == '2':
                test_emotion()
            elif choice == '3':
                test_gaze()
            elif choice == '4':
                print("Exiting...")
                break
            else:
                print("Invalid choice! Please enter 1-4.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
