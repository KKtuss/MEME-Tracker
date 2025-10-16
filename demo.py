#!/usr/bin/env python3
"""
Demo script showing how to use the CameraTracker programmatically
without the GUI.
"""

import cv2
import time
from camera_tracker import CameraTracker

def demo_basic_detection():
    """Basic demo of pose detection without image assignment"""
    print("Starting basic pose detection demo...")
    print("Press 'q' to quit, 's' to save current pose info")
    
    # Initialize tracker
    tracker = CameraTracker()
    tracker.load_pose_mappings()
    
    # Start camera
    try:
        tracker.start_camera()
        print("Camera started successfully!")
    except Exception as e:
        print(f"Failed to start camera: {e}")
        return
    
    frame_count = 0
    last_pose = None
    
    try:
        while True:
            frame, pose_key = tracker.get_frame()
            
            if frame is None:
                continue
            
            frame_count += 1
            
            # Display pose information
            if pose_key != last_pose:
                last_pose = pose_key
                pose_name = pose_key.replace("_", " ").title() if pose_key else "None"
                print(f"Detected pose: {pose_name}")
            
            # Add text overlay
            cv2.putText(frame, f"Pose: {pose_name if pose_key else 'None'}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit, 's' to save pose info", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Meme Tracker Demo', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and pose_key:
                print(f"Saved pose info: {pose_key}")
                # You could save this to a file or database here
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        # Cleanup
        tracker.stop_camera()
        cv2.destroyAllWindows()
        print("Demo completed!")

def demo_with_image_assignment():
    """Demo showing how to assign and display images based on poses"""
    print("Starting image assignment demo...")
    print("This demo shows how to assign images to poses programmatically")
    
    # Initialize tracker
    tracker = CameraTracker()
    tracker.load_pose_mappings()
    
    # Example: Assign some demo images (you would replace these with actual image paths)
    demo_assignments = {
        "smile_left_hand": "images/smile_left.png",  # Replace with actual image
        "smile_right_hand": "images/smile_right.png",  # Replace with actual image
        "look_left": "images/look_left.png",  # Replace with actual image
        "look_right": "images/look_right.png",  # Replace with actual image
    }
    
    print("Assigning demo images to poses...")
    for pose, image_path in demo_assignments.items():
        # In a real scenario, you would check if the image exists
        tracker.assign_image_to_pose(pose, image_path)
        print(f"  {pose}: {image_path}")
    
    print("\nTo see the full image assignment functionality, run the GUI:")
    print("  py main_app.py")

def main():
    """Main demo function"""
    print("=" * 60)
    print("MEME TRACKER DEMO")
    print("=" * 60)
    print()
    print("Choose a demo:")
    print("1. Basic pose detection (camera feed)")
    print("2. Show image assignment example")
    print("3. Both")
    print()
    
    try:
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            demo_basic_detection()
        elif choice == "2":
            demo_with_image_assignment()
        elif choice == "3":
            demo_with_image_assignment()
            print("\n" + "=" * 40)
            demo_basic_detection()
        else:
            print("Invalid choice. Running basic detection demo...")
            demo_basic_detection()
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed: {e}")

if __name__ == "__main__":
    main()
