import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
from typing import Dict, List, Tuple, Optional

class CameraTrackerOpenCV:
    """
    Simplified camera tracker using only OpenCV for face and hand detection.
    This version works with Python 3.13 and doesn't require MediaPipe.
    """
    
    def __init__(self):
        # Initialize OpenCV classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Hand detection using contour analysis (simplified approach)
        self.hand_detector = None  # We'll use contour detection for hands
        
        # Pose-to-image mapping
        self.pose_images: Dict[str, str] = {}
        self.current_image_path: Optional[str] = None
        self.displayed_image = None
        
        # Detection thresholds
        self.smile_threshold = 3  # Minimum number of smile detections
        self.eye_attention_threshold = 0.3  # Threshold for eye position detection
        
        # Camera
        self.cap = None
        self.is_running = False
        
        # Detection state tracking
        self.smile_counter = 0
        
    def load_pose_mappings(self, filepath: str = "pose_mappings.json"):
        """Load pose-to-image mappings from JSON file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.pose_images = json.load(f)
        else:
            # Default mappings (facial features only)
            self.pose_images = {
                "smile_look_left": "",
                "smile_look_right": "",
                "smile_look_center": "",
                "look_left": "",
                "look_right": "",
                "look_center": ""
            }
            self.save_pose_mappings(filepath)
    
    def save_pose_mappings(self, filepath: str = "pose_mappings.json"):
        """Save pose-to-image mappings to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.pose_images, f, indent=2)
    
    def detect_smile(self, gray, faces) -> bool:
        """Simple, reliable smile detection"""
        smile_detections = 0
        
        for (x, y, w, h) in faces:
            # Focus on the mouth area (lower 50% of face)
            roi_gray = gray[y+int(h*0.5):y+h, x:x+w]
            if roi_gray.size > 0:
                # Single, well-tuned detection method
                smiles = self.smile_cascade.detectMultiScale(
                    roi_gray, 
                    scaleFactor=1.3,
                    minNeighbors=5,
                    minSize=(20, 20),
                    maxSize=(roi_gray.shape[0]//2, roi_gray.shape[1]//2)
                )
                smile_detections += len(smiles)
        
        # Simple smoothing
        self.smile_counter = 0.8 * self.smile_counter + 0.2 * smile_detections
        return self.smile_counter > 0.8
    
    def detect_eye_direction(self, gray, faces) -> str:
        """Detect eye direction (left, right, center) based on eye positions"""
        if len(faces) == 0:
            return "center"
        
        # Get the largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        
        # Look for eyes in the upper half of the face
        roi_gray = gray[y:y+h//2, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 2:
            # Sort eyes by x position
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye, right_eye = eyes[0], eyes[1]
            
            # Calculate eye centers
            left_eye_center = (left_eye[0] + left_eye[2]//2, left_eye[1] + left_eye[3]//2)
            right_eye_center = (right_eye[0] + right_eye[2]//2, right_eye[1] + right_eye[3]//2)
            
            # Calculate relative positions
            face_center_x = w // 2
            left_eye_rel_x = (left_eye_center[0] - face_center_x) / face_center_x
            right_eye_rel_x = (right_eye_center[0] - face_center_x) / face_center_x
            
            # Determine direction based on eye positions
            avg_eye_offset = (left_eye_rel_x + right_eye_rel_x) / 2
            
            if avg_eye_offset < -self.eye_attention_threshold:
                return "left"
            elif avg_eye_offset > self.eye_attention_threshold:
                return "right"
        
        return "center"
    
    
    def classify_pose(self, gray, frame) -> str:
        """Classify the current pose based on detected features"""
        # Detect faces with stricter parameters to reduce false positives
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,   # Less sensitive to reduce false positives
            minNeighbors=8,    # Higher threshold for better accuracy
            minSize=(80, 80),  # Larger minimum size
            maxSize=(400, 400), # Maximum size to avoid false positives
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return "no_face"
        
        # Detect smile
        is_smiling = self.detect_smile(gray, faces)
        
        # Detect eye direction
        eye_direction = self.detect_eye_direction(gray, faces)
        
        # Create pose key based on facial features only
        if is_smiling:
            if eye_direction == "left":
                return "smile_look_left"
            elif eye_direction == "right":
                return "smile_look_right"
            else:
                return "smile_look_center"
        else:
            if eye_direction == "left":
                return "look_left"
            elif eye_direction == "right":
                return "look_right"
            else:
                return "look_center"
    
    def start_camera(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        self.is_running = True
    
    def stop_camera(self):
        """Stop camera capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()
    
    def get_frame(self) -> Tuple[np.ndarray, Optional[str]]:
        """Get current frame and detected pose"""
        if not self.is_running or not self.cap:
            return None, None
        
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Classify pose
        pose_key = self.classify_pose(gray, frame)
        
        # Draw detection overlays
        self.draw_detections(frame, gray)
        
        return frame, pose_key
    
    def draw_detections(self, frame, gray):
        """Draw detection overlays on frame with improved visualization"""
        # Draw faces with same parameters as detection
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(80, 80),
            maxSize=(400, 400)
        )
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Draw eyes
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # Draw smiles using simple detection
            roi_gray_mouth = gray[y+int(h*0.5):y+h, x:x+w]
            roi_color_mouth = frame[y+int(h*0.5):y+h, x:x+w]
            if roi_gray_mouth.size > 0:
                smiles = self.smile_cascade.detectMultiScale(
                    roi_gray_mouth, 
                    scaleFactor=1.3,
                    minNeighbors=5,
                    minSize=(20, 20)
                )
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color_mouth, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
                    cv2.putText(roi_color_mouth, "Smile", (sx, sy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Add detection info
        cv2.putText(frame, f"Smile Counter: {self.smile_counter:.1f}", 
                   (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Faces: {len(faces)}", 
                   (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def get_image_for_pose(self, pose_key: str) -> Optional[str]:
        """Get image path for given pose key"""
        return self.pose_images.get(pose_key, None)
    
    def assign_image_to_pose(self, pose_key: str, image_path: str):
        """Assign an image to a specific pose"""
        self.pose_images[pose_key] = image_path
        self.save_pose_mappings()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_camera()

if __name__ == "__main__":
    # This will be used for testing the tracker
    tracker = CameraTrackerOpenCV()
    tracker.load_pose_mappings()
    
    print("OpenCV Camera Tracker initialized successfully!")
    print("Available poses:", list(tracker.pose_images.keys()))
