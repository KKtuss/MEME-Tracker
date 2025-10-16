import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from tensorflow import keras

class CameraTrackerTrained:
    """
    Camera tracker using a trained CNN model for smile detection.
    Loads a model trained specifically for the user's facial expressions.
    """
    
    def __init__(self):
        # Initialize OpenCV classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Pose-to-image mapping
        self.pose_images: Dict[str, str] = {}
        self.current_image_path: Optional[str] = None
        self.displayed_image = None
        
        # Detection thresholds
        self.smile_threshold = 0.5
        self.eye_attention_threshold = 0.3
        
        # Camera
        self.cap = None
        self.is_running = False
        
        # Detection state tracking
        self.smile_confidence = 0.0
        
        # Load trained model
        self.smile_model = None
        self.load_trained_model()
        
    def load_trained_model(self):
        """Load the trained smile detection model"""
        model_path = "trained_smile_model.h5"
        
        if os.path.exists(model_path):
            try:
                self.smile_model = keras.models.load_model(model_path)
                print("Loaded trained smile detection model successfully!")
            except Exception as e:
                print(f"Error loading trained model: {e}")
                print("Please train a model first using train_smile_detector.py")
                self.smile_model = None
        else:
            print(f"Trained model not found at {model_path}")
            print("Please train a model first using train_smile_detector.py")
            self.smile_model = None
    
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
    
    def detect_smile_trained(self, face_roi_gray) -> float:
        """Detect smile using trained CNN model"""
        if self.smile_model is None:
            return 0.0
        
        try:
            # Resize face to match model input size (64x64)
            face_resized = cv2.resize(face_roi_gray, (64, 64))
            
            # Normalize and reshape for model
            face_normalized = face_resized.astype('float32') / 255.0
            face_input = face_normalized.reshape(1, 64, 64, 1)
            
            # Predict
            prediction = self.smile_model.predict(face_input, verbose=0)
            smile_confidence = prediction[0][1]  # Probability of smile class
            
            return float(smile_confidence)
        except Exception as e:
            print(f"Error in trained smile detection: {e}")
            return 0.0
    
    def detect_smile(self, gray, faces) -> bool:
        """Detect if the person is smiling using trained model"""
        if len(faces) == 0 or self.smile_model is None:
            return False
        
        # Get the largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        
        # Extract face ROI
        face_roi_gray = gray[y:y+h, x:x+w]
        
        # Use trained model to detect smile
        smile_confidence = self.detect_smile_trained(face_roi_gray)
        
        # Update confidence with smoothing
        self.smile_confidence = 0.8 * self.smile_confidence + 0.2 * smile_confidence
        
        return self.smile_confidence > self.smile_threshold
    
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
        # Detect faces with good parameters
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            maxSize=(400, 400)
        )
        
        if len(faces) == 0:
            return "no_face"
        
        # Detect smile using trained model
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
        """Draw detection overlays on frame"""
        # Draw faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=5,
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
            
            # Show trained model prediction
            if self.smile_model is not None:
                face_roi_gray = gray[y:y+h, x:x+w]
                smile_conf = self.detect_smile_trained(face_roi_gray)
                cv2.putText(frame, f"Trained Model: {smile_conf:.3f}", (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Add detection info
        cv2.putText(frame, f"Smile Confidence: {self.smile_confidence:.3f}", 
                   (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show model status
        if self.smile_model is not None:
            cv2.putText(frame, "Model: TRAINED", (10, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Model: NOT TRAINED", (10, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
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
    tracker = CameraTrackerTrained()
    tracker.load_pose_mappings()
    
    print("Trained Camera Tracker initialized successfully!")
    print("Available poses:", list(tracker.pose_images.keys()))
    
    if tracker.smile_model is None:
        print("\nWARNING: No trained model found!")
        print("Please run 'python train_smile_detector.py' to train a model first.")
