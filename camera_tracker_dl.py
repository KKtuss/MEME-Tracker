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
import imutils

class CameraTrackerDL:
    """
    Camera tracker using deep learning for smile detection based on the Smile-Detector project.
    Uses a CNN model for accurate smile detection.
    """
    
    def __init__(self):
        # Initialize OpenCV classifiers for face detection only
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Initialize smile detection model
        self.smile_model = None
        self.load_smile_model()
        
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
        
    def load_smile_model(self):
        """Load or create the smile detection model"""
        model_path = "smile_model.h5"
        
        if os.path.exists(model_path):
            try:
                self.smile_model = keras.models.load_model(model_path)
                print("Loaded existing smile detection model")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.create_smile_model()
        else:
            print("Creating new smile detection model...")
            self.create_smile_model()
    
    def create_smile_model(self):
        """Create a simple CNN model for smile detection"""
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(2, activation='softmax')  # 2 classes: no_smile, smile
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        self.smile_model = model
        
        # Create a simple dummy training to initialize weights
        dummy_data = np.random.random((10, 32, 32, 1))
        dummy_labels = np.random.randint(0, 2, 10)
        self.smile_model.fit(dummy_data, dummy_labels, epochs=1, verbose=0)
        
        # Save the initialized model
        self.smile_model.save("smile_model.h5")
        print("Created and saved new smile detection model")
        
    def detect_smile_dl(self, face_roi) -> float:
        """Detect smile using deep learning model"""
        try:
            # Resize face to 32x32 and convert to grayscale
            face_resized = cv2.resize(face_roi, (32, 32))
            if len(face_resized.shape) == 3:
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_resized
            
            # Normalize and reshape for model
            face_normalized = face_gray.astype('float32') / 255.0
            face_input = face_normalized.reshape(1, 32, 32, 1)
            
            # Predict
            prediction = self.smile_model.predict(face_input, verbose=0)
            smile_confidence = prediction[0][1]  # Probability of smile class
            
            return float(smile_confidence)
        except Exception as e:
            print(f"Error in smile detection: {e}")
            return 0.0
    
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
        """Detect if the person is smiling using deep learning"""
        if len(faces) == 0:
            return False
        
        # Get the largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        
        # Extract face ROI
        face_roi = gray[y:y+h, x:x+w]
        
        # Use deep learning model to detect smile
        smile_confidence = self.detect_smile_dl(face_roi)
        
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
        
        # Detect smile using deep learning
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
            
            # Draw smile confidence
            face_roi = gray[y:y+h, x:x+w]
            smile_conf = self.detect_smile_dl(face_roi)
            cv2.putText(frame, f"Smile: {smile_conf:.2f}", (x, y+h+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Add detection info
        cv2.putText(frame, f"Smile Confidence: {self.smile_confidence:.2f}", 
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
    tracker = CameraTrackerDL()
    tracker.load_pose_mappings()
    
    print("Deep Learning Camera Tracker initialized successfully!")
    print("Available poses:", list(tracker.pose_images.keys()))
