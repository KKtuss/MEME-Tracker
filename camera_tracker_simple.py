import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
from typing import Dict, List, Tuple, Optional

class CameraTrackerSimple:
    """
    Simple but effective camera tracker using mouth shape analysis and brightness detection.
    This approach actually works and responds to real smiles.
    """
    
    def __init__(self):
        # Initialize OpenCV classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Pose-to-image mapping
        self.pose_images: Dict[str, str] = {}
        self.current_image_path: Optional[str] = None
        self.displayed_image = None
        
        # Detection thresholds - more conservative
        self.smile_threshold = 0.6  # Higher threshold to reduce false positives
        self.eye_attention_threshold = 0.3
        
        # Camera
        self.cap = None
        self.is_running = False
        
        # Detection state tracking
        self.smile_confidence = 0.0
        self.mouth_width_history = []
        self.baseline_scores = []  # For calibration
        self.is_calibrated = False
        
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
    
    def analyze_mouth_region(self, face_roi_gray) -> float:
        """Analyze mouth region for smile characteristics"""
        h, w = face_roi_gray.shape
        
        # Focus on the lower 40% of the face (mouth area)
        mouth_region = face_roi_gray[int(h*0.6):h, :]
        
        if mouth_region.size == 0:
            return 0.0
        
        # Method 1: Detect horizontal edges (smiles create more horizontal lines)
        horizontal_kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
        horizontal_edges = cv2.filter2D(mouth_region, -1, horizontal_kernel)
        horizontal_strength = np.mean(np.abs(horizontal_edges))
        
        # Method 2: Detect brightness changes (teeth are bright)
        mouth_brightness = np.mean(mouth_region)
        brightness_variance = np.var(mouth_region)
        
        # Method 3: Detect mouth corners (smiles widen the mouth)
        corners = cv2.goodFeaturesToTrack(mouth_region, maxCorners=10, qualityLevel=0.01, minDistance=10)
        corner_count = len(corners) if corners is not None else 0
        
        # Combine all indicators with more conservative thresholds
        # Normalize and weight the different features
        edge_score = min(horizontal_strength / 80.0, 1.0)  # Higher threshold for edges
        brightness_score = min(brightness_variance / 1500.0, 1.0)  # Higher threshold for brightness
        corner_score = min(corner_count / 8.0, 1.0)  # Higher threshold for corners
        
        # Weighted combination - more conservative
        combined_score = (edge_score * 0.5 + brightness_score * 0.3 + corner_score * 0.2)
        
        return min(combined_score, 1.0)
    
    def detect_smile(self, gray, faces) -> bool:
        """Detect smile using multiple analysis methods"""
        if len(faces) == 0:
            return False
        
        # Get the largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        
        # Extract face ROI
        face_roi_gray = gray[y:y+h, x:x+w]
        
        # Analyze mouth region
        mouth_score = self.analyze_mouth_region(face_roi_gray)
        
        # Calibration: Learn baseline scores for neutral face
        if not self.is_calibrated:
            self.baseline_scores.append(mouth_score)
            if len(self.baseline_scores) > 30:  # Collect 30 samples
                baseline_avg = sum(self.baseline_scores) / len(self.baseline_scores)
                self.baseline_scores = [baseline_avg]  # Store baseline
                self.is_calibrated = True
                print(f"Calibrated baseline score: {baseline_avg:.3f}")
        
        # Adjust score relative to baseline
        if self.is_calibrated:
            baseline = self.baseline_scores[0]
            # Only consider it a smile if significantly above baseline
            adjusted_score = max(0, (mouth_score - baseline) / (1 - baseline + 0.1))
        else:
            adjusted_score = mouth_score
        
        # Additional check: Look for mouth width changes
        mouth_width = self.estimate_mouth_width(face_roi_gray)
        self.mouth_width_history.append(mouth_width)
        
        # Keep only recent measurements
        if len(self.mouth_width_history) > 10:
            self.mouth_width_history.pop(0)
        
        # Calculate mouth width change
        if len(self.mouth_width_history) > 5:
            avg_width = sum(self.mouth_width_history[-5:]) / 5
            recent_width = sum(self.mouth_width_history[-2:]) / 2
            width_change = (recent_width - avg_width) / avg_width if avg_width > 0 else 0
            width_score = max(0, min(width_change * 3, 1))  # More sensitive to width changes
        else:
            width_score = 0
        
        # Combine mouth analysis with width change
        total_score = adjusted_score * 0.8 + width_score * 0.2
        
        # Update confidence with smoothing
        self.smile_confidence = 0.85 * self.smile_confidence + 0.15 * total_score
        
        return self.smile_confidence > self.smile_threshold
    
    def estimate_mouth_width(self, face_roi_gray) -> float:
        """Estimate mouth width from face region"""
        h, w = face_roi_gray.shape
        
        # Focus on mouth area
        mouth_region = face_roi_gray[int(h*0.6):h, :]
        
        # Find horizontal edges
        edges = cv2.Canny(mouth_region, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the widest contour (likely the mouth)
            widest_contour = max(contours, key=lambda c: cv2.boundingRect(c)[2])
            _, _, width, _ = cv2.boundingRect(widest_contour)
            return width / w  # Normalize by face width
        
        return 0.0
    
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
            
            # Draw mouth region analysis
            face_roi_gray = gray[y:y+h, x:x+w]
            mouth_score = self.analyze_mouth_region(face_roi_gray)
            
            # Draw mouth region rectangle
            mouth_y = int(y + h * 0.6)
            cv2.rectangle(frame, (x, mouth_y), (x+w, y+h), (0, 0, 255), 1)
            cv2.putText(frame, f"Mouth: {mouth_score:.2f}", (x, mouth_y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Add detection info
        cv2.putText(frame, f"Smile Confidence: {self.smile_confidence:.2f}", 
                   (10, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show calibration status
        if self.is_calibrated:
            cv2.putText(frame, f"Calibrated (Baseline: {self.baseline_scores[0]:.3f})", 
                       (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Calibrating... ({len(self.baseline_scores)}/30)", 
                       (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
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
    tracker = CameraTrackerSimple()
    tracker.load_pose_mappings()
    
    print("Simple Camera Tracker initialized successfully!")
    print("Available poses:", list(tracker.pose_images.keys()))
