import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
from typing import Dict, List, Tuple, Optional

class CameraTracker:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize face mesh and hands
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Pose-to-image mapping
        self.pose_images: Dict[str, str] = {}
        self.current_image_path: Optional[str] = None
        self.displayed_image = None
        
        # Detection thresholds
        self.smile_threshold = 0.3
        self.eye_attention_threshold = 0.5
        
        # Camera
        self.cap = None
        self.is_running = False
        
    def load_pose_mappings(self, filepath: str = "pose_mappings.json"):
        """Load pose-to-image mappings from JSON file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.pose_images = json.load(f)
        else:
            # Default mappings
            self.pose_images = {
                "smile_left_hand": "",
                "smile_right_hand": "",
                "smile_both_hands": "",
                "neutral_left_hand": "",
                "neutral_right_hand": "",
                "neutral_both_hands": "",
                "look_left": "",
                "look_right": "",
                "look_center": ""
            }
            self.save_pose_mappings(filepath)
    
    def save_pose_mappings(self, filepath: str = "pose_mappings.json"):
        """Save pose-to-image mappings to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.pose_images, f, indent=2)
    
    def detect_smile(self, landmarks) -> bool:
        """Detect if the person is smiling based on facial landmarks"""
        # Get key points for smile detection
        # Upper lip points: 12, 13, 14, 15, 16
        # Lower lip points: 17, 18, 19, 20, 21
        upper_lip_center = landmarks[13]
        lower_lip_center = landmarks[17]
        
        # Calculate mouth opening
        mouth_opening = abs(upper_lip_center.y - lower_lip_center.y)
        
        # Get corner points of mouth
        left_corner = landmarks[61]  # Left corner of mouth
        right_corner = landmarks[291]  # Right corner of mouth
        
        # Calculate mouth width
        mouth_width = abs(left_corner.x - right_corner.x)
        
        # Smile detection: mouth width increases and opening decreases
        return mouth_width > self.smile_threshold
    
    def detect_eye_direction(self, landmarks) -> str:
        """Detect eye direction (left, right, center)"""
        # Left eye landmarks
        left_eye_inner = landmarks[133]
        left_eye_outer = landmarks[33]
        
        # Right eye landmarks  
        right_eye_inner = landmarks[362]
        right_eye_outer = landmarks[263]
        
        # Calculate eye center positions
        left_eye_center_x = (left_eye_inner.x + left_eye_outer.x) / 2
        right_eye_center_x = (right_eye_inner.x + right_eye_outer.x) / 2
        
        # Calculate iris positions (approximate)
        left_iris_x = landmarks[468].x  # Left iris center
        right_iris_x = landmarks[473].x  # Right iris center
        
        # Determine direction based on iris position relative to eye center
        left_offset = left_iris_x - left_eye_center_x
        right_offset = right_iris_x - right_eye_center_x
        
        avg_offset = (left_offset + right_offset) / 2
        
        if avg_offset < -self.eye_attention_threshold:
            return "left"
        elif avg_offset > self.eye_attention_threshold:
            return "right"
        else:
            return "center"
    
    def detect_hand_positions(self, hand_landmarks) -> str:
        """Detect hand positions (left, right, both, none)"""
        if len(hand_landmarks) == 0:
            return "none"
        elif len(hand_landmarks) == 1:
            # Determine if it's left or right hand based on landmark positions
            hand = hand_landmarks[0]
            wrist_x = hand.landmark[0].x  # Wrist landmark
            return "left" if wrist_x < 0.5 else "right"
        else:
            return "both"
    
    def classify_pose(self, face_landmarks, hand_landmarks) -> str:
        """Classify the current pose based on detected features"""
        # Detect smile
        is_smiling = self.detect_smile(face_landmarks)
        
        # Detect eye direction
        eye_direction = self.detect_eye_direction(face_landmarks)
        
        # Detect hand positions
        hand_position = self.detect_hand_positions(hand_landmarks)
        
        # Create pose key based on combination
        if is_smiling:
            if hand_position == "left":
                return "smile_left_hand"
            elif hand_position == "right":
                return "smile_right_hand"
            elif hand_position == "both":
                return "smile_both_hands"
            else:
                return "smile_no_hands"
        else:
            if hand_position == "left":
                return "neutral_left_hand"
            elif hand_position == "right":
                return "neutral_right_hand"
            elif hand_position == "both":
                return "neutral_both_hands"
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
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process face mesh
        face_results = self.face_mesh.process(rgb_frame)
        
        # Process hands
        hand_results = self.hands.process(rgb_frame)
        
        # Draw face mesh
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                    None, self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
        
        # Draw hands
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Classify pose if face is detected
        pose_key = None
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            hand_landmarks = hand_results.multi_hand_landmarks or []
            pose_key = self.classify_pose(face_landmarks.landmark, hand_landmarks)
        
        return frame, pose_key
    
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
    tracker = CameraTracker()
    tracker.load_pose_mappings()
    
    print("Camera Tracker initialized successfully!")
    print("Available poses:", list(tracker.pose_images.keys()))

