import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import time
import threading
import json
import os
from typing import Dict, Optional

# Import our custom modules (without emotion detector for now)
from facial_landmarks import FacialLandmarks
from gaze_tracker import GazeTracker

class SimpleExpressionApp:
    """
    Simplified facial expression detection app focusing on landmarks and gaze tracking.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Facial Expression Detection")
        self.root.geometry("1200x800")
        
        # Initialize detection modules
        self.landmarks_detector = FacialLandmarks()
        self.gaze_tracker = GazeTracker()
        
        # Application state
        self.is_running = False
        self.cap = None
        self.current_frame = None
        
        # Detection results
        self.last_detection_results = {
            "landmarks": None,
            "gaze": None,
            "timestamp": 0
        }
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Create GUI
        self.create_widgets()
        
        # Start update loop
        self.update_display()
        
    def create_widgets(self):
        """Create the main GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Start/Stop button
        self.start_button = ttk.Button(control_frame, text="Start Camera", command=self.toggle_camera)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        # FPS display
        self.fps_label = ttk.Label(control_frame, text="FPS: 0", font=("Arial", 10))
        self.fps_label.grid(row=0, column=1, padx=20)
        
        # Detection status
        self.status_label = ttk.Label(control_frame, text="Camera: Stopped", font=("Arial", 10))
        self.status_label.grid(row=0, column=2, padx=20)
        
        # Camera display
        camera_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="5")
        camera_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        self.camera_label = ttk.Label(camera_frame, text="Camera not started", background="black", foreground="white")
        self.camera_label.grid(row=0, column=0)
        
        # Results display
        results_frame = ttk.LabelFrame(main_frame, text="Detection Results", padding="10")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create results display
        self.create_results_display(results_frame)
        
        # Instructions panel
        instructions_frame = ttk.LabelFrame(main_frame, text="Instructions", padding="10")
        instructions_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        instructions_text = """
🎯 DETECTION FEATURES:
• Face Detection: Blue rectangles around faces
• Eye Detection: Green rectangles around eyes  
• Smile Detection: Red rectangles around smiles
• Gaze Tracking: Yellow circles show pupil positions
• Eye Status: Shows if eyes are open or closed

👀 TESTING:
• Open/close your eyes - should detect correctly
• Look left/right - gaze direction should change
• Smile - should detect smile regions
• Move your head - face tracking should follow

🔧 FIXES APPLIED:
• Improved eye openness detection
• Filtered out false eye detections (third eye issue)
• Better gaze direction calculation
• Fixed eye detection accuracy
        """
        
        self.instructions_label = ttk.Label(instructions_frame, text=instructions_text, 
                                           font=("Arial", 10), justify="left")
        self.instructions_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
    
    def create_results_display(self, parent):
        """Create the results display panel"""
        # Gaze results
        gaze_frame = ttk.LabelFrame(parent, text="Gaze Tracking", padding="5")
        gaze_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.gaze_label = ttk.Label(gaze_frame, text="No gaze detected", font=("Arial", 12, "bold"))
        self.gaze_label.grid(row=0, column=0, sticky=(tk.W))
        
        self.gaze_details = ttk.Label(gaze_frame, text="", font=("Arial", 10))
        self.gaze_details.grid(row=1, column=0, sticky=(tk.W))
        
        # Landmarks results
        landmarks_frame = ttk.LabelFrame(parent, text="Facial Landmarks", padding="5")
        landmarks_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.landmarks_label = ttk.Label(landmarks_frame, text="No landmarks detected", font=("Arial", 12, "bold"))
        self.landmarks_label.grid(row=0, column=0, sticky=(tk.W))
        
        self.landmarks_details = ttk.Label(landmarks_frame, text="", font=("Arial", 10))
        self.landmarks_details.grid(row=1, column=0, sticky=(tk.W))
        
        # Eye status
        eye_frame = ttk.LabelFrame(parent, text="Eye Status", padding="5")
        eye_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        self.eye_label = ttk.Label(eye_frame, text="No eye status", font=("Arial", 12, "bold"))
        self.eye_label.grid(row=0, column=0, sticky=(tk.W))
        
        self.eye_details = ttk.Label(eye_frame, text="", font=("Arial", 10))
        self.eye_details.grid(row=1, column=0, sticky=(tk.W))
    
    def toggle_camera(self):
        """Start or stop camera"""
        if not self.is_running:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise Exception("Could not open camera")
                
                self.is_running = True
                self.start_button.config(text="Stop Camera")
                self.status_label.config(text="Camera: Running")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            
            self.start_button.config(text="Start Camera")
            self.status_label.config(text="Camera: Stopped")
            self.camera_label.config(image="", text="Camera stopped")
    
    def update_display(self):
        """Update camera feed and detection results"""
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Mirror effect
                self.current_frame = frame
                
                # Perform detections
                self.perform_detections(frame)
                
                # Draw overlays
                frame = self.draw_overlays(frame)
                
                # Update camera display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_tk = ImageTk.PhotoImage(frame_pil)
                
                self.camera_label.config(image=frame_tk, text="")
                self.camera_label.image = frame_tk  # Keep a reference
                
                # Update results display
                self.update_results_display()
                
                # Update FPS
                self.update_fps()
        
        # Schedule next update
        self.root.after(30, self.update_display)  # ~30 FPS
    
    def perform_detections(self, frame):
        """Perform all detection analyses"""
        current_time = time.time()
        
        # Get facial landmarks
        landmark_data = self.landmarks_detector.get_landmark_data(frame)
        
        # Get gaze analysis if face detected
        gaze_result = None
        
        if landmark_data["faces_detected"] > 0:
            landmark = landmark_data["landmarks"][0]
            face_coords = landmark["face"]
            face_roi = landmark["face_roi"]
            eyes = landmark["eyes"]
            
            # Gaze tracking
            gaze_data = self.gaze_tracker.track_gaze(frame, face_roi, face_coords)
            gaze_result = gaze_data["gaze"]
        
        # Update detection results
        self.last_detection_results = {
            "landmarks": landmark_data,
            "gaze": gaze_result,
            "timestamp": current_time
        }
    
    def draw_overlays(self, frame):
        """Draw detection overlays on frame"""
        # Draw landmarks
        if self.last_detection_results["landmarks"]:
            frame = self.landmarks_detector.draw_landmarks(frame, self.last_detection_results["landmarks"])
        
        # Draw gaze overlay
        if (self.last_detection_results["gaze"] and 
            self.last_detection_results["landmarks"] and 
            self.last_detection_results["landmarks"]["landmarks"]):
            
            eyes = self.last_detection_results["landmarks"]["landmarks"][0]["eyes"]
            frame = self.gaze_tracker.draw_gaze_overlay(frame, self.last_detection_results["gaze"], eyes)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def update_results_display(self):
        """Update the results display panel"""
        # Update gaze results
        if self.last_detection_results["gaze"]:
            gaze = self.last_detection_results["gaze"]
            self.gaze_label.config(text=self.gaze_tracker.get_gaze_text(gaze))
            
            if not gaze["is_eyes_closed"]:
                details = f"Direction: {gaze['direction'].title()}\n"
                details += f"Confidence: {gaze['confidence']:.1%}"
                self.gaze_details.config(text=details)
            else:
                self.gaze_details.config(text="Eyes closed")
        else:
            self.gaze_label.config(text="No gaze detected")
            self.gaze_details.config(text="")
        
        # Update landmarks results
        if self.last_detection_results["landmarks"]:
            landmarks = self.last_detection_results["landmarks"]
            self.landmarks_label.config(text=f"Faces: {landmarks['faces_detected']}")
            
            if landmarks["faces_detected"] > 0:
                landmark = landmarks["landmarks"][0]
                details = f"Eyes: {len(landmark['eyes'])}\n"
                details += f"Smiles: {len(landmark['smiles'])}"
                self.landmarks_details.config(text=details)
            else:
                self.landmarks_details.config(text="No landmarks detected")
        else:
            self.landmarks_label.config(text="No landmarks detected")
            self.landmarks_details.config(text="")
        
        # Update eye status
        if self.last_detection_results["landmarks"] and self.last_detection_results["landmarks"]["faces_detected"] > 0:
            eye_analysis = self.last_detection_results["landmarks"]["landmarks"][0]["eye_analysis"]
            self.eye_label.config(text=f"Eyes: {'Open' if eye_analysis['both_eyes_open'] else 'Closed'}")
            
            details = f"Left Eye: {'Open' if eye_analysis['left_eye_open'] else 'Closed'}\n"
            details += f"Right Eye: {'Open' if eye_analysis['right_eye_open'] else 'Closed'}"
            self.eye_details.config(text=details)
        else:
            self.eye_label.config(text="No eye status")
            self.eye_details.config(text="")
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_label.config(text=f"FPS: {self.current_fps:.1f}")
            
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def on_closing(self):
        """Handle application closing"""
        if self.is_running:
            self.toggle_camera()
        self.root.destroy()

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = SimpleExpressionApp(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    root.mainloop()

if __name__ == "__main__":
    main()
