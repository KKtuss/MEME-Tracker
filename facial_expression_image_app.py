import cv2
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import time

from facial_landmarks import FacialLandmarks
from gaze_tracker import GazeTracker
# from emotion_detector import EmotionDetector  # Commented out for now

class FacialExpressionImageApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Facial Expression Image Viewer")
        self.root.geometry("1200x800")
        
        # Initialize detectors
        self.landmarks_detector = FacialLandmarks()
        self.gaze_tracker = GazeTracker()
        # self.emotion_detector = EmotionDetector()  # Commented out for now
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Image storage
        self.images = {
            'eyes_open': None,
            'eyes_closed': None,
            'looking_left': None,
            'looking_right': None,
            'looking_center': None,
            'smiling': None,
            'neutral': None
        }
        
        # Current state
        self.current_expression = "neutral"
        self.is_running = False
        
        self.setup_ui()
        self.load_default_images()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Camera feed
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Camera feed label
        self.camera_label = ttk.Label(left_panel, text="Camera Feed", font=('Arial', 12, 'bold'))
        self.camera_label.pack(pady=5)
        
        # Camera display
        self.camera_display = ttk.Label(left_panel, background='black')
        self.camera_display.pack(pady=5)
        
        # Control buttons
        control_frame = ttk.Frame(left_panel)
        control_frame.pack(pady=10)
        
        self.start_button = ttk.Button(control_frame, text="Start Detection", command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Status display
        self.status_label = ttk.Label(left_panel, text="Status: Stopped", font=('Arial', 10))
        self.status_label.pack(pady=5)
        
        # Expression info
        self.expression_label = ttk.Label(left_panel, text="Current Expression: None", font=('Arial', 10))
        self.expression_label.pack(pady=5)
        
        # Right panel - Image display and controls
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Image display
        self.image_label = ttk.Label(right_panel, text="No Image Selected", font=('Arial', 12, 'bold'))
        self.image_label.pack(pady=5)
        
        self.image_display = ttk.Label(right_panel, background='lightgray')
        self.image_display.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        
        # Image controls
        controls_frame = ttk.LabelFrame(right_panel, text="Image Controls", padding=10)
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Create image selection buttons
        expressions = [
            ('eyes_open', 'Eyes Open'),
            ('eyes_closed', 'Eyes Closed'),
            ('looking_left', 'Looking Left'),
            ('looking_right', 'Looking Right'),
            ('looking_center', 'Looking Center'),
            ('smiling', 'Smiling'),
            ('neutral', 'Neutral')
        ]
        
        for i, (key, label) in enumerate(expressions):
            btn_frame = ttk.Frame(controls_frame)
            btn_frame.grid(row=i//2, column=i%2, padx=5, pady=2, sticky=tk.W)
            
            ttk.Button(btn_frame, text=f"Select {label}", 
                      command=lambda k=key: self.select_image(k)).pack(side=tk.LEFT)
            
            self.image_status = ttk.Label(btn_frame, text="Not set")
            self.image_status.pack(side=tk.LEFT, padx=5)
        
        # Clear all button
        ttk.Button(controls_frame, text="Clear All Images", command=self.clear_all_images).grid(row=4, column=0, columnspan=2, pady=10)
        
        # Auto-trigger checkbox
        self.auto_trigger_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls_frame, text="Auto-trigger images", 
                       variable=self.auto_trigger_var).grid(row=5, column=0, columnspan=2, pady=5)
        
    def load_default_images(self):
        """Load default placeholder images"""
        # Create a simple colored rectangle for each expression type
        default_images = {
            'eyes_open': self.create_default_image("Eyes Open", "green"),
            'eyes_closed': self.create_default_image("Eyes Closed", "blue"),
            'looking_left': self.create_default_image("Looking Left", "orange"),
            'looking_right': self.create_default_image("Looking Right", "purple"),
            'looking_center': self.create_default_image("Looking Center", "yellow"),
            'smiling': self.create_default_image("Smiling", "red"),
            'neutral': self.create_default_image("Neutral", "gray")
        }
        
        for key, img in default_images.items():
            self.images[key] = img
            
    def create_default_image(self, text, color):
        """Create a default colored image with text"""
        img = Image.new('RGB', (400, 300), color)
        # You could add text here if needed
        return img
        
    def select_image(self, expression_type):
        """Select an image for a specific expression"""
        file_path = filedialog.askopenfilename(
            title=f"Select image for {expression_type}",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        
        if file_path:
            try:
                img = Image.open(file_path)
                # Resize to fit display
                img = img.resize((400, 300), Image.Resampling.LANCZOS)
                self.images[expression_type] = img
                print(f"Loaded image for {expression_type}: {file_path}")
            except Exception as e:
                print(f"Error loading image: {e}")
                
    def clear_all_images(self):
        """Clear all loaded images"""
        self.load_default_images()
        print("All images cleared, using defaults")
        
    def start_detection(self):
        """Start the facial expression detection"""
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Running")
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        
    def stop_detection(self):
        """Stop the facial expression detection"""
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")
        
    def detection_loop(self):
        """Main detection loop"""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)  # Mirror effect
            
            # Get landmark data
            landmark_data = self.landmarks_detector.get_landmark_data(frame)
            
            if landmark_data["faces_detected"] > 0:
                landmark = landmark_data["landmarks"][0]
                face_coords = landmark["face"]
                eyes = landmark["eyes"]
                eye_analysis = landmark["eye_analysis"]
                
                # Analyze gaze direction
                gaze_result = self.gaze_tracker.analyze_gaze_direction(frame, eyes, face_coords)
                
                # Determine current expression
                new_expression = self.determine_expression(eye_analysis, gaze_result)
                
                # Update display if expression changed
                if new_expression != self.current_expression and self.auto_trigger_var.get():
                    self.current_expression = new_expression
                    self.root.after(0, self.update_expression_display)
                
                # Draw landmarks on frame
                frame = self.landmarks_detector.draw_landmarks(frame, landmark_data)
                
                # Add expression text to frame
                cv2.putText(frame, f"Expression: {self.current_expression}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Update status
                self.root.after(0, lambda: self.expression_label.config(
                    text=f"Current Expression: {self.current_expression}"))
            else:
                self.root.after(0, lambda: self.expression_label.config(
                    text="Current Expression: No face detected"))
            
            # Update camera display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_pil = frame_pil.resize((640, 480), Image.Resampling.LANCZOS)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            self.root.after(0, lambda: self.camera_display.config(image=frame_tk))
            self.root.after(0, lambda: setattr(self.camera_display, 'image', frame_tk))
            
            time.sleep(0.03)  # ~30 FPS
            
    def determine_expression(self, eye_analysis, gaze_result):
        """Determine the current facial expression based on detection results"""
        # Check if eyes are closed
        if gaze_result.get("is_eyes_closed", False):
            return "eyes_closed"
            
        # Check gaze direction
        gaze_direction = gaze_result.get("direction", "center")
        if gaze_direction == "left":
            return "looking_left"
        elif gaze_direction == "right":
            return "looking_right"
        elif gaze_direction == "center":
            return "looking_center"
            
        # Check if eyes are open (fallback)
        if eye_analysis.get("both_eyes_open", False):
            return "eyes_open"
            
        return "neutral"
        
    def update_expression_display(self):
        """Update the image display based on current expression"""
        if self.current_expression in self.images and self.images[self.current_expression] is not None:
            # Convert PIL image to PhotoImage
            img_tk = ImageTk.PhotoImage(self.images[self.current_expression])
            
            # Update the display
            self.image_display.config(image=img_tk)
            self.image_display.image = img_tk  # Keep a reference
            
            print(f"Displaying image for: {self.current_expression}")
            
    def on_closing(self):
        """Handle application closing"""
        self.stop_detection()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()
        
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

if __name__ == "__main__":
    app = FacialExpressionImageApp()
    app.run()
