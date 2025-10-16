import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
from camera_tracker_trained import CameraTrackerTrained

class MemeTrackerAppTrained:
    def __init__(self, root):
        self.root = root
        self.root.title("Meme Tracker - Trained Deep Learning Model")
        self.root.geometry("1200x800")
        
        # Initialize tracker
        self.tracker = CameraTrackerTrained()
        self.tracker.load_pose_mappings()
        
        # GUI variables
        self.is_running = False
        self.current_pose = None
        self.displayed_image = None
        
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
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Start/Stop button
        self.start_button = ttk.Button(control_frame, text="Start Camera", command=self.toggle_camera)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        # Current pose label
        self.pose_label = ttk.Label(control_frame, text="Current Pose: None", font=("Arial", 12, "bold"))
        self.pose_label.grid(row=0, column=1, padx=20)
        
        # Model status label
        model_status = "TRAINED" if self.tracker.smile_model is not None else "NOT TRAINED"
        self.info_label = ttk.Label(control_frame, text=f"Model: {model_status} - Deep Learning CNN", 
                                   font=("Arial", 10))
        self.info_label.grid(row=0, column=2, padx=20)
        
        # Training button
        self.train_button = ttk.Button(control_frame, text="Train Model", command=self.open_trainer)
        self.train_button.grid(row=0, column=3, padx=(10, 0))
        
        # Camera display
        camera_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="5")
        camera_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        self.camera_label = ttk.Label(camera_frame, text="Camera not started", background="black", foreground="white")
        self.camera_label.grid(row=0, column=0)
        
        # Image display
        image_frame = ttk.LabelFrame(main_frame, text="Assigned Image", padding="5")
        image_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.image_label = ttk.Label(image_frame, text="No image assigned", background="lightgray")
        self.image_label.grid(row=0, column=0)
        
        # Instructions panel
        instructions_frame = ttk.LabelFrame(main_frame, text="Instructions", padding="10")
        instructions_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        if self.tracker.smile_model is None:
            instructions_text = """
TRAINING REQUIRED: No trained model found!

To use this app:
1. Click 'Train Model' button
2. Follow the training process:
   - Collect 20+ samples of NO SMILE (press 'n')
   - Collect 20+ samples of SMILE (press 's')
   - Train the model (press 't')
3. Test the trained model
4. Return to this app and start the camera

The model will learn YOUR specific facial expressions for accurate detection.
            """
        else:
            instructions_text = """
MODEL READY: Trained model loaded successfully!

The CNN model has been trained on your facial expressions.
Start the camera to begin pose detection.
            """
        
        self.instructions_label = ttk.Label(instructions_frame, text=instructions_text, 
                                           font=("Arial", 10), justify="left")
        self.instructions_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Pose configuration panel
        config_frame = ttk.LabelFrame(main_frame, text="Pose Configuration", padding="10")
        config_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Create pose configuration widgets
        self.create_pose_config_widgets(config_frame)
        
    def create_pose_config_widgets(self, parent):
        """Create widgets for configuring pose-to-image mappings"""
        # Create notebook for tabs
        notebook = ttk.Notebook(parent)
        notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # Group poses by category (facial features only)
        pose_categories = {
            "Smile + Eye Direction": ["smile_look_left", "smile_look_right", "smile_look_center"],
            "Eye Direction Only": ["look_left", "look_right", "look_center"]
        }
        
        for category, poses in pose_categories.items():
            # Create tab frame
            tab_frame = ttk.Frame(notebook)
            notebook.add(tab_frame, text=category)
            
            # Create scrollable frame for poses
            canvas = tk.Canvas(tab_frame)
            scrollbar = ttk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Add pose configuration widgets
            for i, pose in enumerate(poses):
                self.create_single_pose_widget(scrollable_frame, pose, i)
            
            # Pack canvas and scrollbar
            canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
            
            tab_frame.columnconfigure(0, weight=1)
            tab_frame.rowconfigure(0, weight=1)
    
    def create_single_pose_widget(self, parent, pose, row):
        """Create widget for a single pose configuration"""
        frame = ttk.Frame(parent, padding="5")
        frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Pose name label
        pose_name = pose.replace("_", " ").title()
        label = ttk.Label(frame, text=pose_name, width=20)
        label.grid(row=0, column=0, padx=(0, 10))
        
        # Current image path label
        current_path = self.tracker.pose_images.get(pose, "")
        path_label = ttk.Label(frame, text=current_path or "No image assigned", 
                              foreground="blue" if current_path else "red")
        path_label.grid(row=0, column=1, padx=(0, 10), sticky=(tk.W, tk.E))
        
        # Browse button
        browse_button = ttk.Button(frame, text="Browse", 
                                  command=lambda p=pose, l=path_label: self.browse_image(p, l))
        browse_button.grid(row=0, column=2, padx=(0, 10))
        
        # Clear button
        clear_button = ttk.Button(frame, text="Clear", 
                                 command=lambda p=pose, l=path_label: self.clear_image(p, l))
        clear_button.grid(row=0, column=3)
        
        # Configure column weights
        frame.columnconfigure(1, weight=1)
    
    def browse_image(self, pose, path_label):
        """Browse for image file and assign to pose"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.gif *.bmp *.tiff"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title=f"Select image for {pose.replace('_', ' ').title()}",
            filetypes=filetypes
        )
        
        if filename:
            self.tracker.assign_image_to_pose(pose, filename)
            path_label.config(text=filename, foreground="blue")
            messagebox.showinfo("Success", f"Image assigned to {pose.replace('_', ' ').title()}")
    
    def clear_image(self, pose, path_label):
        """Clear image assignment for pose"""
        self.tracker.assign_image_to_pose(pose, "")
        path_label.config(text="No image assigned", foreground="red")
    
    def open_trainer(self):
        """Open the training application"""
        try:
            import subprocess
            subprocess.Popen(['py', 'train_smile_detector.py'])
            messagebox.showinfo("Training", "Training application opened in a new window.\nFollow the instructions to train your model.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open training application: {str(e)}")
    
    def toggle_camera(self):
        """Start or stop camera"""
        if self.tracker.smile_model is None:
            messagebox.showwarning("No Model", "Please train a model first before starting the camera.")
            return
        
        if not self.is_running:
            try:
                self.tracker.start_camera()
                self.is_running = True
                self.start_button.config(text="Stop Camera")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
        else:
            self.tracker.stop_camera()
            self.is_running = False
            self.start_button.config(text="Start Camera")
            self.camera_label.config(image="", text="Camera stopped")
            self.image_label.config(image="", text="No image assigned")
    
    def update_display(self):
        """Update camera feed and image display"""
        if self.is_running:
            frame, pose_key = self.tracker.get_frame()
            
            if frame is not None:
                # Update camera display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_tk = ImageTk.PhotoImage(frame_pil)
                
                self.camera_label.config(image=frame_tk, text="")
                self.camera_label.image = frame_tk  # Keep a reference
                
                # Update pose detection
                if pose_key != self.current_pose:
                    self.current_pose = pose_key
                    pose_name = pose_key.replace("_", " ").title() if pose_key else "None"
                    self.pose_label.config(text=f"Current Pose: {pose_name}")
                    
                    # Update image display
                    self.update_image_display(pose_key)
        
        # Schedule next update
        self.root.after(30, self.update_display)  # ~30 FPS
    
    def update_image_display(self, pose_key):
        """Update the displayed image based on current pose"""
        if pose_key and pose_key in self.tracker.pose_images:
            image_path = self.tracker.pose_images[pose_key]
            if image_path and os.path.exists(image_path):
                try:
                    # Load and resize image
                    img = Image.open(image_path)
                    img = img.resize((400, 300), Image.Resampling.LANCZOS)
                    img_tk = ImageTk.PhotoImage(img)
                    
                    self.image_label.config(image=img_tk, text="")
                    self.image_label.image = img_tk  # Keep a reference
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    self.image_label.config(image="", text=f"Error loading image")
            else:
                self.image_label.config(image="", text="No image assigned")
        else:
            self.image_label.config(image="", text="No image assigned")
    
    def on_closing(self):
        """Handle application closing"""
        if self.is_running:
            self.tracker.stop_camera()
        self.root.destroy()

def main():
    import os
    
    # Create images directory if it doesn't exist
    if not os.path.exists("images"):
        os.makedirs("images")
    
    root = tk.Tk()
    app = MemeTrackerAppTrained(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    root.mainloop()

if __name__ == "__main__":
    main()
