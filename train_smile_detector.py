import cv2
import numpy as np
import os
import json
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class SmileDetectorTrainer:
    """
    Train a CNN model to detect smiles from live camera data.
    Collects training data and trains a custom model.
    """
    
    def __init__(self):
        self.model = None
        self.training_data = []
        self.training_labels = []
        self.data_dir = "smile_training_data"
        self.model_path = "trained_smile_model.h5"
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(f"{self.data_dir}/no_smile", exist_ok=True)
        os.makedirs(f"{self.data_dir}/smile", exist_ok=True)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        print("Smile Detector Trainer initialized!")
        print("Commands:")
        print("  'n' or '0' - Capture NO SMILE")
        print("  's' or '1' - Capture SMILE") 
        print("  't' - Train model")
        print("  'q' - Quit")
    
    def create_model(self):
        """Create a CNN model for smile detection"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(64, 64, 1)),
            
            # Convolutional layers
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer (2 classes: no_smile, smile)
            layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def capture_training_data(self):
        """Capture training data from camera"""
        print("\n=== Data Collection Mode ===")
        print("Position your face in the camera and:")
        print("Press 'n' for NO SMILE, 's' for SMILE, 'q' to quit")
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            
            # Draw face detection
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (64, 64))
                
                # Show instructions
                cv2.putText(frame, "Press 'n' for NO SMILE", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press 's' for SMILE", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press 'q' to quit", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Captured: {frame_count} samples", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Smile Training Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n') or key == ord('0'):  # No smile
                if len(faces) > 0:
                    face = faces[0]  # Take the largest face
                    x, y, w, h = face
                    face_roi = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, (64, 64))
                    
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{self.data_dir}/no_smile/no_smile_{timestamp}.jpg"
                    cv2.imwrite(filename, face_resized)
                    
                    # Add to training data
                    self.training_data.append(face_resized.reshape(64, 64, 1))
                    self.training_labels.append(0)  # 0 = no smile
                    frame_count += 1
                    print(f"Captured NO SMILE sample #{frame_count}")
            
            elif key == ord('s') or key == ord('1'):  # Smile
                if len(faces) > 0:
                    face = faces[0]  # Take the largest face
                    x, y, w, h = face
                    face_roi = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, (64, 64))
                    
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{self.data_dir}/smile/smile_{timestamp}.jpg"
                    cv2.imwrite(filename, face_resized)
                    
                    # Add to training data
                    self.training_data.append(face_resized.reshape(64, 64, 1))
                    self.training_labels.append(1)  # 1 = smile
                    frame_count += 1
                    print(f"Captured SMILE sample #{frame_count}")
        
        cv2.destroyAllWindows()
        print(f"\nData collection complete! Captured {frame_count} samples.")
        print(f"Training data: {len(self.training_data)} samples")
    
    def load_existing_data(self):
        """Load existing training data from saved images"""
        print("Loading existing training data...")
        
        # Load no smile images
        no_smile_dir = f"{self.data_dir}/no_smile"
        for filename in os.listdir(no_smile_dir):
            if filename.endswith('.jpg'):
                img_path = os.path.join(no_smile_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, (64, 64))
                    self.training_data.append(img_resized.reshape(64, 64, 1))
                    self.training_labels.append(0)
        
        # Load smile images
        smile_dir = f"{self.data_dir}/smile"
        for filename in os.listdir(smile_dir):
            if filename.endswith('.jpg'):
                img_path = os.path.join(smile_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, (64, 64))
                    self.training_data.append(img_resized.reshape(64, 64, 1))
                    self.training_labels.append(1)
        
        print(f"Loaded {len(self.training_data)} training samples")
        print(f"No smile: {self.training_labels.count(0)} samples")
        print(f"Smile: {self.training_labels.count(1)} samples")
    
    def train_model(self):
        """Train the CNN model"""
        if len(self.training_data) < 20:
            print("Not enough training data! Need at least 20 samples.")
            print("Please collect more data first.")
            return
        
        print("\n=== Training Model ===")
        
        # Convert to numpy arrays
        X = np.array(self.training_data, dtype=np.float32) / 255.0  # Normalize
        y = np.array(self.training_labels)
        
        # Create model
        self.model = self.create_model()
        
        # Split data for validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=16,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        # Save the trained model
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        
        print("Training complete!")
    
    def test_model(self):
        """Test the trained model with live camera"""
        if self.model is None:
            if os.path.exists(self.model_path):
                print("Loading trained model...")
                self.model = keras.models.load_model(self.model_path)
            else:
                print("No trained model found! Please train first.")
                return
        
        print("\n=== Testing Model ===")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (64, 64))
                face_normalized = face_resized.reshape(1, 64, 64, 1) / 255.0
                
                # Predict
                prediction = self.model.predict(face_normalized, verbose=0)
                smile_prob = prediction[0][1]  # Probability of smile
                
                # Display prediction
                label = "SMILE" if smile_prob > 0.5 else "NO SMILE"
                color = (0, 255, 0) if smile_prob > 0.5 else (0, 0, 255)
                
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Confidence: {smile_prob:.3f}", (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow('Smile Detection Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def run_interactive_training(self):
        """Run interactive training session"""
        print("=== Smile Detector Training System ===")
        
        # Load existing data
        self.load_existing_data()
        
        while True:
            print("\nOptions:")
            print("1. Collect more training data")
            print("2. Train model")
            print("3. Test model")
            print("4. Exit")
            
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                self.capture_training_data()
            elif choice == '2':
                self.train_model()
            elif choice == '3':
                self.test_model()
            elif choice == '4':
                break
            else:
                print("Invalid choice!")
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    trainer = SmileDetectorTrainer()
    trainer.run_interactive_training()
