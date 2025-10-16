import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from deepface import DeepFace
import time

class EmotionDetector:
    """
    Emotion detection using DeepFace library.
    Analyzes facial expressions to detect emotions like happy, sad, angry, etc.
    """
    
    def __init__(self):
        self.emotion_models = ['emotion']  # DeepFace models to use
        self.actions = ['emotion']  # Actions to analyze
        self.backends = ['opencv']  # Backend for face detection
        
        # Emotion mapping with emojis
        self.emotion_emojis = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'fear': '😨',
            'disgust': '🤢',
            'surprise': '😲',
            'neutral': '😐'
        }
        
        # Performance optimization
        self.last_analysis_time = 0
        self.analysis_interval = 0.5  # Analyze every 0.5 seconds
        self.last_result = None
        
        # Face detection cache
        self.face_cache = {}
        self.cache_timeout = 1.0  # Cache faces for 1 second
    
    def analyze_emotion(self, frame: np.ndarray, face_coords: Optional[Tuple[int, int, int, int]] = None) -> Dict:
        """
        Analyze emotion in the frame or specific face region
        """
        current_time = time.time()
        
        # Check if we should skip analysis for performance
        if current_time - self.last_analysis_time < self.analysis_interval:
            return self.last_result or self._get_default_result()
        
        try:
            # If face coordinates provided, crop the face region
            if face_coords:
                x, y, w, h = face_coords
                # Add some padding around the face
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2 * padding)
                h = min(frame.shape[0] - y, h + 2 * padding)
                
                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size == 0 or face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                    return self._get_default_result()
                
                # Use the cropped face for analysis
                analysis_frame = face_crop
            else:
                analysis_frame = frame
            
            # Perform emotion analysis
            result = DeepFace.analyze(
                analysis_frame,
                actions=self.actions,
                models=self.emotion_models,
                detector_backend=self.backends[0],
                enforce_detection=False,
                silent=True
            )
            
            # Process the result
            if isinstance(result, list):
                result = result[0]  # Take first face if multiple detected
            
            emotion_result = self._process_emotion_result(result)
            emotion_result['analysis_time'] = current_time
            
            # Cache the result
            self.last_result = emotion_result
            self.last_analysis_time = current_time
            
            return emotion_result
            
        except Exception as e:
            print(f"Emotion analysis error: {e}")
            return self._get_default_result()
    
    def _process_emotion_result(self, result: Dict) -> Dict:
        """Process DeepFace result into our format"""
        emotions = result.get('emotion', {})
        
        # Get dominant emotion
        dominant_emotion = result.get('dominant_emotion', 'neutral')
        
        # Get confidence scores
        emotion_scores = {}
        for emotion, score in emotions.items():
            emotion_scores[emotion] = float(score)
        
        # Calculate smile confidence (happy emotion)
        smile_confidence = emotion_scores.get('happy', 0.0) / 100.0
        
        return {
            'dominant_emotion': dominant_emotion,
            'emotion_emoji': self.emotion_emojis.get(dominant_emotion, '😐'),
            'emotion_scores': emotion_scores,
            'smile_confidence': smile_confidence,
            'is_smiling': smile_confidence > 0.3,  # Threshold for smile detection
            'confidence': max(emotion_scores.values()) / 100.0,
            'analysis_successful': True
        }
    
    def _get_default_result(self) -> Dict:
        """Return default result when analysis fails"""
        return {
            'dominant_emotion': 'neutral',
            'emotion_emoji': '😐',
            'emotion_scores': {'neutral': 100.0},
            'smile_confidence': 0.0,
            'is_smiling': False,
            'confidence': 0.0,
            'analysis_successful': False
        }
    
    def get_emotion_text(self, emotion_result: Dict) -> str:
        """Get formatted emotion text with emoji"""
        if not emotion_result['analysis_successful']:
            return "No emotion detected"
        
        dominant = emotion_result['dominant_emotion']
        emoji = emotion_result['emotion_emoji']
        confidence = emotion_result['confidence']
        
        return f"{emoji} {dominant.title()} ({confidence:.1%})"
    
    def get_smile_status(self, emotion_result: Dict) -> str:
        """Get smile status text"""
        if emotion_result['is_smiling']:
            confidence = emotion_result['smile_confidence']
            return f"😊 Smiling ({confidence:.1%})"
        else:
            return "😐 Not smiling"
    
    def draw_emotion_overlay(self, frame: np.ndarray, emotion_result: Dict, position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """Draw emotion information on the frame"""
        x, y = position
        
        # Draw emotion text
        emotion_text = self.get_emotion_text(emotion_result)
        cv2.putText(frame, emotion_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw smile status
        smile_text = self.get_smile_status(emotion_result)
        cv2.putText(frame, smile_text, (x, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw detailed emotion scores
        if emotion_result['analysis_successful']:
            y_offset = y + 70
            for emotion, score in emotion_result['emotion_scores'].items():
                if score > 5:  # Only show emotions with >5% confidence
                    emoji = self.emotion_emojis.get(emotion, '😐')
                    text = f"{emoji} {emotion}: {score:.1f}%"
                    cv2.putText(frame, text, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    y_offset += 20
        
        return frame
    
    def analyze_multiple_faces(self, frame: np.ndarray, face_coords_list: List[Tuple[int, int, int, int]]) -> List[Dict]:
        """Analyze emotions for multiple faces"""
        results = []
        
        for face_coords in face_coords_list:
            result = self.analyze_emotion(frame, face_coords)
            results.append(result)
        
        return results
    
    def get_emotion_statistics(self, emotion_results: List[Dict]) -> Dict:
        """Get statistics from multiple emotion analyses"""
        if not emotion_results:
            return {}
        
        # Count emotions
        emotion_counts = {}
        total_smiles = 0
        total_confidence = 0
        
        for result in emotion_results:
            if result['analysis_successful']:
                emotion = result['dominant_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                if result['is_smiling']:
                    total_smiles += 1
                
                total_confidence += result['confidence']
        
        avg_confidence = total_confidence / len(emotion_results) if emotion_results else 0
        smile_rate = total_smiles / len(emotion_results) if emotion_results else 0
        
        return {
            'emotion_counts': emotion_counts,
            'total_faces': len(emotion_results),
            'smiling_faces': total_smiles,
            'smile_rate': smile_rate,
            'average_confidence': avg_confidence
        }

if __name__ == "__main__":
    # Test the emotion detector
    cap = cv2.VideoCapture(0)
    detector = EmotionDetector()
    
    print("Testing Emotion Detection...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror effect
        
        # Analyze emotion
        emotion_result = detector.analyze_emotion(frame)
        
        # Draw overlay
        frame = detector.draw_emotion_overlay(frame, emotion_result)
        
        cv2.imshow('Emotion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
