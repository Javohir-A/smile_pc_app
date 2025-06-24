# src/processors/emotion_recognizer.py - RESPONSIVE VERSION (Simplified)
import cv2
import logging
import time
import os
import numpy as np
from collections import deque, defaultdict
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from src.models.face_embedding import SearchResult
from src.models.stream import *
from src.config import AppConfig

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU

logger = logging.getLogger(__name__)

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logging.warning("DeepFace not available. Install with: pip install deepface")

@dataclass
class SimpleEmotionHistory:
    """Simple emotion smoothing without blocking changes"""
    face_id: str
    recent_emotions: deque  # Last 3 predictions
    current_emotion: str = "neutral"
    last_update: float = 0.0
    
class DeepFaceEmotionRecognizer:
    """DeepFace-based emotion recognition - RESPONSIVE VERSION"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.available = DEEPFACE_AVAILABLE
        
        self.emotion_cache = {}
        self.cache_timeout = 0.3  # Reduced from 0.5s for more responsiveness
        
        # SIMPLE smoothing - just average last 3 predictions
        self.emotion_histories: Dict[str, SimpleEmotionHistory] = {}
        self.history_size = 3  # Only last 3 predictions
        self.min_confidence = 0.25  # Very low threshold
        
        # Enhanced emotion mapping to boost real emotions
        self.emotion_weights = {
            'happy': 1.4,      # Really boost happy/smile detection
            'neutral': 0.8,    # Reduce neutral slightly to allow other emotions
            'sad': 1.2,        # Boost sad detection  
            'angry': 1.1,      # Boost angry detection
            'surprise': 1.0,   # Normal weight
            'fear': 0.9,       # Slight reduction
            'disgust': 0.8     # Reduce disgust
        }
        
        # Frame processing - process every frame for responsiveness
        self.process_every_n_frames = 1  # Process every frame!
        self.frame_skip_counter = 0
        
        # Cleanup
        self.cleanup_interval = 30.0
        self.last_cleanup = time.time()
        
        if not self.available:
            logger.warning("DeepFace not available - emotion recognition disabled")
        else:
            logger.info("Responsive DeepFace emotion recognizer initialized")
            logger.info(f"  Simple smoothing with {self.history_size} frame history")
            logger.info(f"  Min confidence: {self.min_confidence}")
            logger.info(f"  Processing every frame for maximum responsiveness")
    
    def predict_emotion(self, face_roi: np.ndarray, face_id: str = None) -> Tuple[str, float, Dict[str, float]]:
        """Predict emotion with simple smoothing - RESPONSIVE"""
        if not self.available:
            return 'neutral', 0.5, {'neutral': 50.0}
        
        if not face_id:
            face_id = "unknown"
        
        current_time = time.time()
        
        # Check cache first (but with shorter timeout)
        if face_id in self.emotion_cache:
            cached_data, cache_time = self.emotion_cache[face_id]
            if current_time - cache_time < self.cache_timeout:
                emotion, confidence, probs = cached_data
                return self._apply_simple_smoothing(face_id, emotion, confidence, probs, current_time)
        
        try:
            # Get new prediction from DeepFace
            raw_emotion, raw_confidence, raw_probabilities = self._get_deepface_prediction(face_roi)
            
            if raw_emotion is None:
                # Return last known emotion or neutral
                if face_id in self.emotion_histories:
                    last_emotion = self.emotion_histories[face_id].current_emotion
                    return last_emotion, 0.5, {last_emotion: 50.0}
                return 'neutral', 0.5, {'neutral': 50.0}
            
            # Cache the result
            self.emotion_cache[face_id] = ((raw_emotion, raw_confidence, raw_probabilities), current_time)
            
            # Apply simple smoothing
            return self._apply_simple_smoothing(face_id, raw_emotion, raw_confidence, raw_probabilities, current_time)
            
        except Exception as e:
            logger.debug(f"DeepFace emotion recognition error: {e}")
            if face_id in self.emotion_histories:
                last_emotion = self.emotion_histories[face_id].current_emotion
                return last_emotion, 0.5, {last_emotion: 50.0}
            return 'neutral', 0.5, {'neutral': 50.0}
    
    def _get_deepface_prediction(self, face_roi: np.ndarray) -> Tuple[Optional[str], float, Optional[Dict[str, float]]]:
        """Get prediction from DeepFace with enhanced emotion weighting"""
        try:
            # Ensure face ROI is large enough
            if face_roi.shape[0] < 48 or face_roi.shape[1] < 48:
                face_roi = cv2.resize(face_roi, (64, 64))
            
            # Convert to RGB
            if len(face_roi.shape) == 3:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
            
            # Analyze with DeepFace
            result = DeepFace.analyze(
                img_path=face_rgb,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(result, list):
                result = result[0]
            
            emotion_data = result.get('emotion', {})
            if not emotion_data:
                return None, 0.0, None
            
            # Apply emotion weights to boost non-neutral emotions
            weighted_emotions = {}
            for emotion, probability in emotion_data.items():
                weight = self.emotion_weights.get(emotion, 1.0)
                weighted_emotions[emotion] = probability * weight
            
            # Find best weighted emotion
            best_emotion = max(weighted_emotions.keys(), key=lambda k: weighted_emotions[k])
            best_confidence = emotion_data.get(best_emotion, 0.0) / 100.0
            
            # Boost confidence for non-neutral emotions
            if best_emotion != 'neutral':
                best_confidence = min(1.0, best_confidence * 1.2)
            
            logger.debug(f"DeepFace: {best_emotion} ({best_confidence:.2f}) from {emotion_data}")
            
            return best_emotion, best_confidence, emotion_data
            
        except Exception as e:
            logger.debug(f"DeepFace error: {e}")
            return None, 0.0, None
    
    def _apply_simple_smoothing(self, face_id: str, raw_emotion: str, raw_confidence: float, 
                               raw_probabilities: Dict[str, float], current_time: float) -> Tuple[str, float, Dict[str, float]]:
        """Apply very simple smoothing - just average last few predictions"""
        
        # Skip very low confidence predictions
        if raw_confidence < self.min_confidence:
            if face_id in self.emotion_histories:
                history = self.emotion_histories[face_id]
                return history.current_emotion, raw_confidence, {history.current_emotion: raw_confidence * 100}
            return 'neutral', raw_confidence, {'neutral': raw_confidence * 100}
        
        # Get or create history
        if face_id not in self.emotion_histories:
            self.emotion_histories[face_id] = SimpleEmotionHistory(
                face_id=face_id,
                recent_emotions=deque(maxlen=self.history_size),
                current_emotion="neutral",
                last_update=current_time
            )
        
        history = self.emotion_histories[face_id]
        history.last_update = current_time
        
        # Add new prediction
        history.recent_emotions.append((raw_emotion, raw_confidence))
        
        # Calculate smoothed emotion
        if len(history.recent_emotions) == 1:
            # First prediction - use it directly if confident enough
            if raw_confidence > 0.4:
                history.current_emotion = raw_emotion
                logger.info(f"Face {face_id}: First emotion {raw_emotion} ({raw_confidence:.2f})")
            
        elif len(history.recent_emotions) >= 2:
            # Look for consistent emotion in recent predictions
            recent_emotion_counts = defaultdict(float)
            total_weight = 0
            
            # Weight recent predictions more
            for i, (emotion, confidence) in enumerate(history.recent_emotions):
                weight = (i + 1) * confidence  # More recent + higher confidence = more weight
                recent_emotion_counts[emotion] += weight
                total_weight += weight
            
            # Find most weighted emotion
            if total_weight > 0:
                best_emotion = max(recent_emotion_counts.keys(), 
                                 key=lambda k: recent_emotion_counts[k])
                best_score = recent_emotion_counts[best_emotion] / total_weight
                
                # Change emotion if new one is strong enough
                if best_emotion != history.current_emotion:
                    if best_score > 0.4:  # Require 40% weighted score
                        logger.info(f"Face {face_id}: Emotion change {history.current_emotion} → {best_emotion} "
                                   f"(score: {best_score:.2f})")
                        history.current_emotion = best_emotion
                    else:
                        logger.debug(f"Face {face_id}: Not enough evidence for {best_emotion} "
                                    f"(score: {best_score:.2f})")
        
        # Cleanup old histories
        self._periodic_cleanup(current_time)
        
        return history.current_emotion, raw_confidence, {history.current_emotion: raw_confidence * 100}
    
    def _periodic_cleanup(self, current_time: float):
        """Clean up old emotion histories"""
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        cleanup_threshold = 60.0
        to_remove = []
        
        for face_id, history in self.emotion_histories.items():
            if current_time - history.last_update > cleanup_threshold:
                to_remove.append(face_id)
        
        for face_id in to_remove:
            del self.emotion_histories[face_id]
            if face_id in self.emotion_cache:
                del self.emotion_cache[face_id]
            logger.debug(f"Cleaned up emotion history for face {face_id}")
        
        self.last_cleanup = current_time
    
    def adjust_sensitivity(self, more_sensitive: bool = True):
        """Adjust emotion sensitivity"""
        if more_sensitive:
            self.min_confidence = max(0.1, self.min_confidence - 0.1)
            self.cache_timeout = max(0.1, self.cache_timeout - 0.1)
            # Boost non-neutral emotions more
            self.emotion_weights['happy'] = min(2.0, self.emotion_weights['happy'] + 0.2)
            self.emotion_weights['sad'] = min(2.0, self.emotion_weights['sad'] + 0.1)
            self.emotion_weights['angry'] = min(2.0, self.emotion_weights['angry'] + 0.1)
            self.emotion_weights['neutral'] = max(0.5, self.emotion_weights['neutral'] - 0.1)
            
            logger.info(f"🔥 INCREASED sensitivity: min_confidence={self.min_confidence:.2f}, "
                       f"happy_weight={self.emotion_weights['happy']:.1f}")
        else:
            self.min_confidence = min(0.5, self.min_confidence + 0.1)
            self.cache_timeout = min(1.0, self.cache_timeout + 0.1)
            # Reduce emotion weights
            self.emotion_weights['happy'] = max(1.0, self.emotion_weights['happy'] - 0.2)
            self.emotion_weights['sad'] = max(1.0, self.emotion_weights['sad'] - 0.1)
            self.emotion_weights['angry'] = max(1.0, self.emotion_weights['angry'] - 0.1)
            self.emotion_weights['neutral'] = min(1.2, self.emotion_weights['neutral'] + 0.1)
            
            logger.info(f"🧊 DECREASED sensitivity: min_confidence={self.min_confidence:.2f}, "
                       f"happy_weight={self.emotion_weights['happy']:.1f}")
    
    def force_emotion_reset(self, face_id: str = None):
        """Reset emotion histories"""
        if face_id:
            if face_id in self.emotion_histories:
                del self.emotion_histories[face_id]
                logger.info(f"Reset emotion history for face {face_id}")
        else:
            self.emotion_histories.clear()
            self.emotion_cache.clear()
            logger.info("Reset ALL emotion histories and cache")
    
    def get_face_emotion_stats(self, face_id: str) -> Optional[Dict[str, Any]]:
        """Get emotion stats for debugging"""
        if face_id not in self.emotion_histories:
            return None
        
        history = self.emotion_histories[face_id]
        
        recent_emotions = [emotion for emotion, _ in history.recent_emotions]
        recent_confidences = [conf for _, conf in history.recent_emotions]
        
        return {
            'face_id': face_id,
            'current_emotion': history.current_emotion,
            'recent_emotions': recent_emotions,
            'recent_confidences': recent_confidences,
            'prediction_count': len(history.recent_emotions),
            'last_update': history.last_update
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system stats"""
        emotion_distribution = defaultdict(int)
        for history in self.emotion_histories.values():
            emotion_distribution[history.current_emotion] += 1
        
        return {
            'active_faces': len(self.emotion_histories),
            'emotion_distribution': dict(emotion_distribution),
            'cache_size': len(self.emotion_cache),
            'current_weights': dict(self.emotion_weights),
            'configuration': {
                'min_confidence': self.min_confidence,
                'cache_timeout': self.cache_timeout,
                'history_size': self.history_size
            }
        }
        
def normalize_emotion(emotion: str) -> str:
    """Normalize emotion to our 3 categories"""
    emotion_map = {
        'happy': 'smile',
        'joy': 'smile',
        'neutral': 'normal',
        'surprise': 'normal',
        'sad': 'upset',
        'angry': 'upset',
        'fear': 'upset',
        'disgust': 'upset'
    }
    return emotion_map.get(emotion.lower(), 'normal')
