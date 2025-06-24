import logging 
from typing import Optional, List
import cv2
import time
import numpy as np

from src.config.settings import AppConfig

logger = logging.getLogger(__name__)

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logging.warning("face_recognition not available. Install with: pip install face-recognition")


class FaceRecognizer:
    """Face recognition using face_recognition library"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.available = FACE_RECOGNITION_AVAILABLE
        self.embedding_cache = {}
        self.cache_timeout = 1.0  # Cache embeddings for 1 second
        
        if not self.available:
            logger.warning("face_recognition not available - face recognition disabled")
        else:
            logger.info("Face recognizer initialized successfully")
    
    def extract_face_embedding(self, face_roi: np.ndarray, face_id: str = None) -> Optional[List[float]]:
        """Extract face embedding from face ROI"""
        if not self.available:
            logger.debug("Face recognition not available, skipping embedding extraction")
            return None
        
        # Check cache first
        current_time = time.time()
        if face_id and face_id in self.embedding_cache:
            cached_embedding, cache_time = self.embedding_cache[face_id]
            if current_time - cache_time < self.cache_timeout:
                logger.debug(f"Using cached embedding for face {face_id}")
                return cached_embedding
        
        try:
            # Validate input
            if face_roi is None or face_roi.size == 0:
                logger.debug("Invalid face ROI provided")
                return None
            
            # Convert to RGB (face_recognition expects RGB)
            if len(face_roi.shape) == 3:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
            
            # Ensure minimum size for face recognition
            min_size = 100
            if face_rgb.shape[0] < min_size or face_rgb.shape[1] < min_size:
                # Calculate scale factor to reach minimum size
                scale = max(min_size / face_rgb.shape[0], min_size / face_rgb.shape[1])
                new_width = int(face_rgb.shape[1] * scale)
                new_height = int(face_rgb.shape[0] * scale)
                face_rgb = cv2.resize(face_rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                logger.debug(f"Resized face ROI from {face_roi.shape} to {face_rgb.shape}")
            
            # Get face encodings
            logger.debug(f"Extracting face encoding from image shape: {face_rgb.shape}")
            encodings = face_recognition.face_encodings(face_rgb, num_jitters=1, model='small')
            
            if encodings and len(encodings) > 0:
                embedding = encodings[0].tolist()
                logger.debug(f"Successfully extracted face embedding with {len(embedding)} dimensions")
                
                # Cache result
                if face_id:
                    self.embedding_cache[face_id] = (embedding, current_time)
                    logger.debug(f"Cached embedding for face {face_id}")
                
                return embedding
            else:
                logger.debug("No face encodings found in the provided ROI")
                return None
            
        except Exception as e:
            logger.error(f"Face embedding extraction error for face {face_id}: {e}")
            return None