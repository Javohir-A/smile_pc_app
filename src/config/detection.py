# config/detection.py
from dataclasses import dataclass, field
from typing import Dict, Tuple
import os
from .database import BaseConfig

@dataclass
class DetectionConfig(BaseConfig):
    """Face detection and emotion recognition configuration"""
    # Model paths
    model_prototxt: str = "src/utils/deploy.prototxt"
    model_weights: str = "src/utils/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    
    # Detection settings
    detection_size: Tuple[int, int] = (416, 416)
    detection_interval: int = 3
    distance_threshold: float = 0.6
    confidence_threshold: float = 0.5
    
    # Emotion settings
    emotion_map: Dict[str, str] = field(default_factory=dict)
    emotion_detection_interval: float = 0.5  # seconds
    min_emotion_duration: float = 1.0  # minimum duration to record emotion change
    
    # Performance settings
    max_faces_per_frame: int = 10
    face_recognition_timeout: int = 5  # seconds
    
    def __post_init__(self):
        if not self.emotion_map:
            self.emotion_map = {
                'happy': 'Smile',
                'neutral': 'Normal',
                'sad': 'Upset',
                'fear': 'Upset',
                'angry': 'Upset',
                'surprise': 'Normal',
                'disgust': 'Upset'
            }
    
    @classmethod
    def from_env(cls) -> 'DetectionConfig':
        # Parse detection size from environment
        detection_size = (416, 416)
        size_env = os.getenv('DETECTION_SIZE', '416,416')
        if size_env:
            try:
                width, height = map(int, size_env.split(','))
                detection_size = (width, height)
            except ValueError:
                pass  # Use default
        
        return cls(
            model_prototxt=os.getenv('MODEL_PROTOTXT', cls.model_prototxt),
            model_weights=os.getenv('MODEL_WEIGHTS', cls.model_weights),
            detection_size=detection_size,
            detection_interval=cls.get_env_int('DETECTION_INTERVAL', 3),
            distance_threshold=cls.get_env_float('DISTANCE_THRESHOLD', 0.6),
            confidence_threshold=cls.get_env_float('CONFIDENCE_THRESHOLD', 0.5),
            emotion_detection_interval=cls.get_env_float('EMOTION_DETECTION_INTERVAL', 0.5),
            min_emotion_duration=cls.get_env_float('MIN_EMOTION_DURATION', 1.0),
            max_faces_per_frame=cls.get_env_int('MAX_FACES_PER_FRAME', 10),
            face_recognition_timeout=cls.get_env_int('FACE_RECOGNITION_TIMEOUT', 5)
        )
