from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class FaceRecognition:
    """Face recognition result"""
    guid: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    confidence: Optional[float] = None
    distance: Optional[float] = None

@dataclass
class BoundingBox:
    """Represents a bounding box for detected faces"""
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    confidence: float = 0.0

@dataclass
class FaceDetection:
    """Face detection result"""
    bounding_box: BoundingBox
                # (top, right, bottom, left)
    face_location: Tuple[int, int, int, int]  

@dataclass
class EmotionResult:
    """Emotion detection result"""
    dominant_emotion: str = None
    emotion_probabilities: Dict[str, float] = None
    emotion_category: str = None
    starting_time: float = None
    ending_time: float = None
    percentage: float = None
    type: list = None

@dataclass
class ProcessedFace:
    """Complete face processing result"""
    detection: FaceDetection
    recognition: FaceRecognition
    emotion: EmotionResult
    
    @property
    def bounding_box(self) -> BoundingBox:
        return self.detection.bounding_box
    
    @property
    def name(self) -> str:
        return self.recognition.name
    
    @property
    def dominant_emotion(self) -> str:
        return self.emotion.dominant_emotion
