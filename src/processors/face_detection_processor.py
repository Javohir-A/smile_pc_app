# src/processors/face_detection_processor.py - OPTIMIZED FOR REAL-TIME STREAMING
import cv2
import numpy as np
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .emotion_recognizer import DeepFaceEmotionRecognizer
from .face_recognizer import FaceRecognizer

from src.config import AppConfig
from src.di.dependencies import DependencyContainer
from src.models.stream import *

logger = logging.getLogger(__name__)

@dataclass
class FaceDetection:
    """Represents a detected face with recognition info"""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    landmarks: Optional[List[tuple]] = None
    emotion: Optional[str] = None
    emotion_confidence: Optional[float] = None
    emotion_probabilities: Optional[Dict[str, float]] = None
    face_id: Optional[str] = None
    recognition_confidence: Optional[float] = None
    # New fields for recognition
    human_guid: Optional[str] = None
    human_name: Optional[str] = None
    human_type: Optional[str] = None
    is_recognized: bool = False
    face_embedding: Optional[List[float]] = None

@dataclass
class DetectionResult:
    """Result of face detection processing"""
    stream_id: str
    timestamp: float
    frame_number: int
    faces: List[FaceDetection]
    processing_time: float    

@dataclass
class CachedFaceData:
    """Cached face recognition and emotion data"""
    human_guid: Optional[str] = None
    human_name: Optional[str] = None
    human_type: Optional[str] = None
    recognition_confidence: float = 0.0
    is_recognized: bool = False
    emotion: str = "neutral"
    emotion_confidence: float = 0.5
    last_update: float = 0.0
    update_count: int = 0

class FaceDetectionProcessor:
    """Enhanced face detection processor optimized for real-time streaming"""
    
    def __init__(self, config: AppConfig, container: DependencyContainer):
        self.config = config
        self.container = container
        self.net = None
        self.emotion_recognizer = None
        self.face_recognizer = None
        self.face_usecase = None
        self.detection_callbacks: List[callable] = []
        self.face_tracker = {}  # For tracking faces across frames
        self.next_face_id = 1
        self._lock = threading.Lock()
        self.min_face_size: Optional[int] = 80
        
        # Store detection results for each stream
        self.stream_detections: Dict[str, DetectionResult] = {}
        self.detection_lock = threading.Lock()
        
        # ULTRA OPTIMIZATION: Aggressive caching and processing intervals
        self.face_cache: Dict[str, CachedFaceData] = {}  # Cache all face data
        self.cache_timeout = 3.0  # Keep cache for 3 seconds
        
        # Recognition cache
        self.recognition_cache = {}
        self.recognition_cache_timeout = 2.0  # Cache for 2 seconds
        
        # Processing intervals to reduce load
        self.emotion_process_every = 5  # Process emotion every 5 frames
        self.recognition_process_every = 10  # Process face recognition every 10 frames
        self.frame_counter = 0
        
        # Performance optimization
        self.background_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="face_detection")
        self.pending_background_tasks = set()
        
        # PERFORMANCE: Reduce logging frequency
        self.log_counter = 0
        self.log_every_n_frames = 100  # Log detailed info only every 30 frames
        
        # Initialize all components
        self._initialize_models()
        logger.info("Enhanced FaceDetectionProcessor with recognition initialized")
    
    def _initialize_models(self):
        """Initialize face detection, emotion recognition, and face recognition models"""
        try:
            # Load DNN face detection model
            if not self._load_face_detection_model():
                logger.error("Failed to load face detection model")
                return
            
            # Initialize DeepFace emotion recognizer
            try:
                self.emotion_recognizer = DeepFaceEmotionRecognizer(self.config)
                self.emotion_recognizer.adjust_sensitivity(more_sensitive=False)
                
                self.emotion_recognizer.cache_timeout = 2.0
                
                logger.info("Emotion recognizer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize emotion recognizer: {e}")
                self.emotion_recognizer = None
            
            # Initialize face recognizer
            try:
                self.face_recognizer = FaceRecognizer(self.config)
                self.face_recognizer.cache_timeout = 3.0
                
                logger.info("Face recognizer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize face recognizer: {e}")
                self.face_recognizer = None
            
            # Get face usecase for recognition
            self._initialize_face_usecase()
            
            logger.info("All face detection models initialized successfully")
            
        except Exception as e:
            logger.error(f"Critical error initializing face detection models: {e}")
    
    def _initialize_face_usecase(self):
        """Initialize face usecase with proper error handling and validation"""
        try:
            logger.info("Attempting to get face usecase from dependency container...")
            self.face_usecase = self.container.get_face_usecase()
            
            if self.face_usecase is None:
                logger.warning("Face usecase is None from container")
                raise Exception("Face usecase returned None from container")
            
            # Quick test without detailed logging
            test_embedding = [0.0] * 128
            try:
                test_results = self.face_usecase.search_similar_faces(test_embedding, limit=1, threshold=0.9)
                logger.info(f"Face usecase test successful")
            except Exception as test_e:
                logger.warning(f"Face usecase test failed (may be normal if database is empty)")
            
            logger.info("Face usecase initialized successfully")
            
        except Exception as e:
            logger.warning(f"Could not get face usecase from container: {e}")

        # Final validation
        if self.face_usecase is None:
            logger.error("CRITICAL: Face usecase could not be initialized - face recognition will be disabled")
        else:
            logger.info("Face usecase is ready for face recognition")
    
    def _load_face_detection_model(self) -> bool:
        """Load the DNN face detection model with optimized settings"""
        try:
            prototxt_path = Path(self.config.detection.model_prototxt)
            model_path = Path(self.config.detection.model_weights)
            
            if not prototxt_path.is_file():
                raise FileNotFoundError(f"Missing prototxt file: {prototxt_path}")
            if not model_path.is_file():
                raise FileNotFoundError(f"Missing model file: {model_path}")
            
            logger.info(f"Loading DNN model from {prototxt_path} and {model_path}")
            self.net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
            
            # Optimize DNN backend if available
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                logger.info("Using CUDA backend for face detection")
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                logger.info("Using CPU backend for face detection")
            
            # Lower confidence threshold for better angle detection
            self.min_confidence = getattr(self.config.detection, 'confidence_threshold', 0.4)
            
            logger.info(f"DNN face detection model loaded successfully (confidence threshold: {self.min_confidence})")
            return True
            
        except Exception as e:
            logger.error(f"DNN model loading failed: {e}")
            return False
    
    def add_detection_callback(self, callback: callable):
        """Add a callback function to be called when faces are detected"""
        self.detection_callbacks.append(callback)
        logger.info(f"Added detection callback: {callback.__name__}")
        
    def _get_cached_face_data(self, face_id: str) -> CachedFaceData:
        """Get cached face data or create new cache entry"""
        current_time = time.time()
        
        if face_id in self.face_cache:
            cached_data = self.face_cache[face_id]
            # Check if cache is still valid
            if current_time - cached_data.last_update < self.cache_timeout:
                return cached_data
        
        # Create new cache entry
        self.face_cache[face_id] = CachedFaceData(last_update=current_time)
        return self.face_cache[face_id]    
    
    def _update_face_cache_async(self, face_id: str, face_roi_expanded: np.ndarray, face_roi: np.ndarray):
        """Update face cache with recognition and emotion data in background"""
        def background_update():
            try:
                current_time = time.time()
                cached_data = self.face_cache.get(face_id)
                if not cached_data:
                    return
                
                # Face recognition (expensive operation)
                if (self.face_recognizer and self.face_recognizer.available and 
                    self.face_usecase and not cached_data.is_recognized):
                    
                    face_embedding = self.face_recognizer.extract_face_embedding(face_roi_expanded, face_id)
                    if face_embedding:
                        search_results = self.face_usecase.search_similar_faces(
                            face_embedding, limit=1, threshold=0.6
                        )
                        
                        if search_results and len(search_results) > 0:
                            best_match = search_results[0]
                            if best_match.recognition_confidence >= 0.86:
                                cached_data.human_guid = best_match.human_guid
                                cached_data.human_name = best_match.name
                                cached_data.human_type = best_match.human_type
                                cached_data.recognition_confidence = best_match.recognition_confidence
                                cached_data.is_recognized = True
                                
                                # Log only significant recognitions
                                if self.log_counter % 50 == 0:
                                    logger.info(f"Background: Face {face_id} recognized as {best_match.name}")
                
                # Emotion recognition (expensive operation)
                if self.emotion_recognizer:
                    emotion, emotion_conf, _ = self.emotion_recognizer.predict_emotion(face_roi, face_id)
                    if emotion and emotion_conf > 0.3:
                        cached_data.emotion = emotion
                        cached_data.emotion_confidence = emotion_conf
                
                cached_data.last_update = current_time
                cached_data.update_count += 1
                
            except Exception as e:
                if self.log_counter % 100 == 0:
                    logger.error(f"Background face update error: {e}")
            finally:
                # Remove from pending tasks
                self.pending_background_tasks.discard(face_id)
        
        # Only start background task if not already running for this face
        if face_id not in self.pending_background_tasks:
            self.pending_background_tasks.add(face_id)
            self.background_executor.submit(background_update)
    
    def _recognize_face(self, face_embedding: List[float], face_id: str) -> Tuple[Optional[str], Optional[str], Optional[str], float]:
        """Recognize face using face usecase with improved error handling"""
        if not self.face_usecase:
            return None, None, None, 0.0
        
        if not face_embedding:
            return None, None, None, 0.0
        
        # Check cache first
        current_time = time.time()
        if face_id in self.recognition_cache:
            cached_data, cache_time = self.recognition_cache[face_id]
            if current_time - cache_time < self.recognition_cache_timeout:
                return cached_data
        
        try:
            # Search for similar faces with proper parameters
            search_results = self.face_usecase.search_similar_faces(
                face_embedding,
                limit=3,  # Get top 3 matches for better accuracy
                threshold=0.6  # Lower threshold for better recall
            )
            
            if search_results and len(search_results) > 0:
                # Get the best match
                best_match = search_results[0]
                
                # Only accept matches above a certain confidence threshold
                if best_match.recognition_confidence >= 0.86:  
                    result = (best_match.human_guid, best_match.name, best_match.human_type, best_match.recognition_confidence)
                    
                    # Cache the successful result
                    self.recognition_cache[face_id] = (result, current_time)
                    
                    # REDUCED LOGGING: Only log on first recognition
                    if face_id not in self.recognition_cache:
                        logger.info(f"Face {face_id} recognized as {best_match.name} ({best_match.human_type})")
                    return result
            
            return None, None, None, 0.0
            
        except Exception as e:
            # REDUCED LOGGING: Only log errors occasionally
            if self.log_counter % 100 == 0:
                logger.error(f"Face recognition error for face {face_id}: {e}")
            return None, None, None, 0.0
    
    def process_frame(self, frame_data: FrameData) -> Optional[DetectionResult]:
        """Process a single frame - ULTRA-OPTIMIZED for minimal lag"""
        if not self.net:
            return None
        
        start_time = time.time()
        self.frame_counter += 1
        self.log_counter += 1
        
        try:
            frame = frame_data.frame
            h, w = frame.shape[:2]
            
            # Create blob for DNN (this is fast)
            blob = cv2.dnn.blobFromImage(
                frame, 1.0, (300, 300), [104, 117, 123], False, False
            )
            
            # Face detection (this is reasonably fast)
            self.net.setInput(blob)
            detections = self.net.forward()
            
            # Process detections with aggressive optimization
            face_detections = []
            confidence_threshold = self.min_confidence
            
            should_log = (self.log_counter % self.log_every_n_frames == 0)
            
            for i, detection in enumerate(detections[0, 0, :, :]):
                confidence = detection[2]
                
                if confidence > confidence_threshold:
                    # Calculate bounding box (fast)
                    x1 = max(0, int(detection[3] * w))
                    y1 = max(0, int(detection[4] * h))
                    x2 = min(w, int(detection[5] * w))
                    y2 = min(h, int(detection[6] * h))
                    
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Skip tiny faces
                    if width < 30 or height < 30:
                        continue
                    
                    # Get face ID (fast)
                    face_id = self._assign_face_id(x1, y1, width, height)
                    
                    # Get cached face data (very fast)
                    cached_data = self._get_cached_face_data(face_id)
                    
                    # Extract face ROIs (fast)
                    expand_ratio = 0.2
                    expand_x = int(width * expand_ratio / 2)
                    expand_y = int(height * expand_ratio / 2)
                    
                    ex1 = max(0, x1 - expand_x)
                    ey1 = max(0, y1 - expand_y)
                    ex2 = min(w, x2 + expand_x)
                    ey2 = min(h, y2 + expand_y)
                    
                    face_roi_expanded = frame[ey1:ey2, ex1:ex2]
                    face_roi = frame[y1:y2, x1:x2]
                    
                    # CRITICAL OPTIMIZATION: Only do expensive operations occasionally and in background
                    should_process_recognition = (self.frame_counter % self.recognition_process_every == 0)
                    should_process_emotion = (self.frame_counter % self.emotion_process_every == 0)
                    
                    if (should_process_recognition or should_process_emotion) and face_roi.size > 0:
                        # Start background processing for expensive operations
                        self._update_face_cache_async(face_id, face_roi_expanded, face_roi)
                    
                    # Create face detection object using cached data (very fast)
                    face_detection = FaceDetection(
                        x=x1,
                        y=y1,
                        width=width,
                        height=height,
                        confidence=float(confidence),
                        emotion=cached_data.emotion,
                        emotion_confidence=cached_data.emotion_confidence,
                        emotion_probabilities=None,  # Skip for performance
                        face_id=face_id,
                        human_guid=cached_data.human_guid,
                        human_name=cached_data.human_name,
                        human_type=cached_data.human_type,
                        is_recognized=cached_data.is_recognized,
                        recognition_confidence=cached_data.recognition_confidence,
                        face_embedding=None  # Skip for performance
                    )
                    
                    face_detections.append(face_detection)
            
            processing_time = time.time() - start_time
            
            # Create result
            result = DetectionResult(
                stream_id=frame_data.stream_id,
                timestamp=frame_data.timestamp,
                frame_number=frame_data.frame_number,
                faces=face_detections,
                processing_time=processing_time
            )
            
            # Store detection results (fast)
            with self.detection_lock:
                self.stream_detections[frame_data.stream_id] = result
            
            # Call callbacks in background (non-blocking)
            if face_detections:
                for callback in self.detection_callbacks:
                    try:
                        self.background_executor.submit(callback, result, frame_data)
                    except Exception:
                        pass  # Ignore callback errors to avoid slowing down
            
            # Minimal logging
            if face_detections and should_log:
                recognized_count = len([f for f in face_detections if f.is_recognized])
                logger.info(f"Stream {frame_data.stream_id}: {len(face_detections)} faces "
                           f"({recognized_count} recognized) in {processing_time:.3f}s")
            
            return result
        
        except Exception as e:
            if self.log_counter % 200 == 0:  # Very rare error logging
                logger.error(f"Frame processing error: {e}")
            return None
    
    def get_stream_detections(self, stream_id: str) -> Optional[DetectionResult]:
        """Get the latest detection results for a stream"""
        with self.detection_lock:
            return self.stream_detections.get(stream_id)
    
    def _assign_face_id(self, x: int, y: int, width: int, height: int) -> str:
        """Assign a face ID for tracking purposes - OPTIMIZED"""
        center_x = x + width // 2
        center_y = y + height // 2
        current_time = time.time()
        
        # Simple distance check without lock for performance
        min_distance = float('inf')
        closest_id = None
        
        for face_id, (prev_x, prev_y, _) in self.face_tracker.items():
            distance = (center_x - prev_x) ** 2 + (center_y - prev_y) ** 2  # Skip sqrt for speed
            if distance < min_distance and distance < 10000:  # 100^2 pixel threshold
                min_distance = distance
                closest_id = face_id
        
        if closest_id:
            self.face_tracker[closest_id] = (center_x, center_y, current_time)
            return closest_id
        else:
            new_id = f"face_{self.next_face_id:03d}"
            self.next_face_id += 1
            self.face_tracker[new_id] = (center_x, center_y, current_time)
            
            # Clean up old face IDs less frequently
            if self.frame_counter % 150 == 0:  # Cleanup every 150 frames instead of every frame
                to_remove = [
                    fid for fid, (_, _, timestamp) in self.face_tracker.items()
                    if current_time - timestamp > 10.0  # Longer timeout
                ]
                for fid in to_remove:
                    self.face_tracker.pop(fid, None)
                    self.face_cache.pop(fid, None)  # Also clean cache
            
            return new_id
        
    def _notify_callbacks(self, result: DetectionResult, frame_data: FrameData):
        """Notify all registered callbacks about detection results"""
        for callback in self.detection_callbacks:
            try:
                # Run callback in thread pool to avoid blocking
                self.executor.submit(callback, result, frame_data)
            except Exception as e:
                if self.log_counter % 100 == 0:
                    logger.error(f"Error in detection callback: {e}")
    
    def draw_detections(self, frame: np.ndarray, detections: List[FaceDetection], 
                       show_emotions: bool = True, show_probabilities: bool = False) -> np.ndarray:
        """Draw detection results on frame with names and enhanced visualization"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            # Draw bounding box with color based on recognition status
            if detection.is_recognized:
                color = (0, 255, 0)  # Green for recognized faces
                thickness = 3
            else:
                color = self._get_emotion_color(detection.emotion)
                thickness = 2
            
            # Draw main bounding box
            cv2.rectangle(
                annotated_frame,
                (detection.x, detection.y),
                (detection.x + detection.width, detection.y + detection.height),
                color,
                thickness
            )
            
            # Prepare main label text
            label_parts = []
            
            # Show name if recognized
            if detection.is_recognized and detection.human_name:
                label_parts.append(f"{detection.human_name}")
                if detection.recognition_confidence > 0:
                    label_parts.append(f"({detection.recognition_confidence:.1%})")
            else:
                if detection.face_id:
                    label_parts.append(f"Unknown #{detection.face_id.split('_')[-1]}")
            
            from .emotion_recognizer import normalize_emotion            
            # Add emotion info
            if show_emotions and detection.emotion:
                emotion_text = f"{normalize_emotion(detection.emotion).title()}"
                if detection.emotion_confidence:
                    emotion_text += f" {detection.emotion_confidence:.0%}"
                label_parts.append(emotion_text)
            # Draw main label
            if label_parts:
                # Combine label parts
                if detection.is_recognized:
                    # For recognized faces, show name on top line, emotion on second line
                    name_label = label_parts[0] + (f" {label_parts[1]}" if len(label_parts) > 1 and "%" in label_parts[1] else "")
                    emotion_label = label_parts[-1] if show_emotions and len(label_parts) > 1 else None
                    
                    # Draw name label
                    font_scale = 0.8
                    font_thickness = 2
                    label_size = cv2.getTextSize(name_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    
                    # Draw name background
                    cv2.rectangle(
                        annotated_frame,
                        (detection.x, detection.y - label_size[1] - 20),
                        (detection.x + label_size[0] + 10, detection.y - 5),
                        color,
                        -1
                    )
                    
                    # Draw name text
                    cv2.putText(
                        annotated_frame,
                        name_label,
                        (detection.x + 5, detection.y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        font_thickness
                    )
                    
                    # Draw emotion label below name if available
                    if emotion_label:
                        emotion_size = cv2.getTextSize(emotion_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                        cv2.rectangle(
                            annotated_frame,
                            (detection.x, detection.y + detection.height + 5),
                            (detection.x + emotion_size[0] + 10, detection.y + detection.height + 25),
                            self._get_emotion_color(detection.emotion),
                            -1
                        )
                        
                        cv2.putText(
                            annotated_frame,
                            emotion_label,
                            (detection.x + 5, detection.y + detection.height + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            1
                        )
                else:
                    # For unknown faces, show all info in one label
                    main_label = " | ".join(label_parts)
                    label_size = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Draw label background
                    cv2.rectangle(
                        annotated_frame,
                        (detection.x, detection.y - label_size[1] - 15),
                        (detection.x + label_size[0] + 10, detection.y - 5),
                        color,
                        -1
                    )
                    
                    # Draw main label text
                    cv2.putText(
                        annotated_frame,
                        main_label,
                        (detection.x + 5, detection.y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
        
        return annotated_frame
    
    def _get_emotion_color(self, emotion: str) -> tuple:
        """Get color for emotion visualization"""
        colors = {
            'happy': (0, 255, 0), 'sad': (255, 0, 0), 'angry': (0, 0, 255),
            'surprise': (0, 255, 255), 'fear': (128, 0, 128),
            'disgust': (0, 128, 0), 'neutral': (128, 128, 128)
        }
        return colors.get(emotion.lower() if emotion else 'neutral', (255, 255, 255))
        
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        with self._lock:
            recognized_faces = [
                fid for fid in self.face_tracker.keys()
                if fid in self.recognition_cache and self.recognition_cache[fid][0][0] is not None
            ]
            return {
                'active_faces': len(self.face_tracker),
                'recognized_faces': len(recognized_faces),
                'total_faces_detected': self.next_face_id - 1,
                'face_ids': list(self.face_tracker.keys()),
                'deepface_available': self.emotion_recognizer.available if self.emotion_recognizer else False,
                'face_recognition_available': self.face_recognizer.available if self.face_recognizer else False,
                'face_usecase_available': self.face_usecase is not None,
                'models_loaded': {
                    'dnn_model': self.net is not None,
                    'emotion_recognizer': self.emotion_recognizer is not None,
                    'face_recognizer': self.face_recognizer is not None,
                    'face_usecase': self.face_usecase is not None
                }
            }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.background_executor:
            self.background_executor.shutdown(wait=True)
        self.face_tracker.clear()
        self.recognition_cache.clear()
        with self.detection_lock:
            self.stream_detections.clear()
        logger.info("FaceDetectionProcessor cleaned up")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()
    
    def _assess_face_quality(self, face_region: np.ndarray, face_detection: FaceDetection) -> dict:
        """
        Assess face quality before saving unknown person
        Returns quality metrics and decision
        """
        quality_score = 0.0
        quality_reasons = []
        
        # 1. Size check (face should be large enough)
        min_face_size = self.min_face_size  # pixels
        face_area = face_detection.width * face_detection.height
        if face_area < min_face_size * min_face_size:
            quality_reasons.append("face_too_small")
        else:
            quality_score += 0.25
        
        # 2. Aspect ratio check (avoid stretched faces)
        aspect_ratio = face_detection.width / face_detection.height
        if 0.7 <= aspect_ratio <= 1.3:  # Normal face proportions
            quality_score += 0.25
        else:
            quality_reasons.append("abnormal_aspect_ratio")
        
        # 3. Blur detection using Laplacian variance
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score > 100:  # Adjust threshold based on testing
            quality_score += 0.25
        else:
            quality_reasons.append("too_blurry")
        
        # 4. Confidence threshold
        if face_detection.confidence > 0.7:
            quality_score += 0.25
        else:
            quality_reasons.append("low_detection_confidence")
        
        return {
            'quality_score': quality_score,
            'is_good_quality': quality_score >= 0.75,  # Require 3/4 criteria
            'reasons': quality_reasons,
            'blur_score': blur_score,
            'face_area': face_area
        }