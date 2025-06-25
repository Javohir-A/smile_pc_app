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
from ucode_sdk import new, Config

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

    human_guid: Optional[str] = None
    human_name: Optional[str] = None
    human_type: Optional[str] = None
    is_recognized: bool = False
    face_embedding: Optional[List[float]] = None
    pose_info: Optional[dict] = None  # ADD THIS LINE
    
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
    pose_info: Optional[dict] = None  

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

        self.ucode_api = new(config=Config(app_id=config.ucode.app_id, base_url=config.ucode.base_url))
        
        self.unknown_person_manager = UnknownPersonManager(config)
        
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
        self.pose_process_every = 10
        self.pose_cache_ttl = 1.0

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
                # Ensure pose_info exists (for backward compatibility)
                if not hasattr(cached_data, 'pose_info') or cached_data.pose_info is None:
                    cached_data.pose_info = self._get_default_pose_info()
                return cached_data
        
        # Create new cache entry with pose_info
        new_cache_data = CachedFaceData(
            last_update=current_time,
            pose_info=self._get_default_pose_info()
        )
        self.face_cache[face_id] = new_cache_data
        return new_cache_data       
    
    def _update_face_cache_async(self, face_id: str, face_roi_expanded: np.ndarray, 
                            face_roi: np.ndarray, should_process_pose: bool = False):
        """Update face cache with recognition and emotion data in background"""
        def background_update():
            try:
                current_time = time.time()
                cached_data = self.face_cache.get(face_id)
                if not cached_data:
                    logger.debug(f"No cached data for face {face_id}")
                    return

                # Debug logging
                logger.debug(f"Processing face {face_id}: emotion={cached_data.emotion}, recognized={cached_data.is_recognized}")

                # Ensure pose_info exists
                if not hasattr(cached_data, 'pose_info') or cached_data.pose_info is None:
                    cached_data.pose_info = self._get_default_pose_info()

                # Face recognition (expensive operation) - only for unrecognized faces
                if (self.face_recognizer and self.face_recognizer.available and
                    self.face_usecase and not cached_data.is_recognized):
                    
                    logger.debug(f"Attempting recognition for face {face_id}")
                    face_embedding = self.face_recognizer.extract_face_embedding(face_roi_expanded, face_id)
                    if face_embedding:
                        logger.debug(f"Got embedding for face {face_id}, searching database")
                        search_results = self.face_usecase.search_similar_faces(
                            face_embedding, limit=1, threshold=0.6
                        )
                        if search_results and len(search_results) > 0:
                            best_match = search_results[0]
                            logger.debug(f"Found match for face {face_id}: {best_match.name} (confidence: {best_match.recognition_confidence})")
                            if best_match.recognition_confidence >= 0.86:
                                cached_data.human_guid = best_match.human_guid
                                cached_data.human_name = best_match.name
                                cached_data.human_type = best_match.human_type
                                cached_data.recognition_confidence = best_match.recognition_confidence
                                cached_data.is_recognized = True
                                logger.info(f"Background: Face {face_id} recognized as {best_match.name}")

                # Emotion recognition (expensive operation) - PROCESS FOR ALL FACES
                if self.emotion_recognizer and self.emotion_recognizer.available:
                    logger.debug(f"Processing emotion for face {face_id}")
                    emotion, emotion_conf, *_ = self.emotion_recognizer.predict_emotion(face_roi, face_id)
                    if emotion and emotion_conf > 0.3:
                        logger.debug(f"Updated emotion for face {face_id}: {emotion} ({emotion_conf})")
                        cached_data.emotion = emotion
                        cached_data.emotion_confidence = emotion_conf
                    else:
                        logger.debug(f"Low confidence emotion for face {face_id}: {emotion} ({emotion_conf})")

                cached_data.last_update = current_time
                cached_data.update_count += 1

            except Exception as e:
                logger.error(f"Background face update error for {face_id}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
            finally:
                # Remove from pending tasks
                self.pending_background_tasks.discard(face_id)

        # Only start background task if not already running for this face
        if face_id not in self.pending_background_tasks:
            self.pending_background_tasks.add(face_id)
            self.background_executor.submit(background_update) 
           
    def _extract_face_regions(self, frame: np.ndarray, detection: np.ndarray, w: int, h: int) -> tuple:
        """Extract face regions from detection - FAST"""
        x1 = max(0, int(detection[3] * w))
        y1 = max(0, int(detection[4] * h))
        x2 = min(w, int(detection[5] * w))
        y2 = min(h, int(detection[6] * h))
        
        width = x2 - x1
        height = y2 - y1
        
        # Skip tiny faces
        if width < 30 or height < 30:
            return None, None, None, None, None, None
        
        # Calculate expanded ROI
        expand_ratio = 0.2
        expand_x = int(width * expand_ratio / 2)
        expand_y = int(height * expand_ratio / 2)
        
        ex1 = max(0, x1 - expand_x)
        ey1 = max(0, y1 - expand_y)
        ex2 = min(w, x2 + expand_x)
        ey2 = min(h, y2 + expand_y)
        
        face_roi_expanded = frame[ey1:ey2, ex1:ex2]
        face_roi = frame[y1:y2, x1:x2]
        
        return x1, y1, width, height, face_roi_expanded, face_roi

    def _should_process_expensive_operations(self) -> tuple:
        """Determine if expensive operations should run this frame"""
        should_process_recognition = (self.frame_counter % self.recognition_process_every == 0)
        should_process_emotion = (self.frame_counter % self.emotion_process_every == 0)
        should_process_pose = (self.frame_counter % getattr(self, 'pose_process_every', 10) == 0)
        
        return should_process_recognition, should_process_emotion, should_process_pose

        
# Modify _process_single_detection to include unknown person processing:

    def _process_single_detection(self, frame: np.ndarray, detection: np.ndarray, 
                                w: int, h: int, should_process_recognition: bool,
                                should_process_emotion: bool, should_process_pose: bool) -> Optional[FaceDetection]:
        """Process a single face detection - OPTIMIZED"""
        confidence = detection[2]
        
        if confidence <= self.min_confidence:
            return None
        
        # Extract face regions
        result = self._extract_face_regions(frame, detection, w, h)
        if result[0] is None:  # Skip tiny faces
            return None
        
        x1, y1, width, height, face_roi_expanded, face_roi = result
        
        # Get face ID and cached data
        face_id = self._assign_face_id(x1, y1, width, height)
        cached_data = self._get_cached_face_data(face_id)
        
        # Background processing for expensive operations
        if (should_process_recognition or should_process_emotion or should_process_pose) and face_roi.size > 0:
            self._update_face_cache_async(face_id, face_roi_expanded, face_roi, 
                                        should_process_pose)
        
        # Create face detection object
        face_detection = self._create_face_detection_object(x1, y1, width, height, confidence, face_id, cached_data)
        
        # Process unknown faces (run this in background to avoid slowing down main thread)
        if not face_detection.is_recognized and should_process_pose:
            try:
                self.background_executor.submit(
                    self._process_unknown_face, face_roi, face_detection
                )
            except Exception:
                pass  # Don't let unknown face processing errors affect main pipeline
        
        return face_detection

    def process_frame(self, frame_data: FrameData) -> Optional[DetectionResult]:
        """Process a single frame - ULTRA-OPTIMIZED with pose detection"""
        if not self.net:
            return None
        
        start_time = time.time()
        self.frame_counter += 1
        self.log_counter += 1
        
        if self.frame_counter % 18000 == 0:  # ~5 minutes at 60fps
            try:
                self.background_executor.submit(self.unknown_person_manager.cleanup_old_first_seen)
            except Exception:
                pass
        
        try:
            frame = frame_data.frame
            h, w = frame.shape[:2]
            
            # Face detection
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
            self.net.setInput(blob)
            detections = self.net.forward()
            
            # Determine processing strategy
            should_process_recognition, should_process_emotion, should_process_pose = self._should_process_expensive_operations()
            should_log = (self.log_counter % self.log_every_n_frames == 0)
            
            # Process all detections
            face_detections = []
            for i, detection in enumerate(detections[0, 0, :, :]):
                face_detection = self._process_single_detection(
                    frame, detection, w, h, 
                    should_process_recognition, should_process_emotion, should_process_pose
                )
                if face_detection:
                    face_detections.append(face_detection)
            
            # Create and store result
            result = self._create_detection_result(frame_data, face_detections, start_time)
            self._store_and_notify(result, frame_data, face_detections, should_log)
            
            return result
            
        except Exception as e:
            if self.log_counter % 200 == 0:
                logger.error(f"Frame processing error: {e}")
            return None

    def _create_detection_result(self, frame_data: FrameData, face_detections: list, start_time: float) -> DetectionResult:
        """Create DetectionResult object"""
        processing_time = time.time() - start_time
        return DetectionResult(
            stream_id=frame_data.stream_id,
            timestamp=frame_data.timestamp,
            frame_number=frame_data.frame_number,
            faces=face_detections,
            processing_time=processing_time
        )

    def _store_and_notify(self, result: DetectionResult, frame_data: FrameData, 
                        face_detections: list, should_log: bool):
        """Store results and notify callbacks"""
        # Store detection results
        with self.detection_lock:
            self.stream_detections[frame_data.stream_id] = result
        
        # Background callbacks
        if face_detections:
            for callback in self.detection_callbacks:
                try:
                    self.background_executor.submit(callback, result, frame_data)
                except Exception:
                    pass
        
        # Minimal logging
        if face_detections and should_log:
            recognized_count = len([f for f in face_detections if f.is_recognized])
            logger.info(f"Stream {frame_data.stream_id}: {len(face_detections)} faces "
                    f"({recognized_count} recognized) in {result.processing_time:.3f}s")    

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
    
# Make sure this import is at the top of the file or in the draw_detections method:
    from .emotion_recognizer import normalize_emotion

    # In the draw_detections method, ensure emotion display works:
# Fix the draw_detections method - the logic for showing emotions on recognized faces is broken:

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
            
            # Prepare labels
            name_parts = []
            emotion_parts = []
            
            # Build name label
            if detection.is_recognized and detection.human_name:
                name_parts.append(f"{detection.human_name}")
                if detection.recognition_confidence > 0:
                    name_parts.append(f"({detection.recognition_confidence:.1%})")
            else:
                if detection.face_id:
                    name_parts.append(f"Unknown #{detection.face_id.split('_')[-1]}")
            
            # Build emotion label (separate from name)
            if show_emotions and detection.emotion:
                try:
                    # Try to import normalize_emotion, fallback if not available
                    try:
                        from .emotion_recognizer import normalize_emotion
                        emotion_text = f"{normalize_emotion(detection.emotion).title()}"
                    except ImportError:
                        emotion_text = f"{detection.emotion.title()}"
                    
                    if detection.emotion_confidence:
                        emotion_text += f" {detection.emotion_confidence:.0%}"
                    emotion_parts.append(emotion_text)
                except Exception as e:
                    # Fallback emotion display
                    emotion_text = f"{detection.emotion}"
                    if detection.emotion_confidence:
                        emotion_text += f" {detection.emotion_confidence:.0%}"
                    emotion_parts.append(emotion_text)
            
            # Draw labels
            if name_parts or emotion_parts:
                if detection.is_recognized:
                    # For recognized faces: name on top, emotion below
                    if name_parts:
                        name_label = " ".join(name_parts)
                        
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
                    
                    # Draw emotion label below the face (FIXED: Always show emotion if available)
                    if emotion_parts:
                        emotion_label = " ".join(emotion_parts)
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
                    # For unknown faces: show all info in one label
                    all_parts = name_parts + emotion_parts
                    main_label = " | ".join(all_parts)
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
        """Enhanced quality assessment including face pose"""
        quality_score = 0.0
        quality_reasons = []
        
        # 1. Size check
        min_face_size = 80
        face_area = face_detection.width * face_detection.height
        if face_area >= min_face_size * min_face_size:
            quality_score += 0.2
        else:
            quality_reasons.append("face_too_small")
        
        # 2. Aspect ratio check  
        aspect_ratio = face_detection.width / face_detection.height
        if 0.7 <= aspect_ratio <= 1.3:
            quality_score += 0.2
        else:
            quality_reasons.append("abnormal_aspect_ratio")
        
        # 3. Blur detection
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score > 100:
            quality_score += 0.2
        else:
            quality_reasons.append("too_blurry")
        
        # 4. Detection confidence
        if face_detection.confidence > 0.7:
            quality_score += 0.2
        else:
            quality_reasons.append("low_detection_confidence")
        
        # 5. NEW: Face pose assessment
        pose_info = self._detect_face_pose(face_region, face_detection)
        if pose_info['is_frontal']:
            quality_score += 0.2
            pose_bonus = pose_info['pose_quality'] * 0.1  # Up to 0.1 bonus for excellent pose
            quality_score += pose_bonus
        else:
            quality_reasons.append(f"face_not_frontal_yaw_{pose_info['yaw']:.1f}_pitch_{pose_info['pitch']:.1f}")
        
        return {
            'quality_score': quality_score,
            'is_good_quality': quality_score >= 0.8,  # Raised threshold due to pose requirement
            'reasons': quality_reasons,
            'blur_score': blur_score,
            'face_area': face_area,
            'pose_info': pose_info
        }

    def _detect_face_pose(self, face_region: np.ndarray, face_detection: FaceDetection) -> dict:
        """
        Detect face pose/angle to determine if face is suitable for recognition
        Returns pose information and quality assessment
        """
        pose_info = {
            'yaw': 0.0,      # Left-right rotation
            'pitch': 0.0,    # Up-down rotation  
            'roll': 0.0,     # Tilt rotation
            'is_frontal': False,
            'pose_quality': 0.0,
            'pose_score': 0.0
        }
        
        try:
            # Method 1: Using facial landmarks (recommended)
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
            
            # Detect facial landmarks
            landmarks = self._get_facial_landmarks(gray)
            
            if landmarks is not None:
                pose_info = self._calculate_pose_from_landmarks(landmarks, face_region.shape)
            
            # Method 2: Fallback - use simple geometric analysis
            else:
                pose_info = self._estimate_pose_geometric(face_region)
                
        except Exception as e:
            logger.warning(f"Face pose detection failed: {e}")
        
        return pose_info

    def _get_facial_landmarks(self, gray_face: np.ndarray) -> Optional[np.ndarray]:
        """Extract facial landmarks using dlib or MediaPipe"""
        try:
            # Option A: Using dlib (more accurate but requires model download)
            if hasattr(self, 'landmark_predictor'):
                faces = self.face_detector(gray_face)
                if len(faces) > 0:
                    landmarks = self.landmark_predictor(gray_face, faces[0])
                    return np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
            
            # Option B: Using MediaPipe (lighter, built-in)
            elif hasattr(self, 'mp_face_mesh'):
                results = self.mp_face_mesh.process(cv2.cvtColor(gray_face, cv2.COLOR_GRAY2RGB))
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    h, w = gray_face.shape
                    return np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark])
            
        except Exception as e:
            logger.debug(f"Landmark detection failed: {e}")
        
        return None

    def _calculate_pose_from_landmarks(self, landmarks: np.ndarray, face_shape: tuple) -> dict:
        """Calculate face pose angles from facial landmarks"""
        try:
            # Key landmark points for pose estimation
            # Using 68-point dlib model indices
            nose_tip = landmarks[30]           # Nose tip
            left_eye_corner = landmarks[36]    # Left eye outer corner  
            right_eye_corner = landmarks[45]   # Right eye outer corner
            left_mouth = landmarks[48]         # Left mouth corner
            right_mouth = landmarks[54]        # Right mouth corner
            chin = landmarks[8]                # Chin point
            
            # Calculate face center
            face_center_x = (left_eye_corner[0] + right_eye_corner[0]) / 2
            face_center_y = (left_eye_corner[1] + right_eye_corner[1]) / 2
            
            # Calculate yaw (left-right rotation)
            eye_distance = np.linalg.norm(right_eye_corner - left_eye_corner)
            nose_offset = nose_tip[0] - face_center_x
            yaw_ratio = nose_offset / (eye_distance / 2) if eye_distance > 0 else 0
            yaw_angle = np.arcsin(np.clip(yaw_ratio, -1, 1)) * 180 / np.pi
            
            # Calculate pitch (up-down rotation)
            nose_to_chin = np.linalg.norm(chin - nose_tip)
            expected_nose_chin_ratio = 1.2  # Typical ratio for frontal face
            pitch_ratio = nose_to_chin / (eye_distance * expected_nose_chin_ratio) if eye_distance > 0 else 1
            pitch_angle = (1 - pitch_ratio) * 30  # Approximate pitch in degrees
            
            # Calculate roll (tilt)
            eye_vector = right_eye_corner - left_eye_corner
            roll_angle = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi
            
            # Determine if face is frontal enough for recognition
            yaw_threshold = 25   # degrees
            pitch_threshold = 20 # degrees  
            roll_threshold = 15  # degrees
            
            is_frontal = (abs(yaw_angle) < yaw_threshold and 
                        abs(pitch_angle) < pitch_threshold and 
                        abs(roll_angle) < roll_threshold)
            
            # Calculate pose quality score (0-1)
            yaw_score = max(0, 1 - abs(yaw_angle) / yaw_threshold)
            pitch_score = max(0, 1 - abs(pitch_angle) / pitch_threshold)
            roll_score = max(0, 1 - abs(roll_angle) / roll_threshold)
            pose_quality = (yaw_score + pitch_score + roll_score) / 3
            
            return {
                'yaw': yaw_angle,
                'pitch': pitch_angle,
                'roll': roll_angle,
                'is_frontal': is_frontal,
                'pose_quality': pose_quality,
                'pose_score': pose_quality,
                'landmarks_count': len(landmarks)
            }
            
        except Exception as e:
            logger.warning(f"Pose calculation failed: {e}")
            return self._get_default_pose_info()

    def _estimate_pose_geometric(self, face_region: np.ndarray) -> dict:
        """
        Fallback method: estimate pose using simple geometric features
        Less accurate but doesn't require landmark detection
        """
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
            h, w = gray.shape
            
            # Detect eyes using Haar cascades (simpler fallback)
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(eyes) >= 2:
                # Sort eyes by x-coordinate (left to right)
                eyes = sorted(eyes, key=lambda x: x[0])
                left_eye = eyes[0]
                right_eye = eyes[1] if len(eyes) > 1 else eyes[0]
                
                # Calculate eye centers
                left_center = (left_eye[0] + left_eye[2]//2, left_eye[1] + left_eye[3]//2)
                right_center = (right_eye[0] + right_eye[2]//2, right_eye[1] + right_eye[3]//2)
                
                # Calculate roll angle
                eye_vector = (right_center[0] - left_center[0], right_center[1] - left_center[1])
                roll_angle = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi
                
                # Estimate yaw from eye size difference (rough approximation)
                left_eye_area = left_eye[2] * left_eye[3]
                right_eye_area = right_eye[2] * right_eye[3]
                eye_ratio = left_eye_area / right_eye_area if right_eye_area > 0 else 1
                
                # If one eye is much smaller, face might be turned
                yaw_angle = (1 - eye_ratio) * 30 if abs(1 - eye_ratio) > 0.2 else 0
                
                # Simple frontality check
                is_frontal = abs(roll_angle) < 15 and abs(yaw_angle) < 20
                pose_quality = 0.6 if is_frontal else 0.3  # Lower confidence for geometric method
                
                return {
                    'yaw': yaw_angle,
                    'pitch': 0.0,  # Can't estimate pitch reliably with this method
                    'roll': roll_angle,
                    'is_frontal': is_frontal,
                    'pose_quality': pose_quality,
                    'pose_score': pose_quality,
                    'method': 'geometric'
                }
        
        except Exception as e:
            logger.warning(f"Geometric pose estimation failed: {e}")
        
        return self._get_default_pose_info()

    def _get_default_pose_info(self) -> dict:
        """Return default pose info when detection fails"""
        return {
            'yaw': 0.0,
            'pitch': 0.0, 
            'roll': 0.0,
            'is_frontal': False,  # Conservative default
            'pose_quality': 0.0,
            'pose_score': 0.0,
            'method': 'default'
        }
    
    def _detect_face_pose_fast(self, face_roi: np.ndarray) -> dict:
        """Fast pose detection using simple geometric features"""
        try:
            if face_roi.size == 0:
                return self._get_default_pose_info()
            
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            h, w = gray.shape
            
            # Use simple eye detection for pose estimation
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 3, minSize=(10, 10))
            
            if len(eyes) >= 2:
                # Sort eyes by x-coordinate
                eyes = sorted(eyes, key=lambda x: x[0])
                left_eye = eyes[0]
                right_eye = eyes[1]
                
                # Calculate eye centers
                left_center = (left_eye[0] + left_eye[2]//2, left_eye[1] + left_eye[3]//2)
                right_center = (right_eye[0] + right_eye[2]//2, right_eye[1] + right_eye[3]//2)
                
                # Calculate angles
                eye_distance = np.sqrt((right_center[0] - left_center[0])**2 + 
                                    (right_center[1] - left_center[1])**2)
                
                # Roll angle
                roll_angle = np.arctan2(right_center[1] - left_center[1], 
                                    right_center[0] - left_center[0]) * 180 / np.pi
                
                # Rough yaw estimation from eye sizes
                left_area = left_eye[2] * left_eye[3]
                right_area = right_eye[2] * right_eye[3]
                area_ratio = left_area / right_area if right_area > 0 else 1
                yaw_angle = (1 - area_ratio) * 20 if abs(1 - area_ratio) > 0.3 else 0
                
                # Frontality check
                is_frontal = (abs(roll_angle) < 15 and abs(yaw_angle) < 25)
                pose_quality = 0.8 if is_frontal else 0.4
                
                return {
                    'yaw': yaw_angle,
                    'pitch': 0.0,
                    'roll': roll_angle,
                    'is_frontal': is_frontal,
                    'pose_quality': pose_quality,
                    'pose_score': pose_quality,
                    'method': 'fast_geometric'
                }
        
        except Exception:
            pass
        
        return self._get_default_pose_info()

    def _create_empty_cache_entry(self) -> dict:
        """Create empty cache entry with all fields"""
        return {
            'human_guid': None,
            'human_name': None,
            'human_type': None,
            'is_recognized': False,
            'recognition_confidence': 0.0,
            'emotion': None,
            'emotion_confidence': 0.0,
            'pose_info': self._get_default_pose_info(),
            'last_updated': 0,
            'recognition_updated': 0,
            'emotion_updated': 0,
            'pose_updated': 0
        }
        

    def _create_face_detection_object(self, x1: int, y1: int, width: int, height: int, 
                                    confidence: float, face_id: str, cached_data) -> FaceDetection:
        """Create FaceDetection object from processed data - FAST"""
        
        # Debug logging
        if cached_data.is_recognized:
            logger.debug(f"Creating recognized face detection for {face_id}: "
                        f"name={cached_data.human_name}, emotion={cached_data.emotion}, "
                        f"emotion_conf={cached_data.emotion_confidence}")
        
        return FaceDetection(
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
            face_embedding=None,  # Skip for performance
        )
        
 # Add this method to FaceDetectionProcessor:

    def _process_unknown_face(self, face_roi: np.ndarray, face_detection: FaceDetection) -> bool:
        """Process unknown face with quality control"""
        if face_detection.is_recognized:
            return False  # Not an unknown face
        
        # Assess quality including pose
        quality_assessment = self._assess_face_quality(face_roi, face_detection)
        
        # Check if we should capture this face
        if not self.unknown_person_manager.should_capture_unknown_face(
            face_detection.face_id, quality_assessment
        ):
            return False
        
        # If quality is good, save the face image
        if quality_assessment['is_good_quality']:
            self.unknown_person_manager.add_face_image(
                face_detection.face_id, face_roi, quality_assessment
            )
            logger.info(f"Captured high-quality unknown face {face_detection.face_id} "
                    f"(quality: {quality_assessment['quality_score']:.2f})")
        
        # Check if we're ready to save to database
        if self.unknown_person_manager.is_ready_for_database(face_detection.face_id):
            return self._save_unknown_person_to_database(face_detection.face_id)
        
        return False
    
    def _save_unknown_person_to_database(self, face_id: str) -> bool:
        """Save unknown person: best image -> UCode -> create human -> save face embedding"""
        import os, json, requests
        
        try:
            if face_id in self.unknown_person_manager.saved_faces:
                return True
            
            # Get best images
            best_images = self.unknown_person_manager.get_best_images(face_id, count=3)
            if not best_images:
                logger.warning(f"No images available for unknown face {face_id}")
                return False
            
            # Get the SINGLE best quality image (first one is highest quality)
            best_image_data = best_images[0]
            face_image = best_image_data['image']
            quality = best_image_data['quality']
            timestamp = best_image_data['timestamp']
            
            # Create temporary storage for the best image only
            temp_dir = "storage/temp_unknown"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save best image temporarily
            current_timestamp = int(time.time())
            filename = f"unknown_{face_id}_{current_timestamp}_quality_{quality['quality_score']:.2f}.jpg"
            temp_filepath = os.path.join(temp_dir, filename)
            
            cv2.imwrite(temp_filepath, face_image)
            logger.info(f"Saved best quality image: {filename}")
            
            # STEP 1: Upload to UCode
            image_url = "https://cdn.u-code.io/"
            try:
                uploadRes = self.ucode_api.files().upload(temp_filepath).exec()
                
                if uploadRes and len(uploadRes) >= 1:
                    create_response = uploadRes[0]  
                    
                if hasattr(create_response, 'data') and create_response.data:
                        file_link = create_response.data.get('link')
                        if file_link:
                            # construct full URL for download
                            image_url += file_link
                else:
                    return False
                    
                logger.info(f"Successfully uploaded to UCode: {image_url}")
                
            except Exception as e:
                logger.error(f"Exception during UCode upload for {face_id}: {e}")
                return False
            finally:
                # Clean up temp file
                try:
                    os.remove(temp_filepath)
                except:
                    pass
            
            # STEP 2: Create human in UCode
            human_guid = ""
            try:
                human_api_data = {
                    "full_name": f"Unknown_Client_{current_timestamp}",
                    # "company_id": "",  # Add if needed
                    # "branch_id": "",   # Add if needed
                    "photos": [image_url],
                    "type": ["client"]
                }
                
                resp = self.ucode_api.items("human").create(human_api_data).exec()
                if resp and len(resp) >= 1:
                    human_create_response = resp[0]
                
                human_guid = human_create_response.data_container.data["guid"]
                    
                logger.info(f"Successfully created human in UCode: {human_guid}")
                
            except Exception as e:
                logger.error(f"Exception during UCode human creation for {face_id}: {e}")
                return False
            
            # STEP 3: Save face embedding via external API
            try:

                face_api_data = {
                    "human_guid": human_guid,
                    "name": f"Unknown_Client_{current_timestamp}",
                    "human_type": "client",
                    "image_url": image_url,
                    "metadata": self._make_json_serializable({
                        "face_id": face_id,
                        "detection_timestamp": current_timestamp,
                        "original_timestamp": timestamp,
                        "quality_score": quality['quality_score'],
                        "blur_score": quality['blur_score'],
                        "face_area": quality['face_area'],
                        "pose_info": quality['pose_info'],
                        "is_frontal": quality['pose_info']['is_frontal'],
                        "reasons": quality['reasons']
                    })
                }
                # Get the external API URL from config
                external_api_url = getattr(self.config, 'EXTERNAL_FACE_API_URL', 'https://tabassum.mini-tweet.uz/api/v1/faces/from-url')
                
                response = requests.post(external_api_url, json=face_api_data, timeout=30)
                
                if response.status_code == 200 or response.status_code == 201:
                    logger.info(f"Successfully saved face embedding for {face_id} via external API")
                    response_data = response.json()
                    logger.info(f"External API response: {response_data}")
                else:
                    logger.error(f"External API failed for {face_id}: {response.status_code} - {response.text}")
                    return False
                    
            except Exception as e:
                logger.error(f"Exception during external API call for {face_id}: {e}")
                return False
            
            # STEP 4: Save metadata locally for tracking
            try:
                # Create storage directory for successful saves
                success_dir = "storage/unknown_faces_saved"
                os.makedirs(success_dir, exist_ok=True)
                
                success_metadata = {
                    'face_id': face_id,
                    'human_guid': human_guid,
                    'image_url': image_url,
                    'detection_time': current_timestamp,
                    'original_timestamp': timestamp,
                    'quality_score': quality['quality_score'],
                    'blur_score': quality['blur_score'],
                    'pose_info': quality['pose_info'],
                    'external_api_url': external_api_url,
                    'success': True
                }
                
                success_file = os.path.join(success_dir, f"success_{face_id}_{current_timestamp}.json")
                with open(success_file, 'w') as f:
                    json.dump(success_metadata, f, indent=2)
                    
            except Exception as e:
                logger.warning(f"Failed to save success metadata for {face_id}: {e}")
                # Don't fail the whole process for this
            
            # Mark as saved to prevent duplicate saves
            self.unknown_person_manager.saved_faces.add(face_id)
            logger.info(f"Successfully completed full pipeline for unknown person {face_id}")
            
            # Clean up after processing
            self.unknown_person_manager.cleanup_face(face_id)
            return True
            
        except Exception as e:
            logger.error(f"Error in save pipeline for unknown person {face_id}: {e}")
            return False
    
    def _make_json_serializable(self, obj):
            """Convert object to JSON-serializable format"""
            if isinstance(obj, dict):
                return {key: self._make_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [self._make_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.int32, np.int64, int)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64, float)):
                return float(obj)
            else:
                return str(obj)

class UnknownPersonManager:
    """Manages unknown face detection and quality assessment"""
    
    def __init__(self, config:AppConfig):
        self.unknown_faces_buffer = {}  # face_id -> list of quality assessments
        self.capture_attempts = {}      # face_id -> attempt count
        self.unknown_face_images = {}   # face_id -> list of face images
        self.face_first_seen = {}  # face_id -> timestamp
        self.recognition_grace_period = 3.0
        
        # Configuration
        self.max_attempts = getattr(config.detection, 'UNKNOWN_FACE_MAX_ATTEMPTS', 5)
        self.min_good_captures = getattr(config.detection, 'UNKNOWN_FACE_MIN_CAPTURES', 2)
        self.min_quality_score = getattr(config.detection, 'UNKNOWN_FACE_MIN_QUALITY', 0.75)
        self.require_frontal = getattr(config.detection, 'UNKNOWN_FACE_REQUIRE_FRONTAL', True)
        self.saved_faces = set()  
        
        logger.info(f"UnknownPersonManager initialized: max_attempts={self.max_attempts}, "
                   f"min_captures={self.min_good_captures}, min_quality={self.min_quality_score}")
    
    def should_capture_unknown_face(self, face_id: str, quality_assessment: dict) -> bool:
        """Decide whether to capture this unknown face"""
        current_time = time.time()
        
        # Track when this face was first seen
        if face_id not in self.face_first_seen:
            self.face_first_seen[face_id] = current_time
        
        # Give recognition system time to work (grace period)
        time_since_first_seen = current_time - self.face_first_seen[face_id]
        if time_since_first_seen < self.recognition_grace_period:
            logger.info(f"Face {face_id} still in recognition grace period "
                        f"({time_since_first_seen:.1f}s < {self.recognition_grace_period}s)")
            return False
        
        #delete from first time seen to free memory after human is out of first time seen grace period
        # if face_id in self.face_first_seen:
        #     del self.face_first_seen[face_id]
        
        # Initialize tracking for new face
        if face_id not in self.unknown_faces_buffer:
            self.unknown_faces_buffer[face_id] = []
            self.capture_attempts[face_id] = 0
            self.unknown_face_images[face_id] = []
        
        # Check if we've exceeded max attempts
        if self.capture_attempts[face_id] >= self.max_attempts:
            return False
        
        # Count good quality captures so far
        good_captures = len([q for q in self.unknown_faces_buffer[face_id] if q['is_good_quality']])
        
        # If we have enough good captures, stop
        if good_captures >= self.min_good_captures:
            return False
        
        # Check pose requirement
        pose_info = quality_assessment.get('pose_info', {})
        if self.require_frontal and not pose_info.get('is_frontal', False):
            logger.debug(f"Skipping unknown face {face_id} - not frontal "
                        f"(yaw: {pose_info.get('yaw', 0):.1f}°)")
            return False
        
        # Increment attempt counter
        self.capture_attempts[face_id] += 1
        self.unknown_faces_buffer[face_id].append(quality_assessment)
        
        # Capture if quality is good
        return quality_assessment['is_good_quality']
    
    def add_face_image(self, face_id: str, face_image: np.ndarray, quality_assessment: dict):
        """Store a high-quality face image"""
        if face_id in self.unknown_face_images:
            self.unknown_face_images[face_id].append({
                'image': face_image.copy(),
                'quality': quality_assessment,
                'timestamp': time.time()
            })
            logger.debug(f"Stored face image for {face_id} "
                        f"(quality: {quality_assessment['quality_score']:.2f})")
    
    # def is_ready_for_database(self, face_id: str) -> bool:
    #     """Check if we have enough good quality captures to save to database"""
    #     if face_id not in self.unknown_faces_buffer:
    #         return False
        
    #     good_captures = len([q for q in self.unknown_faces_buffer[face_id] if q['is_good_quality']])
    #     return good_captures >= self.min_good_captures
    
    def is_ready_for_database(self, face_id: str) -> bool:
        """Check if we have enough good quality captures to save to database"""
        if face_id in self.saved_faces:
            return False  # Already saved this face
            
        if face_id not in self.unknown_faces_buffer:
            return False
    
        good_captures = len([q for q in self.unknown_faces_buffer[face_id] if q['is_good_quality']])
        return good_captures >= self.min_good_captures
    
    def get_best_images(self, face_id: str, count: int = 3) -> List[dict]:
        """Get the best quality images for a face"""
        if face_id not in self.unknown_face_images:
            return []
        
        # Sort by quality score
        images = self.unknown_face_images[face_id]
        sorted_images = sorted(images, key=lambda x: x['quality']['quality_score'], reverse=True)
        return sorted_images[:count]
    
    def cleanup_face(self, face_id: str):
        """Clean up data for a face that's been processed"""
        self.unknown_faces_buffer.pop(face_id, None)
        self.capture_attempts.pop(face_id, None)
        self.unknown_face_images.pop(face_id, None)
        logger.debug(f"Cleaned up unknown face data for {face_id}")
        
    def cleanup_old_first_seen(self):
        """Clean up old first_seen entries to prevent memory leaks"""
        current_time = time.time()
        to_remove = []
        
        for face_id, first_seen_time in self.face_first_seen.items():
            if current_time - first_seen_time > (self.recognition_grace_period + 30):  # Extra buffer
                to_remove.append(face_id)
        
        for face_id in to_remove:
            del self.face_first_seen[face_id]
            logger.debug(f"Cleaned up old first_seen entry for {face_id}")
