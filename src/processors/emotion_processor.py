# src/processors/emotion_processor.py - MEMORY QUEUE WITH RETRY LOGIC + CONFIDENCE AVERAGING

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Set, List
from dataclasses import dataclass, field
import threading
import requests
from uuid import UUID, uuid4
from queue import Queue, Empty
import numpy as np
from collections import deque

from src.processors.face_detection_processor import DetectionResult, FaceDetection
from src.processors.video_storage_processor import VideoStorageProcessor
from src.services.video_upload_service import VideoUploadService
from src.config import AppConfig

logger = logging.getLogger(__name__)

@dataclass
class ConfidenceReading:
    """Single confidence reading with timestamp"""
    confidence: float
    timestamp: datetime
    emotion: str

@dataclass
class EmotionTracker:
    """Enhanced emotion tracking for a person with confidence averaging"""
    human_id: UUID
    human_name: str
    human_type: str
    current_emotion: str
    start_time: datetime
    last_seen: datetime
    confidence: float  # Keep this for backward compatibility
    
    # NEW: Confidence tracking for averaging
    confidence_readings: deque = field(default_factory=lambda: deque(maxlen=100))  # Last 100 readings
    initial_confidence: float = 0.0
    latest_confidence: float = 0.0
    
    def add_confidence_reading(self, confidence: float, emotion: str, timestamp: datetime):
        """Add a new confidence reading"""
        safe_conf = safe_float(confidence)
        self.confidence_readings.append(ConfidenceReading(
            confidence=safe_conf,
            timestamp=timestamp,
            emotion=emotion
        ))
        self.latest_confidence = safe_conf
        self.confidence = safe_conf  # Update the old field for compatibility
        
        # Set initial confidence if this is the first reading
        if len(self.confidence_readings) == 1:
            self.initial_confidence = safe_conf
    
    def get_average_confidence(self) -> float:
        """Calculate average confidence over the emotion duration"""
        if not self.confidence_readings:
            return safe_float(self.confidence)  # Fallback to old confidence
        
        # Calculate weighted average - more recent readings have slightly more weight
        total_weight = 0.0
        weighted_sum = 0.0
        
        for i, reading in enumerate(self.confidence_readings):
            # Weight increases linearly (later readings get slightly more weight)
            weight = 1.0 + (i * 0.1)  # 1.0, 1.1, 1.2, etc.
            weighted_sum += reading.confidence * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else safe_float(self.confidence)
    
    def get_confidence_stats(self) -> Dict[str, float]:
        """Get detailed confidence statistics"""
        if not self.confidence_readings:
            # Fallback to old confidence system
            conf = safe_float(self.confidence)
            return {
                'average': conf,
                'initial': conf,
                'latest': conf,
                'min': conf,
                'max': conf,
                'readings_count': 1
            }
        
        confidences = [r.confidence for r in self.confidence_readings]
        
        return {
            'average': self.get_average_confidence(),
            'initial': self.initial_confidence,
            'latest': self.latest_confidence,
            'min': min(confidences),
            'max': max(confidences),
            'readings_count': len(confidences)
        }

@dataclass
class PendingEmotion:
    """Emotion waiting for video URL or timeout"""
    emotion_id: str
    human_id: UUID
    human_name: str
    human_type: str
    emotion_type: str
    confidence: float  # This will now be the AVERAGED confidence
    timestamp: datetime
    duration_minutes: float
    camera_id: str
    created_at: datetime
    video_url: Optional[str] = None
    sent_to_api: bool = False
    api_response_id: Optional[str] = None  # For retry updates
    timeout_at: datetime = None
    retry_count: int = 0
    
    # NEW: Additional confidence details (optional, for debugging/analytics)
    confidence_details: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.timeout_at is None:
            self.timeout_at = self.created_at + timedelta(seconds=60)

def safe_float(value) -> float:
    """Convert any numeric value to Python float for JSON serialization"""
    if value is None:
        return 0.0
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def safe_string(value) -> str:
    """Convert any value to safe string for JSON serialization"""
    if value is None:
        return ""
    return str(value)

class SimpleEmotionProcessor:
    """Memory-based emotion processor with queue, timeout, retry logic, and confidence averaging"""
    
    # Map emotions to 3 categories
    EMOTION_MAP = {
        'happy': 'smile',
        'joy': 'smile',
        'neutral': 'normal',
        'surprise': 'normal',
        'sad': 'upset',
        'angry': 'upset',
        'fear': 'upset',
        'disgust': 'upset'
    }
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.trackers: Dict[UUID, EmotionTracker] = {}
        self.lock = threading.Lock()
        
        # Configuration
        self.timeout_seconds = 3.0
        self.min_duration = 0.5
        self.video_wait_timeout = 30.0  # 30 seconds timeout for video
        self.max_retry_attempts = 3
        
        # NEW: Confidence tracking settings
        self.min_confidence_readings = 3  # Minimum readings for reliable average
        self.confidence_update_interval = 0.5  # Update confidence every 0.5 seconds
        
        # API settings
        self.api_url = getattr(config, 'api_base_url', 'https://tabassum.mini-tweet.uz/api/v1')
        
        # Memory-based emotion queue
        self.pending_emotions: Dict[str, PendingEmotion] = {}  # emotion_id -> PendingEmotion
        self.video_url_queue: Queue = Queue()  # For video URLs from upload service
        self.pending_lock = threading.Lock()
        
        # Video storage components
        self.video_processor = VideoStorageProcessor(config)
        self.upload_service = VideoUploadService(config)
        
        # Set up video upload callback
        self.video_processor.add_upload_callback(self._handle_video_ready)
        
        # Start upload service
        try:
            self.upload_service.start()
        except Exception as e:
            logger.error(f"Failed to start upload service: {e}")
        
        # Start background threads
        self._start_emotion_processor_thread()
        self._start_timeout_checker_thread()
        
        logger.info(f"🎥 Memory queue emotion processor with confidence averaging initialized")
        logger.info(f"API: {self.api_url}")
        logger.info(f"Video timeout: {self.video_wait_timeout}s")
        logger.info(f"Max retries: {self.max_retry_attempts}")
        logger.info(f"Confidence averaging: enabled with {self.min_confidence_readings} min readings")
    
    def _start_emotion_processor_thread(self):
        """Start background thread to process emotion queue"""
        def emotion_processor():
            logger.info("📋 Emotion queue processor started")
            while True:
                try:
                    # Process video URLs from upload service
                    try:
                        video_data = self.video_url_queue.get(timeout=1.0)
                        self._process_video_url(video_data)
                    except Empty:
                        pass
                    
                    # Process emotions ready to be sent
                    self._process_ready_emotions()
                    
                    time.sleep(0.5)  # Check every 500ms
                    
                except Exception as e:
                    logger.error(f"Error in emotion processor thread: {e}")
                    time.sleep(1)
        
        processor_thread = threading.Thread(target=emotion_processor, daemon=True)
        processor_thread.start()
    
    def _start_timeout_checker_thread(self):
        """Start background thread to check for timeouts"""
        def timeout_checker():
            logger.info("⏰ Emotion timeout checker started")
            while True:
                try:
                    current_time = datetime.now()
                    timeout_emotions = []
                    
                    with self.pending_lock:
                        for emotion_id, emotion in self.pending_emotions.items():
                            if not emotion.sent_to_api and current_time >= emotion.timeout_at:
                                timeout_emotions.append(emotion_id)
                    
                    # Send timeout emotions without video
                    for emotion_id in timeout_emotions:
                        with self.pending_lock:
                            if emotion_id in self.pending_emotions:
                                emotion = self.pending_emotions[emotion_id]
                                logger.warning(f"⏰ Timeout: Sending {emotion.human_name} emotion without video")
                                self._send_emotion_to_api(emotion)
                    
                    time.sleep(5)  # Check every 5 seconds
                    
                except Exception as e:
                    logger.error(f"Error in timeout checker: {e}")
                    time.sleep(5)
        
        timeout_thread = threading.Thread(target=timeout_checker, daemon=True)
        timeout_thread.start()
    
    def process_detections(self, detection_result: DetectionResult, frame=None):
        """Process detections and track emotions - MAIN ENTRY POINT"""
        current_time = datetime.now()
        
        # Process video storage if frame provided
        if frame is not None:
            try:
                recognized_faces = [f for f in detection_result.faces if f.is_recognized and f.human_guid]
                if recognized_faces:
                    self.video_processor.process_detection(detection_result, frame)
            except Exception as e:
                logger.error(f"Error processing video: {e}")
        
        # Get all currently detected people with emotions
        current_detected_people: Set[UUID] = set()
        with self.lock:
            # First pass: Process all detected faces
            for face in detection_result.faces:
                if face.is_recognized and face.human_guid and face.emotion:
                    try:
                        human_id = UUID(face.human_guid)
                        current_detected_people.add(human_id)
                        emotion_category = self._get_emotion_category(face.emotion)
                        
                        if human_id in self.trackers:
                            # Person was already being tracked
                            tracker = self.trackers[human_id]
                            
                            # Check if emotion changed
                            if tracker.current_emotion != emotion_category:
                                logger.info(f"🔄 {tracker.human_name}: {tracker.current_emotion} → {emotion_category}")
                                # Emotion changed - queue previous emotion and start new one
                                self._queue_emotion_for_processing(tracker, current_time, detection_result.stream_id)
                                self._start_new_emotion(human_id, face.human_name, face.human_type, emotion_category, current_time, face.emotion_confidence)
                            else:
                                # Same emotion - ADD CONFIDENCE READING and update last seen time
                                tracker.add_confidence_reading(
                                    face.emotion_confidence or 0.5,
                                    face.emotion,
                                    current_time
                                )
                                tracker.last_seen = current_time
                                
                                # Log confidence updates occasionally for debugging
                                if len(tracker.confidence_readings) % 20 == 0:  # Every 20 readings
                                    stats = tracker.get_confidence_stats()
                                    logger.debug(f"📊 {tracker.human_name} ({emotion_category}): "
                                               f"avg={stats['average']:.2f}, readings={stats['readings_count']}")
                        else:
                            # New person detected - start tracking
                            logger.info(f"👋 NEW: {face.human_name} started {emotion_category}")
                            self._start_new_emotion(human_id, face.human_name, face.human_type, emotion_category, current_time, face.emotion_confidence)
                            
                    except Exception as e:
                        logger.error(f"Error processing face {face.human_name}: {e}")
            
            # Second pass: Check for people who disappeared
            disappeared_people = []
            for human_id, tracker in self.trackers.items():
                if human_id not in current_detected_people:
                    time_since_seen = (current_time - tracker.last_seen).total_seconds()
                    if time_since_seen > self.timeout_seconds:
                        logger.info(f"👻 DISAPPEARED: {tracker.human_name} after {time_since_seen:.1f}s")
                        disappeared_people.append(human_id)
            
            # Queue emotions for disappeared people and remove them
            for human_id in disappeared_people:
                tracker = self.trackers[human_id]
                self._queue_emotion_for_processing(tracker, current_time, detection_result.stream_id)
                del self.trackers[human_id]
    
    def _queue_emotion_for_processing(self, tracker: EmotionTracker, end_time: datetime, camera_id: str):
        """Queue emotion for processing with AVERAGED confidence"""
        duration = (end_time - tracker.start_time).total_seconds()
        duration_minutes = duration / 60.0
        
        if duration < self.min_duration:
            logger.debug(f"⏭️ Skipping short emotion for {tracker.human_name}: {duration:.1f}s")
            return
        
        # GET CONFIDENCE STATISTICS - this is the key enhancement
        confidence_stats = tracker.get_confidence_stats()
        averaged_confidence = confidence_stats['average']
        
        # Log the confidence averaging result
        logger.info(f"📊 {tracker.human_name} ({tracker.current_emotion}): "
                   f"Duration {duration:.1f}s, "
                   f"Confidence: avg={averaged_confidence:.2f} "
                   f"(from {confidence_stats['readings_count']} readings, "
                   f"initial={confidence_stats['initial']:.2f}, "
                   f"final={confidence_stats['latest']:.2f})")
        
        # Create pending emotion with AVERAGED confidence
        emotion_id = str(uuid4())
        pending_emotion = PendingEmotion(
            emotion_id=emotion_id,
            human_id=tracker.human_id,
            human_name=safe_string(tracker.human_name),
            human_type=safe_string(tracker.human_type),
            emotion_type=safe_string(tracker.current_emotion),
            confidence=safe_float(averaged_confidence),  # USE AVERAGED CONFIDENCE HERE!
            timestamp=tracker.start_time,
            duration_minutes=safe_float(duration_minutes),
            camera_id=safe_string(camera_id),
            created_at=datetime.now(),
            confidence_details=confidence_stats  # Store detailed stats for debugging
        )
        
        with self.pending_lock:
            self.pending_emotions[emotion_id] = pending_emotion
        
        logger.info(f"📋 Queued emotion: {tracker.human_name} - {tracker.current_emotion} "
                   f"({duration:.1f}s, avg confidence: {averaged_confidence:.2f})")
    
    def _handle_video_ready(self, video_record):
        """Handle when a video is ready for upload"""
        logger.info(f"🎥 Video ready for upload: {video_record.human_name} - {video_record.emotion_type}")
        
        try:
            # Ensure upload service is running
            if not self.upload_service.running:
                self.upload_service.start()
            
            # Queue for upload
            self.upload_service.queue_upload(video_record)
            
            # Queue video data for processing
            video_data = {
                'human_id': video_record.human_id,
                'emotion_type': video_record.emotion_type,
                'video_record': video_record
            }
            self.video_url_queue.put(video_data)
            
        except Exception as e:
            logger.error(f"Error handling video ready: {e}")
    
    def _process_video_url(self, video_data):
        """Process video URL when upload completes"""
        try:
            video_record = video_data['video_record']
            human_id = video_data['human_id']
            emotion_type = video_data['emotion_type']
            
            # Wait for upload to complete (with timeout)
            max_wait = 60  # 60 seconds max wait for upload
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                if video_record.uploaded and video_record.file_url:
                    # Upload completed - assign URL to matching pending emotion
                    self._assign_video_url_to_emotion(human_id, emotion_type, video_record.file_url)
                    break
                time.sleep(1)
            
            if not video_record.uploaded:
                logger.warning(f"⏰ Video upload timeout for {human_id} - {emotion_type}")
                
        except Exception as e:
            logger.error(f"Error processing video URL: {e}")
    
    def _assign_video_url_to_emotion(self, human_id: UUID, emotion_type: str, video_url: str):
        """Assign video URL to matching pending emotion"""
        with self.pending_lock:
            # Find matching pending emotion (most recent for this person and emotion)
            matching_emotions = [
                emotion for emotion in self.pending_emotions.values()
                if emotion.human_id == human_id 
                and emotion.emotion_type == emotion_type 
                and not emotion.sent_to_api
                and emotion.video_url is None
            ]
            
            if matching_emotions:
                # Get most recent emotion
                latest_emotion = max(matching_emotions, key=lambda e: e.created_at)
                latest_emotion.video_url = video_url
                logger.info(f"📹 Assigned video URL to {latest_emotion.human_name} - {emotion_type}")
    
    def _process_ready_emotions(self):
        """Process emotions that are ready to be sent"""
        ready_emotions = []
        
        with self.pending_lock:
            for emotion_id, emotion in self.pending_emotions.items():
                # Send if: has video URL OR timeout reached OR no video expected
                if not emotion.sent_to_api:
                    if emotion.video_url or datetime.now() >= emotion.timeout_at:
                        ready_emotions.append(emotion_id)
        
        # Send ready emotions
        for emotion_id in ready_emotions:
            with self.pending_lock:
                if emotion_id in self.pending_emotions:
                    emotion = self.pending_emotions[emotion_id]
                    self._send_emotion_to_api(emotion)
    
    def _send_emotion_to_api(self, emotion: PendingEmotion):
        """Send emotion to API with FIXED JSON serialization and averaged confidence"""
        try:
            # emotion.confidence is now the AVERAGED confidence!
            emotion_data = {
                "human_id": safe_string(emotion.human_id),
                "human_type": safe_string(emotion.human_type),
                "emotion_type": safe_string(emotion.emotion_type),
                "confidence": safe_float(emotion.confidence),  # This is now averaged!
                "camera_id": safe_string(emotion.camera_id),
                "timestamp": emotion.timestamp.isoformat() + "Z",
                "duration_minutes": safe_float(emotion.duration_minutes)
            }
            
            # Add video URL if available
            if emotion.video_url:
                emotion_data["video_url"] = safe_string(emotion.video_url)
                logger.info(f"📹 Sending {emotion.human_name} emotion WITH video URL (avg confidence: {emotion.confidence:.2f})")
            else:
                logger.info(f"📹 Sending {emotion.human_name} emotion WITHOUT video URL (avg confidence: {emotion.confidence:.2f})")
            
            # Log confidence details for debugging
            if emotion.confidence_details:
                details = emotion.confidence_details
                logger.debug(f"📊 Confidence details: avg={details['average']:.2f}, "
                            f"initial={details['initial']:.2f}, final={details['latest']:.2f}, "
                            f"readings={details['readings_count']}")
            
            response = requests.post(
                f"{self.api_url}/emotions/detect",
                json=emotion_data,
                timeout=10.0,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code in [200, 201]:
                # Mark as sent
                emotion.sent_to_api = True
                
                # Extract response ID for potential updates
                try:
                    response_data = response.json()
                    emotion.api_response_id = response_data.get('id') or response_data.get('emotional_report_id')
                except:
                    pass
                
                video_info = " (with video)" if emotion.video_url else " (no video)"
                readings_info = ""
                if emotion.confidence_details:
                    readings_info = f" from {emotion.confidence_details['readings_count']} readings"
                
                logger.info(f"✅ SUCCESS: Sent {emotion.emotion_type} for {emotion.human_name}{video_info} "
                           f"(avg confidence: {emotion.confidence:.2f}{readings_info})")
                
                # If sent without video, set up retry mechanism
                if not emotion.video_url:
                    self._setup_retry_for_video_update(emotion)
                
            else:
                logger.error(f"❌ API ERROR {response.status_code}: {response.text}")
                emotion.retry_count += 1
                
                if emotion.retry_count < self.max_retry_attempts:
                    # Retry after delay
                    emotion.timeout_at = datetime.now() + timedelta(seconds=10)
                    logger.info(f"🔄 Will retry sending emotion for {emotion.human_name}")
                else:
                    emotion.sent_to_api = True  # Give up after max retries
                    logger.error(f"❌ Max retries reached for {emotion.human_name}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ NETWORK ERROR sending emotion: {e}")
            emotion.retry_count += 1
        except Exception as e:
            logger.error(f"❌ ERROR sending emotion: {e}")
            # Log more details for debugging
            logger.error(f"Emotion data: human_id={emotion.human_id}, type={emotion.emotion_type}, confidence={emotion.confidence}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            emotion.sent_to_api = True  # Mark as sent to avoid infinite loop
    
    def _setup_retry_for_video_update(self, emotion: PendingEmotion):
        """Setup retry mechanism to update emotion with video URL later"""
        def retry_video_update():
            # Wait additional time for video to be processed
            time.sleep(60)  # Wait 1 minute
            
            with self.pending_lock:
                if emotion.emotion_id in self.pending_emotions:
                    # Check if we now have a video URL
                    if not emotion.video_url:
                        # Try to find video URL for this emotion
                        self._try_find_video_for_emotion(emotion)
                    
                    if emotion.video_url and emotion.api_response_id:
                        # Update emotion with video URL via API
                        self._update_emotion_with_video(emotion)
        
        # Start retry in background
        threading.Thread(target=retry_video_update, daemon=True).start()
    
    def _try_find_video_for_emotion(self, emotion: PendingEmotion):
        """Try to find video URL for emotion from recent uploads"""
        # This could check recent uploads or video storage for matching videos
        # For now, just log that we're trying
        logger.info(f"🔍 Searching for video URL for {emotion.human_name} - {emotion.emotion_type}")
    
    def _update_emotion_with_video(self, emotion: PendingEmotion):
        """Update previously sent emotion with video URL"""
        try:
            update_data = {
                "video_url": safe_string(emotion.video_url)
            }
            
            # Use UPDATE API endpoint (you'll need to implement this)
            response = requests.put(
                f"{self.api_url}/emotions/{emotion.api_response_id}",
                json=update_data,
                timeout=10.0,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code in [200, 204]:
                logger.info(f"✅ Updated {emotion.human_name} emotion with video URL")
            else:
                logger.error(f"❌ Failed to update emotion with video: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error updating emotion with video: {e}")
    
    def _get_emotion_category(self, emotion: str) -> str:
        """Convert emotion to one of 3 categories"""
        return self.EMOTION_MAP.get(emotion.lower(), 'normal')
    
    def _start_new_emotion(self, human_id: UUID, human_name: str, human_type: str, emotion: str, current_time: datetime, confidence: float):
        """Start tracking new emotion with initial confidence reading"""
        tracker = EmotionTracker(
            human_id=human_id,
            human_type=safe_string(human_type),
            human_name=safe_string(human_name),
            current_emotion=safe_string(emotion),
            start_time=current_time,
            last_seen=current_time,
            confidence=safe_float(confidence)
        )
        
        # Add initial confidence reading
        tracker.add_confidence_reading(confidence or 0.5, emotion, current_time)
        
        self.trackers[human_id] = tracker
        logger.debug(f"▶️ Started tracking {emotion} for {human_name} (initial confidence: {confidence:.2f})")

    def get_stats(self) -> dict:
        """Get current statistics with confidence information"""
        with self.lock:
            active_emotions = {}
            total_confidence_readings = 0
            
            for tracker in self.trackers.values():
                duration = (datetime.now() - tracker.start_time).total_seconds()
                confidence_stats = tracker.get_confidence_stats()
                total_confidence_readings += confidence_stats['readings_count']
                
                active_emotions[tracker.human_name] = {
                    'emotion': tracker.current_emotion,
                    'duration': f"{duration:.1f}s",
                    'confidence': {
                        'current_avg': f"{confidence_stats['average']:.2f}",
                        'latest': f"{confidence_stats['latest']:.2f}",
                        'readings': confidence_stats['readings_count']
                    }
                }
        
        with self.pending_lock:
            pending_count = len(self.pending_emotions)
            sent_count = len([e for e in self.pending_emotions.values() if e.sent_to_api])
            with_video_count = len([e for e in self.pending_emotions.values() if e.video_url])
        
        return {
            'active_trackers': len(self.trackers),
            'total_confidence_readings': total_confidence_readings,
            'pending_emotions': pending_count,
            'sent_emotions': sent_count,
            'emotions_with_video': with_video_count,
            'active_emotions': active_emotions,
            'video_queue_size': self.video_url_queue.qsize(),
            'config': {
                'timeout_seconds': self.timeout_seconds,
                'video_wait_timeout': self.video_wait_timeout,
                'max_retries': self.max_retry_attempts,
                'min_confidence_readings': self.min_confidence_readings,
                'api_url': self.api_url
            }
        }
    
    def force_send_all(self):
        """Force send all pending emotions"""
        with self.pending_lock:
            unsent_emotions = [e for e in self.pending_emotions.values() if not e.sent_to_api]
            logger.info(f"🚨 FORCE SENDING {len(unsent_emotions)} pending emotions")
            
            for emotion in unsent_emotions:
                self._send_emotion_to_api(emotion)
        
        with self.lock:
            # Also send currently active emotions
            current_time = datetime.now()
            for tracker in list(self.trackers.values()):
                self._queue_emotion_for_processing(tracker, current_time, "force_send")
            self.trackers.clear()
    
    def cleanup(self):
        """Cleanup and send remaining data"""
        logger.info("🧹 Cleaning up memory queue emotion processor...")
        
        try:
            # Force send all pending emotions
            self.force_send_all()
            
            # Cleanup video processor
            if self.video_processor:
                self.video_processor.cleanup()
            
            # Stop upload service
            if self.upload_service:
                self.upload_service.stop()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        logger.info("Memory queue emotion processor cleaned up")