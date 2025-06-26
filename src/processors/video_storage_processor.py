# src/processors/video_storage_processor.py - ASYNC VERSION (Non-blocking FFmpeg)
import cv2
import os
import logging
import threading
import time
import subprocess
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID, uuid4
from collections import deque
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import queue

from src.processors.face_detection_processor import DetectionResult, FaceDetection
from src.models.video_storage import VideoRecord
from src.config import AppConfig
from .emotion_recognizer import normalize_emotion

logger = logging.getLogger(__name__)

@dataclass
class PersonVideoSession:
    """Track video session for a person's emotion"""
    human_id: UUID
    human_name: str
    human_type: str
    emotion: str
    start_time: datetime
    frames: deque
    camera_id: str
    last_seen: datetime

class AsyncVideoStorageProcessor:
    """Async video processor that doesn't block real-time streaming"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.video_dir = "videos"
        self.temp_dir = "temp_videos"
        self.min_video_duration = 0.5  # Minimum 0.5 seconds
        self.max_video_duration = 600.0  # Maximum 10 minutes
        self.timeout_seconds = 5.0  # Person disappeared timeout
        
        # Active video sessions per person per emotion
        self.active_sessions: Dict[UUID, PersonVideoSession] = {}
        self.upload_callbacks: List[callable] = []
        
        self.lock = threading.Lock()
        
        # Ensure directories exist
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Check if ffmpeg is available for better compression
        self.ffmpeg_available = self._check_ffmpeg()
        
        # ASYNC: Video processing queue and background worker
        self.video_processing_queue = queue.Queue()
        self.video_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="video_processor")
        self.processing_workers_started = False
        
        logger.info(f"🎥 Async video processor initialized")
        logger.info(f"   Video dir: {self.video_dir}")
        logger.info(f"   FFmpeg available: {self.ffmpeg_available}")
        logger.info(f"   Background processing: Enabled")
        
        # Start background video processing workers
        self._start_background_workers()
    
    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available for video processing"""
        try:
            # Check FFmpeg availability
            result = subprocess.run(['ffmpeg', '-version'], 
                                capture_output=True, check=True, text=True)
            logger.info(f"FFmpeg detected: {result.stdout.split()[2]}")
            
            # Check for H.264 encoder specifically
            codecs_result = subprocess.run(['ffmpeg', '-encoders'], 
                                        capture_output=True, text=True)
            if 'libx264' in codecs_result.stdout:
                logger.info("✅ H.264 encoder (libx264) available")
                return True
            else:
                logger.warning("⚠️ H.264 encoder not available, falling back to OpenCV")
                return False
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("FFmpeg not found - using OpenCV only (may have browser compatibility issues)")
            return False

    
    def _start_background_workers(self):
        """Start background workers for video processing"""
        if self.processing_workers_started:
            return
            
        self.processing_workers_started = True
        
        def video_processing_worker():
            """Background worker that processes video conversion without blocking"""
            logger.info("🎬 Video processing worker started")
            
            while True:
                try:
                    # Get video session from queue (blocking with timeout)
                    try:
                        session_data = self.video_processing_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    
                    session, end_time = session_data
                    
                    # Process video in background without blocking main thread
                    try:
                        logger.info(f"🔄 Background processing video for {session.human_name} - {session.emotion}")
                        video_record = self._create_web_compatible_video_async(session, end_time)
                        
                        if video_record:
                            logger.info(f"✅ Background video completed: {session.human_name} - {session.emotion}")
                            
                            # Notify upload callbacks
                            for callback in self.upload_callbacks:
                                try:
                                    # Run callbacks in executor to avoid blocking
                                    self.video_executor.submit(callback, video_record)
                                except Exception as e:
                                    logger.error(f"Error in upload callback: {e}")
                        else:
                            logger.warning(f"❌ Background video processing failed for {session.human_name}")
                    
                    except Exception as e:
                        logger.error(f"Error in background video processing: {e}")
                    
                    finally:
                        self.video_processing_queue.task_done()
                
                except Exception as e:
                    logger.error(f"Error in video processing worker: {e}")
                    time.sleep(1)
        
        # Start worker threads
        for i in range(2):  # 2 background workers
            worker_thread = threading.Thread(
                target=video_processing_worker, 
                daemon=True, 
                name=f"video_worker_{i}"
            )
            worker_thread.start()
            
        logger.info("🎬 Background video processing workers started")
    
    def add_upload_callback(self, callback: callable):
        """Add callback for when videos are ready to upload"""
        self.upload_callbacks.append(callback)
        logger.info(f"Added video upload callback: {callback.__name__}")
    
    def process_detection(self, detection_result: DetectionResult, frame: any):
        """Process detection and manage video sessions - NON-BLOCKING"""
        current_time = datetime.now()
        
        with self.lock:
            # Only process if we have recognized faces
            recognized_faces = [f for f in detection_result.faces if f.is_recognized and f.human_guid]
            
            if not recognized_faces:
                # No recognized faces - check for session timeouts only
                self._check_session_timeouts(current_time, detection_result.stream_id)
                return
            
            # Track currently detected people
            current_people = set()
            
            for face in recognized_faces:
                human_id = UUID(face.human_guid)
                emotion = normalize_emotion(face.emotion)
                current_people.add(human_id)
                
                # Draw face annotations on frame copy (quick operation)
                annotated_frame = self._draw_face_annotation(frame.copy(), face)
                
                if human_id in self.active_sessions:
                    session = self.active_sessions[human_id]
                    
                    # Check if emotion changed
                    if session.emotion != emotion:
                        logger.info(f"🔄 {session.human_name}: {session.emotion} → {emotion}")
                        
                        # Queue current session for background processing (NON-BLOCKING)
                        self._queue_session_for_processing(session, current_time)
                        
                        # Start new session immediately
                        self._start_video_session(
                            human_id, face.human_name, face.human_type, 
                            emotion, current_time, detection_result.stream_id
                        )
                    else:
                        # Same emotion - update session
                        session.last_seen = current_time
                        session.frames.append({
                            'frame': annotated_frame,
                            'timestamp': current_time
                        })
                        
                        # Check max duration
                        duration = (current_time - session.start_time).total_seconds()
                        if duration >= self.max_video_duration:
                            logger.info(f"⏰ Max duration reached for {session.human_name} ({duration:.1f}s)")
                            self._queue_session_for_processing(session, current_time)
                            # Start new session
                            self._start_video_session(
                                human_id, face.human_name, face.human_type,
                                emotion, current_time, detection_result.stream_id
                            )
                else:
                    # New person detected - start video session
                    logger.info(f"👋 NEW: {face.human_name} started {emotion} - beginning video recording")
                    self._start_video_session(
                        human_id, face.human_name, face.human_type,
                        emotion, current_time, detection_result.stream_id
                    )
                
                # Add frame to current session (quick operation)
                if human_id in self.active_sessions:
                    self.active_sessions[human_id].frames.append({
                        'frame': annotated_frame,
                        'timestamp': current_time
                    })
            
            # Check for people who disappeared
            self._check_session_timeouts(current_time, detection_result.stream_id, current_people)
    
    def _queue_session_for_processing(self, session: PersonVideoSession, end_time: datetime):
        """Queue session for background processing - NON-BLOCKING"""
        duration = (end_time - session.start_time).total_seconds()
        
        if duration < self.min_video_duration:
            logger.debug(f"⏭️ Skipping short video for {session.human_name}: {duration:.1f}s")
            return
        
        if len(session.frames) < 1:
            logger.debug(f"⏭️ Not enough frames for {session.human_name}: {len(session.frames)}")
            return
        
        try:
            # Add to processing queue - this is very fast and non-blocking
            self.video_processing_queue.put((session, end_time), block=False)
            logger.info(f"📋 Queued video for background processing: {session.human_name} - {session.emotion} ({duration:.1f}s)")
        except queue.Full:
            logger.warning(f"⚠️ Video processing queue full - dropping video for {session.human_name}")
    
    def _draw_face_annotation(self, frame, face: FaceDetection):
        """Draw face detection annotations on frame - OPTIMIZED FOR SPEED"""
        # Use simpler drawing for performance
        color = (0, 255, 0) if face.is_recognized else (0, 0, 255)
        
        # Draw bounding box
        cv2.rectangle(frame, (face.x, face.y), 
                     (face.x + face.width, face.y + face.height), color, 2)
        
        # Simple label - name and emotion only
        if face.human_name and face.emotion:
            label = f"{face.human_name}: {face.emotion}"
            cv2.putText(frame, label, (face.x, face.y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame
    
    def _start_video_session(self, human_id: UUID, human_name: str, human_type: str,
                           emotion: str, start_time: datetime, camera_id: str):
        """Start a new video recording session"""
        session = PersonVideoSession(
            human_id=human_id,
            human_name=human_name,
            human_type=human_type,
            emotion=emotion,
            start_time=start_time,
            frames=deque(maxlen=900),  # 30 seconds at 30fps
            camera_id=camera_id,
            last_seen=start_time
        )
        
        self.active_sessions[human_id] = session
        logger.debug(f"▶️ Started video session: {human_name} - {emotion}")
    
    def _create_web_compatible_video_async(self, session: PersonVideoSession, end_time: datetime) -> Optional[VideoRecord]:
        """Create web browser compatible video file in background thread"""
        
        # Generate filename
        timestamp_str = session.start_time.strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in session.human_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_name}_{session.emotion}_{timestamp_str}_{session.camera_id}.mp4"
        
        if not session.frames:
            logger.error(f"No frames to write for {session.human_name}")
            return None
        
        if self.ffmpeg_available:
            return self._create_video_with_ffmpeg_async(session, end_time, filename)
        else:
            return self._create_video_with_opencv_async(session, end_time, filename)

    def _create_video_with_ffmpeg_async(self, session: PersonVideoSession, end_time: datetime, filename: str) -> Optional[VideoRecord]:
        """Enhanced FFmpeg video creation with better error handling"""
        
        temp_file = os.path.join(self.temp_dir, f"temp_{filename}")
        final_file = os.path.join(self.video_dir, filename)
        
        # Get frame dimensions
        first_frame = session.frames[0]['frame']
        height, width = first_frame.shape[:2]
        
        # Create temporary raw video with OpenCV using most compatible codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v for temp file
        fps = 15.0  # Reduced FPS for better compatibility
        
        temp_writer = cv2.VideoWriter(temp_file, fourcc, fps, (width, height))
         
        if not temp_writer.isOpened():
            logger.error(f"Failed to create temporary video writer")
            return None
        
        try:
            # Write frames to temporary file
            frames_written = 0
            for frame_data in session.frames:
                temp_writer.write(frame_data['frame'])
                frames_written += 1
            
            temp_writer.release()
            
            # Verify temp file was created
            if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
                logger.error(f"Temporary video file creation failed: {temp_file}")
                return None
            
            # Enhanced FFmpeg command with fallback options
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', temp_file,
                '-c:v', 'libx264',           # Try H.264 first
                '-preset', 'ultrafast',      # Fastest encoding
                '-crf', '28',                # Reasonable quality
                '-profile:v', 'baseline',    # Maximum compatibility
                '-level', '3.0',             # Web compatibility
                '-pix_fmt', 'yuv420p',       # Universal pixel format
                '-movflags', '+faststart',   # Progressive download
                '-r', str(fps),              # Frame rate
                '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
                '-fflags', '+genpts',        # Generate presentation timestamps
                '-y',                        # Overwrite output file
                final_file
            ]
            
            # Run FFmpeg with enhanced error handling
            logger.debug(f"🔧 Converting video with FFmpeg: {filename}")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                logger.warning(f"FFmpeg H.264 encoding failed, trying fallback: {result.stderr}")
                
                # Fallback: Try with different codec
                fallback_cmd = [
                    'ffmpeg',
                    '-i', temp_file,
                    '-c:v', 'mpeg4',            # Fallback to MPEG-4
                    '-preset', 'ultrafast',
                    '-qscale:v', '3',           # Good quality
                    '-r', str(fps),
                    '-y',
                    final_file
                ]
                
                result = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode != 0:
                    logger.error(f"FFmpeg fallback also failed: {result.stderr}")
                    return None
            
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            # Verify final file
            if not os.path.exists(final_file):
                logger.error(f"FFmpeg output file not created: {final_file}")
                return None
                
            file_size = os.path.getsize(final_file)
            if file_size == 0:
                logger.error(f"FFmpeg output file is empty: {final_file}")
                os.remove(final_file)
                return None
            
            duration = (end_time - session.start_time).total_seconds()
            video_record = VideoRecord(
                guid=uuid4(),
                human_id=session.human_id,
                human_name=session.human_name,
                emotion_type=session.emotion,
                start_time=session.start_time,
                end_time=end_time,
                duration_seconds=duration,
                file_path=final_file,
                camera_id=session.camera_id,
                uploaded=False,
                created_at=datetime.now()
            )
            
            logger.info(f"✅ Created video with FFmpeg: {filename} ({duration:.1f}s, {frames_written} frames, {file_size} bytes)")
            return video_record
            
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg encoding timeout for {filename}")
            return None
        except Exception as e:
            logger.error(f"Error in FFmpeg video creation: {e}")
            # Clean up temporary files
            for temp_path in [temp_file, final_file]:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            return None

    def _create_video_with_opencv_async(self, session: PersonVideoSession, end_time: datetime, filename: str) -> Optional[VideoRecord]:
        """Improved OpenCV video creation with better codec handling"""
        
        file_path = os.path.join(self.video_dir, filename)
        
        # Get frame dimensions
        first_frame = session.frames[0]['frame']
        height, width = first_frame.shape[:2]
        
        # Enhanced codec priority list for better compatibility
        codecs_to_try = [
            # Try basic MP4 codecs first (most compatible)
            ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # MPEG-4
            ('XVID', cv2.VideoWriter_fourcc(*'XVID')),  # Xvid
            # Try H.264 variants (if available)
            ('avc1', cv2.VideoWriter_fourcc(*'avc1')),  # H.264 variant 1
            ('H264', cv2.VideoWriter_fourcc(*'H264')),  # H.264 variant 2
            ('X264', cv2.VideoWriter_fourcc(*'X264')),  # x264
            # Fallback to basic codecs
            ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),  # Motion JPEG
        ]
        
        fps = 15.0  # Reduced FPS for better compatibility
        out = None
        successful_codec = None
        
        for codec_name, fourcc in codecs_to_try:
            try:
                out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
                if out.isOpened():
                    # Test write a frame to ensure codec actually works
                    test_frame = first_frame.copy()
                    out.write(test_frame)
                    successful_codec = codec_name
                    logger.info(f"✅ Using codec: {codec_name}")
                    break
                else:
                    if out:
                        out.release()
                    out = None
            except Exception as e:
                logger.debug(f"Codec {codec_name} failed: {e}")
                if out:
                    out.release()
                out = None
        
        if not out or not successful_codec:
            logger.error("❌ Failed to create video writer with any codec")
            return None
        
        try:
            # Reset video writer for actual recording
            out.release()
            fourcc = cv2.VideoWriter_fourcc(*successful_codec)
            out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.error(f"Failed to reinitialize video writer with {successful_codec}")
                return None
            
            # Write all frames
            frames_written = 0
            for frame_data in session.frames:
                try:
                    out.write(frame_data['frame'])
                    frames_written += 1
                except Exception as e:
                    logger.warning(f"Failed to write frame {frames_written}: {e}")
                    continue
            
            out.release()
            
            # Verify file was created and has content
            if not os.path.exists(file_path):
                logger.error(f"Video file was not created: {file_path}")
                return None
                
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logger.error(f"Video file is empty: {file_path}")
                os.remove(file_path)
                return None
            
            duration = (end_time - session.start_time).total_seconds()
            video_record = VideoRecord(
                guid=uuid4(),
                human_id=session.human_id,
                human_name=session.human_name,
                emotion_type=session.emotion,
                start_time=session.start_time,
                end_time=end_time,
                duration_seconds=duration,
                file_path=file_path,
                camera_id=session.camera_id,
                uploaded=False,
                created_at=datetime.now()
            )
            
            logger.info(f"✅ Created video with OpenCV: {filename} ({successful_codec}, {duration:.1f}s, {frames_written} frames, {file_size} bytes)")
            return video_record
            
        except Exception as e:
            logger.error(f"Error writing video frames with OpenCV: {e}")
            if out:
                out.release()
            # Clean up failed file
            if os.path.exists(file_path):
                os.remove(file_path)
            return None
        
    def _check_session_timeouts(self, current_time: datetime, camera_id: str, current_people: set = None):
        """Check for session timeouts and queue videos for processing"""
        if current_people is None:
            current_people = set()
        
        timeout_duration = timedelta(seconds=self.timeout_seconds)
        sessions_to_finish = []
        
        for human_id, session in self.active_sessions.items():
            if human_id not in current_people:
                time_since_seen = current_time - session.last_seen
                if time_since_seen >= timeout_duration:
                    logger.info(f"⏰ Timeout: {session.human_name} disappeared - queuing video")
                    sessions_to_finish.append(human_id)
        
        # Queue timed out sessions for processing
        for human_id in sessions_to_finish:
            session = self.active_sessions[human_id]
            self._queue_session_for_processing(session, current_time)
            del self.active_sessions[human_id]
    

    def cleanup(self):
        """Cleanup and finish all active sessions"""
        with self.lock:
            current_time = datetime.now()
            logger.info(f"🧹 Cleaning up {len(self.active_sessions)} active video sessions...")
            
            # Queue all active sessions for processing
            for session in list(self.active_sessions.values()):
                self._queue_session_for_processing(session, current_time)
            
            self.active_sessions.clear()
        
        # Wait for background processing to complete (with timeout)
        logger.info("⏳ Waiting for background video processing to complete...")
        try:
            # Wait for queue to be processed
            start_time = time.time()
            while not self.video_processing_queue.empty() and (time.time() - start_time) < 30:
                time.sleep(0.5)
            
            # Shutdown executor
            self.video_executor.shutdown(wait=True, timeout=10)
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        # Clean up temporary directory
        try:
            for temp_file in os.listdir(self.temp_dir):
                temp_path = os.path.join(self.temp_dir, temp_file)
                if os.path.isfile(temp_path):
                    os.remove(temp_path)
                    logger.debug(f"Removed temp file: {temp_file}")
        except Exception as e:
            logger.warning(f"Error cleaning temp directory: {e}")
        
        logger.info("Async video storage processor cleaned up")
    
    def get_stats(self) -> dict:
        """Get processor statistics"""
        with self.lock:
            total_frames = sum(len(session.frames) for session in self.active_sessions.values())
            active_sessions_info = {
                str(session.human_id): {
                    'name': session.human_name,
                    'emotion': session.emotion,
                    'duration': (datetime.now() - session.start_time).total_seconds(),
                    'frames': len(session.frames)
                }
                for session in self.active_sessions.values()
            }
            
            return {
                'active_sessions': len(self.active_sessions),
                'total_buffered_frames': total_frames,
                'sessions_info': active_sessions_info,
                'video_dir': self.video_dir,
                'ffmpeg_available': self.ffmpeg_available,
                'encoding_method': 'Async FFmpeg + H.264' if self.ffmpeg_available else 'Async OpenCV',
                'processing_queue_size': self.video_processing_queue.qsize(),
                'background_workers': 'Running' if self.processing_workers_started else 'Stopped'
            }

# Backward compatibility alias
VideoStorageProcessor = AsyncVideoStorageProcessor