# src/services/video_upload_service.py - FIXED VERSION
import logging
import os
import threading
import time
from typing import List, Optional, Callable
from queue import Queue

from src.models.video_storage import VideoRecord

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

# Import ucode-sdk based on your example
try:
    from ucode_sdk.config import Config
    from ucode_sdk.sdk import new
    UCODE_AVAILABLE = True
except ImportError:
    UCODE_AVAILABLE = False
    logger.warning("ucode-sdk not available. Install with: pip install ucode-sdk")

class VideoUploadService:
    """Service for uploading videos to storage using ucode-sdk with upload callbacks"""
    
    def __init__(self, config):
        self.config = config    
        self.upload_queue = Queue()
        self.running = False
        self.upload_thread = None
        self.ucode_available = UCODE_AVAILABLE
        
        # Upload configuration
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
        # Upload completion callbacks - (success: bool, file_url: Optional[str])
        self.upload_callbacks: List[Callable] = []
        
        # ucode-sdk configuration
        self.app_id = getattr(config, 'ucode_app_id', "P-QlWnuJCdfy32dsQoIjXHQNScO7DR2TdL")
        self.base_url = getattr(config, 'ucode_base_url', "")
        self.base_url = "https://api.client.u-code.io"
        
        # Initialize SDK
        if self.ucode_available:
            try:
                ucode_config = Config(app_id=self.app_id, base_url=self.base_url)
                self.sdk = new(ucode_config)
                logger.info(f"🔗 ucode-sdk initialized: {self.base_url}")
            except Exception as e:
                logger.error(f"Failed to initialize ucode-sdk: {e}")
                self.ucode_available = False
        
        if not self.ucode_available:
            logger.warning("Video upload service disabled - ucode-sdk not available")
        else:
            logger.info("📤 Video upload service initialized")
    
    def add_upload_callback(self, callback: Callable):
        """Add callback for upload completion (success, file_url)"""
        self.upload_callbacks.append(callback)
    
    def start(self):
        """Start the upload service"""
        if not self.ucode_available:
            logger.warning("Cannot start video upload service - ucode-sdk not available")
            return
        
        self.running = True
        self.upload_thread = threading.Thread(target=self._upload_worker, daemon=True)
        self.upload_thread.start()
        logger.info("📤 Video upload service started")
    
    def stop(self):
        """Stop the upload service"""
        self.running = False
        if self.upload_thread:
            self.upload_thread.join(timeout=10)
        logger.info("📤 Video upload service stopped")
    
    def queue_upload(self, video_record: VideoRecord):
        """Queue a video for upload"""
        if not self.ucode_available:
            logger.warning(f"Cannot queue upload - ucode-sdk not available: {video_record.file_path}")
            return
        
        self.upload_queue.put(video_record)
        logger.info(f"📋 Queued video for upload: {video_record.human_name} - {video_record.emotion_type}")
    
    def _upload_worker(self):
        """Background worker thread for uploading videos"""
        logger.info("📤 Upload worker thread started")
        
        while self.running:
            try:
                # Get video from queue (with timeout to allow checking running flag)
                try:
                    video_record = self.upload_queue.get(timeout=1.0)
                except:
                    continue
                
                # Upload the video
                success, file_url = self._upload_video(video_record)
                
                if success:
                    logger.info(f"✅ Successfully uploaded: {video_record.human_name} - {video_record.emotion_type}")
                    logger.info(f"🔗 Video URL: {file_url}")
                    
                    # Update video record
                    video_record.uploaded = True
                    video_record.file_url = file_url
                    
                    # Notify callbacks
                    for callback in self.upload_callbacks:
                        try:
                            callback(True, file_url)
                        except Exception as e:
                            logger.error(f"Error in upload callback: {e}")
                    
                    self._cleanup_local_file(video_record.file_path)
                else:
                    logger.error(f"❌ Failed to upload: {video_record.human_name} - {video_record.emotion_type}")
                    
                    for callback in self.upload_callbacks:
                        try:
                            callback(False, None)
                        except Exception as e:
                            logger.error(f"Error in upload callback: {e}")
                
                self.upload_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in upload worker: {e}")
                time.sleep(1)
        
        logger.info("📤 Upload worker thread ended")
    
    def _upload_video(self, video_record: VideoRecord) -> tuple[bool, Optional[str]]:
        """Upload a single video using ucode-sdk"""
        if not os.path.exists(video_record.file_path):
            logger.error(f"Video file not found: {video_record.file_path}")
            return False, None
        
        retries = 0
        while retries < self.max_retries:
            try:
                logger.info(f"📤 Uploading video (attempt {retries + 1}): {video_record.file_path}")
                
                result = self.sdk.files().upload(video_record.file_path).exec()
                
                logger.debug(f"📦 Upload result: {result}")
                
                if result and len(result) >= 1:
                    create_response = result[0]  
                    
                    if hasattr(create_response, 'data') and create_response.data:
                        file_link = create_response.data.get('link')
                        if file_link:
                            # construct full URL for download
                            file_url = f"https://cdn.u-code.io/{file_link}"
                            
                            logger.info(f"✅ Upload successful!")
                            logger.info(f"📋 File info: {create_response.data.get('file_name_download', 'unknown')}")
                            logger.info(f"📏 File size: {create_response.data.get('file_size', 0)} bytes")
                            
                            return True, file_url
                
                logger.error(f"Upload failed - no valid response from ucode-sdk")
                    
            except Exception as e:
                logger.error(f"Upload attempt {retries + 1} failed: {e}")
                retries += 1
                
                if retries < self.max_retries:
                    logger.info(f"⏳ Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
        
        logger.error(f"❌ Upload failed after {self.max_retries} attempts: {video_record.file_path}")
        return False, None
    
    def _cleanup_local_file(self, file_path: str):
        """Clean up local video file after successful upload"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"🧹 Cleaned up local file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up file {file_path}: {e}")
    
    def get_queue_size(self) -> int:
        """Get current upload queue size"""
        return self.upload_queue.qsize()
    
    def get_stats(self) -> dict:
        """Get upload service statistics"""
        return {
            "available": self.ucode_available,
            "running": self.running,
            "queue_size": self.get_queue_size(),
            "max_retries": self.max_retries,
            "app_id": self.app_id,
            "base_url": self.base_url
        }