# src/config/video.py
from dataclasses import dataclass, field
from typing import List, Tuple
import os
from .base import BaseConfig

@dataclass
class VideoConfig(BaseConfig):
    """Video generation and upload configuration"""
    
    # Video generation settings
    enabled: bool = True
    trigger_emotions: List[str] = field(default_factory=lambda: ["upset", "smile", "normal"])  # Only these 3 emotions
    recognized_humans_only: bool = True  # Only record videos for recognized humans
    pre_event_seconds: int = 0  # No buffer - exact emotion duration only
    post_event_seconds: int = 0  # No buffer - exact emotion duration only  
    min_emotion_duration: float = 0.5  # Minimum emotion duration to trigger recording
    video_buffer_seconds: int = 10  # Keep 10 seconds of frames in buffer
    video_duration_seconds: int = 5  # Create 5-second clips
    
    video_wait_timeout: int = 15  # How long to wait for video before sending emotion
    
    # Video quality settings
    fps: int = 15
    resolution: Tuple[int, int] = (1280, 720)
    codec: str = "mp4v"
    video_format: str = "mp4"
    quality: int = 80  # 0-100 quality scale
    max_file_size_mb: int = 100  # Maximum video file size
    
    # Buffer settings
    buffer_size_seconds: int = 60  # Keep 60 seconds of frames in circular buffer
    max_concurrent_recordings: int = 5  # Max simultaneous recordings
    
    # Storage settings
    temp_video_dir: str = "/tmp/emotion_videos"
    cleanup_after_upload: bool = True
    keep_failed_uploads: bool = False
    
    # Ucode SDK settings
    ucode_app_id: str = ""
    ucode_base_url: str = "https://api.admin.u-code.io"
    upload_timeout: int = 300  # 5 minutes
    max_upload_retries: int = 3
    retry_delay: float = 5.0
    
    # Video metadata
    include_face_annotations: bool = True
    include_emotion_overlay: bool = True
    include_timestamp: bool = True
    blur_unknown_faces: bool = True
    
    # Performance settings
    encoding_threads: int = 2
    async_upload: bool = True
    compress_before_upload: bool = True
    
    def __post_init__(self):
        # Ensure temp directory exists
        os.makedirs(self.temp_video_dir, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'VideoConfig':
        # Parse trigger emotions from environment
        trigger_emotions = ["upset", "smile", "normal"]  # Only these 3 emotions for recognized humans
        emotions_env = os.getenv('VIDEO_TRIGGER_EMOTIONS', '')
        if emotions_env:
            trigger_emotions = [e.strip() for e in emotions_env.split(',') if e.strip()]
        
        # Parse resolution from environment
        resolution = (1280, 720)
        resolution_env = os.getenv('VIDEO_RESOLUTION', '1280,720')
        if resolution_env:
            try:
                width, height = map(int, resolution_env.split(','))
                resolution = (width, height)
            except ValueError:
                pass  # Use default
        
        return cls(
            enabled=cls.get_env_bool('VIDEO_GENERATION_ENABLED', True),
            trigger_emotions=trigger_emotions,
            recognized_humans_only=cls.get_env_bool('VIDEO_RECOGNIZED_HUMANS_ONLY', True),
            pre_event_seconds=0,  # Always 0 - exact emotion duration only
            post_event_seconds=0,  # Always 0 - exact emotion duration only
            min_emotion_duration=cls.get_env_float('VIDEO_MIN_EMOTION_DURATION', 0.5),
            
            fps=cls.get_env_int('VIDEO_FPS', 15),
            resolution=resolution,
            codec=os.getenv('VIDEO_CODEC', 'mp4v'),
            video_format=os.getenv('VIDEO_FORMAT', 'mp4'),
            quality=cls.get_env_int('VIDEO_QUALITY', 80),
            max_file_size_mb=cls.get_env_int('VIDEO_MAX_FILE_SIZE_MB', 100),
            
            buffer_size_seconds=cls.get_env_int('VIDEO_BUFFER_SIZE_SECONDS', 60),
            max_concurrent_recordings=cls.get_env_int('VIDEO_MAX_CONCURRENT_RECORDINGS', 5),
            
            temp_video_dir=os.getenv('VIDEO_TEMP_DIR', '/tmp/emotion_videos'),
            cleanup_after_upload=cls.get_env_bool('VIDEO_CLEANUP_AFTER_UPLOAD', True),
            keep_failed_uploads=cls.get_env_bool('VIDEO_KEEP_FAILED_UPLOADS', False),
            
            ucode_app_id=os.getenv('UCODE_APP_ID', 'P-QlWnuJCdfy32dsQoIjXHQNScO7DR2TdL'),
            ucode_base_url=os.getenv('UCODE_BASE_URL', 'https://api.admin.u-code.io'),
            upload_timeout=cls.get_env_int('VIDEO_UPLOAD_TIMEOUT', 300),
            max_upload_retries=cls.get_env_int('VIDEO_MAX_UPLOAD_RETRIES', 3),
            retry_delay=cls.get_env_float('VIDEO_RETRY_DELAY', 5.0),
            
            include_face_annotations=cls.get_env_bool('VIDEO_INCLUDE_FACE_ANNOTATIONS', True),
            include_emotion_overlay=cls.get_env_bool('VIDEO_INCLUDE_EMOTION_OVERLAY', True),
            include_timestamp=cls.get_env_bool('VIDEO_INCLUDE_TIMESTAMP', True),
            blur_unknown_faces=cls.get_env_bool('VIDEO_BLUR_UNKNOWN_FACES', True),
            
            encoding_threads=cls.get_env_int('VIDEO_ENCODING_THREADS', 2),
            async_upload=cls.get_env_bool('VIDEO_ASYNC_UPLOAD', True),
            compress_before_upload=cls.get_env_bool('VIDEO_COMPRESS_BEFORE_UPLOAD', True)
        )
    
    def validate(self) -> bool:
        """Validate video configuration"""
        errors = []
        
        if not self.ucode_app_id:
            errors.append("Ucode App ID is required for video upload")
        
        if not self.ucode_base_url:
            errors.append("Ucode Base URL is required for video upload")
        
        if self.pre_event_seconds < 0 or self.post_event_seconds < 0:
            errors.append("Pre/post event seconds must be positive")
        
        if self.fps <= 0 or self.fps > 60:
            errors.append("FPS must be between 1 and 60")
        
        if not self.trigger_emotions:
            errors.append("At least one trigger emotion must be specified")
        
        # Validate that only upset, smile, normal are used
        valid_emotions = ["upset", "smile", "normal"]
        invalid_emotions = [e for e in self.trigger_emotions if e.lower() not in valid_emotions]
        if invalid_emotions:
            errors.append(f"Invalid trigger emotions: {invalid_emotions}. Only {valid_emotions} are supported.")
        
        if not os.path.exists(self.temp_video_dir):
            try:
                os.makedirs(self.temp_video_dir, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create temp video directory: {e}")
        
        if errors:
            print("Video configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    

# Add to src/config/settings.py or create src/config/video.py

from dataclasses import dataclass
import os
from .base import BaseConfig

@dataclass
class VideoConfig(BaseConfig):
    """Video recording and upload configuration"""
    # Video storage settings
    enable_video_recording: bool = True
    video_storage_dir: str = "data/emotion_videos"
    video_buffer_seconds: int = 10  # Keep 10 seconds of frames in buffer
    video_duration_seconds: int = 5  # Create 5-second clips
    video_fps: int = 15
    video_quality: str = "medium"  # low, medium, high
    
    # Upload settings
    use_local_video_storage: bool = True  # Use local storage vs cloud
    video_upload_url: str = ""
    upload_timeout_seconds: int = 30
    max_upload_retries: int = 3
    
    # API integration
    api_base_url: str = "https://tabassum.mini-tweet.uz/api/v1"
    video_wait_timeout: int = 15  # How long to wait for video before sending emotion
    
    @classmethod
    def from_env(cls) -> 'VideoConfig':
        return cls(
            enable_video_recording=cls.get_env_bool('ENABLE_VIDEO_RECORDING', True),
            video_storage_dir=os.getenv('VIDEO_STORAGE_DIR', 'data/emotion_videos'),
            video_buffer_seconds=cls.get_env_int('VIDEO_BUFFER_SECONDS', 10),
            video_duration_seconds=cls.get_env_int('VIDEO_DURATION_SECONDS', 5),
            video_fps=cls.get_env_int('VIDEO_FPS', 15),
            video_quality=os.getenv('VIDEO_QUALITY', 'medium'),
            use_local_video_storage=cls.get_env_bool('USE_LOCAL_VIDEO_STORAGE', True),
            video_upload_url=os.getenv('VIDEO_UPLOAD_URL', ''),
            upload_timeout_seconds=cls.get_env_int('UPLOAD_TIMEOUT_SECONDS', 30),
            max_upload_retries=cls.get_env_int('MAX_UPLOAD_RETRIES', 3),
            api_base_url=os.getenv('API_BASE_URL', 'https://tabassum.mini-tweet.uz/api/v1'),
            video_wait_timeout=cls.get_env_int('VIDEO_WAIT_TIMEOUT', 15)
        )
