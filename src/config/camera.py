# config/camera.py
from dataclasses import dataclass, field
from typing import List
import os
from .database import BaseConfig

@dataclass
class StreamConfig(BaseConfig):
    """Individual stream configuration"""
    url: str
    name: str = ""
    enabled: bool = True
    priority: int = 1
    
    @classmethod
    def from_env(cls) -> 'StreamConfig':
        # This would typically be used for single stream setups
        return cls(
            url=os.getenv('STREAM_URL', ''),
            name=os.getenv('STREAM_NAME', 'default'),
            enabled=cls.get_env_bool('STREAM_ENABLED', True),
            priority=cls.get_env_int('STREAM_PRIORITY', 1)
        )

@dataclass
class CameraConfig(BaseConfig):
    """Camera and streaming configuration"""
    rtsp_urls: List[str] = field(default_factory=list)
    fallback_url: str = "rtsp://localhost:8554/mystream"
    max_reconnects: int = 5
    reconnect_delay: float = 2.0
    max_errors: int = 10
    camera_poll_interval: int = 3600  # seconds
    stream_timeout: int = 30
    buffer_size: int = 1000000
    detection_interval: float = 0
    discovery_interval_hours: float = 0.05  # How often to check for new cameras 0.05 3 minutes
    reduce_motion_blur = True
    show_frame_info = True
    force_color_conversion = True
    suppress_format_warnings = True
    
    
    def __post_init__(self):
        if not self.rtsp_urls:
            self.rtsp_urls = [
                "rtsp://localhost:8554/mystream"
                "rtsp://odmen:1qaz2wsx@10.0.34.193:554/Streaming/Channels/102?rtsp_transport=tcp",
                "rtsp://odmen:1qaz2wsx@10.0.12.162:554/Streaming/Channels/102?rtsp_transport=tcp",
                # "rtsp://admin:Abc12345@192.168.150.116:554/Streaming/Channels/102?rtsp_transport=tcp", #out of service 
                "rtsp://admin:abc12345@192.168.150.175:554/Streaming/Channels/102?rtsp_transport=tcp"
            ]
    
    @classmethod
    def from_env(cls) -> 'CameraConfig':
        # Parse multiple RTSP URLs from environment
        rtsp_urls = []
        rtsp_urls_env = os.getenv('RTSP_URLS', '')
        if rtsp_urls_env:
            rtsp_urls = [url.strip() for url in rtsp_urls_env.split(',') if url.strip()]
        
        return cls(
            rtsp_urls=rtsp_urls,
            fallback_url=os.getenv('FALLBACK_URL', cls.fallback_url),
            max_reconnects=cls.get_env_int('MAX_RECONNECTS', 5),
            reconnect_delay=cls.get_env_float('RECONNECT_DELAY', 2.0),
            max_errors=cls.get_env_int('MAX_ERRORS', 10),
            camera_poll_interval=cls.get_env_int('CAMERA_POLL_INTERVAL', 3600),
            stream_timeout=cls.get_env_int('STREAM_TIMEOUT', 30),
            buffer_size=cls.get_env_int('BUFFER_SIZE', 1000000),
            discovery_interval_hours=cls.get_env_float("CAMERA_DISCOVERY_INTERVAL_HOURS", 1)
        )
