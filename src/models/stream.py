from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import Optional

class StreamStatus(Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    ACTIVE = "active"
    ERROR = "error"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"

@dataclass
class StreamInfo:
    """Information about a stream"""
    stream_id: str
    url: str
    name: str
    status: StreamStatus
    last_frame_time: float
    error_count: int
    reconnect_count: int
    fps: float = 0.0
    resolution: tuple = (0, 0)
    detect_emotion: bool = False
    
@dataclass
class FrameData:
    """Frame data with metadata"""
    stream_id: str
    frame: np.ndarray
    timestamp: float
    frame_number: int
    stream_info: StreamInfo
    processed_frame: Optional[np.ndarray] = None  # For display with annotations
    