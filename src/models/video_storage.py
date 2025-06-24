# src/models/video_storage.py
from dataclasses import dataclass
from typing import Optional
from uuid import UUID
from datetime import datetime

@dataclass
class VideoRecord:
    """Video record for emotion-based video storage (no database needed)"""
    guid: Optional[UUID] = None
    human_id: UUID = None
    human_name: str = None
    emotion_type: str = None  # smile, upset, normal
    start_time: datetime = None
    end_time: datetime = None
    duration_seconds: float = None
    file_path: str = None
    file_url: Optional[str] = None  # URL after upload to ucode
    camera_id: str = None
    mini_pc_id: Optional[UUID] = None
    uploaded: bool = False
    created_at: Optional[datetime] = None