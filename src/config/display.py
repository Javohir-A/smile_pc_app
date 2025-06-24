# config/display.py
from dataclasses import dataclass
from .database import BaseConfig

@dataclass
class DisplayConfig(BaseConfig):
    """Display and GUI configuration"""
    display_width: int = 1920
    display_height: int = 1080
    enable_gui: bool = True
    show_fps: bool = True
    show_detection_boxes: bool = True
    show_emotion_labels: bool = True
    fullscreen: bool = False
    reduce_motion_blur: bool = True
    show_frame_info: bool = True
    show_emotions: bool = True
    @classmethod
    def from_env(cls) -> 'DisplayConfig':
        return cls(
            display_width=cls.get_env_int('DISPLAY_WIDTH', 1920),
            display_height=cls.get_env_int('DISPLAY_HEIGHT', 1080),
            enable_gui=cls.get_env_bool('ENABLE_GUI', False),
            show_fps=cls.get_env_bool('SHOW_FPS', True),
            show_detection_boxes=cls.get_env_bool('SHOW_DETECTION_BOXES', True),
            show_emotion_labels=cls.get_env_bool('SHOW_EMOTION_LABELS', True),
            fullscreen=cls.get_env_bool('FULLSCREEN', False)
        )
