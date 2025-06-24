from dataclasses import dataclass
from typing import Optional
from uuid import UUID

@dataclass
class Camera:
    guid: Optional[UUID] = None
    mini_pc_id: Optional[UUID] = None
    rtsp_url: Optional[str] = None
    detect_emotion: bool = False
    detect_hands: bool = False
    detect_uniform: bool = False
    detect_mask: bool = False
    voice_detect: bool = False
    branch_id: Optional[int] = None
    company_id: Optional[int] = None
    port: Optional[int] = None
    ip_address: Optional[str] = None
    password: Optional[str] = None
    username: Optional[str] = None
    
    def generate_rtsp_url(self) -> str:
        """
        Generates an RTSP URL for a camera object.
        Handles optional authentication cleanly.
        """
        auth_part = ""
        if self.username and self.password:
            auth_part = f"{self.username}:{self.password}@"
        
        return (
            f"rtsp://{auth_part}{self.ip_address}:{self.port}{'/Streaming/Channels/102' if len(auth_part) > 2 else '/mystream'}"
        )
