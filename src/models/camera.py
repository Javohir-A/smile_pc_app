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
    
    # def generate_rtsp_url(self) -> str:
    #     """
    #     Generates an RTSP URL for a camera object.
    #     Handles optional authentication cleanly.
    #     """
    #     auth_part = ""
    #     if self.username and self.password:
    #         auth_part = f"{self.username}:{self.password}@"
        
    #     return (
    #         f"rtsp://{auth_part}{self.ip_address}:{self.port}{'/Streaming/Channels/101' if len(auth_part) > 2 else '/mystream'}"
    #     )

    def generate_rtsp_url(self, protocol: str = "rtsp") -> str:
        """
        Generates a streaming URL for a camera using the given protocol.
        Supports RTSP, HTTP, HTTPS, RTMP, etc.

        Args:
            protocol (str): Protocol to use (e.g., 'rtsp', 'http', 'https', 'rtmp').

        Returns:
            str: Formatted streaming URL.
        """
        if not self.ip_address or not self.port:
            raise ValueError("Camera IP address and port must be set")

        auth_part = ""
        if self.username and self.password:
            auth_part = f"{self.username}:{self.password}@"

        # Example stream paths by protocol
        default_paths = {
            "rtsp": "/Streaming/Channels/102",
            "http": "/video",
            "https": "/secure-video",
            "rtmp": "/live/stream",
        }

        path = default_paths.get(protocol.lower(), "/mystream")

        return f"{protocol}://{auth_part}{self.ip_address}:{self.port}{path}"
