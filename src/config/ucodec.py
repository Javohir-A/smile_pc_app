from dataclasses import dataclass
from.base import BaseConfig
import os
from ucode_sdk.config import Config
from ucode_sdk.sdk import new   

@dataclass
class UcodeSdkConfig(BaseConfig):
    base_url: str = "https://api.client.u-code.io"
    app_id: str = ""
    file_upload_base_url: str = "https://cdn.u-code.io/ucode/"
    
    @classmethod
    def from_env(cls) -> 'UcodeSdkConfig':
        return cls(
            base_url=os.getenv("UCODE_BASE_URL", "https://api.client.u-code.io"),
            app_id=os.getenv("UCODE_APP_ID", ""),
            file_upload_base_url=os.getenv("FILE_UPLOAD_BASE_URL", "https://cdn.u-code.io/ucode/")
        )
