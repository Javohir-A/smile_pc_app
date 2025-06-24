# config/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import json
from typing import Any, Dict

@dataclass
class BaseConfig(ABC):
    """Base configuration class with common functionality"""
    
    @classmethod
    @abstractmethod
    def from_env(cls) -> 'BaseConfig':
        """Create configuration from environment variables"""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return self.__dict__
    
    @staticmethod
    def get_env_bool(key: str, default: bool = False) -> bool:
        """Get boolean value from environment variable"""
        return os.getenv(key, str(default)).lower() in ('true', '1', 'yes', 'on')
    
    @staticmethod
    def get_env_int(key: str, default: int) -> int:
        """Get integer value from environment variable"""
        return int(os.getenv(key, str(default)))
    
    @staticmethod
    def get_env_float(key: str, default: float) -> float:
        """Get float value from environment variable"""
        return float(os.getenv(key, str(default)))
    
    @staticmethod
    def get_env_dict(key: str, default: dict) -> dict:
        """Get dictionary value from environment variable (JSON string)"""
        env_value = os.getenv(key)
        if env_value:
            try:
                return json.loads(env_value)
            except (json.JSONDecodeError, TypeError):
                pass
        return default