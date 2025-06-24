# config/external.py
from dataclasses import dataclass, field
import os
from .base import BaseConfig

@dataclass
class MilvusIndexConfig(BaseConfig):
    name: str = "face_embedding"
    metric_type: str = "L2"
    index_type: str = "IVF_FLAT"
    params: dict = field(default_factory=lambda: {"nlist": 128})
    
    @classmethod
    def from_env(cls) -> 'MilvusIndexConfig':
        return cls(
            name=os.getenv("MILVUS_INDEX_NAME", "face_embedding"),
            metric_type=os.getenv("MILVUS_METRIC_TYPE", "L2"),
            index_type=os.getenv("MILVUS_INDEX_TYPE", "IVF_FLAT"),
            params=cls.get_env_dict("MILVUS_PARAMS", {"nlist": 128})
        )

@dataclass
class MilvusConfig(BaseConfig):
    """Milvus vector database configuration"""
    host: str = "localhost"
    port: int = 19530
    alias: str = "default"
    collection: str = "face_collection"
    timeout: int = 30
    embedding_dim: int = 128
    index: MilvusIndexConfig = field(default_factory=lambda: MilvusIndexConfig())
    
    @classmethod
    def from_env(cls) -> 'MilvusConfig':
        return cls(
            host=os.getenv('MILVUS_HOST', "localhost"),
            port=cls.get_env_int('MILVUS_PORT', 19530),
            alias=os.getenv('MILVUS_ALIAS', "default"),
            collection=os.getenv('MILVUS_COLLECTION', "face_collection"),
            timeout=cls.get_env_int('MILVUS_TIMEOUT', 30),
            embedding_dim=cls.get_env_int('MILVUS_EMBEDDING_DIM', 128),
            index=MilvusIndexConfig.from_env()
        )

@dataclass
class APIConfig(BaseConfig):
    """External API configuration"""
    emotion_service_url: str = ""
    emotion_service_timeout: int = 10
    ucode_admin_url: str = ""
    ucode_admin_timeout: int = 15
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    @classmethod
    def from_env(cls) -> 'APIConfig':
        return cls(
            emotion_service_url=os.getenv('EMOTION_SERVICE_URL', ""),
            emotion_service_timeout=cls.get_env_int('EMOTION_SERVICE_TIMEOUT', 10),
            ucode_admin_url=os.getenv('UCODE_ADMIN_URL', ""),
            ucode_admin_timeout=cls.get_env_int('UCODE_ADMIN_TIMEOUT', 15),
            retry_attempts=cls.get_env_int('API_RETRY_ATTEMPTS', 3),
            retry_delay=cls.get_env_float('API_RETRY_DELAY', 1.0)
        )

@dataclass
class ExternalConfig(BaseConfig):
    """External services configuration"""
    milvus: MilvusConfig = field(default_factory=lambda: MilvusConfig())
    api: APIConfig = field(default_factory=lambda: APIConfig())
    
    def __post_init__(self):
        if self.milvus is None:
            self.milvus = MilvusConfig.from_env()
        if self.api is None:
            self.api = APIConfig.from_env()
    
    @classmethod
    def from_env(cls) -> 'ExternalConfig':
        return cls(
            milvus=MilvusConfig.from_env(),
            api=APIConfig.from_env()
        )