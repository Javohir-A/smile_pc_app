# src/models/face_embedding.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class FaceEmbedding:
    # Fields WITHOUT default values must come first
    human_guid: str
    name: str
    human_type: str
    face_embedding: List[float]
    
    # Fields WITH default values must come after
    id: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create_new(cls, human_guid: str, name: str, human_type: str, 
                   face_embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> 'FaceEmbedding':
        """Create a new FaceEmbedding instance with generated timestamp"""
        return cls(
            human_guid=human_guid,
            name=name,
            human_type=human_type,
            face_embedding=face_embedding,
            metadata=metadata or {}
        )
@dataclass
class SearchResult:
    id: str
    human_guid: str
    name: str
    similarity_score: float
    metadata: Optional[Dict[str, Any]] = None
    distance: float = None
    recognition_confidence: float = None
    human_type: str = None
    
    @classmethod
    def from_face_embedding(cls, face_embedding: FaceEmbedding, 
                           similarity_score: float) -> 'SearchResult':
        """Create SearchResult from FaceEmbedding"""
        return cls(
            id=face_embedding.id,
            human_guid=face_embedding.human_guid,
            name=face_embedding.name,
            similarity_score=similarity_score,
            metadata=face_embedding.metadata,
            human_type=face_embedding.human_type
        )
