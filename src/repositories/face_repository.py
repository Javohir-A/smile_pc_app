from abc import ABC, abstractmethod
from src.models.face_embedding import FaceEmbedding, SearchResult
from typing import Optional, Dict, List, Any


class FaceRepository(ABC):
    @abstractmethod
    def create(self, human_guid: str, name: str, human_type: str, face_embedding: List[float], 
               metadata: Optional[Dict[str, Any]] = None) -> FaceEmbedding:
        """Create a new face embedding record"""
        pass
    
    @abstractmethod
    def get_by_id(self, record_id: str) -> Optional[FaceEmbedding]:
        """Get a face embedding record by its ID"""
        pass
    
    @abstractmethod
    def get_by_human_guid(self, human_guid: str) -> List[FaceEmbedding]:
        """Get all face embedding records for a specific user"""
        pass
    
    @abstractmethod
    def update(self, record_id: str, name: Optional[str] = None, 
            human_type: Optional[str] = None,
            face_embedding: Optional[List[float]] = None,
            metadata: Optional[Dict[str, Any]] = None) -> Optional[FaceEmbedding]:
        """Update an existing face embedding record"""
        pass
    
    @abstractmethod
    def delete(self, record_id: str) -> bool:
        """Delete a face embedding record by ID. Returns True if deleted, False if not found"""
        pass
    
    @abstractmethod
    def list_all(self, limit: int = 100, offset: int = 0) -> List[FaceEmbedding]:
        """List all face embedding records with pagination"""
        pass
    
    @abstractmethod
    def search_similar(self, face_embedding: List[float], limit: int = 10, 
                      threshold: float = 0.8) -> List[SearchResult]:
        """Search for similar face embeddings using vector similarity"""
        pass
    
    @abstractmethod
    def exists(self, record_id: str) -> bool:
        """Check if a face embedding record exists"""
        pass
    
    @abstractmethod
    def count_by_human_guid(self, human_guid: str) -> int:
        """Count the number of face embeddings for a specific user"""
        pass
    
    @abstractmethod
    def get_embedding_by_id(self, record_id: str) -> Optional[List[float]]:
        """Get only the face embedding vector by record ID (for performance)"""
        pass
