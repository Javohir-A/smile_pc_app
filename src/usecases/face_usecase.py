from typing import List, Optional, Dict, Any
from src.repositories.face_repository import FaceRepository
from src.models.face_embedding import FaceEmbedding, SearchResult

class FaceUseCase:
    def __init__(self, face_repository: FaceRepository):
        self.face_repository = face_repository
    
    def create_face(self, human_guid: str, name: str, human_type, face_embedding: List[float], 
                   metadata: Optional[Dict[str, Any]] = None) -> FaceEmbedding:
        """Create a new face embedding"""
        try:
            return self.face_repository.create(human_guid, name, human_type, face_embedding, metadata)
        except ValueError as e:
            raise ValueError(f"Invalid face embedding: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to create face: {str(e)}")
    
    def get_face(self, face_id: str) -> Optional[FaceEmbedding]:
        """Get a face embedding by ID"""
        return self.face_repository.get_by_id(face_id)
    
    def get_faces_by_user(self, human_guid: str) -> List[FaceEmbedding]:
        """Get all face embeddings for a specific user"""
        return self.face_repository.get_by_human_guid(human_guid)
    
    def update_face(self, face_id: str, name: Optional[str] = None,
                human_type: Optional[str] = None, 
                face_embedding: Optional[List[float]] = None,
                metadata: Optional[Dict[str, Any]] = None) -> Optional[FaceEmbedding]:
        """Update a face embedding"""
        try:
            return self.face_repository.update(face_id, name, human_type, face_embedding, metadata)
        except ValueError as e:
            raise ValueError(f"Invalid face embedding: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to update face: {str(e)}")
    
    def delete_face(self, face_id: str) -> bool:
        """Delete a face embedding"""
        return self.face_repository.delete(face_id)
    
    def list_faces(self, limit: int = 100, offset: int = 0) -> List[FaceEmbedding]:
        """List all face embeddings with pagination"""
        return self.face_repository.list_all(limit, offset)
    
    def search_similar_faces(self, face_embedding: List[float], limit: int = 10, 
                           threshold: float = 0.8) -> List[SearchResult]:
        """Search for similar face embeddings"""
        try:
            return self.face_repository.search_similar(face_embedding, limit, threshold)
        except ValueError as e:
            raise ValueError(f"Invalid face embedding: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to search faces: {str(e)}")
    
    def count_user_faces(self, human_guid: str) -> int:
        """Count face embeddings for a specific user"""
        return self.face_repository.count_by_human_guid(human_guid)
    
    def get_face_embedding_only(self, face_id: str) -> Optional[List[float]]:
        """Get only the face embedding vector (for performance)"""
        return self.face_repository.get_embedding_by_id(face_id)
    
    def face_exists(self, face_id: str) -> bool:
        """Check if a face embedding exists"""
        return self.face_repository.exists(face_id)

