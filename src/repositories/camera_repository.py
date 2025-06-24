# src/repositories/camera_repository.py (update existing)
from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID
from src.models.camera import Camera
from src.models.filters import GetListFilter

class CameraRepository(ABC):
    @abstractmethod
    def get_camera(self, camera_id: UUID) -> Optional[Camera]:
        pass

    @abstractmethod
    def list_all(self) -> List[Camera]:
        pass
    
    @abstractmethod
    def list(self, filters: GetListFilter) -> List[Camera]:
        pass
    
    @abstractmethod
    def create_camera(self, camera: Camera) -> Camera:
        pass
    
    @abstractmethod
    def update_camera(self, camera: Camera) -> Camera:
        pass
    
    @abstractmethod
    def delete_camera(self, camera_id: UUID) -> bool:
        pass
    
    @abstractmethod
    def assign_camera_to_mini_pc(self, camera_id:UUID, mini_pc_id:UUID) -> Camera:
        pass
    
    @abstractmethod
    def unassign_camera_from_mini_pc(self, camera_id:UUID) -> Camera:
        pass