# src/usecases/camera_usecase.py (update existing)
import logging
from typing import List, Optional
from uuid import UUID
from src.repositories.camera_repository import CameraRepository
from src.models.camera import Camera
from src.models.filters import GetListFilter, Filter

logger = logging.getLogger(__name__)

class CameraUseCase:
    def __init__(self, camera_repository: CameraRepository):
        self.camera_repo = camera_repository
    
    def get_camera(self, camera_id: UUID) -> Optional[Camera]:
        """Get camera by ID"""
        try:
            return self.camera_repo.get_camera(camera_id)
        except Exception as e:
            logger.error(f"Error getting camera: {e}")
            raise
    
    def list_cameras(self, filters: GetListFilter) -> List[Camera]:
        """List cameras with filters"""
        try:
            return self.camera_repo.list(filters)
        except Exception as e:
            logger.error(f"Error listing cameras: {e}")
            raise
    
    def create_camera(self, camera: Camera) -> Camera:
        """Create a new camera"""
        try:
            return self.camera_repo.create_camera(camera)
        except Exception as e:
            logger.error(f"Error creating camera: {e}")
            raise
    
    def update_camera(self, camera: Camera) -> Camera:
        """Update camera"""
        try:
            return self.camera_repo.update_camera(camera)
        except Exception as e:
            logger.error(f"Error updating camera: {e}")
            raise
    
    def delete_camera(self, camera_id: UUID) -> bool:
        """Delete camera"""
        try:
            return self.camera_repo.delete_camera(camera_id)
        except Exception as e:
            logger.error(f"Error deleting camera: {e}")
            raise
    
    def get_cameras_by_mini_pc(self, mini_pc_id: UUID) -> List[Camera]:
        """Get cameras managed by a specific Mini PC"""
        try:
            return self.camera_repo.list(GetListFilter(filters=[Filter(column="mini_pc_id", type="eq", value=mini_pc_id)]))
        except Exception as e:
            logger.error(f"Error getting cameras by Mini PC: {e}")
            raise

    def assign_camera_to_mini_pc(self, camera_id:UUID, mini_pc_id:UUID) -> Camera:
        try:
            return self.camera_repo.update_camera(Camera(guid=camera_id, mini_pc_id=mini_pc_id))
        except Exception as e:
            logger.error(f"Error assining mini pc to camera: {e}")
            raise
        
    def unassign_camera_from_mini_pc(self, camera_id:UUID) -> Camera:
        try:
            return self.camera_repo.update_camera(Camera(camera_id=camera_id, mini_pc_id=UUID()))
        except Exception as e:
            logger.error(f"Error unassigning camera from mini PC: {e}")
            raise