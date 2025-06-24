# src/repositories/relational_db/camera_repository_impl.py (update existing)
from typing import List, Optional
from uuid import UUID
from sqlalchemy.orm import Session
from src.models.camera import Camera
from src.repositories.camera_repository import CameraRepository
from src.db.models import CameraModel
from src.models.filters import GetListFilter
from src.helpers.query_builder import apply_filters, apply_ordering, apply_pagination
import logging

logger = logging.getLogger(__name__)

class CameraRepositoryImpl(CameraRepository):
    def __init__(self, session: Session):
        self.session = session

    def get_camera(self, camera_id: UUID) -> Optional[Camera]:
        try:
            cam = self.session.query(CameraModel).filter_by(guid=camera_id).first()
            return self._to_domain(cam) if cam else None
        except Exception as e:
            logger.error(f"Error getting camera {camera_id}: {e}")
            raise

    def list_all(self) -> List[Camera]:
        try:
            cameras = self.session.query(CameraModel).all()
            return [self._to_domain(cam) for cam in cameras]
        except Exception as e:
            logger.error(f"Error listing all cameras: {e}")
            raise
    
    def list(self, filters: GetListFilter) -> List[Camera]:
        try:
            query = self.session.query(CameraModel)
            
            if filters.filters:
                query = apply_filters(query, CameraModel, filters.filters)
            
            if filters.order_by:
                query = apply_ordering(query, CameraModel, filters.order_by)
            
            query = apply_pagination(query, filters.page, filters.limit)
            results = query.all()
            
            return [self._to_domain(cam) for cam in results]
            
        except Exception as e:
            logger.error(f"Error listing cameras with filters: {e}")
            raise

    def create_camera(self, camera: Camera) -> Camera:
        try:
            camera_model = CameraModel(
                detect_emotion=camera.detect_emotion,
                detect_hands=camera.detect_hands,
                detect_uniform=camera.detect_uniform,
                detect_mask=camera.detect_mask,
                voice_detect=camera.voice_detect,
                branch_id=camera.branch_id,
                company_id=camera.company_id,
                port=camera.port,
                ip_address=camera.ip_address,
                password=camera.password,
                username=camera.username,
                mini_pc_id=getattr(camera, 'mini_pc_id', None)  # Add this line
            )
        
            self.session.add(camera_model)
            self.session.commit()
            self.session.refresh(camera_model)

            return self._to_domain(camera_model)
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating camera: {e}")
            raise
    
    def update_camera(self, camera: Camera) -> Camera:
        try:
            camera_model = self.session.query(CameraModel).filter_by(guid=camera.guid).first()
            if not camera_model:
                raise ValueError(f"Camera with ID {camera.guid} not found")
            
            # Update fields
            camera_model.detect_emotion = camera.detect_emotion
            camera_model.detect_hands = camera.detect_hands
            camera_model.detect_uniform = camera.detect_uniform
            camera_model.detect_mask = camera.detect_mask
            camera_model.voice_detect = camera.voice_detect
            camera_model.branch_id = camera.branch_id
            camera_model.company_id = camera.company_id
            camera_model.port = camera.port
            camera_model.ip_address = camera.ip_address
            camera_model.password = camera.password
            camera_model.username = camera.username
            if hasattr(camera, 'mini_pc_id'):
                camera_model.mini_pc_id = camera.mini_pc_id
            
            self.session.commit()
            self.session.refresh(camera_model)
            
            return self._to_domain(camera_model)
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating camera: {e}")
            raise
    
    def delete_camera(self, camera_id: UUID) -> bool:
        try:
            camera_model = self.session.query(CameraModel).filter_by(guid=camera_id).first()
            if not camera_model:
                return False
            
            self.session.delete(camera_model)
            self.session.commit()
            return True
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error deleting camera: {e}")
            raise

    def get_cameras_by_mini_pc(self, mini_pc_id: UUID) -> List[Camera]:
        """Get all cameras managed by a specific Mini PC"""
        try:
            cameras = self.session.query(CameraModel).filter_by(mini_pc_id=mini_pc_id).all()
            return [self._to_domain(cam) for cam in cameras]
        except Exception as e:
            logger.error(f"Error getting cameras for Mini PC {mini_pc_id}: {e}")
            raise
    
    def assign_camera_to_mini_pc(self, camera_id: UUID, mini_pc_id: UUID) -> Camera:
        """Assign a camera to a Mini PC"""
        try:
            camera_model = self.session.query(CameraModel).filter_by(guid=camera_id).first()
            if not camera_model:
                raise ValueError(f"Camera with ID {camera_id} not found")
            
            camera_model.mini_pc_id = mini_pc_id
            self.session.commit()
            self.session.refresh(camera_model)
            
            return self._to_domain(camera_model)
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error assigning camera {camera_id} to Mini PC {mini_pc_id}: {e}")
            raise
    
    def unassign_camera_from_mini_pc(self, camera_id: UUID) -> Camera:
        """Remove camera assignment from Mini PC"""
        try:
            camera_model = self.session.query(CameraModel).filter_by(guid=camera_id).first()
            if not camera_model:
                raise ValueError(f"Camera with ID {camera_id} not found")
            
            camera_model.mini_pc_id = None
            self.session.commit()
            self.session.refresh(camera_model)
            
            return self._to_domain(camera_model)
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error unassigning camera {camera_id}: {e}")
            raise

    def _to_domain(self, model: CameraModel) -> Camera:
        """Convert SQLAlchemy model to domain model"""
        camera = Camera(
            guid=model.guid,
            detect_emotion=model.detect_emotion,
            detect_hands=model.detect_hands,
            detect_uniform=model.detect_uniform,
            detect_mask=model.detect_mask,
            voice_detect=model.voice_detect,
            branch_id=model.branch_id,
            company_id=model.company_id,
            port=model.port,
            ip_address=model.ip_address,
            username=model.username,
            password=model.password
        )
        
        # Add mini_pc_id if it exists
        if hasattr(model, 'mini_pc_id') and model.mini_pc_id:
            setattr(camera, 'mini_pc_id', model.mini_pc_id)
            
        return camera