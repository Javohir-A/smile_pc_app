# src/usecases/mini_pc_usecase.py
import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from src.repositories.mini_pc_repository import MiniPCRepository
from src.usecases.camer_usecase import CameraUseCase
from src.models.mini_pc import MiniPC
from src.models.camera import Camera
from src.models.filters import GetListFilter, Filter

logger = logging.getLogger(__name__)

class MiniPCUseCase:
    def __init__(self, mini_pc_repository: MiniPCRepository, camera_usecase: CameraUseCase):
        self.mini_pc_repo = mini_pc_repository
        self.camera_usecase = camera_usecase
    
    def create_mini_pc(self, mini_pc: MiniPC) -> MiniPC:
        """Create a new Mini PC"""
        try:
            return self.mini_pc_repo.create(mini_pc)
        except Exception as e:
            logger.error(f"Error creating Mini PC: {e}")
            raise
    
    def get_mini_pc_by_id(self, mini_pc_id: UUID) -> Optional[MiniPC]:
        """Get Mini PC by ID"""
        try:
            return self.mini_pc_repo.get_by_id(mini_pc_id)
        except Exception as e:
            logger.error(f"Error getting Mini PC by ID: {e}")
            raise
    
    def get_mini_pc_by_mac(self, mac_address: str) -> Optional[MiniPC]:
        """Get Mini PC by MAC address"""
        try:
            return self.mini_pc_repo.get_by_mac_address(mac_address)
        except Exception as e:
            logger.error(f"Error getting Mini PC by MAC: {e}")
            raise
    
    def list_mini_pcs(self, filters: GetListFilter) -> List[MiniPC]:
        """List Mini PCs with filters"""
        try:
            return self.mini_pc_repo.list(filters)
        except Exception as e:
            logger.error(f"Error listing Mini PCs: {e}")
            raise
    
    def update_mini_pc(self, mini_pc: MiniPC) -> MiniPC:
        """Update Mini PC"""
        try:
            return self.mini_pc_repo.update(mini_pc)
        except Exception as e:
            logger.error(f"Error updating Mini PC: {e}")
            raise
    
    def delete_mini_pc(self, mini_pc_id: UUID) -> bool:
        """Delete Mini PC"""
        try:
            return self.mini_pc_repo.delete(mini_pc_id)
        except Exception as e:
            logger.error(f"Error deleting Mini PC: {e}")
            raise
    
    def get_mini_pc_cameras(self, mini_pc_id: Optional[UUID] = None, mac_address: Optional[str] = None) -> List[Camera]:
        """Get all cameras connected to a Mini PC"""
        try:
            if mini_pc_id:
                # Validate Mini PC exists first
                mini_pc = self.mini_pc_repo.get_by_id(mini_pc_id)
                if not mini_pc:
                    raise ValueError(f"Mini PC with ID {mini_pc_id} not found")
                
                return self.camera_usecase.get_cameras_by_mini_pc(mini_pc_id)
            
            elif mac_address:
                # Find Mini PC by MAC, then get cameras
                mini_pc = self.mini_pc_repo.get_by_mac_address(mac_address)
                if not mini_pc:
                    raise ValueError(f"Mini PC with MAC {mac_address} not found")
                
                return self.camera_usecase.get_cameras_by_mini_pc(mini_pc.guid)
            
            else:
                raise ValueError("Either mini_pc_id or mac_address must be provided")
                
        except Exception as e:
            logger.error(f"Error getting cameras for Mini PC: {e}")
            raise
    
    def get_mini_pc_with_cameras(self, mini_pc_id: UUID) -> Dict[str, Any]:
        """Get Mini PC info with its cameras"""
        try:
            mini_pc = self.mini_pc_repo.get_by_id(mini_pc_id)
            if not mini_pc:
                raise ValueError(f"Mini PC with ID {mini_pc_id} not found")
            
            cameras = self.camera_usecase.get_cameras_by_mini_pc(mini_pc_id)
            
            return {
                "mini_pc": mini_pc,
                "cameras": cameras,
                "cameras_count": len(cameras),
                "active_cameras_count": len([c for c in cameras if getattr(c, 'is_active', True)])
            }
            
        except Exception as e:
            logger.error(f"Error getting Mini PC with cameras: {e}")
            raise
    
    def assign_camera_to_mini_pc(self, camera_id: UUID, mini_pc_id: UUID) -> Camera:
        """Assign a camera to a Mini PC"""
        try:
            # Validate Mini PC exists
            mini_pc = self.mini_pc_repo.get_by_id(mini_pc_id)
            if not mini_pc:
                raise ValueError(f"Mini PC with ID {mini_pc_id} not found")
            
            return self.camera_usecase.assign_camera_to_mini_pc(camera_id, mini_pc_id)
            
        except Exception as e:
            logger.error(f"Error assigning camera to Mini PC: {e}")
            raise
    
    def unassign_camera_from_mini_pc(self, camera_id: UUID) -> Camera:
        """Remove camera assignment from Mini PC"""
        try:
            return self.camera_usecase.unassign_camera_from_mini_pc(camera_id)
        except Exception as e:
            logger.error(f"Error unassigning camera from Mini PC: {e}")
            raise
    
    def get_branch_mini_pcs(self, branch_id: UUID) -> List[MiniPC]:
        """Get all Mini PCs for a specific branch"""
        try:
            filters = GetListFilter(
                filters=[Filter(column="branch_id", type="eq", value=str(branch_id))]
            )
            return self.mini_pc_repo.list(filters)
        except Exception as e:
            logger.error(f"Error getting Mini PCs for branch: {e}")
            raise
    
    def get_company_mini_pcs(self, company_id: UUID) -> List[MiniPC]:
        """Get all Mini PCs for a specific company"""
        try:
            filters = GetListFilter(
                filters=[Filter(column="company_id", type="eq", value=str(company_id))]
            )
            return self.mini_pc_repo.list(filters)
        except Exception as e:
            logger.error(f"Error getting Mini PCs for company: {e}")
            raise
    
    def get_active_mini_pcs(self) -> List[MiniPC]:
        """Get all active Mini PCs"""
        try:
            filters = GetListFilter(
                filters=[Filter(column="is_active", type="eq", value="true")]
            )
            return self.mini_pc_repo.list(filters)
        except Exception as e:
            logger.error(f"Error getting active Mini PCs: {e}")
            raise
