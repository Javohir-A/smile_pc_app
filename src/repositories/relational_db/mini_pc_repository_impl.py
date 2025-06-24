# src/repositories/relational_db/mini_pc_repository_impl.py
import logging
from typing import List, Optional
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from src.repositories.mini_pc_repository import MiniPCRepository
from src.models.mini_pc import MiniPC
from src.models.filters import GetListFilter
from src.db.models import MiniPCModel
from src.helpers.query_builder import apply_filters, apply_ordering, apply_pagination

logger = logging.getLogger(__name__)

class MiniPCRepositoryImpl(MiniPCRepository):
    def __init__(self, db: Session):
        self.db = db
    
    def list(self, filters: GetListFilter) -> List[MiniPC]:
        try:
            query = self.db.query(MiniPCModel)
            
            if filters.filters:
                query = apply_filters(query, MiniPCModel, filters.filters)
            
            if filters.order_by:
                query = apply_ordering(query, MiniPCModel, filters.order_by)
            
            query = apply_pagination(query, filters.page, filters.limit)
            results = query.all()
            
            return [self._to_domain(mini_pc) for mini_pc in results]
            
        except Exception as e:
            logger.error(f"Error listing Mini PCs: {e}")
            raise
    
    def get_by_id(self, mini_pc_id: UUID) -> Optional[MiniPC]:
        try:
            mini_pc_model = self.db.query(MiniPCModel).filter_by(guid=mini_pc_id).first()
            return self._to_domain(mini_pc_model) if mini_pc_model else None
        except Exception as e:
            logger.error(f"Error getting Mini PC by ID {mini_pc_id}: {e}")
            raise
    
    def get_by_mac_address(self, mac_address: str) -> Optional[MiniPC]:
        try:
            mini_pc_model = self.db.query(MiniPCModel).filter_by(mac_address=mac_address).first()
            return self._to_domain(mini_pc_model) if mini_pc_model else None
        except Exception as e:
            logger.error(f"Error getting Mini PC by MAC {mac_address}: {e}")
            raise
    
    def create(self, mini_pc: MiniPC) -> MiniPC:
        try:
            mini_pc_model = MiniPCModel(
                device_name=mini_pc.device_name,
                mac_address=mini_pc.mac_address,
                ip_address=mini_pc.ip_address,
                port=mini_pc.port,
                branch_id=mini_pc.branch_id,
                company_id=mini_pc.company_id,
                is_active=mini_pc.is_active
            )
            
            self.db.add(mini_pc_model)
            self.db.commit()
            self.db.refresh(mini_pc_model)
            
            return self._to_domain(mini_pc_model)
            
        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"Integrity error creating Mini PC: {e}")
            raise ValueError("Mini PC with this MAC address already exists")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating Mini PC: {e}")
            raise
    
    def update(self, mini_pc: MiniPC) -> MiniPC:
        try:
            mini_pc_model = self.db.query(MiniPCModel).filter_by(guid=mini_pc.guid).first()
            if not mini_pc_model:
                raise ValueError(f"Mini PC with ID {mini_pc.guid} not found")
            
            # Update fields
            if mini_pc.device_name is not None:
                mini_pc_model.device_name = mini_pc.device_name
            if mini_pc.mac_address is not None:
                mini_pc_model.mac_address = mini_pc.mac_address
            if mini_pc.ip_address is not None:
                mini_pc_model.ip_address = mini_pc.ip_address
            if mini_pc.port is not None:
                mini_pc_model.port = mini_pc.port
            if mini_pc.branch_id is not None:
                mini_pc_model.branch_id = mini_pc.branch_id
            if mini_pc.company_id is not None:
                mini_pc_model.company_id = mini_pc.company_id
            if mini_pc.is_active is not None:
                mini_pc_model.is_active = mini_pc.is_active
            
            self.db.commit()
            self.db.refresh(mini_pc_model)
            
            return self._to_domain(mini_pc_model)
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating Mini PC: {e}")
            raise
    
    def delete(self, mini_pc_id: UUID) -> bool:
        try:
            mini_pc_model = self.db.query(MiniPCModel).filter_by(guid=mini_pc_id).first()
            if not mini_pc_model:
                return False
            
            self.db.delete(mini_pc_model)
            self.db.commit()
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting Mini PC: {e}")
            raise
    
    def _to_domain(self, model: MiniPCModel) -> MiniPC:
        """Convert SQLAlchemy model to domain model"""
        return MiniPC(
            guid=model.guid,
            device_name=model.device_name,
            mac_address=model.mac_address,
            ip_address=model.ip_address,
            port=model.port,
            branch_id=model.branch_id,
            company_id=model.company_id,
            is_active=model.is_active,
            created_at=model.created_at
        )
