from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID
from src.models.mini_pc import MiniPC
from src.models.filters import GetListFilter

class MiniPCRepository(ABC):
    @abstractmethod
    def list(self, filters: GetListFilter) -> List[MiniPC]:
        pass
    
    @abstractmethod
    def get_by_id(self, mini_pc_id: UUID) -> Optional[MiniPC]:
        pass
    
    @abstractmethod
    def get_by_mac_address(self, mac_address: str) -> Optional[MiniPC]:
        pass
    
    @abstractmethod
    def create(self, mini_pc: MiniPC) -> MiniPC:
        pass
    
    @abstractmethod
    def update(self, mini_pc: MiniPC) -> MiniPC:
        pass
    
    @abstractmethod
    def delete(self, mini_pc_id: UUID) -> bool:
        pass
