# src/models/mini_pc.py
from dataclasses import dataclass
from typing import Optional
from uuid import UUID
from datetime import datetime

@dataclass
class MiniPC:
    guid: Optional[UUID] = None
    device_name: Optional[str] = None
    mac_address: Optional[str] = None
    ip_address: Optional[str] = None
    port: Optional[int] = None
    branch_id: Optional[UUID] = None
    company_id: Optional[UUID] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
