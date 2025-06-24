# src/db/models.py
from sqlalchemy import Column, String, Boolean, Integer, Date
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.declarative import declarative_base
import uuid
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, ARRAY
from sqlalchemy import Column, Float, String, DateTime, func
# from datetime import datetime

Base = declarative_base()

class CameraModel(Base):
    __tablename__ = "camera_"

    guid = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    detect_emotion = Column(Boolean, default=False)
    detect_hands = Column(Boolean, default=False)
    detect_uniform = Column(Boolean, default=False)
    detect_mask = Column(Boolean, default=False)
    voice_detect = Column(Boolean, default=False)
    branch_id = Column(PG_UUID(as_uuid=True), nullable=True)
    company_id = Column(PG_UUID(as_uuid=True), nullable=True)
    port = Column(Integer, nullable=False)
    ip_address = Column(String, nullable=False)
    username = Column(String, nullable=False)
    password = Column(String, nullable=False)
    
    # ADD THIS LINE (make sure NO comma after the last existing line above)
    mini_pc_id = Column(PG_UUID(as_uuid=True), nullable=True)

# And add the MiniPCModel if you haven't already:
class MiniPCModel(Base):
    __tablename__ = "mini_pc"

    guid = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    device_name = Column(String(100), nullable=False)
    mac_address = Column(String(17), unique=True, nullable=False)
    ip_address = Column(String(45), nullable=False)
    port = Column(Integer, nullable=False, default=8080)
    branch_id = Column(PG_UUID(as_uuid=True), nullable=False)
    company_id = Column(PG_UUID(as_uuid=True), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=func.now())