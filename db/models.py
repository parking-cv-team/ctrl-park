from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base
import datetime


class CameraSource(Base):
    __tablename__ = "camera_sources"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)
    uri = Column(String(255), nullable=False)

    frames = relationship("ProcessedFrame", back_populates="source")


class ProcessedFrame(Base):
    __tablename__ = "processed_frames"
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("camera_sources.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    meta = Column(String(255), default="{}")
    frame_id = Column(Integer, nullable=False)

    source = relationship("CameraSource", back_populates="frames")
