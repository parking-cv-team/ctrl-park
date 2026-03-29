from sqlalchemy import Column, Float, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from db.database import Base
import datetime


class CameraSource(Base):
    __tablename__ = "camera_sources"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)
    uri = Column(String(255), unique=True, nullable=False, index=True)
    frame_width = Column(Integer, nullable=True)
    frame_height = Column(Integer, nullable=True)
    homography = Column(JSON, nullable=True)  # 3×3 matrix [[a,b,c],[d,e,f],[g,h,i]]

    zones = relationship("Zone", back_populates="camera", cascade="all, delete-orphan")


class MappedZone(Base):
    """Physical parking slot in the unified multi-camera metric reference frame."""
    __tablename__ = "mapped_zones"

    id = Column(Integer, primary_key=True, index=True)
    polygon_global_metric = Column(JSON, nullable=False)  # [[x_m, y_m], ...] in cam-0 global space

    zones = relationship("Zone", back_populates="mapped_zone")


class Zone(Base):
    __tablename__ = "zones"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(64), nullable=False, index=True)
    polygon = Column(JSON, nullable=False)           # pixel coords [[x, y], ...] — used by worker.py
    category = Column(String(32), nullable=True)    # parking lot | road | etc
    camera_id = Column(
        Integer, ForeignKey("camera_sources.id"), nullable=False, index=True
    )
    polygon_metric = Column(JSON, nullable=True)    # metric coords [[x_m, y_m], ...] (camera-local)
    mapped_zone_id = Column(
        Integer, ForeignKey("mapped_zones.id"), nullable=True, index=True
    )

    camera = relationship("CameraSource", back_populates="zones")
    mapped_zone = relationship("MappedZone", back_populates="zones")
    occupancies = relationship("ZoneOccupancy", back_populates="zone")
    detections = relationship("Detection", back_populates="zone")


class Detection(Base):

    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("camera_sources.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    tracker_id = Column(Integer, nullable=True)
    global_id = Column(Integer, nullable=True, index=True)

    class_id = Column(Integer, nullable=False)
    class_name = Column(String(32), nullable=False)
    confidence = Column(Float, nullable=False)
    event_type = Column(String(32), default="detection", nullable=False)  # "detection" or "departure"

    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    cx = Column(Float, nullable=False)
    cy = Column(Float, nullable=False)
    bbox_width = Column(Float, nullable=False)
    bbox_height = Column(Float, nullable=False)
    bbox_area = Column(Float, nullable=False)

    zone_id = Column(Integer, ForeignKey("zones.id"), nullable=True)

    zone = relationship("Zone", back_populates="detections")
    zone_occupancies = relationship("ZoneOccupancy", back_populates="detection")


class ZoneOccupancy(Base):
    __tablename__ = "zone_occupancy"

    id = Column(Integer, primary_key=True, index=True)

    detection_id = Column(Integer, ForeignKey("detections.id"), nullable=False)
    zone_id = Column(Integer, ForeignKey("zones.id"), nullable=False, index=True)
    tracker_id = Column(Integer, nullable=True)

    detection = relationship("Detection", back_populates="zone_occupancies")
    zone = relationship("Zone", back_populates="occupancies")
