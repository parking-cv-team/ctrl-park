from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from db import init_db, SessionLocal, CameraSource, Zone
from db.models import Detection
from datetime import datetime, timedelta, timezone
from sqlalchemy import func

load_dotenv()

app = FastAPI(title="Ctrl+Park API")

init_db()


class CameraInput(BaseModel):
    name: str
    uri: str



@app.post("/camera")
def create_camera(camera: CameraInput):
    db = SessionLocal()
    existing = db.query(CameraSource).filter(CameraSource.uri == camera.uri).first()
    if existing:
        raise HTTPException(status_code=400, detail="Camera already registered")
    source = CameraSource(name=camera.name, uri=camera.uri)
    db.add(source)
    db.commit()
    db.refresh(source)
    return {"id": source.id, "name": source.name, "uri": source.uri}


@app.get("/analytics/recent")
def recent_analytics(limit=50):
    limit = int(limit)
    if not limit:
        raise HTTPException(status_code=400, detail="Limit must be an integer")
    if limit < 1:
        raise HTTPException(status_code=400, detail="Limit must be positive")

    db = SessionLocal()
    items = db.query(Detection).order_by(Detection.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": it.id,
            "camera_id": it.camera_id,
            "timestamp": it.timestamp,
            
        }
        for it in items
    ]

@app.get("/analytics/cameras")
def cameras(limit=50):
    limit = int(limit)
    if not limit:
        raise HTTPException(status_code=400, detail="Limit must be an integer")
    if limit < 1:
        raise HTTPException(status_code=400, detail="Limit must be positive")

    db = SessionLocal()
    items = db.query(CameraSource)
    return [
        {
            "id": it.id,
            "name": it.name,
            "uri": it.uri,
        }
        for it in items
    ]

@app.get("/analytics/cameras/recent")
def recent_analytics(camera_id,limit=50):
    limit = int(limit)
    if not limit:
        raise HTTPException(status_code=400, detail="Limit must be an integer")
    if limit < 1:
        raise HTTPException(status_code=400, detail="Limit must be positive")

    db = SessionLocal()
    one_minute_ago = datetime.now(timezone.utc) - timedelta(minutes=1)
    items = db.query(func.count(Detection.id)). \
        filter(Detection.timestamp >= one_minute_ago).filter(Detection.camera_id == camera_id). \
        filter(Detection.class_name == "car").scalar()
    return items

@app.get("/analytics/zones")
def cameras(camera_id,limit=50):
    limit = int(limit)
    if not limit:
        raise HTTPException(status_code=400, detail="Limit must be an integer")
    if limit < 1:
        raise HTTPException(status_code=400, detail="Limit must be positive")

    
    db = SessionLocal()
    items = db.query(Zone).filter(Zone.camera_id == camera_id).limit(limit).all()
    zones = [
        {
            "id": it.id,
            "name": str(it.name),
            "category": it.category,
            "camera_id": it.camera_id,
        } for it in items]
    
    one_minute_ago = datetime.now(timezone.utc) - timedelta(minutes=1)
    items = []
    for z in zones:
        occupancy = db.query(func.count(Detection.id)). \
            filter(Detection.timestamp >= one_minute_ago).filter(Detection.camera_id == camera_id). \
            filter(Detection.class_name == "car").filter(Detection.zone_id == z["id"]).scalar()
        if occupancy == 0:
            items.append({"zone":z["name"],"occupancy":"not occupied"})
        else:
            items.append({"zone":z["name"],"occupancy":"occupied"})
        

    return items