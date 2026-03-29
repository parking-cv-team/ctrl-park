from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from db import init_db, SessionLocal, CameraSource, Zone
from db.models import Detection
from datetime import datetime, timedelta, timezone
from sqlalchemy import func,distinct

import pandas as pd
import matplotlib.pyplot as plt 
import io 

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
    db.close()
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
    db.close()
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
    db.close()
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
    items = db.query(func.count(distinct(Detection.id))). \
        filter(Detection.timestamp >= one_minute_ago).filter(Detection.camera_id == camera_id). \
        filter(Detection.class_name == "car").scalar()
    db.close()
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

    db.close()
    return items

@app.get("/analytics/trajectory_analysis")
def trajectory_analysis(camera_id):
    with SessionLocal() as db:
        rows_cars = (db.query(Detection.id, Detection.cx, Detection.cy).
            filter(Detection.camera_id == camera_id). \
            filter(Detection.class_name == "car")
        )
        
        rows_pedestrians = (db.query(Detection.id, Detection.cx, Detection.cy).
            filter(Detection.camera_id == camera_id).
            filter(Detection.class_name == "pedestrian")
        )

    df_cars = pd.DataFrame(rows_cars)
    df_pedestrians = pd.DataFrame(rows_pedestrians)

    fig, ax = plt.subplots()

    ax.scatter(
        df_cars['cx'], df_cars['cy'], color = "red"
    )

    ax.scatter(
        df_pedestrians['cx'], df_pedestrians['cy'], color = "blue"
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    im_bytes = buf.getvalue()

    buf.close()
    plt.close(fig)

    return Response(content=im_bytes, media_type="image/png")

