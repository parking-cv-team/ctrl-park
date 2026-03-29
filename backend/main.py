from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from db import init_db, SessionLocal, CameraSource, Zone
from db.models import Detection
from datetime import datetime, timedelta, timezone
from sqlalchemy import func,distinct

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 
import cv2

matplotlib.use("Agg")

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
def trajectory_analysis(camera_id, frame=None):
    with SessionLocal() as db:
        rows_cars_parked = (db.query(Detection.id, Detection.tracker_id, Detection.cx, Detection.cy).
            filter(Detection.camera_id == camera_id). \
            filter(Detection.class_name == "car"). \
            filter(Detection.zone_id != None)
        )

        rows_cars_moving = (db.query(Detection.id, Detection.tracker_id, Detection.cx, Detection.cy).
            filter(Detection.camera_id == camera_id). \
            filter(Detection.class_name == "car"). \
            filter(Detection.zone_id == None)
        )
        
        rows_pedestrians = (db.query(Detection.id, Detection.tracker_id, Detection.cx, Detection.cy).
            filter(Detection.camera_id == camera_id).
            filter(Detection.class_name == "pedestrian")
        )

    df_cars_parked = (pd.DataFrame(rows_cars_parked))
    df_cars_moving = (pd.DataFrame(rows_cars_moving))
    df_pedestrians = (pd.DataFrame(rows_pedestrians))

    fig, axes = plt.subplots(1, 3, figsize=(25, 7))

    ax = axes[0]

    if frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(frame_rgb, cmap="gray")

    ax.scatter(
        df_cars_parked['cx'], df_cars_parked['cy'], color = "red", marker="s", alpha=0.3, label="Parked car", s=20
    )

    ax.scatter(
        df_cars_moving['cx'], df_cars_moving['cy'], color = "red", alpha=0.5, label="Moving Car", s=5
    )

    ax.scatter(
        df_pedestrians['cx'], df_pedestrians['cy'], color = "blue", alpha=0.5, label="Moving Pedestrian", s=5
    )

    ax.set_xlabel("x position (pixel)")
    ax.set_ylabel("y position (pixel)")
    ax.set_title("Detected Objects Scatterplot and Trajectories")

    for track_id, g in df_pedestrians.sort_values("id").groupby("tracker_id"):
            if len(g) < 2:
                ax.scatter(g["cx"], g["cy"], color="blue", s=10, alpha=0.7)
                continue

            ax.plot(g["cx"], g["cy"], color="blue", linewidth=1, alpha=0.5)
            ax.scatter(g["cx"].iloc[-1], g["cy"].iloc[-1], color="blue", s=18, alpha=0.9)

            ax.annotate(
                "",
                xy=(g["cx"].iloc[-1], g["cy"].iloc[-1]),
                xytext=(g["cx"].iloc[-2], g["cy"].iloc[-2]),
                arrowprops=dict(arrowstyle="->", color="blue", lw=1, alpha=0.5)
)

    for track_id, g in df_cars_moving.sort_values("id").groupby("tracker_id"):
        if len(g) < 2:
            ax.scatter(g["cx"], g["cy"], color="red", s=14, alpha=0.8)
            continue

        ax.plot(g["cx"], g["cy"], color="red", linewidth=1.5, alpha=0.6)
        ax.scatter(g["cx"].iloc[-1], g["cy"].iloc[-1], color="red", s=24, alpha=0.95)

        ax.annotate(
           "",
            xy=(g["cx"].iloc[-1], g["cy"].iloc[-1]),
            xytext=(g["cx"].iloc[-2], g["cy"].iloc[-2]),
            arrowprops=dict(arrowstyle="->", color="red", lw=1, alpha=0.5))

    ax.legend()
    ax.grid()

    # plot 2d histogram
    ax2 = axes[1]

    h = ax2.hist2d(df_cars_parked["cx"], df_cars_parked["cy"],
                   range=[ax.get_xlim(), ax.get_ylim()],
                   cmap="plasma")
    fig.colorbar(h[3], ax=ax2, label="Count")
    ax2.set_title("Density Heatmap (parked cars only)")

    # plot 2d histogram overall
    all_df = pd.concat([
        df_cars_moving[["cx", "cy"]],
        df_pedestrians[["cx", "cy"]]
    ], ignore_index=True)

    ax3 = axes[2]

    h_other = ax3.hist2d(all_df["cx"], all_df["cy"],
                   range=[ax.get_xlim(), ax.get_ylim()],
                   cmap = "plasma")
    fig.colorbar(h_other[3], ax=ax3, label="Count")
    ax3.set_title("Density Heatmap (moving cars + pedestrians)")

    # invert y axis-es to convert coordinates
    ax.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()

    # convert to bytes and return API response

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    im_bytes = buf.getvalue()

    buf.close()
    plt.close(fig)

    return Response(content=im_bytes, media_type="image/png")

