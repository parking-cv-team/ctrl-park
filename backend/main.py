from unittest import result

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from db import init_db, SessionLocal, CameraSource, Zone, ZoneOccupancy, MappedZone
import db
import db
from db.models import Detection
from datetime import datetime, timedelta, timezone
from sqlalchemy import func, distinct, text
import numpy as np

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cv2
import base64
import numpy as np
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")

import io

load_dotenv()

app = FastAPI(title="Ctrl+Park API")

# Add CORS middleware to allow requests from Streamlit and other origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()


class CameraInput(BaseModel):
    name: str
    uri: str


class TrajectoryRequest(BaseModel):
    camera_id: int
    frame: str = None  # base64-encoded PNG


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
def cameras():
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
def recent_analytics_number_of_cars(camera_id):

    db = SessionLocal()

    items = (
        db.query(Detection.id, Detection.tracker_id, Detection.event_type)
        .filter(Detection.camera_id == camera_id)
        .filter(Detection.class_name == "car")
        .order_by(Detection.id.desc())
    )

    db.close()
    return [
        {"id": it.id, "tracker_id": it.tracker_id, "event_type": it.event_type}
        for it in items
    ]


@app.get("/analytics/cameras/recent/outside_zones")
def recent_analytics_cars_outside_zones(camera_id):
    db = SessionLocal()

    items = (
        db.query(
            Detection.id,
            Detection.tracker_id,
            Detection.class_name,
            Detection.event_type,
            Detection.zone_id,
            Detection.cx,
            Detection.cy,
            Detection.timestamp,
        )
        .filter(Detection.camera_id == camera_id)
        .filter(Detection.class_name != "pedestrian")
        .order_by(Detection.id.desc())
    )
    db.close()
    return [
        {
            "id": it.id,
            "tracker_id": it.tracker_id,
            "class": it.class_name,
            "cx": it.cx,
            "cy": it.cy,
            "time": it.timestamp,
            "event_type": it.event_type,
        }
        for it in items
        if it.zone_id is None
    ]


@app.get("/analytics/zones")
def cameras(camera_id):

    db = SessionLocal()
    items = db.query(Zone).filter(Zone.camera_id == camera_id).all()
    zones = [
        {
            "id": it.id,
            "name": str(it.name),
            "category": it.category,
            "camera_id": it.camera_id,
        }
        for it in items
    ]

    items = []
    for z in zones:
        occupancy = (
            db.query(ZoneOccupancy)
            .filter(ZoneOccupancy.zone_id == z["id"])
            .order_by(ZoneOccupancy.id.desc())
        )

        o = [{"id": it.id, "tracker": it.tracker_id} for it in occupancy]

        if len(o) == 0 or o[0]["tracker"] is None:
            items.append({"zone": z["name"], "occupancy": "not occupied"})
        else:
            items.append({"zone": z["name"], "occupancy": "occupied"})

    db.close()
    return items


@app.post("/analytics/trajectory_analysis")
def trajectory_analysis(body: TrajectoryRequest):
    # TODO: treat edge cases, like no parked cars at all/ no pedestrians at all/ no moving cars at all
    camera_id = body.camera_id
    db = SessionLocal()

    rows_cars_parked = (
        db.query(Detection.id, Detection.tracker_id, Detection.cx, Detection.cy)
        .filter(Detection.camera_id == camera_id)
        .filter(Detection.class_name == "car")
        .filter(Detection.zone_id != None)
    )

    rows_cars_moving = (
        db.query(Detection.id, Detection.tracker_id, Detection.cx, Detection.cy)
        .filter(Detection.camera_id == camera_id)
        .filter(Detection.class_name == "car")
        .filter(Detection.zone_id == None)
    )

    rows_pedestrians = (
        db.query(Detection.id, Detection.tracker_id, Detection.cx, Detection.cy)
        .filter(Detection.camera_id == camera_id)
        .filter(Detection.class_name == "pedestrian")
    )

    df_cars_parked = pd.DataFrame(rows_cars_parked)
    df_cars_moving = pd.DataFrame(rows_cars_moving)
    df_pedestrians = pd.DataFrame(rows_pedestrians)

    fig = plt.figure(layout="constrained", figsize=(17, 12))
    fig.suptitle("Detections Scatterplot and Heatmap(s)")
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    if body.frame is not None:
        img_bytes = base64.b64decode(body.frame)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ax.invert_yaxis()
        ax.imshow(frame_rgb)
        ax2.imshow(frame_rgb, alpha=0.5)
        ax3.imshow(frame_rgb, alpha=0.5)

    ax.scatter(
        df_cars_parked["cx"],
        df_cars_parked["cy"],
        color="red",
        marker="s",
        alpha=0.3,
        label="Parked car",
        s=20,
    )

    ax.scatter(
        df_cars_moving["cx"],
        df_cars_moving["cy"],
        color="red",
        alpha=0.5,
        label="Moving Car",
        s=5,
    )

    ax.scatter(
        df_pedestrians["cx"],
        df_pedestrians["cy"],
        color="blue",
        alpha=0.5,
        label="Moving Pedestrian",
        s=5,
    )

    ax.set_xlabel("x position (pixel)")
    ax.set_ylabel("y position (pixel)")
    ax2.set_xlabel("x position (pixel)")
    ax2.set_ylabel("y position (pixel)")
    ax3.set_xlabel("x position (pixel)")
    ax3.set_ylabel("y position (pixel)")

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
            arrowprops=dict(arrowstyle="->", color="blue", lw=1, alpha=0.5),
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
            arrowprops=dict(arrowstyle="->", color="red", lw=1, alpha=0.5),
        )

    ax.legend()
    ax.grid()

    # plot 2d histogram

    print(ax.get_xlim(), ax.get_ylim())

    h, xe1, ye1 = np.histogram2d(
        df_cars_parked["cx"],
        df_cars_parked["cy"],
        range=[ax.get_xlim(), ax.get_ylim()[::-1]],
        bins=25,
    )

    h_blur = gaussian_filter(h, sigma=2)

    ax2.imshow(
        h_blur.T,
        origin="lower",
        extent=[xe1[0], xe1[-1], ye1[0], ye1[-1]],
        cmap="magma",
        aspect="auto",
        alpha=0.7,
    )

    ax2.set_title("Density Heatmap (parked cars only)")

    # plot 2d histogram overall
    all_df = pd.concat(
        [df_cars_moving[["cx", "cy"]], df_pedestrians[["cx", "cy"]]], ignore_index=True
    )

    h, xe1, ye1 = np.histogram2d(
        all_df["cx"], all_df["cy"], range=[ax.get_xlim(), ax.get_ylim()[::-1]], bins=25
    )

    h_blur = gaussian_filter(h, sigma=2)

    ax3.imshow(
        h_blur.T,
        origin="lower",
        extent=[xe1[0], xe1[-1], ye1[0], ye1[-1]],
        cmap="magma",
        aspect="auto",
        alpha=0.7,
    )

    ax3.set_title("Density Heatmap (moving cars + pedestrians)")

    # invert y axis-es to convert coordinates
    ax2.invert_yaxis()
    ax3.invert_yaxis()

    # convert to bytes and return API response

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    im_bytes = buf.getvalue()

    buf.close()
    plt.close(fig)
    db.close()
    return Response(content=im_bytes, media_type="image/png")


@app.get("/analytics/metrics_report/kpi")
def metrics_report_kpi(camera_id, t_start, t_end):
    db = SessionLocal()
    # 1. total tracked by class
    total_tracked_by_class_q = (
        db.query(
            Detection.class_name.label("class_name"),
            func.count(distinct(Detection.tracker_id)).label("total_tracked"),
        )
        .filter(
            Detection.camera_id == camera_id,
            Detection.timestamp > t_start,
            Detection.timestamp < t_end,
            Detection.tracker_id.isnot(None),
        )
        .group_by(Detection.class_name)
    )

    # 2. average confidence by class
    avg_confidence_by_class_q = (
        db.query(
            Detection.class_name.label("class_name"),
            func.avg(Detection.confidence).label("avg_confidence"),
        )
        .filter(
            Detection.camera_id == camera_id,
            Detection.timestamp > t_start,
            Detection.timestamp < t_end,
        )
        .group_by(Detection.class_name)
    )

    # 3. total zones
    # Assumes Zone has camera_id. If not, this must be changed.
    total_zones_q = db.query(func.count(Zone.id).label("total_zones")).filter(
        Zone.camera_id == camera_id
    )

    # 4. maximum and average occupations
    # Inner query: number of occupied zones/detections per timestamp
    occupations_subq = (
        db.query(
            Detection.timestamp.label("ts"),
            func.count(Detection.id).label("occupation_count"),
        )
        .filter(
            Detection.camera_id == camera_id,
            Detection.timestamp > t_start,
            Detection.timestamp < t_end,
            Detection.zone_id.isnot(None),
        )
        .group_by(Detection.timestamp)
        .subquery()
    )

    max_occupations_q = db.query(
        func.max(occupations_subq.c.occupation_count).label("max_occupations")
    )

    avg_occupations_q = db.query(
        func.avg(occupations_subq.c.occupation_count).label("avg_occupations")
    )

    # 5. average tracking time by class
    track_subquery = (
        db.query(
            Detection.tracker_id.label("t_id"),
            Detection.class_name.label("class_name"),
            func.timestampdiff(
                text("SECOND"),
                func.min(Detection.timestamp),
                func.max(Detection.timestamp),
            ).label("tdiff"),
        )
        .filter(
            Detection.camera_id == camera_id,
            Detection.timestamp > t_start,
            Detection.timestamp < t_end,
            Detection.tracker_id.isnot(None),
            Detection.timestamp.isnot(None),
        )
        .group_by(Detection.tracker_id, Detection.class_name)
        .subquery()
    )

    avg_track_time_q = db.query(
        track_subquery.c.class_name,
        func.avg(track_subquery.c.tdiff).label("avg_track_time_seconds"),
    ).group_by(track_subquery.c.class_name)

    # 7. number of departures

    departures_subquery = (
        db.query(
            Detection.tracker_id.label("tracker"),
            Detection.class_name.label("class_name"),
        )
        .filter(
            Detection.camera_id == camera_id,
            Detection.timestamp > t_start,
            Detection.timestamp < t_end,
            Detection.event_type == "departure",
        )
        .group_by(Detection.tracker_id, Detection.class_name)
        .subquery()
    )

    n_departures = db.query(
        departures_subquery.c.class_name,
        func.count(departures_subquery.c.tracker).label("number_of_departures"),
    ).group_by(departures_subquery.c.class_name)

    # 8. number of new detections
    detections_subquery = (
        db.query(
            Detection.tracker_id.label("tracker"),
            Detection.class_name.label("class_name"),
        )
        .filter(
            Detection.camera_id == camera_id,
            Detection.timestamp > t_start,
            Detection.timestamp < t_end,
            Detection.event_type == "detection",
        )
        .group_by(Detection.tracker_id, Detection.class_name)
        .subquery()
    )

    n_tracked_detect = db.query(
        detections_subquery.c.class_name,
        func.count(detections_subquery.c.tracker).label("number of tracked items"),
    ).group_by(detections_subquery.c.class_name)

    to_ret = {
        "total_tracked_by_class": pd.DataFrame(total_tracked_by_class_q).to_dict(
            orient="records"
        ),
        "avg_confidence_by_class": pd.DataFrame(avg_confidence_by_class_q).to_dict(
            orient="records"
        ),
        "total_zones": pd.DataFrame(total_zones_q).to_dict(orient="records"),
        "max_occupations": pd.DataFrame(max_occupations_q).to_dict(orient="records"),
        "avg_occupations": pd.DataFrame(avg_occupations_q).to_dict(orient="records"),
        "avg_track_time": pd.DataFrame(avg_track_time_q).to_dict(orient="records"),
        "n_departures": pd.DataFrame(n_departures).to_dict(orient="records"),
        "n_tracked_det": pd.DataFrame(n_tracked_detect).to_dict(orient="records"),
    }
    db.close()
    return to_ret


@app.get("/analytics/metrics_report/timeseries")
def metrics_report_timeseries(camera_id, t_start, t_end):
    db = SessionLocal()
    confidence_ts = (
        db.query(
            Detection.timestamp.label("t"),
            Detection.class_name.label("class_name"),
            func.avg(Detection.confidence).label("avg_confidence"),
        )
        .filter(
            Detection.camera_id == camera_id,
            Detection.timestamp > t_start,
            Detection.timestamp < t_end,
        )
        .group_by(Detection.timestamp, Detection.class_name)
        .all()
    )

    tracked_objects_ts = (
        db.query(
            Detection.timestamp.label("t"),
            Detection.class_name.label("class_name"),
            func.count(func.distinct(Detection.tracker_id)).label("num_tracked"),
        )
        .filter(
            Detection.camera_id == camera_id,
            Detection.timestamp > t_start,
            Detection.timestamp < t_end,
            Detection.tracker_id.isnot(None),
            Detection.class_name.in_(["car", "pedestrian"]),
        )
        .group_by(Detection.timestamp, Detection.class_name)
        .all()
    )

    parked_vehicles_ts = (
        db.query(
            Detection.timestamp.label("t"),
            func.count(func.distinct(Detection.tracker_id)).label(
                "num_parked_vehicles"
            ),
        )
        .filter(
            Detection.camera_id == camera_id,
            Detection.timestamp > t_start,
            Detection.timestamp < t_end,
            Detection.tracker_id.isnot(None),
            Detection.zone_id.isnot(None),
            Detection.class_name == "car",
        )
        .group_by(Detection.timestamp)
        .all()
    )
    db.close()
    return {
        "ts_confidence": pd.DataFrame(confidence_ts).to_dict(orient="records"),
        "ts_objects": pd.DataFrame(tracked_objects_ts).to_dict(orient="records"),
        "ts_parked": pd.DataFrame(parked_vehicles_ts).to_dict(orient="records"),
    }


@app.get("/analytics/zones/poly")
def zones_from_cameras(camera_id, limit=50):
    limit = int(limit)
    if not limit:
        raise HTTPException(status_code=400, detail="Limit must be an integer")
    if limit < 1:
        raise HTTPException(status_code=400, detail="Limit must be positive")

    db = SessionLocal()
    items = db.query(Zone).filter(Zone.camera_id == camera_id).limit(limit).all()
    db.close()

    return items


@app.get("/mapped_zones/poly")
def mapped_zones(single_camera: bool = False):
    db = SessionLocal()
    if not single_camera:
        items = db.query(MappedZone).all()
    else:
        items = [
            {
                "id": zone.id,
                "polygon_global_metric": zone.polygon_metric,
            }
            for zone in db.query(Zone).all()
        ]
    db.close()

    return items


@app.get("/mapped_zones/status")
def get_mapped_zones_status(single_camera: bool = False):
    db = SessionLocal()
    if not single_camera:
        all_mapped_zones = db.query(MappedZone).all()
    else:
        all_mapped_zones = [
            (
                {
                    "id": zone.id,
                    "polygon_global_metric": zone.polygon_metric,
                }
            )
            for zone in db.query(Zone).all()
        ]
    db.close()

    result = []
    for mapped_zone in all_mapped_zones:
        # Find all camera-specific zones that reference this mapped zone
        if not single_camera:
            camera_zones = (
            db.query(Zone.id).filter(Zone.mapped_zone_id == (mapped_zone['id'] if single_camera else mapped_zone.id)).all()
        )
            zone_ids = [z[0] for z in camera_zones]
        else:
            zone_ids = [mapped_zone['id'] if single_camera else mapped_zone.id]  # In single camera mode, the mapped zone ID is the same as the zone ID

        if not zone_ids:
            # No zones tied to this mapped zone, so we consider it free because it was never marked occupied
            result.append(
                {
                    "mapped_zone_id": mapped_zone['id'] if single_camera else mapped_zone.id,
                    "last_occupancy_time": None,
                    "is_occupied": False,
                }
            )
            continue

        # Find the most recent occupancy across all zones referencing this mapped zone
        latest_occupancy = (
            db.query(ZoneOccupancy, Detection.timestamp)
            .join(Detection, ZoneOccupancy.detection_id == Detection.id)
            .filter(ZoneOccupancy.zone_id.in_(zone_ids))
            .order_by(ZoneOccupancy.id.desc())
            .first()
        )

        if latest_occupancy is None:
            result.append(
                {
                    "mapped_zone_id": mapped_zone['id'] if single_camera else mapped_zone.id,
                    "last_occupancy_time": None,
                    "is_occupied": False,
                }
            )
        else:
            occupancy_record, timestamp = latest_occupancy
            is_occupied = occupancy_record.tracker_id is not None

            result.append(
                {
                    "mapped_zone_id": mapped_zone['id'] if single_camera else mapped_zone.id,
                    "last_occupancy_time": timestamp.isoformat() if timestamp else None,
                    "is_occupied": is_occupied,
                }
            )

    db.close()
    return result
