from unittest import result

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from db import init_db, SessionLocal, CameraSource, Zone, ZoneOccupancy, MappedZone
from db.models import Detection
from sqlalchemy import func, distinct, text
import numpy as np
from typing import List
from pathlib import Path
import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cv2
import base64
from scipy.ndimage import gaussian_filter
from fastapi import Depends
from sqlalchemy.orm import Session
from processing.merge_cameras import estimate_transforms,compute_global_transforms,ensure_topdown,save_merged_topdown,polygon_iou,_canonical_quad
from processing.merge_cameras import apply_affine_pts

matplotlib.use("Agg")

import io

load_dotenv()

_pipeline_proc = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if _pipeline_proc and _pipeline_proc.poll() is None:
        _pipeline_proc.terminate()
        try:
            _pipeline_proc.wait(timeout=5)
        except Exception:
            _pipeline_proc.kill()

app = FastAPI(title="Ctrl+Park API", lifespan=lifespan)

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

# DB manager to avoid connection leaks
def get_db():
    db = SessionLocal()
    try:
        yield db 
    finally:
        db.close()

@app.post("/camera")
def create_camera(camera: CameraInput, db: Session = Depends(get_db)):
    existing = db.query(CameraSource).filter(CameraSource.uri == camera.uri).first()
    if existing:
        raise HTTPException(status_code=400, detail="Camera already registered")
    source = CameraSource(name=camera.name, uri=camera.uri)
    db.add(source)
    db.commit()
    db.refresh(source)
    return {"id": source.id, "name": source.name, "uri": source.uri}


@app.get("/analytics/recent")
def recent_analytics(limit=50, db: Session = Depends(get_db)):
    limit = int(limit)
    if not limit:
        raise HTTPException(status_code=400, detail="Limit must be an integer")
    if limit < 1:
        raise HTTPException(status_code=400, detail="Limit must be positive")

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
def cameras(db: Session = Depends(get_db)):
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
def recent_analytics_number_of_cars(camera_id, db: Session = Depends(get_db)):

    items = (
        db.query(Detection.id, Detection.tracker_id, Detection.event_type)
        .filter(Detection.camera_id == camera_id)
        .filter(Detection.class_name == "car")
        .order_by(Detection.id.desc())
    )

    return [
        {"id": it.id, "tracker_id": it.tracker_id, "event_type": it.event_type}
        for it in items
    ]


@app.get("/analytics/cameras/recent/outside_zones")
def recent_analytics_cars_outside_zones(camera_id, db: Session = Depends(get_db)):
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
def cameras(camera_id, db: Session = Depends(get_db)):

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

    return items


@app.post("/analytics/trajectory_analysis")
def trajectory_analysis(body: TrajectoryRequest, db: Session = Depends(get_db)):
    camera_id = body.camera_id

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
    return Response(content=im_bytes, media_type="image/png")


@app.get("/analytics/metrics_report/kpi")
def metrics_report_kpi(camera_id, t_start, t_end, db: Session = Depends(get_db)):
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

    return to_ret


@app.get("/analytics/metrics_report/timeseries")
def metrics_report_timeseries(camera_id, t_start, t_end, db: Session = Depends(get_db)):
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
    return {
        "ts_confidence": pd.DataFrame(confidence_ts).to_dict(orient="records"),
        "ts_objects": pd.DataFrame(tracked_objects_ts).to_dict(orient="records"),
        "ts_parked": pd.DataFrame(parked_vehicles_ts).to_dict(orient="records"),
    }


@app.get("/analytics/zones/poly")
def zones_from_cameras(camera_id, limit=50, db: Session = Depends(get_db)):
    limit = int(limit)
    if not limit:
        raise HTTPException(status_code=400, detail="Limit must be an integer")
    if limit < 1:
        raise HTTPException(status_code=400, detail="Limit must be positive")

    db = SessionLocal()
    items = db.query(Zone).filter(Zone.camera_id == camera_id).limit(limit).all()

    return items


@app.get("/mapped_zones/poly")
def mapped_zones(camera_id: int = 0, single_camera: bool = False, db: Session = Depends(get_db)):
    if not single_camera:
        items = db.query(MappedZone).all()
    else:
        items = [
            {
                "id": zone.id,
                "polygon_global_metric": zone.polygon_metric,
            }
            for zone in db.query(Zone).filter(Zone.camera_id == camera_id).all()
        ]

    return items


@app.get("/mapped_zones/status")
def get_mapped_zones_status(camera_id: int = 0, single_camera: bool = False, db: Session = Depends(get_db)):
    print(camera_id, single_camera)
    camera_id = int(camera_id)

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
            for zone in db.query(Zone).filter(Zone.camera_id == camera_id).all()
        ]

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
    return result

@app.get("/camera/config")
def get_camera_config(cam_name: str):
    
    configs_dir, _ = get_config_paths()
    path = configs_dir / f"{cam_name}.json"

    try:
        with path.open() as f:
            cfg = json.load(f)
    except:
        cfg = None
    if cfg is not None:
        initial_params = {
            "rows":          cfg["rows"],
            "cols":          cfg["cols"],
            "slot_w":        cfg["slot_w"],
            "slot_h":        cfg["slot_h"],
            "row_gap":       cfg.get("row_gap", 0.0),
            "files_per_row": cfg.get("files_per_row", 1),
            "frame_idx":     cfg.get("frame_idx", 0),
        }
    else:
        initial_params  = None
    return initial_params


def get_config_paths():
    
    configs_dir = Path(os.getenv("CALIBRATION_CONFIGS_DIR", "calibration/configs"))
    output_dir  = Path(os.getenv("CALIBRATION_OUTPUT_DIR",  "calibration/output"))
    configs_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return configs_dir, output_dir

class ConfigData(BaseModel):
    frame_idx: int
    rows: int
    cols: int
    slot_w: float
    slot_h: float
    row_gap: float
    files_per_row: int
    control_points_pixel: List[List[float]] # Assuming list of [x, y] coordinates
    cam_name: str

@app.post("/camera/config/save")
def save_camera_config(data: ConfigData):
    cam_name = data.cam_name
    data = data.model_dump(exclude="cam_name")
    
    configs_dir, _ = get_config_paths()
    path = configs_dir / f"{cam_name}.json"
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return 


class TopdownData(BaseModel):
    cam_name: str
    img: list

@app.post("/camera/topdown/save")
def save_camera_topdown(data: TopdownData):
    cam_name = data.cam_name
    img = np.array(data.img, dtype=np.uint8)
    _, output_dir = get_config_paths()
    
    topdown_path = output_dir / f"{cam_name}_topdown.png"

    cv2.imwrite(topdown_path,img)
    return


class ZonesToSaveToDB(BaseModel):
    uri: str
    cam_name: str
    frame_w: int
    frame_h: int
    H: list
    slots: list
    active: dict
    params: dict


@app.post('/camera/config/zones')
def save_zones_to_db(data: ZonesToSaveToDB, db: Session = Depends(get_db)):
    uri = data.uri
    cam_name = data.cam_name
    frame_h = data.frame_h
    frame_w = data.frame_w    
    H = np.array(data.H)
    slots = data.slots
    active = data.active
    params = data.params
    try:
        camera = db.query(CameraSource).filter(CameraSource.uri == uri).first()
        if not camera:
            camera = CameraSource(name=cam_name, uri=uri)
            db.add(camera)
            db.flush()
        else:
            camera.name = cam_name
        camera.frame_width  = frame_w
        camera.frame_height = frame_h
        camera.homography   = H.tolist()
        db.query(Zone).filter(Zone.camera_id == camera.id).delete(
            synchronize_session=False
        )

        slot_w       = params["slot_w"]
        slot_h       = params["slot_h"]
        row_gap      = params.get("row_gap", 0.0)
        files_per_row = params.get("files_per_row", 1)

        for s in sorted(slots, key=lambda x: (x["row"], x["col"])):
            key = f"{s['row']}_{s['col']}"
            if not active.get(key, False):
                continue
            poly_detect = s.get("polygon_detection_px")
            if poly_detect is None:
                continue
            poly_metric = slot_metric_polygon(
                s["row"], s["col"], slot_w, slot_h, row_gap, files_per_row
            )
            db.add(Zone(
                name=key,
                polygon=[[int(p[0]), int(p[1])] for p in poly_detect],
                polygon_metric=poly_metric,
                camera_id=camera.id,
                category="parking lot",
            ))

        db.commit()
        n_zones = sum(
            1 for s in slots
            if active.get(f"{s['row']}_{s['col']}", False)
            and s.get("polygon_detection_px") is not None
        )
    finally:
        pass 


def slot_metric_polygon(row: int, col: int,
                        slot_w: float, slot_h: float,
                        row_gap: float = 0.0,
                        files_per_row: int = 1) -> list:
                        
    fi  = row % files_per_row
    lr  = row // files_per_row
    x0  = col * slot_w
    y0  = slot_y_origin(fi, lr, slot_h, row_gap, files_per_row)
    return [
        [x0,          y0         ],
        [x0 + slot_w, y0         ],
        [x0 + slot_w, y0 + slot_h],
        [x0,          y0 + slot_h],
    ]

def slot_y_origin(file_idx: int, logical_row: int,
                  slot_h: float, row_gap: float,
                  files_per_row: int) -> float:
                  
    return logical_row * (files_per_row * slot_h + row_gap) + file_idx * slot_h


@app.get("/ping")
def ping(e):
    print(e)
    return "ping"


class SelectedIds(BaseModel):
    selected_ids: list

@app.get("/camera/selected_data")
def get_selected_Camera_data(data: SelectedIds, db: Session = Depends(get_db)):
    selected_ids = data.selected_ids
    try:
        q = (
            db.query(CameraSource)
            .filter(CameraSource.homography.isnot(None))
            .order_by(CameraSource.id)
        )
        if selected_ids is not None:
            q = q.filter(CameraSource.id.in_(selected_ids))
        cameras = q.all()
        if not cameras:
            return None

        cam_data_list = []
        for camera in cameras:
            cam_name = camera.name
            zones = (
                db.query(Zone)
                .filter(
                    Zone.camera_id == camera.id,
                    Zone.polygon_metric.isnot(None),
                )
                .all()
            )
            if not zones:
                print(f"[WARN] Camera '{cam_name}' has no zones with polygon_metric — skipping.")
                continue

            
            # Build slot list
            slots = []
            for zone in zones:
                try:
                    row, col = map(int, zone.name.split("_"))
                except ValueError:
                    print(f"[WARN] Cannot parse row/col from zone name '{zone.name}' — skipping.")
                    continue
                poly_metric = np.array(zone.polygon_metric, dtype=float)
                centroid = poly_metric.mean(axis=0).tolist()
                slots.append({
                    "key": zone.name,
                    "row": row,
                    "col": col,
                    "polygon_detection_px": zone.polygon,
                    "polygon_metric": zone.polygon_metric,
                    "metric_centroid": centroid,
                    "zone_id": zone.id,
                })

            cam_data_list.append({
                "cam_idx": len(cam_data_list),
                "camera_id": camera.id,
                "uri": camera.uri,
                "cam_name": cam_name,
                "slots": slots,
            })

        return cam_data_list
    finally:
        pass



class MergeSaveData(BaseModel):
    step_results: list
    selected_ids: list
    

@app.post("/merge/save")
def save_merge(data: MergeSaveData):
    print(isinstance(data,MergeSaveData))
    selected_ids = data.selected_ids
    step_results = data.step_results
    _,output_dir = get_config_paths()
    topdown_path= output_dir / "merged_topdown.png"
    db = next(get_db())
    cam_data_list = get_selected_Camera_data(SelectedIds(selected_ids=selected_ids), db)
    n = len(cam_data_list)

    if n == 1:
        print("[merge_cameras] Only one camera — generating single-cam top-down.")
        M_id = np.array([[1., 0., 0.], [0., 1., 0.]])
        save_merged_topdown(topdown_path, cam_data_list, [M_id])
        return

    # ------------------------------------------------------------------
    # Estimate transforms
    # ------------------------------------------------------------------
    print("\n[merge_cameras] Estimating transforms…")
    estimate_transforms(step_results, cam_data_list)
    global_transforms = compute_global_transforms(n, step_results)

    # ------------------------------------------------------------------
    # Ensure per-camera top-downs exist
    # ------------------------------------------------------------------
    for cam_data, M in zip(cam_data_list, global_transforms):
        ensure_topdown(output_dir, cam_data, [M])

    # ------------------------------------------------------------------
    # Save merged top-down PNG
    # ------------------------------------------------------------------
    save_merged_topdown(topdown_path, cam_data_list, global_transforms)

    # ------------------------------------------------------------------
    # Save merge results to DB
    # ------------------------------------------------------------------
    db = next(get_db())
    save_merge_to_db(step_results, cam_data_list, global_transforms,db)



IOU_MATCH_THRESHOLD = 0.3 

def save_merge_to_db(step_results: list, cam_data_list: list,
                     global_transforms: list, db: Session = Depends(get_db)) -> int:
    """Auto-match all slots via IoU in global metric space, then write to DB.

    Algorithm:
      1. Clear existing mapped_zone_id for every zone of the cameras being processed.
      2. Project each slot's polygon_metric into global metric space using global_transforms.
      3. For every pair of cameras (i, j), find mutual best-IoU matches above
         IOU_MATCH_THRESHOLD and feed them into Union-Find.
      4. Every slot (matched or singleton) ends up in a group.
      5. For each group write one MappedZone (mean global polygon) and update
         mapped_zone_id on all member zones.

    step_results is accepted for API compatibility but not used by this function;
    the matching is driven entirely by geometry.
    """
    
    # ------------------------------------------------------------------
    # Step 1 & 2 — project all slots to global metric space
    # ------------------------------------------------------------------
    # global_slots[cam_idx][key] = {"slot": slot_dict, "poly_global": ndarray}
    global_slots: dict[int, dict[str, dict]] = {}
    all_nodes: list[tuple] = []
    for cam_data in cam_data_list:
        cam_idx = cam_data["cam_idx"]
        M = global_transforms[cam_idx]
        global_slots[cam_idx] = {}
        for slot in cam_data["slots"]:
            poly_local = np.array(slot["polygon_metric"], dtype=float)
            poly_global = apply_affine_pts(M, poly_local)
            global_slots[cam_idx][slot["key"]] = {
                "slot": slot,
                "poly_global": poly_global,
            }
            all_nodes.append((cam_idx, slot["key"]))

    # ------------------------------------------------------------------
    # Step 3 — Union-Find initialisation (every slot is its own group)
    # ------------------------------------------------------------------
    parent: dict = {node: node for node in all_nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path compression (halving)
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # ------------------------------------------------------------------
    # Step 4 — Mutual best-IoU matching across every camera pair
    # ------------------------------------------------------------------
    cam_indices = [d["cam_idx"] for d in cam_data_list]
    n_pairs_found = 0
    for ii, ci in enumerate(cam_indices):
        for cj in cam_indices[ii + 1:]:
            items_i = list(global_slots[ci].items())   # [(key, entry), ...]
            items_j = list(global_slots[cj].items())
            if not items_i or not items_j:
                continue

            # best match for each slot in cam_i → cam_j
            best_of_i: dict[str, tuple[float, str | None]] = {}
            for ki, vi in items_i:
                best_iou, best_kj = 0.0, None
                for kj, vj in items_j:
                    iou = polygon_iou(vi["poly_global"], vj["poly_global"])
                    if iou > best_iou:
                        best_iou, best_kj = iou, kj
                best_of_i[ki] = (best_iou, best_kj)

            # best match for each slot in cam_j → cam_i
            best_of_j: dict[str, tuple[float, str | None]] = {}
            for kj, vj in items_j:
                best_iou, best_ki = 0.0, None
                for ki, vi in items_i:
                    iou = polygon_iou(vj["poly_global"], vi["poly_global"])
                    if iou > best_iou:
                        best_iou, best_ki = iou, ki
                best_of_j[kj] = (best_iou, best_ki)

            # Accept only mutual best matches above threshold
            for ki, (iou_ij, kj) in best_of_i.items():
                if kj is None or iou_ij < IOU_MATCH_THRESHOLD:
                    continue
                iou_ji, ki_back = best_of_j.get(kj, (0.0, None))
                if ki_back == ki:   # mutual agreement
                    union((ci, ki), (cj, kj))
                    n_pairs_found += 1

    
    # ------------------------------------------------------------------
    # Step 5 — Collect groups (singletons included)
    # ------------------------------------------------------------------
    groups: dict = {}
    for node in all_nodes:
        groups.setdefault(find(node), []).append(node)
    all_groups = list(groups.values())

    # ------------------------------------------------------------------
    # Step 6 — Write to DB
    # ------------------------------------------------------------------
    try:
        # Clear existing mapped_zone_id for every zone of the cameras in play
        camera_ids = [d["camera_id"] for d in cam_data_list]
        for cam_id in camera_ids:
            db.query(Zone).filter(Zone.camera_id == cam_id).update(
                {"mapped_zone_id": None}, synchronize_session=False
            )
        db.flush()

        n_saved = 0
        for group in all_groups:
            polygons = []
            zone_ids = []
            for cam_idx, slot_key in group:
                entry = global_slots.get(cam_idx, {}).get(slot_key)
                if entry is None:
                    continue
                polygons.append(entry["poly_global"])
                zone_ids.append(entry["slot"]["zone_id"])

            if not polygons:
                continue

            # Canonicalize vertex order (TL→TR→BR→BL) before averaging so that
            # vertex[i] of cam-A's polygon corresponds to vertex[i] of cam-B's.
            canonical = [_canonical_quad(p) for p in polygons]
            mean_poly = np.mean(canonical, axis=0).tolist()
            mz = MappedZone(polygon_global_metric=mean_poly)
            db.add(mz)
            db.flush()

            for zid in zone_ids:
                z = db.query(Zone).filter(Zone.id == zid).first()
                if z:
                    z.mapped_zone_id = mz.id

            n_saved += 1

        db.commit()
        n_multi = sum(1 for g in all_groups if len(g) > 1)
        print(f"[merge_cameras] DB: {n_saved} MappedZone(s) written "
              f"({n_multi} multi-camera, {n_saved - n_multi} singleton)")
        return n_saved
    finally:
        pass

@app.post("/start/pipeline")
def start_pipe_line(db: Session =Depends(get_db)):
    global _pipeline_proc
    import subprocess
    import sys
    cams = db.query(CameraSource).all()
    videos  = [c.uri for c in cams]
    cmd = [sys.executable, "-m", "processing.run"]
    if videos:
        cmd += ["--video"] + videos
    _pipeline_proc = subprocess.Popen(cmd, cwd=Path(__file__).parent.parent)
    return {"status": "pipeline started", "streams": len(videos) if videos else 1}

@app.get("/merged")
def has_been_merged(db: Session = Depends(get_db)):
    r=db.query(func.count(MappedZone))

    return r!=0
