import os
import cv2
from queue import Queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, cast
from dotenv import load_dotenv
import numpy as np
import supervision as sv
from db import SessionLocal, CameraSource, Zone, Detection, ZoneOccupancy

from .detect_frame import detect_frame_dual, load_models

load_dotenv()

# Load config from .env
DEFAULT_FRAME_RATE = int(os.getenv("FRAME_RATE", "25"))
LOST_TRACK_BUFFER = int(os.getenv("LOST_TRACK_BUFFER", "30"))
MIN_SAVE_INTERVAL = float(os.getenv("MIN_DETECTION_SAVE_INTERVAL", "1.0"))
BBOX_MOVEMENT_THRESHOLD = int(os.getenv("BBOX_MOVEMENT_THRESHOLD", "5"))
OCCUPANCY_TRANSITION_BUFFER = float(os.getenv("OCCUPANCY_TRANSITION_BUFFER", "2"))
OCCUPANT_ABSENCE_THRESHOLD = int(os.getenv("OCCUPANT_ABSENCE_THRESHOLD_SECONDS", "120"))
OLD_DETECTION_CUTOFF = int(os.getenv("OLD_DETECTION_CUTOFF_SECONDS", "10"))

# Global state
detection_tracker: Dict[Tuple[str, int], Dict] = {}
occupancy_timers: Dict[int, Dict] = {}
live_tracker_ids: Dict[int, set] = {}  # zone_id -> set of active tracker_ids


def _build_zone_polygons(zones: List[Zone]) -> List[Tuple[Zone, sv.PolygonZone]]:
    """Convert zones to polygon structures."""
    return [(z, sv.PolygonZone(polygon=np.array(z.polygon, dtype=np.int64))) for z in zones]

def _poly_iou(x1: float, y1: float, x2: float, y2: float,
              zone_polygon: np.ndarray) -> float:
    """IoU between an axis-aligned bbox rectangle and a convex zone polygon."""
    bbox_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    zone_poly  = zone_polygon.astype(np.float32)
    _, inter   = cv2.intersectConvexConvex(bbox_poly, zone_poly)
    inter_area = cv2.contourArea(inter) if inter is not None and len(inter) > 0 else 0.0
    bbox_area  = (x2 - x1) * (y2 - y1)
    zone_area  = cv2.contourArea(zone_poly)
    union_area = bbox_area + zone_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def _get_zone_ids(detections: sv.Detections, zone_polygons: List[Tuple[Zone, sv.PolygonZone]]) -> List[Optional[int]]:
    """Get zone IDs for each detection."""
    if len(detections) == 0:
        return []
    det_zone_ids: List[Optional[int]] = [None] * len(detections)
    for zone_db, poly_zone in zone_polygons:
        mask = poly_zone.trigger(detections)
        for i, inside in enumerate(mask):
            if inside and det_zone_ids[i] is None:
                det_zone_ids[i] = cast(int, zone_db.id)
    return det_zone_ids


def _detection_passes_filters(
    class_name: str, tracker_id: Optional[int], timestamp: float, 
    x1: float, y1: float, x2: float, y2: float
) -> bool:
    """Check if detection passes all filters."""
    if tracker_id is None:
        return True
    
    key = (class_name, tracker_id)
    if key not in detection_tracker:
        detection_tracker[key] = {"timestamp": timestamp, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
        return True
    
    prev = detection_tracker[key]
    
def _get_zone_ids(
    detections: sv.Detections,
    zone_polygons: List[Tuple[Zone, sv.PolygonZone]],
    source: CameraSource,
) -> List[Optional[int]]:
    """Assign each detection to a zone using multi-signal scoring.

    Algorithm (in priority order):

    1. Pre-filter: drop zones with pixel-space IoU(bbox, zone) < 10%.
       If nothing survives → None.

    2. Ground-contact test: project bbox bottom-centre (cx, y2) to metric
       space via H_world = inv(H_parking). If exactly one candidate zone's
       polygon_metric contains the projected point → return that zone
       immediately (fast path, most common case).

    3. Scoring fallback (ambiguous or degenerate projection):
         score = 0.50 * ground_proximity   # 1.0 inside, 1/(1+dist) outside
               + 0.25 * iou * confidence   # penalise low-confidence boxes
               + 0.15 * (zone_y_max / H)   # depth: lower in image = nearer
               + 0.10 * confidence
       Return highest-scoring zone, or None if best score < 0.15.

    4. Sanity check: if projected point is > 60 m from origin, the
       homography is degenerate for this pixel — fall through to step 3
       without the ground-contact signal.
    """
    if len(detections) == 0:
        return []

    # Pre-compute H_world once for this batch
    H_world = None
    if source.homography is not None:
        H = np.array(source.homography, dtype=np.float64)
        try:
            H_world = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            pass

    frame_h = float(source.frame_height or 1080)
    result: List[Optional[int]] = [None] * len(detections)

    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i]
        confidence = float(detections.confidence[i]) if detections.confidence is not None else 1.0
        cx = (x1 + x2) / 2.0

        # ── Ground contact point in metric space ─────────────────────────
        pt_metric: Optional[Tuple[float, float]] = None
        degenerate = False
        if H_world is not None:
            p = H_world @ np.array([cx, float(y2), 1.0])
            if abs(p[2]) > 1e-9:
                p = p / p[2]
                if np.linalg.norm(p[:2]) <= 60.0:
                    pt_metric = (float(p[0]), float(p[1]))
                else:
                    degenerate = True  # sanity check failed

        # ── Step 1: pre-filter by IoU ─────────────────────────────────────
        candidates: List[Tuple[Zone, float]] = []
        for zone_db, _ in zone_polygons:
            zone_px = np.array(zone_db.polygon, dtype=np.float32)
            iou = _poly_iou(x1, y1, x2, y2, zone_px)
            if iou >= 0.10:
                candidates.append((zone_db, iou))

        if not candidates:
            continue  # result[i] stays None

        # ── Step 2: unambiguous ground-contact match ──────────────────────
        if pt_metric is not None and not degenerate:
            matches = [
                zd for zd, _ in candidates
                if zd.polygon_metric is not None
                and cv2.pointPolygonTest(
                    np.array(zd.polygon_metric, dtype=np.float32),
                    pt_metric, False,
                ) >= 0
            ]
            if len(matches) == 1:
                result[i] = matches[0].id
                continue

        # ── Step 3: scoring fallback ──────────────────────────────────────
        best_id: Optional[int] = None
        best_score = 0.15  # minimum acceptance threshold

        for zone_db, iou in candidates:
            # Ground proximity
            if pt_metric is not None and not degenerate and zone_db.polygon_metric is not None:
                poly_m  = np.array(zone_db.polygon_metric, dtype=np.float32)
                signed  = cv2.pointPolygonTest(poly_m, pt_metric, True)
                ground_prox = 1.0 if signed >= 0 else 1.0 / (1.0 + abs(signed))
            else:
                ground_prox = 0.5  # neutral when metric data unavailable

            # Depth: higher pixel y → closer to camera
            zone_px  = np.array(zone_db.polygon, dtype=np.float32)
            y_prox   = float(zone_px[:, 1].max()) / frame_h

            score = (0.50 * ground_prox
                   + 0.25 * iou * confidence
                   + 0.15 * y_prox
                   + 0.10 * confidence)

            if score > best_score:
                best_score = score
                best_id = zone_db.id

        result[i] = best_id

    return result


def _persist_detections(
    db, source: CameraSource, class_name: str, tracked: sv.Detections, 
    zone_polygons: List[Tuple[Zone, sv.PolygonZone]], timestamp: float
) -> None:
    """Persist detections to database."""
    if len(tracked) == 0:
        return
    
    det_zone_ids = _get_zone_ids(tracked, zone_polygons, source)
    dt = datetime.utcfromtimestamp(timestamp)
    detections_to_add = {}
    
    for i in range(len(tracked)):
        x1, y1, x2, y2 = tracked.xyxy[i]
        
        confidence = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0 # ?
        tracker_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else None

        # Filter duplicate/noise
        if not _detection_passes_filters(class_name, tracker_id, timestamp, x1, y1, x2, y2):
            continue
        
        # Create detection
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        det_row = Detection(
            camera_id=source.id,
            timestamp=dt,
            tracker_id=tracker_id,
            class_id=int(tracked.class_id[i]) if tracked.class_id is not None else 0,
            class_name=class_name,
            confidence=confidence,
            x1=x1, y1=y1, x2=x2, y2=y2,
            cx=cx, cy=cy,
            bbox_width=x2 - x1,
            bbox_height=y2 - y1,
            bbox_area=(x2 - x1) * (y2 - y1),
            zone_id=det_zone_ids[i],
        )
        detections_to_add[det_row] = None
        
        # Handle zone occupancy
        zone_id = det_zone_ids[i]
        if zone_id is not None:
            # Track live tracker IDs per zone
            if zone_id not in live_tracker_ids:
                live_tracker_ids[zone_id] = set()
            if tracker_id is not None:
                live_tracker_ids[zone_id].add(tracker_id)
            
            if zone_id not in occupancy_timers:
                occupancy_timers[zone_id] = {
                    "tracker_id": tracker_id,
                    "new_tracker_id": None,
                    "new_tracker_timestamp": None,
                }
                detections_to_add[det_row] = ZoneOccupancy(
                    detection_id=det_row.id,
                    zone_id=zone_id,
                    tracker_id=tracker_id,
                )
            else:
                # Handle tracker transition
                state = occupancy_timers[zone_id]
                if (state["tracker_id"] != tracker_id and state["new_tracker_id"] != tracker_id):
                    state["new_tracker_id"] = tracker_id
                    state["new_tracker_timestamp"] = timestamp
                elif state["tracker_id"] == tracker_id:
                    state["new_tracker_id"] = None
                    state["new_tracker_timestamp"] = None
                
                if (state["new_tracker_timestamp"] is not None and 
                    timestamp - state["new_tracker_timestamp"] > OCCUPANCY_TRANSITION_BUFFER):
                    state["tracker_id"] = tracker_id
                    state["new_tracker_id"] = None
                    state["new_tracker_timestamp"] = None
                    detections_to_add[det_row] = ZoneOccupancy(
                        detection_id=det_row.id,
                        zone_id=zone_id,
                        tracker_id=tracker_id,
                    )
                
                # Update last seen timestamp
                if state["tracker_id"] == tracker_id:
                    state["last_seen_timestamp"] = timestamp
                    state["last_detection"] = det_row
    
    # Save to DB
    for det_row in detections_to_add:
        db.add(det_row)
    db.flush()
    
    for det_row, zone_occ in detections_to_add.items():
        if zone_occ is not None:
            zone_occ.detection_id = det_row.id
            db.add(zone_occ)



def _check_departures(db, timestamp: float) -> None:
    """Check for vehicles that left and mark slots empty."""
    for zone_id, state in list(occupancy_timers.items()):
        if state.get("tracker_id") is None:
            continue
        
        last_seen = state.get("last_seen_timestamp")
        if last_seen is None or (timestamp - last_seen) < OCCUPANT_ABSENCE_THRESHOLD:
            continue
        
        # Mark slot as empty with departure event
        last_det = state.get("last_detection")
        if last_det is not None:
            # Create a departure Detection record using last known bbox
            dt = datetime.utcfromtimestamp(timestamp)
            departure_record = Detection(
                camera_id=last_det.camera_id,
                timestamp=dt,
                tracker_id=state.get("tracker_id"),
                class_id=last_det.class_id,
                class_name=last_det.class_name,
                confidence=1.0,
                event_type="departure",
                x1=last_det.x1, y1=last_det.y1, x2=last_det.x2, y2=last_det.y2,
                cx=last_det.cx, cy=last_det.cy,
                bbox_width=last_det.bbox_width,
                bbox_height=last_det.bbox_height,
                bbox_area=last_det.bbox_area,
                zone_id=zone_id,
            )
            db.add(departure_record)
            db.flush()
            db.add(ZoneOccupancy(
                detection_id=departure_record.id,
                zone_id=zone_id,
                tracker_id=None,
            ))
        
        # Remove from live trackers
        if zone_id in live_tracker_ids:
            live_tracker_ids[zone_id].discard(state.get("tracker_id"))
        
        state["tracker_id"] = None
        state["last_seen_timestamp"] = None
        state["new_tracker_id"] = None
        state["new_tracker_timestamp"] = None

def processing_loop(in_queue: Queue, in_zones: List[Zone]):
    """Main processing loop."""
    db = SessionLocal()
    camera_cache = {}
    model_car, model_ped = load_models()
    tracker_car = None
    tracker_ped = None
    
    zone_polygons = _build_zone_polygons(in_zones)
    
    while True:
        camera_uri, frame, timestamp, frame_id, source_fps = in_queue.get()
        if frame is None:
            break
        
        # Initialize trackers with actual stream FPS on first frame
        if tracker_car is None:
            frame_rate = max(1, int(source_fps)) if source_fps and source_fps > 0 else DEFAULT_FRAME_RATE
            tracker_car = sv.ByteTrack(frame_rate=frame_rate, lost_track_buffer=LOST_TRACK_BUFFER)
            tracker_ped = sv.ByteTrack(frame_rate=frame_rate, lost_track_buffer=LOST_TRACK_BUFFER)
        
        # Get or create camera source
        if camera_uri not in camera_cache:
            source = db.query(CameraSource).filter(CameraSource.uri == camera_uri).first()
            if not source:
                source = CameraSource(name=camera_uri, uri=camera_uri)
                db.add(source)
                db.commit()
                db.refresh(source)
            camera_cache[camera_uri] = source
        source = camera_cache[camera_uri]
        
        # Detect and track
        raw = detect_frame_dual(frame, model_car, model_ped)
        tracked_cars = tracker_car.update_with_detections(raw["cars"]) # pyright: ignore[reportOptionalMemberAccess] Not none
        tracked_peds = tracker_ped.update_with_detections(raw["pedestrian"]) # pyright: ignore[reportOptionalMemberAccess] Not none
        
        # Persist detections
        _persist_detections(db, source, "car", tracked_cars, zone_polygons, timestamp)
        _persist_detections(db, source, "pedestrian", tracked_peds, zone_polygons, timestamp)
        
        # Check for departures and cleanup
        _check_departures(db, timestamp)
        cutoff = timestamp - OLD_DETECTION_CUTOFF
        for key in list(detection_tracker.keys()):
            if detection_tracker[key]["timestamp"] <= cutoff:
                del detection_tracker[key]
        
        db.commit()
        in_queue.task_done()

    print("PIPELINE FINISHED")
    db.close()