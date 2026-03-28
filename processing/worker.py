import time
from queue import Queue
import os
from datetime import datetime, timedelta
from typing import List
from dotenv import load_dotenv
import numpy as np
import supervision as sv
from db import SessionLocal, CameraSource, Zone, Detection, ZoneOccupancy

from .detect_frame import detect_frame_dual, load_models

#TODO: see camera_ingest needs to be coherent
_FRAME_RATE = 25
_LOST_TRACK_BUFFER = 30

load_dotenv()

def _build_zone_polygons(in_zones: List[Zone]) -> list[tuple[Zone, sv.PolygonZone]]:
    """Convert Zone objects to (Zone, PolygonZone) pairs."""
    result = []
    for z in in_zones:
        polygon = np.array(z.polygon, dtype=np.int32)
        result.append((z, sv.PolygonZone(polygon=polygon)))
    return result


def _zone_for_each_detection(
    detections: sv.Detections,
    zone_polygons: list[tuple[Zone, sv.PolygonZone]],
) -> list[int | None]:
    """Return a list of zone_id (or None) for each detection in order."""
    if len(detections) == 0:
        return []
    det_zone_ids: list[int | None] = [None] * len(detections)
    for zone_db, poly_zone in zone_polygons:
        mask = poly_zone.trigger(detections)
        for i, inside in enumerate(mask):
            if inside and det_zone_ids[i] is None:
                det_zone_ids[i] = zone_db.id
    return det_zone_ids


detection_tracker = {}

occupancy_timers = {}





def process_frame(camera_uri, detections, occupancies, timestamp, frame_id):
    
    #clear old trackers from object that have left the scene
    cutoff = datetime.utcnow() - timedelta(seconds=10)
    for tid in list(detection_tracker.keys()):
        if detection_tracker[tid]["timestamp"] <= cutoff:
            del detection_tracker[tid]


def _persist_detections(
    db,
    source: CameraSource,
    class_name: str,
    tracked: sv.Detections,
    zone_polygons: list[tuple[Zone, sv.PolygonZone]],
    timestamp: float,
    detection_tracker: dict,
    *,
    global_id: int | None = None,
    world_x: float | None = None,
    world_y: float | None = None,
) -> None:

    if len(tracked) == 0:
        return

    det_zone_ids = _zone_for_each_detection(tracked, zone_polygons)
    dt = datetime.utcfromtimestamp(timestamp)
    
    detections_to_add = {}

    for i in range(len(tracked)):
        x1, y1, x2, y2 = tracked.xyxy[i]
        confidence = float(tracked.confidence[i])
        tracker_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else None

        # filter 1: confidence threshold
        if confidence < 0.75:
            continue

        # filter 2 & 3: tracker state check
        if tracker_id is not None:
            key = (class_name, tracker_id)  

            if key not in detection_tracker:
                detection_tracker[key] = {
                    "timestamp": timestamp,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                }
            else:
                prev = detection_tracker[key]

                # filter 2: less than 1 second since last saved
                if (timestamp - prev["timestamp"]) < 1.0:
                    continue

                # filter 3: bbox hasn't moved enough
                if not (
                    abs(prev["x1"] - x1) > 10 or
                    abs(prev["y1"] - y1) > 10 or
                    abs(prev["x2"] - x2) > 10 or
                    abs(prev["y2"] - y2) > 10
                ):
                    continue

                # passed both checks — update state
                detection_tracker[key] = {
                    "timestamp": timestamp,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                }

        # write to DB
        zone_id = det_zone_ids[i]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        det_row = Detection(
            camera_id=source.id,
            timestamp=dt,
            tracker_id=tracker_id,
            global_id=global_id,
            world_x=world_x,
            world_y=world_y,
            class_id=int(tracked.class_id[i]) if tracked.class_id is not None else 0,
            class_name=class_name,
            confidence=confidence,
            x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
            cx=float(cx), cy=float(cy),
            bbox_width=float(x2 - x1),
            bbox_height=float(y2 - y1),
            bbox_area=float((x2 - x1) * (y2 - y1)),
            zone_id=zone_id,
        
        )


       
        detections_to_add[det_row] = None
     


        if det_row.zone_id not in occupancy_timers:
            occupancy_timers[det_row.zone_id] = {
                "tracker_id": det_row.tracker_id,
                "new_tracker_id": None,
                "new_tracker_timestamp": None,
            }
            detections_to_add[det_row] = ZoneOccupancy(
                detection_id=det_row.id,
                zone_id=zone_id,
                tracker_id=tracker_id,
            )   
          
            continue
        if (
            occupancy_timers[det_row.zone_id]["tracker_id"] != det_row.tracker_id
            and occupancy_timers[det_row.zone_id]["new_tracker_id"] != det_row.tracker_id
        ):
            occupancy_timers[det_row.zone_id]["new_detection"] = det_row
        elif occupancy_timers[det_row.zone_id]["tracker_id"] == det_row.tracker_id:
            occupancy_timers[det_row.zone_id]["new_detection"].tracker_id = None
            occupancy_timers[det_row.zone_id]["new_detection"].timestamp = None
        if occupancy_timers[det_row.zone_id]["new_detection"].tracker_timestamp and det_row.timestamp -occupancy_timers[
            det_row.zone_id
        ]["new_detection"].tracker_timestamp > timedelta(minutes=2):
            occupancy_timers[det_row.zone_id]["tracker_id"] = det_row.tracker_id
            occupancy_timers[det_row.zone_id]["new_detection"].tracker_id = None
            occupancy_timers[det_row.zone_id]["new_detection"].tracker_timestamp = None
            detections_to_add[det_row] = ZoneOccupancy(
                detection_id=det_row,
                zone_id=zone_id,
                tracker_id=tracker_id,
            )   
            continue
    
        # ── ADDED: whenever we confirm the current occupant is still present,                                                                                                                
        # record the exact moment and keep a reference to their Detection row.
        # This is what _check_departures will later compare against to decide                                                                                                                 
        # whether the car has been gone long enough to call the slot free.                                                                                                                    
        if (                                                                                                                                                                                  
              det_row.zone_id is not None                                                                                                                                                       
              and det_row.zone_id in occupancy_timers                                                                                                                                           
              and occupancy_timers[det_row.zone_id].get("tracker_id") == det_row.tracker_id
          ):                                                                                                                                                                                    
              occupancy_timers[det_row.zone_id]["last_seen_timestamp"] = timestamp
              occupancy_timers[det_row.zone_id]["last_detection"] = det_row
    
    for det_row in detections_to_add:
        db.add(det_row)
    db.flush()

    for det_row, zone_occ in detections_to_add.items():
        if zone_occ is not None:
            zone_occ.detection_id = det_row.id
            db.add(zone_occ)



def _check_departures(db, timestamp: float) -> None:
    """Check every tracked zone for cars that have silently left.

    When a zone's current occupant has not been seen for >= 2 minutes we
    consider the slot empty. A ZoneOccupancy row with tracker_id=None is
    written to signal the change, and the occupant state is reset so the
    next vehicle that arrives will be treated as a fresh occupancy.

    Must be called once per frame, after _persist_detections for all classes,
    so that last_seen_timestamp is up to date before this check runs.
    """
    for zone_id, state in list(occupancy_timers.items()):

        if state.get("tracker_id") is None:
            continue

        last_seen = state.get("last_seen_timestamp")
        if last_seen is None:
            continue

        if (timestamp - last_seen) < 120.0:
            continue

        last_det = state.get("last_detection")
        if last_det is not None:
            db.add(last_det)
            db.flush()
            db.add(ZoneOccupancy(
                detection_id=last_det.id,
                zone_id=zone_id,
                tracker_id=None,
            ))

        occupancy_timers[zone_id]["tracker_id"] = None
        occupancy_timers[zone_id]["last_seen_timestamp"] = None
        occupancy_timers[zone_id].pop("last_detection", None)
        occupancy_timers[zone_id]["new_tracker_id"] = None
        occupancy_timers[zone_id]["new_tracker_timestamp"] = None



def processing_loop(in_queue: Queue, in_zones: List[Zone]):
    db = SessionLocal()
    camera_cache = {}
    model_car, model_ped = load_models()
    tracker_car = sv.ByteTrack(frame_rate=_FRAME_RATE, lost_track_buffer=_LOST_TRACK_BUFFER)
    tracker_ped = sv.ByteTrack(frame_rate=_FRAME_RATE, lost_track_buffer=_LOST_TRACK_BUFFER)
    zone_polygons = _build_zone_polygons(in_zones)
    occupancy_state = {}       
    detection_tracker = {}    
    
    while True:
        camera_uri, frame, timestamp, frame_id = in_queue.get()
        if frame is None:
            break

        if camera_uri not in camera_cache:
            source = db.query(CameraSource).filter(CameraSource.uri == camera_uri).first()
            if not source:
                source = CameraSource(name=camera_uri, uri=camera_uri)
                db.add(source)
                db.commit()
                db.refresh(source)
            camera_cache[camera_uri] = source
        source = camera_cache[camera_uri]

        # Detect
        raw = detect_frame_dual(frame, model_car, model_ped)
        

        # Track
        tracked_cars = tracker_car.update_with_detections(raw["cars"])
        tracked_peds = tracker_ped.update_with_detections(raw["pedestrian"])

        # Persist — pass detection_tracker, use tracked_cars/tracked_peds (not the old wrong names)
        _persist_detections(db, source, "car", tracked_cars, zone_polygons,
                            timestamp, detection_tracker)
        _persist_detections(db, source, "pedestrian", tracked_peds, zone_polygons,
                            timestamp, detection_tracker)

        # ── ADDED: after persisting all detections for this frame, check
        # whether any zone's occupant has been absent for >= 2 minutes and
        # write a "slot is now free" ZoneOccupancy row if so.
        _check_departures(db, timestamp)

        db.commit()
        in_queue.task_done()