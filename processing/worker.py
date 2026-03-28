import time
from queue import Queue
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from db import SessionLocal, CameraSource, Zone, Detection, ZoneOccupancy

from .detect_frame import detect_frame_dual, load_models

load_dotenv()

detection_tracker = {}
def filter_detections(detections):
    detection_results = []

    for det in detections:
        if det.confidence < 0.75:
            continue
        if det.tracker_id not in detection_tracker:
            detection_tracker[det.tracker_id] = {
                "timestamp": det.timestamp,
                "x1": det.x1,
                "y1": det.y1,
                "x2": det.x2,
                "y2": det.y2,
            }
            detection_results.append(det)
            continue
        if (
            detection_tracker[det.tracker_id]["timestamp"] - det.timestamp
        ).seconds() < 1:
            continue
        if (
            abs(detection_tracker[det.tracker_id]["x1"] - det.x1) > 10
            or abs(detection_tracker[det.tracker_id]["y1"] - det.y1) > 10
            or abs(detection_tracker[det.tracker_id]["x2"] - det.x2) > 10
            or abs(detection_tracker[det.tracker_id]["y2"] - det.y2) > 10
        ):
            detection_tracker[det.tracker_id] = {
                "timestamp": det.timestamp,
                "x1": det.x1,
                "y1": det.y1,
                "x2": det.x2,
                "y2": det.y2,
            }
            detection_results.append(det)
            continue
    return detection_results


occupancy_timers = {}
def filter_occupancies(occupancies):
    occupancy_results = []
    for occ in occupancies:
        if occ.zone_id not in occupancy_timers:
            occupancy_timers[occ.zone_id] = {
                "tracker_id": occ.tracker_id,
                "new_tracker_id": None,
                "new_tracker_timestamp": None,
            }
            occupancy_results.append(occ)
            continue
        if (
            occupancy_timers[occ.zone_id]["tracker_id"] != occ.tracker_id
            and occupancy_timers[occ.zone_id]["new_tracker_id"] != occ.tracker_id
        ):
            occupancy_timers[occ.zone_id]["new_tracker_id"] = occ.tracker_id
            occupancy_timers[occ.zone_id]["new_tracker_timestamp"] = occ.timestamp
        elif occupancy_timers[occ.zone_id]["tracker_id"] == occ.tracker_id:
            occupancy_timers[occ.zone_id]["new_tracker_id"] = None
            occupancy_timers[occ.zone_id]["new_tracker_timestamp"] = None
        if occupancy_timers[occ.zone_id]["new_tracker_timestamp"] and occupancy_timers[
            occ.zone_id
        ]["new_tracker_timestamp"] - occ.timestamp < timedelta(minutes=2):
            occupancy_timers[occ.zone_id]["tracker_id"] = occ.tracker_id
            occupancy_timers[occ.zone_id]["new_tracker_id"] = None
            occupancy_timers[occ.zone_id]["new_tracker_timestamp"] = None
            occupancy_results.append(occ) #maybe offset the 2 minutes by altering the timestamp?
            continue

    return occupancy_results


def process_frame(camera_uri, detections, occupancies, timestamp, frame_id):
    
    #clear old trackers from object that have left the scene
    cutoff = datetime.utcnow() - timedelta(seconds=10)
    for tid in list(detection_tracker.keys()):
        if detection_tracker[tid]["timestamp"] <= cutoff:
            del detection_tracker[tid]


def processing_loop(in_queue: Queue):
    db = SessionLocal()
    camera_cache = {}

    # load models
    model_car, model_ped = load_models()

    while True:
        camera_uri, frame, timestamp, frame_id = in_queue.get()

        if frame is None:
            # eos
            break
        
        """
        if camera_uri not in camera_cache:
            source = (
                db.query(CameraSource).filter(CameraSource.uri == camera_uri).first()
            )
            if not source:
                source = CameraSource(name=camera_uri, uri=camera_uri)
                db.add(source)
                db.commit()
                db.refresh(source)
            camera_cache[camera_uri] = source

        source = camera_cache[camera_uri]
        """

        # TODO upade with actual processing and db storage logic.

        # Predict and obtain two sv.Detection objects
        car_detection, pedestrian_detection = detect_frame_dual(frame, model_car, model_ped, 
                                                                arg_car={'verbose': False},
                                                                arg_ped={'verbose':False})
        

        # Filter and add ID according to the function filter_detections
        # filtered_car_detections = filter_detections(car_detection)
        # filtered_pedestrian_detections = filter_detections(pedestrian_detection)

        

        # do stuff with the sv.Detection objects like tracking or parkign lot detection et cetera

        # Commented as demo objects are no longer with us (rip, ProcessedFrame, you will be missed)
        # data_result = process_frame(camera_uri, frame, timestamp, frame_id)
        # pf = SomeDbObjectOrMultiple(data_result)
        # db.add(pf)

        db.commit()
