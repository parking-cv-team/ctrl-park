import time
from queue import Queue
import os
from dotenv import load_dotenv
from db import SessionLocal, CameraSource, Zone, Detection, ZoneOccupancy

load_dotenv()


def process_frame(camera_uri, frame, timestamp, frame_id):
    # TODO: this is a placeholder
    h, w = frame.shape[:2]
    return {"h": h, "w": w, "timestamp": timestamp, "frame_id": frame_id}


def processing_loop(in_queue: Queue):
    db = SessionLocal()
    camera_cache = {}

    while True:
        camera_uri, frame, timestamp, frame_id = in_queue.get()
        if frame is None:
            # eos
            break

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

        # TODO upade with actual processing and db storage logic.
        # Commented as demo objects are no longer with us (rip, ProcessedFrame, you will be missed)
        # data_result = process_frame(camera_uri, frame, timestamp, frame_id)
        # pf = SomeDbObjectOrMultiple(data_result)
        # db.add(pf)

        db.commit()
