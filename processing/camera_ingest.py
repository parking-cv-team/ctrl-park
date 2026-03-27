import cv2
import time
from queue import Queue
import os
from dotenv import load_dotenv
from .zones import ZoneConfig,load_zone_config
from .draw_zones import draw_parking_from_scratch
from pathlib import Path


load_dotenv()


def get_parking_zones(uri, frame):

    zones_path = Path("parking_slots") / (Path(uri).stem + ".json")

    if not zones_path.exists():
        print(f"[demo_pipeline] No zone config found at {zones_path}. Drawing zones now.")
        draw_parking_from_scratch(uri, frame, str(zones_path))
        if not zones_path.exists():
            return None

    zone_config: ZoneConfig = load_zone_config(zones_path)

    if not zone_config.zones:
        print("[demo_pipeline] Zone config has no zones.")
        return None

    return zone_config.zones

def capture_stream(uri: str, out_queue: Queue):
    """Connect to a camera"""
    cap = cv2.VideoCapture(uri)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {uri}")

    count = 0
    ret, frame = cap.read()
    if ret:
        get_parking_zones(uri,frame)
        out_queue.put((uri, frame, time.time(), count))
        count += 1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_queue.put((uri, frame, time.time(), count))
        count += 1

    cap.release()
    # signal end, arbirtrarily encoded by yours truly, probably needs chaning, so TODO
    out_queue.put((uri, None, None, None))
