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
    
    logging_enabled = os.getenv("USE_LOGGING", "False").lower() == "true"
    max_queue_size = int(os.getenv("MAX_QUEUE_SIZE", "200"))
    queue_threshold = int(os.getenv("QUEUE_RESTART_THRESHOLD", str(max_queue_size // 2)))
    time_sleep = float(os.getenv("SLEEP_TIME", "1.0"))

    if logging_enabled: 
        # logging for debugging purposes, should be removed in the final version
        log_path = os.getenv("QUEUE_SIZE_LOG_PATH", "queue_size.log")
        logfile = open(log_path, "w", encoding="utf-8", buffering=1)

        def _log_queue_size(size):
            logfile.write(f"{time.time():.3f},{uri},{size}\n")
            logfile.flush()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out_queue.put((uri, frame, time.time(), count))
                _log_queue_size(out_queue.qsize())
                if out_queue.qsize() > max_queue_size:
                    while out_queue.qsize() > queue_threshold:
                        time.sleep(time_sleep)
                count += 1
        finally:
            logfile.close()
            cap.release()
    else:
        try:
            while True:
                # Queue manager: checks for the queue size as it gets filled up, if it's too big then the script 
                # waits for a certain amount of time (as in defined in your .env file or as a default it's 1 second)
                ret, frame = cap.read()
                if not ret:
                    break
                out_queue.put((uri, frame, time.time(), count))
                if out_queue.qsize() > max_queue_size:
                    while out_queue.qsize() > queue_threshold:
                        time.sleep(time_sleep)
                count += 1
        finally:
            cap.release()

    # signal end, arbitrarily encoded by yours truly, probably needs chaning, so TODO
    out_queue.put((uri, None, None, None))
