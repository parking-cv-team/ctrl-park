import cv2
import time
from queue import Queue
import os
from dotenv import load_dotenv

from typing import List
from db.models import CameraSource, Zone
from .zones import _load_zone_config_from_db
from .draw_zones import draw_parking_from_scratch
from pathlib import Path


load_dotenv()




def get_parking_zones(uri, frame):
    try:
        zone_config: CameraSource = _load_zone_config_from_db(source=uri)
    except FileNotFoundError:
        print(
            f"[demo_pipeline] No zone config found in DB for {uri}. Drawing zones now."
        )
        draw_parking_from_scratch(uri, frame, None)
        try:
            zone_config = _load_zone_config_from_db(source=uri)
        except FileNotFoundError:
            return None

    if not zone_config.zones:
        print("[demo_pipeline] Zone config has no zones.")
        return None

    return zone_config.zones


def capture_stream(uri: str, out_queue: Queue):
    """Connect to a camera"""
    cap = cv2.VideoCapture(uri)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = float(os.getenv("TARGET_FPS", "3"))
    skip = max(  # if source_fps is unavailable, estimate 30fps and skip to get approximately target_fps
        1, int(round((source_fps if source_fps > 0 else 30) / target_fps))
    )

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {uri}")

    count = 0
    ret, frame = cap.read()

    logging_enabled = os.getenv("USE_LOGGING", "False").lower() == "true"
    max_queue_size = int(os.getenv("MAX_QUEUE_SIZE", "200"))
    queue_threshold = int(
        os.getenv("QUEUE_RESTART_THRESHOLD", str(max_queue_size // 2))
    )
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
                if count % skip == 0:
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
