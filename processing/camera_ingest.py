import cv2
import time
from queue import Queue
import os
from dotenv import load_dotenv
from numpy import size

load_dotenv()


def capture_stream(uri: str, out_queue: Queue):
    """Connect to a camera"""
    cap = cv2.VideoCapture(uri)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {uri}")

    count = 0
    logging_enabled = os.getenv("USE_LOGGING", "False").lower() == "true"
    max_queue_size = int(os.getenv("MAX_QUEUE_SIZE", "200"))
    queue_threshold = int(os.getenv("QUEUE_RESTART_THRESHOLD", str(max_queue_size // 2)))
    time_sleep = int(os.getenv("SLEEP_TIME", "1"))

    if logging_enabled: 
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
                ret, frame = cap.read()
                if not ret:
                    break
                out_queue.put((uri, frame, time.time(), count))
                if out_queue.qsize() > max_queue_size:
                    while out_queue.qsize() > queue_threshold:
                        time.sleep(1)
                count += 1
        finally:
            cap.release()

    # signal end, arbitrarily encoded by yours truly, probably needs chaning, so TODO
    out_queue.put((uri, None, None, None))
