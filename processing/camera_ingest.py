import cv2
import time
from queue import Queue
import os
from dotenv import load_dotenv

load_dotenv()


def capture_stream(uri: str, out_queue: Queue):
    """Connect to a camera"""
    cap = cv2.VideoCapture(uri)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {uri}")

    count = 0
    log_path = os.getenv("QUEUE_SIZE_LOG_PATH", "queue_size.log")

    logfile = open(log_path, "a", encoding="utf-8", buffering=1)

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
            count += 1
    finally:
        logfile.close()
        cap.release()

    # signal end, arbitrarily encoded by yours truly, probably needs chaning, so TODO
    out_queue.put((uri, None, None, None))
