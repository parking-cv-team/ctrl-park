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
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_queue.put((uri, frame, time.time(), count))
        count += 1

    cap.release()
    # signal end, arbirtrarily encoded by yours truly, probably needs chaning, so TODO
    out_queue.put((uri, None, None, None))
