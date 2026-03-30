import argparse
import threading
import cv2
from queue import Queue
import os
from dotenv import load_dotenv
from .worker import processing_loop
from .camera_ingest import capture_stream, get_parking_zones

load_dotenv()


def main():
    #   Example stub camera, replace with valid RTSP/HTTP camera URI
    #   camera_uri = os.getenv("CAMERA_URI", "video/TestVideo.mp4")  
    #   Otherwise, for a camera via actual protocol: rtsp://example.com/stream

    # Parse arguments given through CLI or .env file 
    parser = argparse.ArgumentParser(description="Parking detection pipeline")
    parser.add_argument(
        "--video",
        default=os.getenv("CAMERA_URI", "video/TestVideo.mp4"),
        help="Video file or stream URI (default: CAMERA_URI env var or video/TestVideo.mp4)",
    )

    args = parser.parse_args()

    camera_uri = args.video


    # Load zones upfront so the worker can be started before capture begins.
    cap = cv2.VideoCapture(camera_uri)
    if not cap.isOpened():
        raise Exception(f"Could not open stream: {camera_uri}")

    
    zones = get_parking_zones(camera_uri)

    while zones is None:
        zones = get_parking_zones(camera_uri)

    frame_queue = Queue()

    worker_thread = threading.Thread(
        target=processing_loop, args=(frame_queue, zones), daemon=True
    )
    worker_thread.start()

    capture_stream(camera_uri, frame_queue)
    print("Capture done, waiting for worker to flush")

    frame_queue.join()


if __name__ == "__main__":
    main()
