import argparse
import threading
from queue import Queue
import os
from dotenv import load_dotenv
from .worker import processing_loop
from .camera_ingest import capture_stream
from .draw_zones import launch as launch_zone_drawer

load_dotenv()


def main():
    #   Example stub camera, replace with valid RTSP/HTTP camera URI
    #   camera_uri = os.getenv("CAMERA_URI", "video/TestVideo.mp4")  
    #   Otherwise, for a camera via actual protocol: rtsp://example.com/stream

    parser = argparse.ArgumentParser(description="Parking detection pipeline")
    parser.add_argument(
        "--video",
        default=os.getenv("CAMERA_URI", "video/TestVideo.mp4"),
        help="Video file or stream URI (default: CAMERA_URI env var or video/TestVideo.mp4)",
    )
    args = parser.parse_args()
    camera_uri = args.video

    # Zone annotation — blocks until the user saves the JSON or quits
    zones_path = launch_zone_drawer(camera_uri)
    if zones_path is None:
        print("[run] No zones saved. Aborting pipeline.")
        return
    print(f"[run] Zones loaded from {zones_path}. Starting pipeline...")


    frame_queue = Queue()
    worker_thread = threading.Thread(
        target=processing_loop, args=(frame_queue,), daemon=True
    )
    worker_thread.start()

    capture_stream(camera_uri, frame_queue)
    print("Capture done, waiting for worker to flush")
    frame_queue.join()


if __name__ == "__main__":
    main()
