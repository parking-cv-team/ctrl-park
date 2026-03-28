import argparse
import threading
from queue import Queue
import os
from dotenv import load_dotenv
from .worker import processing_loop
from .camera_ingest import capture_stream

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
