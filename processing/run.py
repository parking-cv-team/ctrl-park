import threading
from queue import Queue
import os
from dotenv import load_dotenv
from .worker import processing_loop
from .camera_ingest import capture_stream

load_dotenv()


def main():
    frame_queue = Queue()
    # Example stub camera, replace with valid RTSP/HTTP camera URI
    camera_uri = os.getenv(
        "CAMERA_URI", "video/TestVideo.mp4"
    )  # Otherwise, for a camera via actual protocol: rtsp://example.com/stream

    worker_thread = threading.Thread(
        target=processing_loop, args=(frame_queue,), daemon=True
    )
    worker_thread.start()

    capture_stream(camera_uri, frame_queue)
    print("Capture done, waiting for worker to flush")
    frame_queue.join()


if __name__ == "__main__":
    main()
