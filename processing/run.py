import argparse
import threading
import cv2
from queue import Queue, Empty
import os
from pathlib import Path
from dotenv import load_dotenv
from .worker import processing_loop
from .camera_ingest import capture_stream, get_parking_zones

load_dotenv()

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def _resolve_uris(paths: list[str]) -> list[str]:
    """Expand any directory path into its contained video files."""
    uris = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            found = sorted(f for f in path.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS)
            if not found:
                raise FileNotFoundError(f"No video files found in directory: {p}")
            uris.extend(str(f) for f in found)
        else:
            uris.append(p)
    return uris


def main() -> None:
    parser = argparse.ArgumentParser(description="Parking detection pipeline")
    parser.add_argument(
        "--video",
        nargs="+",
        default=None,
        help=(
            "One or more video files / stream URIs. "
            "Also accepts a single string with '|'-separated URIs. "
            "Falls back to CAMERA_URIS (comma-separated) or CAMERA_URI env var."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show live annotated frames per camera (cv2.imshow, main thread).",
    )
    args = parser.parse_args()

    # Resolve URIs
    if args.video:
        if len(args.video) == 1 and "|" in args.video[0]:
            raw_uris = args.video[0].split("|")
        else:
            raw_uris = args.video
    elif os.getenv("CAMERA_URIS"):
        raw_uris = [u.strip() for u in os.getenv("CAMERA_URIS").split(",") if u.strip()]
    else:
        raw_uris = [os.getenv("CAMERA_URI", "video/TestVideo.mp4")]

    camera_uris = _resolve_uris(raw_uris)

    display_queue: "Queue | None" = Queue() if args.debug else None

    # Pre-create windows in the main thread before spawning worker threads.
    # On macOS this is required — windows created from other threads crash.
    if args.debug:
        for uri in camera_uris:
            cv2.namedWindow(uri, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(uri, 640, 360)

    ingest_threads = []
    for uri in camera_uris:
        zones = get_parking_zones(uri)
        while zones is None:
            zones = get_parking_zones(uri)

        frame_queue: Queue = Queue()

        worker_thread = threading.Thread(
            target=processing_loop,
            args=(frame_queue, zones),
            kwargs={"display_queue": display_queue},
            daemon=True,
        )
        worker_thread.start()

        ingest_thread = threading.Thread(
            target=capture_stream,
            args=(uri, frame_queue),
            name=f"ingest-{uri}",
        )
        ingest_thread.start()
        ingest_threads.append(ingest_thread)

    if args.debug:
        while any(t.is_alive() for t in ingest_threads):
            try:
                cam_name, frame = display_queue.get(timeout=0.05)
                cv2.imshow(cam_name, frame)
            except Empty:
                pass
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
    else:
        for t in ingest_threads:
            t.join()

    print("All camera workers have finished.")


if __name__ == "__main__":
    main()
