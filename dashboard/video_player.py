import cv2
import time
import numpy as np
import sys
import os
import requests

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


_ZONE_COLORS: list[tuple[int, int, int]] = [
    (0, 255, 0),  # green
    (0, 165, 255),  # orange
    (255, 0, 0),  # blue
    (0, 0, 255),  # red
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
    (128, 0, 128),  # purple
]

def draw_zones(frame: np.ndarray, zones: list) -> np.ndarray:
    """Render zones onto a copy of canvas."""
    overlay = frame.copy()
    out = frame.copy()

    if zones == None:
        return out

    for idx, zone in enumerate(zones):
        color = _ZONE_COLORS[idx % len(_ZONE_COLORS)]
        pts = np.array(zone['polygon']).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)

        alpha = 0.2
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

        centroid = np.array(zone['polygon']).mean(axis=0).astype(int)
        cv2.putText(
            out,
            str(zone['name']),
            tuple(centroid),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    return out

def run_opencv_window(rtsp_url, zones):
    cap = cv2.VideoCapture(rtsp_url)
    cv2.namedWindow("RTSP Stream", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        
        frame = draw_zones(frame, zones)

        cv2.imshow("RTSP Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_zones_to_draw(camera_id):
    try:
        response = requests.get(
            f"{API_BASE}/analytics/zones/poly", params={"camera_id": camera_id}
        )
        response.raise_for_status()
        rows = response.json()
        return list(rows)

    except Exception as e:
        print("DASHBOARD ERROR: COULD NOT FETCH ZONES FOR CAMERA", camera_id)

if __name__ == "__main__":
    rtsp_url = sys.argv[1]
    camera_id = int(sys.argv[2])

    zones = get_zones_to_draw(camera_id)

    run_opencv_window(rtsp_url, zones)