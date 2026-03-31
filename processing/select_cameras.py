"""
select_cameras.py — Interactive grid UI to pick which cameras to include in merge.

Usage (standalone):
    python -m processing.select_cameras

As module:
    from processing.select_cameras import select_cameras
    ids = select_cameras()   # returns list[int] of selected camera IDs
"""

import sys
import math
import os

# Disable GStreamer before cv2 is imported — it crashes on VideoCapture on some systems
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "0")

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

THUMB_W = 320
THUMB_H = 180
PADDING = 12
BORDER  = 4
LABEL_H = 28
COLS    = 3  # max columns in grid

SEL_COLOR   = (60, 220, 60)    # green border when selected
UNSEL_COLOR = (80, 80, 80)     # grey border when unselected
BG_COLOR    = (30, 30, 30)
TEXT_COLOR  = (220, 220, 220)
BTN_COLOR   = (50, 130, 220)
BTN_HOVER   = (80, 160, 255)

CELL_W = THUMB_W + 2 * BORDER + 2 * PADDING
CELL_H = THUMB_H + 2 * BORDER + 2 * PADDING + LABEL_H


def _grab_frame(uri: str) -> np.ndarray:
    """Capture a thumbnail frame in an isolated subprocess.

    Running VideoCapture in a subprocess prevents GStreamer / libav from
    corrupting the parent process's display state, which would cause a
    segfault when merge_cameras later tries to open its own OpenCV window.
    """
    import subprocess
    import sys

    script = (
        "import os; "
        "os.environ.setdefault('OPENCV_VIDEOIO_PRIORITY_GSTREAMER', '0'); "
        "import cv2, sys; "
        f"cap = cv2.VideoCapture({repr(uri)}); "
        "ok, f = cap.read() if cap.isOpened() else (False, None); "
        "cap.release(); "
        "sys.stdout.buffer.write("
        "    f'{f.shape[0]} {f.shape[1]} {f.shape[2]}\\n'.encode() + f.tobytes()"
        ") if (ok and f is not None) else None"
    )
    fallback = np.zeros((THUMB_H, THUMB_W, 3), np.uint8)
    try:
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=8,
        )
        if r.returncode == 0 and r.stdout:
            header, _, data = r.stdout.partition(b"\n")
            h, w, c = (int(x) for x in header.split())
            if len(data) == h * w * c:
                frame = np.frombuffer(data, np.uint8).reshape((h, w, c))
                return cv2.resize(frame, (THUMB_W, THUMB_H))
    except Exception:
        pass

    cv2.putText(fallback, "unavailable", (20, THUMB_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 200), 2)
    return fallback


def _load_cameras():
    from db import SessionLocal
    from db.models import CameraSource
    db = SessionLocal()
    try:
        cams = (
            db.query(CameraSource)
            .filter(CameraSource.homography.isnot(None))
            .order_by(CameraSource.id)
            .all()
        )
        return [{"id": c.id, "name": c.name or str(c.id), "uri": c.uri} for c in cams]
    finally:
        db.close()


def select_cameras() -> list:
    """
    Show an OpenCV grid of all calibrated cameras.
    User clicks thumbnails to toggle selection, then presses Enter or clicks
    'Start Merge'. Returns list of selected camera IDs.
    Returns [] if cancelled (Esc).
    """
    cams = _load_cameras()
    if not cams:
        print("[select_cameras] No calibrated cameras found in DB.")
        return []

    n = len(cams)
    cols = min(COLS, n)
    rows = math.ceil(n / cols)

    print(f"[select_cameras] Loading frames for {n} camera(s)…")
    thumbs = [_grab_frame(c["uri"]) for c in cams]
    selected = [True] * n  # all selected by default

    BTN_H = 48
    BTN_MARGIN = 16
    canvas_w = cols * CELL_W + PADDING
    canvas_h = rows * CELL_H + PADDING + BTN_H + BTN_MARGIN * 2

    btn_x1 = canvas_w // 2 - 120
    btn_x2 = canvas_w // 2 + 120
    btn_y1 = canvas_h - BTN_H - BTN_MARGIN
    btn_y2 = canvas_h - BTN_MARGIN

    hover_btn = False

    def cell_index(mx, my):
        for i in range(n):
            r, c_ = divmod(i, cols)
            x0 = PADDING + c_ * CELL_W
            y0 = PADDING + r  * CELL_H
            x1 = x0 + CELL_W
            y1 = y0 + CELL_H
            if x0 <= mx < x1 and y0 <= my < y1:
                return i
        return -1

    def draw(hover_b=False):
        canvas = np.full((canvas_h, canvas_w, 3), BG_COLOR, np.uint8)

        for i, (cam, thumb) in enumerate(zip(cams, thumbs)):
            r, c_ = divmod(i, cols)
            x0 = PADDING + c_ * CELL_W + PADDING
            y0 = PADDING + r  * CELL_H + PADDING

            color = SEL_COLOR if selected[i] else UNSEL_COLOR
            # border rect
            cv2.rectangle(canvas,
                           (x0 - BORDER, y0 - BORDER),
                           (x0 + THUMB_W + BORDER, y0 + THUMB_H + BORDER),
                           color, BORDER)
            canvas[y0:y0+THUMB_H, x0:x0+THUMB_W] = thumb

            # label
            label = f"[{'x' if selected[i] else ' '}] {cam['name']} (id={cam['id']})"
            cv2.putText(canvas, label,
                        (x0, y0 + THUMB_H + LABEL_H - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        # button
        bcol = BTN_HOVER if hover_b else BTN_COLOR
        cv2.rectangle(canvas, (btn_x1, btn_y1), (btn_x2, btn_y2), bcol, -1)
        cv2.rectangle(canvas, (btn_x1, btn_y1), (btn_x2, btn_y2), (200,200,200), 1)
        n_sel = sum(selected)
        txt = f"Start Merge ({n_sel} selected)"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(canvas, txt,
                    (btn_x1 + (240 - tw) // 2, btn_y1 + (BTN_H + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        hint = "Click to toggle  |  Enter = confirm  |  Esc = cancel"
        cv2.putText(canvas, hint, (PADDING, canvas_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120,120,120), 1)
        return canvas

    WIN = "Select cameras for merge"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    mx, my = 0, 0

    button_clicked = False

    def on_mouse(event, x, y, flags, _):
        nonlocal mx, my, hover_btn, button_clicked
        mx, my = x, y
        hover_btn = (btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2)
        if event == cv2.EVENT_LBUTTONDOWN:
            if hover_btn:
                button_clicked = True
                return
            idx = cell_index(x, y)
            if idx >= 0:
                selected[idx] = not selected[idx]

    cv2.setMouseCallback(WIN, on_mouse)

    result = None
    while True:
        if button_clicked:
            result = [cams[i]["id"] for i in range(n) if selected[i]]
            break

        canvas = draw(hover_btn)
        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:          # Esc
            result = []
            break
        if key in (13, 10):    # Enter
            result = [cams[i]["id"] for i in range(n) if selected[i]]
            break
        # button click closes window — guard against Qt NULL-window crash
        try:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                result = [cams[i]["id"] for i in range(n) if selected[i]]
                break
        except cv2.error:
            result = [cams[i]["id"] for i in range(n) if selected[i]]
            break

    cv2.destroyAllWindows()
    return result if result is not None else []


if __name__ == "__main__":
    ids = select_cameras()
    if not ids:
        print("Cancelled or nothing selected.")
    else:
        print(f"Selected camera IDs: {ids}")