"""Standalone 4-point parking slot annotator.

Click exactly 4 points on the video frame to define each parking spot.
The polygon closes automatically after the 4th click and you are prompted
for a name in the terminal. Repeat for every spot, then press Q to save.

Usage:
    python scripts/draw_zones.py
    python scripts/draw_zones.py --video data/parking_sample.mp4
    python scripts/draw_zones.py --video foo.mp4 --frame 30 --out data/zones/custom.json

Keys:
    Left-click  Add vertex (polygon closes automatically after 4 clicks)
    U           Undo last pending vertex
    Z           Undo last completed zone
    Q           Save all zones to JSON and quit
    Esc         Quit without saving
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from .zones import (
    SlotZone,
    ZoneConfig,
    _default_config_path,
    save_zone_config,
)

_ZONE_COLORS: list[tuple[int, int, int]] = [
    (0, 255, 0),    # green
    (0, 165, 255),  # orange
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
    (128, 0, 128),  # purple
]

_WINDOW = "Parking Zone Drawer"
_HUD = "Click 4 pts/spot | U: undo vertex | Z: undo zone | Q: save & quit | Esc: quit"
_VERTICES_PER_SLOT = 4


def _draw_state(
    canvas: np.ndarray,
    completed: list[SlotZone],
    pending: list[tuple[int, int]],
) -> np.ndarray:
    """Render completed zones and in-progress vertices onto a copy of canvas."""
    out = canvas.copy()
    overlay = canvas.copy()

    for idx, zone in enumerate(completed):
        color = _ZONE_COLORS[idx % len(_ZONE_COLORS)]
        pts = zone.polygon.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
        centroid = zone.polygon.mean(axis=0).astype(int)
        cv2.putText(
            out, zone.name, tuple(centroid),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.3, out, 0.7, 0, out)

    if pending:
        color = _ZONE_COLORS[len(completed) % len(_ZONE_COLORS)]
        for pt in pending:
            cv2.circle(out, pt, 5, color, -1)
        if len(pending) > 1:
            pts_arr = np.array(pending, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [pts_arr], isClosed=False, color=color, thickness=2)

    # HUD bar
    (tw, th), baseline = cv2.getTextSize(_HUD, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(out, (0, 0), (tw + 10, th + baseline + 8), (0, 0, 0), -1)
    cv2.putText(
        out, _HUD, (5, th + 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    # Zone counter (bottom-left)
    counter_text = f"Zones defined: {len(completed)}  |  Pending clicks: {len(pending)}/{_VERTICES_PER_SLOT}"
    h = out.shape[0]
    cv2.putText(
        out, counter_text, (5, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
    )

    return out


def _mouse_callback(event: int, x: int, y: int, flags: int, state: dict) -> None:
    """Append (x, y) to state['pending'] on left-button-down."""
    if event == cv2.EVENT_LBUTTONDOWN:
        state["pending"].append((x, y))
        state["redraw"] = True


def draw_parking_from_scratch(uri, base_frame,out) -> None:
    
    
    out_path = Path(out) if out else _default_config_path(uri)


    frame_height, frame_width = base_frame.shape[:2]
    print(f"\n[draw_zones] {uri}  {frame_width}x{frame_height}")
    print(f"[draw_zones] Output will be saved to: {out_path}")
    print(f"[draw_zones] {_HUD}\n")

    state: dict = {"pending": [], "completed": [], "redraw": True}
    cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(_WINDOW, _mouse_callback, state)
    counter = 0
    while True:
        # Auto-close after 4 clicks
        if len(state["pending"]) == _VERTICES_PER_SLOT:
            slot_idx = len(state["completed"])
            default_name = f"slot_{slot_idx}"
            name = getNewName(counter)
            if not name:
                name = default_name
            polygon = np.array(state["pending"], dtype=np.int32)
            state["completed"].append(SlotZone(name=name, polygon=polygon))
            state["pending"] = []
            state["redraw"] = True
            print(f"[draw_zones] Zone '{name}' saved.\n")
            counter += 1

        if state["redraw"]:
            display = _draw_state(base_frame, state["completed"], state["pending"])
            cv2.imshow(_WINDOW, display)
            state["redraw"] = False

        key = cv2.waitKey(20) & 0xFF

        try:
            window_closed = cv2.getWindowProperty(_WINDOW, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            window_closed = False
        if window_closed:  # window closed via X
            print("[draw_zones] Exiting without saving.")
            return

        if key == ord("u"):  # undo last pending vertex
            if state["pending"]:
                state["pending"].pop()
                state["redraw"] = True

        elif key == ord("z"):  # undo last completed zone
            if state["completed"]:
                removed = state["completed"].pop()
                state["redraw"] = True
                print(f"[draw_zones] Removed zone '{removed.name}'.")

        elif key == ord("q"):  # save and quit
            break

        elif key == 27:  # Esc — quit without saving
            print("[draw_zones] Exiting without saving.")
            #cv2.destroyWindow(_WINDOW)
            return

    cv2.destroyWindow(_WINDOW)

    if state["pending"]:
        print(
            f"[draw_zones] Warning: discarding {len(state['pending'])} "
            "unfinished vertices (zone not completed)."
        )

    if not state["completed"]:
        print("[draw_zones] No zones defined. Nothing saved.")
        return

    config = ZoneConfig(
        source=uri,
        frame_width=frame_width,
        frame_height=frame_height,
        zones=state["completed"],
    )
    save_zone_config(config, out_path)
    print(f"\n[draw_zones] Saved {len(config.zones)} zone(s) to {out_path}")


def getNewName(counter):
    """
    TODO:
    da determinare metodo per differenziare telecamere diverse per suffissi diversi / determinare che si tratta dello stesso spot di un'altra telecamera
    """
    
    
    return "A"+str(counter)
