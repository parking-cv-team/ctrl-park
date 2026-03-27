"""Tkinter-based 4-point parking slot annotator.

Extracts the first frame from a video file and displays it as the annotation
canvas. Click exactly 4 points to define each parking zone; a dialog will ask
for the zone name and category. Repeat for every spot, then click Save.

The output JSON is compatible with the database Zone model:
  { "source": str, "frame_width": int, "frame_height": int,
    "zones": [ {"name": str, "polygon": [[x,y],...], "category": str|null} ] }

Usage:
    python -m processing.draw_zones --video video/testfile.mp4
    python -m processing.draw_zones --video foo.mp4 --out data/zones/custom.json
"""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog

import cv2
from PIL import Image, ImageTk

_ZONE_COLORS = [
    "#00ff00", "#00a5ff", "#ff0000", "#0000ff",
    "#ffff00", "#ff00ff", "#00ffff", "#800080",
]
_CANVAS_MAX_W = 1280
_CANVAS_MAX_H = 720
_VERTICES_PER_SLOT = 4


def _default_out_path(video_path: Path) -> Path:
    return Path("data/zones") / f"{video_path.stem}_zones.json"


def _extract_first_frame(video_path: Path) -> tuple[Image.Image, int, int]:
    """Return (PIL image, original_width, original_height) for frame 0."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame from {video_path}")
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb), w, h

class ZoneDrawer:
    def __init__(self, video_path: Path, out_path: Path) -> None:
        self.video_path = video_path
        self.out_path = out_path

        self.image, self.imgw, self.imgh = _extract_first_frame(video_path)

        # Scale to fit canvas
        aspect = self.imgw / self.imgh
        if aspect > 1:
            self.canvas_w = min(_CANVAS_MAX_W, self.imgw)
            self.canvas_h = int(self.canvas_w / aspect)
        else:
            self.canvas_h = min(_CANVAS_MAX_H, self.imgh)
            self.canvas_w = int(self.canvas_h * aspect)

        self.scale_x = self.imgw / self.canvas_w
        self.scale_y = self.imgh / self.canvas_h

        self.zones: list[dict] = []
        self.current_pts: list[tuple[int, int]] = []

        self._build_ui()

    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title(f"Parking Zone Drawer — {self.video_path.name}")
        self.root.resizable(False, False)

        # Toolbar
        toolbar = tk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, pady=4)

        tk.Button(toolbar, text="Remove Last Zone",
                  command=self._remove_last_zone).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="Save & Quit",
                  command=self._save_and_quit).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="Quit without Saving",
                  command=self.root.destroy).pack(side=tk.LEFT, padx=4)

        # Output path field
        path_frame = tk.Frame(self.root)
        path_frame.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        tk.Label(path_frame, text="Output JSON:").pack(side=tk.LEFT)
        self.out_var = tk.StringVar(value=str(self.out_path))
        tk.Entry(path_frame, textvariable=self.out_var, width=50).pack(side=tk.LEFT, padx=4)

        # Status label
        self.status_var = tk.StringVar(value="Click 4 points to define a zone.")
        tk.Label(self.root, textvariable=self.status_var, anchor=tk.W).pack(
            side=tk.TOP, fill=tk.X, padx=4
        )

        # Canvas
        self.canvas = tk.Canvas(self.root, width=self.canvas_w, height=self.canvas_h, bg="black")
        self.canvas.pack(side=tk.BOTTOM)
        self.canvas.bind("<Button-1>", self._on_click)

        self._tk_image = ImageTk.PhotoImage(self.image.resize((self.canvas_w, self.canvas_h)))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_image)

        self.root.mainloop()

    def _redraw(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_image)
        # Draw completed zones
        for i, zone in enumerate(self.zones):
            color = _ZONE_COLORS[i % len(_ZONE_COLORS)]
            # Scale back to canvas coords
            pts = [
                (int(x / self.scale_x), int(y / self.scale_y))
                for x, y in zone["polygon"]
            ]
            flat = [coord for pt in pts for coord in pt]
            self.canvas.create_polygon(flat, outline=color, fill=color, stipple="gray25", width=2)
            cx = sum(p[0] for p in pts) // 4
            cy = sum(p[1] for p in pts) // 4
            self.canvas.create_text(cx, cy, text=zone["name"], fill=color,
                                    font=("Helvetica", 11, "bold"))
        # Draw pending points
        color = _ZONE_COLORS[len(self.zones) % len(_ZONE_COLORS)]
        for pt in self.current_pts:
            x, y = pt
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill=color, outline=color)
        if len(self.current_pts) > 1:
            flat = [coord for pt in self.current_pts for coord in pt]
            self.canvas.create_line(flat, fill=color, width=2)

    def _on_click(self, event: tk.Event) -> None:
        self.current_pts.append((event.x, event.y))
        self._redraw()
        remaining = _VERTICES_PER_SLOT - len(self.current_pts)
        if remaining > 0:
            self.status_var.set(f"{remaining} more click(s) to close zone.")
            return

        # 4 points reached — ask for name and category
        idx = len(self.zones)
        name = simpledialog.askstring(
            "Zone Name", f"Name for zone {idx}:",
            initialvalue=f"slot_{idx}", parent=self.root
        )
        if name is None:  # user cancelled
            self.current_pts.clear()
            self._redraw()
            self.status_var.set("Zone cancelled. Click 4 points to define a zone.")
            return
        name = name.strip() or f"slot_{idx}"

        category = simpledialog.askstring(
            "Category", "Category (e.g. 'parking lot') — leave blank to skip:",
            parent=self.root
        )
        category = (category or "").strip() or None

        self.zones.append({
            "name": name,
            "polygon": [
                [int(x * self.scale_x), int(y * self.scale_y)]
                for x, y in self.current_pts
            ],
            "category": category,
        })
        self.current_pts.clear()
        self._redraw()
        self.status_var.set(
            f"Zone '{name}' added ({len(self.zones)} total). Click 4 points for the next zone."
        )

    def _remove_last_zone(self) -> None:
        if not self.zones:
            messagebox.showwarning("Warning", "No zones to remove.")
            return
        removed = self.zones.pop()
        self._redraw()
        self.status_var.set(f"Removed zone '{removed['name']}'.")

    def _save_and_quit(self) -> None:
        if not self.zones:
            messagebox.showwarning("Warning", "No zones defined. Nothing to save.")
            return
        out = Path(self.out_var.get().strip())
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": self.video_path.name,
            "frame_width": self.imgw,
            "frame_height": self.imgh,
            "zones": self.zones,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.saved_path = out
        messagebox.showinfo("Saved", f"Saved {len(self.zones)} zone(s) to:\n{out}")
        self.root.destroy()


def launch(video_uri: str, out_path: Path | None = None) -> Path | None:
    """Open the zone drawer for *video_uri* and block until the user saves or quits.

    Returns the Path of the saved JSON file, or None if the user quit without saving.
    Called by the pipeline before processing starts.
    """
    video_path = Path(video_uri)
    if out_path is None:
        # Derive a filesystem-safe stem from the URI (handles RTSP URLs too)
        safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in video_path.stem)
        out_path = Path("data/zones") / f"{safe_stem}_zones.json"

    drawer = ZoneDrawer(video_path, out_path)
    return getattr(drawer, "saved_path", None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw 4-point parking slot zones from a video")
    parser.add_argument("--video", default="video/testfile.mp4", help="Input video file")
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: data/zones/<stem>_zones.json)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_path = Path(args.out) if args.out else _default_out_path(video_path)
    ZoneDrawer(video_path, out_path)


if __name__ == "__main__":
    main()
