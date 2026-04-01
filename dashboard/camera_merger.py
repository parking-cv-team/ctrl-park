"""
merge_cameras.py — Merge multiple camera calibrations into a unified parking map.

Reads all cameras with homography from the DB (populated by calibrate_parking),
lets the user pair corresponding slots between cameras, then:
  - Writes mapped_zones rows to DB (one per physical parking slot)
  - Updates zones.mapped_zone_id for all paired zones
  - Saves merged_topdown.png to CALIBRATION_OUTPUT_DIR

Usage:
    python -m processing.merge_cameras
    python -m processing.merge_cameras --recalibrate
"""
import requests

import os
import sys

# Disable GStreamer before cv2 is imported — it crashes on VideoCapture on some systems
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "0")

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOPDOWN_SCALE = 80  # px/m
PANEL_W       = 340         # width of the side parameter panel (pixels)
PANEL_BG      = (45, 45, 45)

CAM_COLORS = [          # BGR
    (220,  80,  80),    # cam 0 — blue
    (0,   165, 255),    # cam 1 — orange
    ( 80,  80, 220),    # cam 2 — red
    (200, 200,  60),    # cam 3 — cyan
    (200,  60, 200),    # cam 4 — magenta
    ( 60, 200, 200),    # cam 5 — yellow
]

IOU_MATCH_THRESHOLD = 0.3   # minimum IoU for two slots to be considered the same physical spot
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def cam_color(idx):
    return CAM_COLORS[idx % len(CAM_COLORS)]




# ---------------------------------------------------------------------------
# Affine helpers
# ---------------------------------------------------------------------------
def to_3x3(M2x3):
    out = np.eye(3)
    out[:2, :] = M2x3
    return out


def apply_affine_pts(M2x3, pts):
    """pts: Nx2 → Nx2"""
    pts = np.asarray(pts, dtype=float)
    return (M2x3 @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :2]


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def extract_frame(uri: str) -> np.ndarray | None:
    # Run VideoCapture in a subprocess to isolate GStreamer/codec segfaults.
    # Protocol: first 12 bytes = "H W C\n" (shape), then raw pixel bytes.
    
    cap = cv2.VideoCapture(uri)
    if cap is None:
        return None
    for _ in range(30):
        ok,_ = cap.read()
    ok,frame = cap.read()
    cap.release()
    print(ok)
    
    if ok:
        return frame
    
    return None


# ---------------------------------------------------------------------------
# Load cameras from DB
# ---------------------------------------------------------------------------
def load_cameras_from_db(selected_ids: list | None = None) -> list: #TODO
    """Load cameras with homography and polygon_metric zones from DB.

    If selected_ids is provided, only those camera IDs are loaded.
    """
    try:
        r = requests.get(f"{API_BASE}/camera/selected_data", json={"selected_ids":selected_ids})
        r.raise_for_status()
        rows = r.json()
        cam_data =[]
        for i in rows:
            cam_data.append({
                "cam_idx": i["cam_idx"],
                "camera_id": i["camera_id"],
                "uri": i["uri"],
                "cam_name": i["cam_name"],
                "slots": i["slots"],
            })
        return cam_data
    except Exception as e:
        ping(e)
    pass

def ping(e):
    try:
        requests.get(f"{API_BASE}/ping",params={"e":e})
    except:
        pass
# ---------------------------------------------------------------------------
# CameraView
# ---------------------------------------------------------------------------
class CameraView:
    ALPHA = 0.30

    def __init__(self, cam_idx: int, cam_data: dict):
        self.cam_idx  = cam_idx
        self.cam_name = cam_data["cam_name"]
        self.uri      = cam_data["uri"]
        self.frame    = (cam_data["frame"].copy()
                         if cam_data["frame"] is not None
                         else np.zeros((480, 640, 3), np.uint8))
        self.slots    = cam_data["slots"]
        self.color    = cam_color(cam_idx)
        self.selected = None
        self.paired   = {}   # slot key → partner key

    def find_slot_at(self, x, y):
        for s in self.slots:
            poly = s.get("polygon_detection_px")
            if not poly:
                continue
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            if cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0:
                return s["key"]
        return None

    def draw(self, active=False, pair_count=0, need_pairs=2):
        img = self.frame.copy()
        overlay = img.copy()
        for s in self.slots:
            poly = s.get("polygon_detection_px")
            if not poly:
                continue
            key = s["key"]
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            if key in self.paired:
                fill = (60, 200, 60)
            elif key == self.selected:
                fill = (60, 220, 220)
            else:
                fill = self.color
            cv2.fillPoly(overlay, [pts], fill)
        cv2.addWeighted(overlay, self.ALPHA, img, 1 - self.ALPHA, 0, img)
        for s in self.slots:
            poly = s.get("polygon_detection_px")
            if not poly:
                continue
            key = s["key"]
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            if key in self.paired:
                border = (0, 150, 0)
            elif key == self.selected:
                border = (0, 160, 160)
            else:
                border = tuple(max(0, v - 60) for v in self.color)
            cv2.polylines(img, [pts], True, border, 2)
            cx = int(sum(p[0] for p in poly) / len(poly))
            cy = int(sum(p[1] for p in poly) / len(poly))
            lbl = f"r{s['row']}_c{s['col']}"
            cv2.putText(img, lbl, (cx - 20, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, lbl, (cx - 20, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Camera label in top-left corner
        hud = f"Cam {self.cam_idx}: {self.cam_name}"
        if active:
            color = (0, 220, 220)
            hud += " [ACTIVE]"
        else:
            color = (200, 200, 200)
        cv2.putText(img, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, color, 1, cv2.LINE_AA)

        # Active border highlight
        if active:
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1),
                          (0, 220, 220), 3)
        return img


# ---------------------------------------------------------------------------
# PairingState
# ---------------------------------------------------------------------------
class PairingState:
    MIN_PAIRS = 2

    def __init__(self, view_a: CameraView, view_b: CameraView):
        self.view_a  = view_a
        self.view_b  = view_b
        self.pending = {view_a.cam_idx: None, view_b.cam_idx: None}
        self.pairs   = []   # [(key_a, key_b)]

    def click(self, cam_idx, x, y):
        view = self.view_a if cam_idx == self.view_a.cam_idx else self.view_b
        key = view.find_slot_at(x, y)
        if key is None:
            return
        self.pending[cam_idx] = key
        view.selected = key
        ka = self.pending[self.view_a.cam_idx]
        kb = self.pending[self.view_b.cam_idx]
        if ka and kb:
            already = any(p[0] == ka or p[1] == kb for p in self.pairs)
            if not already:
                self.pairs.append((ka, kb))
                self.view_a.paired[ka] = kb
                self.view_b.paired[kb] = ka
                print(f"  Pair {len(self.pairs)}: {ka} ↔ {kb}")
            self.pending[self.view_a.cam_idx] = None
            self.pending[self.view_b.cam_idx] = None
            self.view_a.selected = None
            self.view_b.selected = None

    def undo(self):
        if not self.pairs:
            return
        ka, kb = self.pairs.pop()
        self.view_a.paired.pop(ka, None)
        self.view_b.paired.pop(kb, None)
        print(f"  Undo: {ka} ↔ {kb}")

    @property
    def ready(self):
        return len(self.pairs) >= self.MIN_PAIRS


# ---------------------------------------------------------------------------
# Unified merge UI — single window with sidebar + tiled cameras
# ---------------------------------------------------------------------------
class MergeUI:
    """Single-window merge interface with a sidebar panel and tiled camera views."""

    WIN_NAME = "Merge Cameras"

    def __init__(self, cam_views: list[CameraView]):
        self.cam_views = cam_views
        self.n = len(cam_views)

        # Layout: compute grid for tiling cameras
        self.grid_cols = int(np.ceil(np.sqrt(self.n)))
        self.grid_rows = int(np.ceil(self.n / self.grid_cols))

        # Spanning tree state
        self.linked = {0}
        self.unlinked = set(range(1, self.n))
        self.step_results = []

        # Current pairing state
        self.state: PairingState | None = None
        self.active_a: int | None = None
        self.active_b: int | None = None

        # Pair selection via sidebar buttons
        self.sel_a: int | None = None
        self.sel_b: int | None = None

        # UI phase: "select_pair" or "pairing"
        self.phase = "select_pair"
        self.aborted = False
        self.finished = False

        # Message to display in sidebar (feedback)
        self.message = ""
        self.message_color = (200, 200, 200)

        # Camera tile layout (computed in _draw)
        self.tile_w = 0
        self.tile_h = 0
        self.canvas_h = 0
        self.canvas_w = 0

    # ------------------------------------------------------------------
    # Sidebar panel drawing
    # ------------------------------------------------------------------
    def _draw_panel(self, canvas: np.ndarray):
        panel = canvas[:, :PANEL_W]
        panel[:] = PANEL_BG

        y = 28
        cv2.putText(panel, "MERGE CAMERAS", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 2, cv2.LINE_AA)
        y += 10
        cv2.line(panel, (12, y), (PANEL_W - 12, y), (75, 75, 75), 1)
        y += 20

        # ── Phase: select_pair ─────────────────────────────────────────
        if self.phase == "select_pair":
            # Show linked / unlinked cameras
            cv2.putText(panel, "Linked:", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 210, 120), 1, cv2.LINE_AA)
            linked_str = ", ".join(str(i) for i in sorted(self.linked))
            cv2.putText(panel, linked_str, (80, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
            y += 20

            cv2.putText(panel, "Unlinked:", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (215, 90, 90), 1, cv2.LINE_AA)
            if self.unlinked:
                unlinked_str = ", ".join(str(i) for i in sorted(self.unlinked))
            else:
                unlinked_str = "none"
            cv2.putText(panel, unlinked_str, (90, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
            y += 24

            cv2.line(panel, (12, y), (PANEL_W - 12, y), (75, 75, 75), 1)
            y += 16

            # Camera A selector
            cv2.putText(panel, "Camera A (linked):", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)
            y += 22
            self._cam_a_buttons_y = y
            btn_x = 12
            for idx in sorted(self.linked):
                bw = 44
                active = (self.sel_a == idx)
                cv2.rectangle(panel, (btn_x, y - 14), (btn_x + bw, y + 8),
                              (65, 125, 75) if active else (60, 60, 60), -1)
                cv2.rectangle(panel, (btn_x, y - 14), (btn_x + bw, y + 8),
                              (120, 120, 120), 1)
                cv2.putText(panel, str(idx), (btn_x + 14, y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (255, 255, 255) if active else (150, 150, 150), 1, cv2.LINE_AA)
                btn_x += bw + 6
            y += 22

            # Camera B selector
            cv2.putText(panel, "Camera B (to link):", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)
            y += 22
            self._cam_b_buttons_y = y
            btn_x = 12
            cams_for_b = sorted(self.unlinked) if self.unlinked else sorted(set(range(self.n)) - self.linked)
            self._cam_b_list = cams_for_b
            for idx in cams_for_b:
                bw = 44
                active = (self.sel_b == idx)
                cv2.rectangle(panel, (btn_x, y - 14), (btn_x + bw, y + 8),
                              (65, 125, 75) if active else (60, 60, 60), -1)
                cv2.rectangle(panel, (btn_x, y - 14), (btn_x + bw, y + 8),
                              (120, 120, 120), 1)
                cv2.putText(panel, str(idx), (btn_x + 14, y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (255, 255, 255) if active else (150, 150, 150), 1, cv2.LINE_AA)
                btn_x += bw + 6
            y += 26

            # Start pairing button
            can_start = (self.sel_a is not None and self.sel_b is not None
                         and self.sel_a != self.sel_b)
            self._start_btn_y = y
            cv2.rectangle(panel, (12, y - 14), (PANEL_W - 12, y + 12),
                          (45, 120, 55) if can_start else (50, 50, 50), -1)
            cv2.rectangle(panel, (12, y - 14), (PANEL_W - 12, y + 12),
                          (100, 200, 110) if can_start else (80, 80, 80), 1)
            cv2.putText(panel, "Start Pairing (S)", (70, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255) if can_start else (120, 120, 120), 1, cv2.LINE_AA)
            y += 30

        # ── Phase: pairing ─────────────────────────────────────────────
        elif self.phase == "pairing":
            cv2.putText(panel, f"Pairing: Cam {self.active_a} <-> Cam {self.active_b}",
                        (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 220, 220), 1, cv2.LINE_AA)
            y += 24

            # Pair count
            if self.state:
                cnt = len(self.state.pairs)
                needed = self.state.MIN_PAIRS
                color = (100, 210, 120) if cnt >= needed else (255, 210, 75)
                cv2.putText(panel, f"Pairs: {cnt}/{needed}", (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
                y += 24

                # List current pairs
                for i, (ka, kb) in enumerate(self.state.pairs):
                    pair_txt = f"  {i+1}. {ka} <-> {kb}"
                    cv2.putText(panel, pair_txt, (12, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (180, 220, 180), 1, cv2.LINE_AA)
                    y += 16

            y += 8
            cv2.line(panel, (12, y), (PANEL_W - 12, y), (75, 75, 75), 1)
            y += 16

            # Confirm button
            can_confirm = self.state and self.state.ready
            self._confirm_btn_y = y
            cv2.rectangle(panel, (12, y - 14), (PANEL_W - 12, y + 12),
                          (45, 120, 55) if can_confirm else (50, 50, 50), -1)
            cv2.rectangle(panel, (12, y - 14), (PANEL_W - 12, y + 12),
                          (100, 200, 110) if can_confirm else (80, 80, 80), 1)
            cv2.putText(panel, "Confirm (C)", (100, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255) if can_confirm else (120, 120, 120), 1, cv2.LINE_AA)
            y += 28

            # Undo button
            self._undo_btn_y = y
            cv2.rectangle(panel, (12, y - 14), (PANEL_W - 12, y + 12),
                          (60, 60, 60), -1)
            cv2.rectangle(panel, (12, y - 14), (PANEL_W - 12, y + 12),
                          (100, 100, 100), 1)
            cv2.putText(panel, "Undo Last (U)", (90, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (215, 170, 90), 1, cv2.LINE_AA)
            y += 28

            # Cancel button
            self._cancel_btn_y = y
            cv2.rectangle(panel, (12, y - 14), (PANEL_W - 12, y + 12),
                          (60, 45, 45), -1)
            cv2.rectangle(panel, (12, y - 14), (PANEL_W - 12, y + 12),
                          (120, 80, 80), 1)
            cv2.putText(panel, "Cancel Pairing", (85, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (215, 90, 90), 1, cv2.LINE_AA)
            y += 28

        # ── Steps completed ────────────────────────────────────────────
        y += 8
        cv2.line(panel, (12, y), (PANEL_W - 12, y), (75, 75, 75), 1)
        y += 18
        cv2.putText(panel, "Completed links:", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 170, 170), 1, cv2.LINE_AA)
        y += 18
        if self.step_results:
            for step in self.step_results:
                txt = f"  Cam {step['cam_a']} <-> Cam {step['cam_b']}: {len(step['pairs'])} pairs"
                cv2.putText(panel, txt, (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (180, 220, 180), 1, cv2.LINE_AA)
                y += 16
        else:
            cv2.putText(panel, "  (none yet)", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (120, 120, 120), 1, cv2.LINE_AA)
            y += 16

        # ── Message ────────────────────────────────────────────────────
        if self.message:
            y += 12
            cv2.putText(panel, self.message, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        self.message_color, 1, cv2.LINE_AA)
            y += 16

        # ── Controls ───────────────────────────────────────────────────
        y = self.canvas_h - 100
        cv2.line(panel, (12, y), (PANEL_W - 12, y), (75, 75, 75), 1)
        y += 18
        controls = [
            ("Click slots to pair them", (155, 155, 155)),
            ("S  start pairing",         ( 95, 215,  95)),
            ("C  confirm pairs",         ( 95, 215,  95)),
            ("U  undo last pair",        (215, 170,  90)),
            ("Esc  exit",                (215,  90,  90)),
        ]
        for text, clr in controls:
            cv2.putText(panel, text, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, clr, 1, cv2.LINE_AA)
            y += 16

    # ------------------------------------------------------------------
    # Full canvas draw with tiled cameras
    # ------------------------------------------------------------------
    def _draw(self) -> np.ndarray:
        # Determine tile size from first camera frame
        ref_h, ref_w = self.cam_views[0].frame.shape[:2]

        # Target tile size: fit all cameras in available space
        target_h = max(300, 600 // self.grid_rows)
        scale = target_h / ref_h
        self.tile_w = int(ref_w * scale)
        self.tile_h = int(ref_h * scale)

        right_w = self.grid_cols * self.tile_w
        self.canvas_w = PANEL_W + right_w
        self.canvas_h = max(self.grid_rows * self.tile_h, 500)

        canvas = np.full((self.canvas_h, self.canvas_w, 3), 50, dtype=np.uint8)

        # Draw tiled camera views
        for i, view in enumerate(self.cam_views):
            row = i // self.grid_cols
            col = i % self.grid_cols
            x0 = PANEL_W + col * self.tile_w
            y0 = row * self.tile_h

            is_active = (self.phase == "pairing" and
                         view.cam_idx in (self.active_a, self.active_b))

            cam_img = view.draw(
                active=is_active,
                pair_count=len(self.state.pairs) if self.state else 0,
                need_pairs=PairingState.MIN_PAIRS,
            )
            resized = cv2.resize(cam_img, (self.tile_w, self.tile_h))
            canvas[y0:y0 + self.tile_h, x0:x0 + self.tile_w] = resized

        # Draw sidebar panel
        self._draw_panel(canvas)

        # Vertical separator
        cv2.line(canvas, (PANEL_W, 0), (PANEL_W, self.canvas_h), (80, 80, 80), 1)

        return canvas

    # ------------------------------------------------------------------
    # Mouse callback
    # ------------------------------------------------------------------
    def _mouse_cb(self, event, x, y, flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Panel clicks
        if x < PANEL_W:
            self._handle_panel_click(x, y)
            return

        # Camera tile clicks (only during pairing)
        if self.phase == "pairing" and self.state:
            self._handle_tile_click(x, y)

    def _handle_panel_click(self, x, y):
        if self.phase == "select_pair":
            # Camera A buttons
            btn_y = getattr(self, '_cam_a_buttons_y', 0)
            if btn_y - 14 <= y <= btn_y + 8:
                btn_x = 12
                for idx in sorted(self.linked):
                    bw = 44
                    if btn_x <= x <= btn_x + bw:
                        self.sel_a = idx
                        self.message = f"Selected Camera A = {idx}"
                        self.message_color = (200, 200, 200)
                        return
                    btn_x += bw + 6

            # Camera B buttons
            btn_y = getattr(self, '_cam_b_buttons_y', 0)
            if btn_y - 14 <= y <= btn_y + 8:
                btn_x = 12
                for idx in getattr(self, '_cam_b_list', []):
                    bw = 44
                    if btn_x <= x <= btn_x + bw:
                        self.sel_b = idx
                        self.message = f"Selected Camera B = {idx}"
                        self.message_color = (200, 200, 200)
                        return
                    btn_x += bw + 6

            # Start pairing button
            start_y = getattr(self, '_start_btn_y', 0)
            if start_y - 14 <= y <= start_y + 12 and 12 <= x <= PANEL_W - 12:
                self._try_start_pairing()

        elif self.phase == "pairing":
            # Confirm button
            confirm_y = getattr(self, '_confirm_btn_y', 0)
            if confirm_y - 14 <= y <= confirm_y + 12 and 12 <= x <= PANEL_W - 12:
                self._try_confirm()
                return

            # Undo button
            undo_y = getattr(self, '_undo_btn_y', 0)
            if undo_y - 14 <= y <= undo_y + 12 and 12 <= x <= PANEL_W - 12:
                if self.state:
                    self.state.undo()
                return

            # Cancel button
            cancel_y = getattr(self, '_cancel_btn_y', 0)
            if cancel_y - 14 <= y <= cancel_y + 12 and 12 <= x <= PANEL_W - 12:
                self._cancel_pairing()
                return

    def _handle_tile_click(self, x, y):
        """Translate click on the tiled camera area to a slot click."""
        rel_x = x - PANEL_W
        col = rel_x // self.tile_w
        row = y // self.tile_h
        cam_idx_clicked = row * self.grid_cols + col

        if cam_idx_clicked >= self.n:
            return
        if cam_idx_clicked not in (self.active_a, self.active_b):
            return

        # Convert to original frame coordinates
        ref_h, ref_w = self.cam_views[cam_idx_clicked].frame.shape[:2]
        local_x = rel_x - col * self.tile_w
        local_y = y - row * self.tile_h
        frame_x = local_x * ref_w / self.tile_w
        frame_y = local_y * ref_h / self.tile_h

        self.state.click(cam_idx_clicked, frame_x, frame_y)

    # ------------------------------------------------------------------
    # Pairing control
    # ------------------------------------------------------------------
    def _try_start_pairing(self):
        if self.sel_a is None or self.sel_b is None:
            self.message = "Select both Camera A and B first"
            self.message_color = (215, 90, 90)
            return
        if self.sel_a == self.sel_b:
            self.message = "A and B must be different"
            self.message_color = (215, 90, 90)
            return
        if self.sel_a not in self.linked:
            self.message = f"Cam {self.sel_a} not in linked set"
            self.message_color = (215, 90, 90)
            return

        self.active_a = self.sel_a
        self.active_b = self.sel_b

        va = self.cam_views[self.active_a]
        vb = self.cam_views[self.active_b]
        # Reset view state
        for v in self.cam_views:
            v.selected = None
            v.paired = {}

        self.state = PairingState(va, vb)
        self.phase = "pairing"
        self.message = f"Click slots in Cam {self.active_a} and Cam {self.active_b}"
        self.message_color = (200, 200, 200)
        print(f"Linking cam {self.active_a} ({va.cam_name})"
              f" ↔ cam {self.active_b} ({vb.cam_name})")

    def _try_confirm(self):
        if not self.state or not self.state.ready:
            self.message = f"Need >= {PairingState.MIN_PAIRS} pairs"
            self.message_color = (215, 90, 90)
            return

        print(f"  ✓ {len(self.state.pairs)} pairs confirmed")
        self.step_results.append({
            "cam_a": self.active_a,
            "cam_b": self.active_b,
            "pairs": [(ka, kb) for ka, kb in self.state.pairs],
        })
        self.linked.add(self.active_b)
        self.unlinked.discard(self.active_b)

        # Reset for next pair
        self.state = None
        self.active_a = None
        self.active_b = None
        self.sel_a = None
        self.sel_b = None

        if not self.unlinked:
            self.finished = True
            self.message = "All cameras linked!"
            self.message_color = (100, 210, 120)
        else:
            self.phase = "select_pair"
            self.message = "Pair confirmed. Select next pair."
            self.message_color = (100, 210, 120)

    def _cancel_pairing(self):
        self.state = None
        self.active_a = None
        self.active_b = None
        self.phase = "select_pair"
        self.message = "Pairing cancelled"
        self.message_color = (215, 170, 90)
        for v in self.cam_views:
            v.selected = None
            v.paired = {}

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> list:
        cv2.namedWindow(self.WIN_NAME, cv2.WINDOW_NORMAL)
        # Show initial frame to realize window
        cv2.imshow(self.WIN_NAME, np.zeros((400, 600, 3), dtype=np.uint8))
        cv2.waitKey(50)
        cv2.setMouseCallback(self.WIN_NAME, self._mouse_cb)

        while True:
            canvas = self._draw()
            cv2.imshow(self.WIN_NAME, canvas)

            key = cv2.waitKey(16) & 0xFF

            if key == 27:  # Esc
                self.aborted = True
                break

            if self.phase == "select_pair":
                if key in (ord("s"), ord("S")):
                    self._try_start_pairing()
            elif self.phase == "pairing":
                if key in (ord("c"), ord("C")):
                    self._try_confirm()
                elif key in (ord("u"), ord("U")):
                    if self.state:
                        self.state.undo()

            if self.finished:
                # Show final state briefly
                canvas = self._draw()
                cv2.imshow(self.WIN_NAME, canvas)
                cv2.waitKey(500)
                break

            try:
                if cv2.getWindowProperty(self.WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    self.aborted = True
                    break
            except cv2.error:
                self.aborted = True
                break

        cv2.destroyAllWindows()

        if self.aborted:
            print("[INFO] Aborted.")
            sys.exit(0)

        return self.step_results



def save_merge(step_results,selected_ids):

    
    data = {
        "step_results":step_results,
        "selected_ids":selected_ids
    }
    try:
        r = requests.post(f"{API_BASE}/merge/save",json=data)
        r.raise_for_status()
    except Exception as e:
        ping(e)
    pass

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def merge_cameras(recalibrate = False):
    


    # ------------------------------------------------------------------
    # Camera selection (skipped when --recalibrate to preserve existing behaviour)
    # ------------------------------------------------------------------
    selected_ids = None
    if not recalibrate:
        from dashboard.sel_cam import select_cameras
        selected_ids = select_cameras()
        if selected_ids is None or len(selected_ids) == 0:
            print("[merge_cameras] No cameras selected — aborting.")
            return None
        print(f"[merge_cameras] Selected camera IDs: {selected_ids}")

    # ------------------------------------------------------------------
    # Load cameras from DB
    # ------------------------------------------------------------------
    cam_data_list = load_cameras_from_db(selected_ids=selected_ids)
    
    # Read frame_idx from config file (default 0)


    # Extract representative frame for visualization
    for i in range(len(cam_data_list)):
        uri  = cam_data_list[i]["uri"]
        frame = extract_frame(uri)
        if frame is None:
            frame = np.zeros((480, 640, 3), np.uint8)
        cam_data_list[i]["frame"]=frame
    
   
    # ------------------------------------------------------------------
    # Single-window merge UI with sidebar
    # ------------------------------------------------------------------
    cam_views = [CameraView(d["cam_idx"], d) for d in cam_data_list]
    ui = MergeUI(cam_views)
    step_results = ui.run()

    save_merge(step_results,selected_ids)
