"""Global multi-camera vehicle tracker for MCMOT pipelines.

Projects per-camera detections to a shared 2-D world space (cam0 pixel
coordinates) using pre-computed homographies, clusters nearby world-space
points from different cameras as the same physical vehicle, and maintains
stable ``global_id`` integers across frames and cameras.

Matching strategy — nearest-neighbour in world (cam0) coordinate space:

    IoU-based matching requires projecting bounding boxes into world
    coordinates, which introduces unpredictable distortion near image
    borders when cameras have very different viewpoints or zoom levels.
    For a parking lot the dominant motion model is near-stationary, so
    a simple centre-to-centre Euclidean distance check in world space is
    both more robust and easier to configure than IoU.

    A single ``distance_threshold`` (pixels in cam0 space) governs both
    intra-frame cross-camera clustering and inter-frame track matching.

Usage::

    from processing.global_tracker import GlobalTracker, run as global_run

    gt = GlobalTracker("data/homographies/homographies.json")

    for frame_idx, cam_dets in enumerate(pipeline):
        for cam_id, dets in cam_dets.items():
            gt.update(cam_id, dets, frame_idx)
        world_dets = gt.commit(frame_idx)

    # or using the module-level run() shortcut:
        world_dets = global_run(gt, cam_dets, frame_idx)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import supervision as sv


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass
class WorldDetection:
    """A single vehicle observed across one or more cameras.

    Attributes:
        global_id: Unique integer assigned and maintained by
            :class:`GlobalTracker` across frames.
        world_xy: Vehicle bounding-box centre projected to world
            (cam0 pixel) coordinates.
        source_cameras: IDs of cameras that contributed to this
            detection in the current frame (e.g. ``["cam0", "cam2"]``).
        local_ids: Map from camera ID string to the ByteTrack
            ``tracker_id`` assigned by that camera's
            :class:`~processing.tracking.CameraTracker`.
        frame_idx: Frame index at which this detection was observed.
    """

    global_id: int
    world_xy: tuple[float, float]
    source_cameras: list[str]
    local_ids: dict[str, int]
    frame_idx: int


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass
class _RawCandidate:
    """Projected detection from one camera, before cross-camera merging."""

    world_xy: tuple[float, float]
    cam_id: str
    local_id: int  # ByteTrack tracker_id; -1 if unavailable


@dataclass
class _GlobalTrack:
    """Mutable state for one active global track."""

    global_id: int
    world_xy: tuple[float, float]
    source_cameras: list[str]
    local_ids: dict[str, int]
    last_seen_frame: int


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_homographies(path: Path) -> dict[str, np.ndarray]:
    """Load homography matrices from a calibration JSON file.

    Args:
        path: Path to the JSON file produced by
            ``processing/SCRIPT_calibrate_homography.py``.

    Returns:
        Dict mapping camera ID string (e.g. ``"cam0"``) to a 3×3
        float64 homography matrix.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError: If the JSON is missing the ``"homographies"`` key.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Homographies file not found: {path}\n"
            "Run processing/SCRIPT_calibrate_homography.py first."
        )
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {
        cam_id: np.array(H, dtype=np.float64)
        for cam_id, H in data["homographies"].items()
    }


def _project_point(xy: tuple[float, float], H: np.ndarray) -> tuple[float, float]:
    """Apply a 3×3 homography to a 2-D point.

    Args:
        xy: Input point in pixel coordinates.
        H: 3×3 homography matrix.

    Returns:
        Projected point in the target coordinate system.
    """
    p = np.array([xy[0], xy[1], 1.0], dtype=np.float64)
    q = H @ p
    return float(q[0] / q[2]), float(q[1] / q[2])


def _cluster_cross_camera(
    candidates: list[_RawCandidate],
    threshold: float,
) -> list[list[_RawCandidate]]:
    """Cluster detections from different cameras that map to the same vehicle.

    Uses path-compressed union-find. Two candidates are merged only when
    they come from *different* cameras — ByteTrack already prevents
    duplicate tracks within a single stream.

    Args:
        candidates: Projected detections from all cameras for one frame.
        threshold: Maximum world-space distance for two detections to be
            considered the same physical vehicle.

    Returns:
        List of clusters. Each cluster is a non-empty list of candidates;
        all members are assumed to represent the same physical vehicle.
    """
    n = len(candidates)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    thresh_sq = threshold * threshold
    for i in range(n):
        for j in range(i + 1, n):
            if candidates[i].cam_id == candidates[j].cam_id:
                continue  # never merge same-camera candidates
            dx = candidates[i].world_xy[0] - candidates[j].world_xy[0]
            dy = candidates[i].world_xy[1] - candidates[j].world_xy[1]
            if dx * dx + dy * dy <= thresh_sq:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[_RawCandidate]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(candidates[i])
    return list(groups.values())


def _match_greedy(
    cluster_positions: list[tuple[float, float]],
    active_tracks: dict[int, _GlobalTrack],
    threshold: float,
) -> list[int | None]:
    """Greedy nearest-neighbour matching of new clusters to active tracks.

    Processes clusters in order and assigns each one the nearest unmatched
    active track within *threshold* world-space pixels. Simple and efficient
    for parking lots with at most ~50 concurrent vehicles.

    Args:
        cluster_positions: World-space centroid for each candidate cluster.
        active_tracks: Currently active global tracks, keyed by global_id.
        threshold: Maximum matching distance.

    Returns:
        List of length ``len(cluster_positions)``. Each entry is the matched
        ``global_id`` or ``None`` if no active track was close enough.
    """
    assignments: list[int | None] = [None] * len(cluster_positions)
    used: set[int] = set()

    for ci, cpos in enumerate(cluster_positions):
        best_dist = threshold + 1.0
        best_gid: int | None = None
        for gid, track in active_tracks.items():
            if gid in used:
                continue
            dx = cpos[0] - track.world_xy[0]
            dy = cpos[1] - track.world_xy[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_gid = gid
        if best_gid is not None:
            assignments[ci] = best_gid
            used.add(best_gid)

    return assignments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class GlobalTracker:
    """Multi-camera global vehicle ID manager.

    Call :meth:`update` once per camera per frame, then :meth:`commit` to
    obtain the merged world-space result for that frame.

    Args:
        homographies_path: Path to the homographies JSON produced by
            ``processing/SCRIPT_calibrate_homography.py``.
        distance_threshold: World-space distance (pixels in cam0 coordinate
            space) below which two detections are considered the same
            vehicle. Applied to both intra-frame cross-camera clustering
            and inter-frame track matching.
        lost_track_buffer: Frames a global track can go unmatched before
            it is removed from the active set.
    """

    def __init__(
        self,
        homographies_path: str | Path,
        *,
        distance_threshold: float = 60.0,
        lost_track_buffer: int = 30,
    ) -> None:
        self._H: dict[str, np.ndarray] = _load_homographies(Path(homographies_path))
        self._dist_thresh: float = distance_threshold
        self._lost_buffer: int = lost_track_buffer
        self._next_gid: int = 1
        self._active: dict[int, _GlobalTrack] = {}
        self._frame_buffer: dict[str, sv.Detections] = {}
        self._current_frame: int = -1
        self._last_result: dict[int, WorldDetection] = {}

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def update(
        self,
        cam_id: str,
        detections: sv.Detections,
        frame_idx: int,
    ) -> None:
        """Buffer one camera's tracked detections for the current frame.

        Call this once per camera, then call :meth:`commit` to finalise the
        frame and obtain merged world-space results.

        Args:
            cam_id: Camera identifier matching a key in the homographies
                JSON (e.g. ``"cam0"``). Cameras with no homography entry
                are silently skipped.
            detections: Tracked detections for this camera and frame, with
                ``tracker_id`` populated by
                :class:`~processing.tracking.CameraTracker`.
            frame_idx: Zero-based index of the current frame.
        """
        if frame_idx != self._current_frame:
            self._current_frame = frame_idx
            self._frame_buffer = {}
        self._frame_buffer[cam_id] = detections

    def commit(self, frame_idx: int) -> dict[int, WorldDetection]:
        """Finalise the current frame and return merged global detections.

        Runs intra-frame cross-camera clustering followed by inter-frame
        greedy nearest-neighbour matching. Expires tracks that have not
        been seen for more than ``lost_track_buffer`` frames.

        Args:
            frame_idx: Current frame index. Should match the most recent
                :meth:`update` calls.

        Returns:
            Mapping from ``global_id`` to :class:`WorldDetection` for all
            active tracks visible in this frame.
        """
        candidates = self._build_candidates()
        self._match_and_assign(candidates, frame_idx)
        self._expire_tracks(frame_idx)
        self._last_result = {
            gid: WorldDetection(
                global_id=track.global_id,
                world_xy=track.world_xy,
                source_cameras=list(track.source_cameras),
                local_ids=dict(track.local_ids),
                frame_idx=frame_idx,
            )
            for gid, track in self._active.items()
            if track.last_seen_frame == frame_idx
        }
        return self._last_result

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _build_candidates(self) -> list[_RawCandidate]:
        """Project all buffered detections to world space."""
        candidates: list[_RawCandidate] = []
        for cam_id, dets in self._frame_buffer.items():
            if cam_id not in self._H:
                continue  # camera has no homography — skip silently
            H = self._H[cam_id]
            for i in range(len(dets)):
                x1, y1, x2, y2 = dets.xyxy[i]
                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)
                world_xy = _project_point((cx, cy), H)
                local_id = int(dets.tracker_id[i]) if dets.tracker_id is not None else -1
                candidates.append(
                    _RawCandidate(world_xy=world_xy, cam_id=cam_id, local_id=local_id)
                )
        return candidates

    def _match_and_assign(
        self,
        candidates: list[_RawCandidate],
        frame_idx: int,
    ) -> None:
        """Cluster candidates and assign/update global track IDs."""
        if not candidates:
            return

        clusters = _cluster_cross_camera(candidates, self._dist_thresh)

        cluster_positions: list[tuple[float, float]] = []
        for cluster in clusters:
            wx = sum(c.world_xy[0] for c in cluster) / len(cluster)
            wy = sum(c.world_xy[1] for c in cluster) / len(cluster)
            cluster_positions.append((wx, wy))

        assignments = _match_greedy(cluster_positions, self._active, self._dist_thresh)

        for ci, cluster in enumerate(clusters):
            wx, wy = cluster_positions[ci]
            source_cams = list({c.cam_id for c in cluster})
            local_ids = {c.cam_id: c.local_id for c in cluster if c.local_id >= 0}

            gid = assignments[ci]
            if gid is not None:
                track = self._active[gid]
                track.world_xy = (wx, wy)
                track.source_cameras = source_cams
                track.local_ids = local_ids
                track.last_seen_frame = frame_idx
            else:
                gid = self._next_gid
                self._next_gid += 1
                self._active[gid] = _GlobalTrack(
                    global_id=gid,
                    world_xy=(wx, wy),
                    source_cameras=source_cams,
                    local_ids=local_ids,
                    last_seen_frame=frame_idx,
                )

    def _expire_tracks(self, frame_idx: int) -> None:
        """Remove tracks not seen for more than ``lost_track_buffer`` frames."""
        stale = [
            gid
            for gid, track in self._active.items()
            if frame_idx - track.last_seen_frame > self._lost_buffer
        ]
        for gid in stale:
            del self._active[gid]


# ---------------------------------------------------------------------------
# Module-level run() — required public entry point
# ---------------------------------------------------------------------------


def run(
    tracker: GlobalTracker,
    cam_detections: dict[str, sv.Detections],
    frame_idx: int,
) -> dict[int, WorldDetection]:
    """Run one complete frame through the GlobalTracker.

    Convenience wrapper that calls :meth:`~GlobalTracker.update` for every
    camera and then :meth:`~GlobalTracker.commit`.

    Args:
        tracker: The stateful :class:`GlobalTracker` instance.
        cam_detections: Dict mapping camera ID string to tracked
            ``sv.Detections`` for this frame.
        frame_idx: Zero-based index of the current frame.

    Returns:
        Mapping from ``global_id`` to :class:`WorldDetection`.
    """
    for cam_id, dets in cam_detections.items():
        tracker.update(cam_id, dets, frame_idx)
    return tracker.commit(frame_idx)
