"""ByteTrack-based multi-object tracking logic.

Provides both a functional API (``make_tracker`` / ``run``) for single-camera
use and a class-based API (``CameraTracker``) for multi-camera pipelines.
"""

import supervision as sv


def make_tracker(
    frame_rate: int = 30,
    lost_track_buffer: int = 30,
    minimum_matching_threshold: float = 0.8,
    minimum_consecutive_frames: int = 1,
) -> sv.ByteTrack:
    """Create a new ByteTrack tracker instance.

    ByteTrack is stateful — it maintains track IDs across frames. Call this
    factory once per video stream and per object class, then pass the same
    instance to every ``run()`` call for that stream.

    Keeping one tracker per class (e.g. one for cars, one for pedestrians)
    prevents ID collisions between classes.

    Args:
        frame_rate: Frame rate of the input video. Used internally by
            ByteTrack to compute the maximum number of frames a lost track is
            kept alive before being discarded.
        lost_track_buffer: Number of frames to keep a track alive after it
            disappears from the scene. Higher values help with temporary
            occlusions; lower values free memory faster.
        minimum_matching_threshold: IoU threshold for matching detections to
            existing tracks. Lower values are more permissive.
        minimum_consecutive_frames: Number of consecutive frames a detection
            must appear in before a new track ID is confirmed.

    Returns:
        A configured ``sv.ByteTrack`` instance ready to process frames.
    """
    return sv.ByteTrack(
        frame_rate=frame_rate,
        lost_track_buffer=lost_track_buffer,
        minimum_matching_threshold=minimum_matching_threshold,
        minimum_consecutive_frames=minimum_consecutive_frames,
    )


def run(detections: sv.Detections, tracker: sv.ByteTrack) -> sv.Detections:
    """Apply ByteTrack to detections for a single frame.

    Updates the tracker state and returns the same detections annotated with
    stable ``tracker_id`` values. Detections that do not match any existing
    track start a new one; tracks without a matching detection are kept alive
    for up to ``lost_track_buffer`` frames before being dropped.

    Args:
        detections: Detections for the current frame, as returned by
            ``processing.detect_frame.run()``. May be empty (``sv.Detections.empty()``).
        tracker: The stateful ByteTrack instance for this stream and class.
            Must be the same object across all frames of a single stream.

    Returns:
        A new ``sv.Detections`` object with ``tracker_id`` populated.
        Only detections matched or confirmed by the tracker are returned —
        the count may differ from the input.
    """
    return tracker.update_with_detections(detections)


class CameraTracker:
    """Per-camera ByteTrack tracker with an associated camera identifier.

    Encapsulates one :class:`supervision.ByteTrack` instance for a single
    camera stream. The ``camera_id`` attribute is an integer label used to
    distinguish cameras in multi-camera pipelines; it does not modify
    ByteTrack's internal track-ID namespace (local IDs remain independent
    integers per tracker instance). Cross-camera ID merging is handled
    downstream by :class:`processing.global_tracker.GlobalTracker`.

    Backward-compatible with :func:`make_tracker` and :func:`run` — existing
    single-camera code does not need to change.

    Args:
        camera_id: Integer label for this camera (e.g. 0, 1, 2, 3).
        frame_rate: FPS of the input video. Used by ByteTrack for its
            internal timing model.
        lost_track_buffer: Frames to keep a lost track alive after it
            disappears from the scene.
        minimum_matching_threshold: IoU threshold for detection-to-track
            matching inside ByteTrack.
        minimum_consecutive_frames: Frames before a new track is confirmed.
    """

    def __init__(
        self,
        camera_id: int,
        *,
        frame_rate: int = 30,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 1,
    ) -> None:
        self.camera_id: int = camera_id
        self._tracker: sv.ByteTrack = sv.ByteTrack(
            frame_rate=frame_rate,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Apply ByteTrack to one frame's detections.

        Args:
            detections: Raw detections for the current frame. May be empty
                (``sv.Detections.empty()``).

        Returns:
            Detections annotated with stable local ``tracker_id`` values.
            Only confirmed tracks are returned; count may differ from input.
        """
        return self._tracker.update_with_detections(detections)
