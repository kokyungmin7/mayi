from __future__ import annotations

import colorsys
import logging
import re
from pathlib import Path

import cv2
import numpy as np

from mayi.anchor_bank.bank import AnchorBank
from mayi.models.types import Detection, KeypointIndex, Track, TrackState
from mayi.tracking.track_manager import TrackManager

logger = logging.getLogger(__name__)

STATE_COLORS: dict[TrackState, tuple[int, int, int]] = {
    TrackState.PENDING: (0, 255, 255),
    TrackState.ACTIVE: (0, 220, 0),
    TrackState.SUSPICIOUS: (0, 165, 255),
    TrackState.REASSIGNED: (255, 0, 255),
    TrackState.LOST: (0, 0, 255),
    TrackState.RECOVERED: (255, 255, 0),
    TrackState.EXPIRED: (128, 128, 128),
    TrackState.DEAD: (64, 64, 64),
}

_GOLDEN_RATIO = 0.618033988749895
_NO_ID_COLOR = (180, 180, 180)


def _id_to_color(global_id: str) -> tuple[int, int, int]:
    """Generate a visually distinct BGR color from a global person ID."""
    m = re.search(r"\d+", global_id)
    idx = int(m.group()) if m else abs(hash(global_id))
    hue = (idx * _GOLDEN_RATIO) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.80, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))

SKELETON_CONNECTIONS: list[tuple[int, int]] = [
    (KeypointIndex.NOSE, KeypointIndex.LEFT_EYE),
    (KeypointIndex.NOSE, KeypointIndex.RIGHT_EYE),
    (KeypointIndex.LEFT_EYE, KeypointIndex.LEFT_EAR),
    (KeypointIndex.RIGHT_EYE, KeypointIndex.RIGHT_EAR),
    (KeypointIndex.LEFT_SHOULDER, KeypointIndex.RIGHT_SHOULDER),
    (KeypointIndex.LEFT_SHOULDER, KeypointIndex.LEFT_ELBOW),
    (KeypointIndex.LEFT_ELBOW, KeypointIndex.LEFT_WRIST),
    (KeypointIndex.RIGHT_SHOULDER, KeypointIndex.RIGHT_ELBOW),
    (KeypointIndex.RIGHT_ELBOW, KeypointIndex.RIGHT_WRIST),
    (KeypointIndex.LEFT_SHOULDER, KeypointIndex.LEFT_HIP),
    (KeypointIndex.RIGHT_SHOULDER, KeypointIndex.RIGHT_HIP),
    (KeypointIndex.LEFT_HIP, KeypointIndex.RIGHT_HIP),
    (KeypointIndex.LEFT_HIP, KeypointIndex.LEFT_KNEE),
    (KeypointIndex.LEFT_KNEE, KeypointIndex.LEFT_ANKLE),
    (KeypointIndex.RIGHT_HIP, KeypointIndex.RIGHT_KNEE),
    (KeypointIndex.RIGHT_KNEE, KeypointIndex.RIGHT_ANKLE),
]

KP_CONF_THRESHOLD = 0.5


class VideoWriter:
    """Annotates frames and optionally writes them to a video file."""

    def __init__(
        self,
        output_path: str | Path | None = None,
        fps: float = 30.0,
        frame_size: tuple[int, int] = (1920, 1080),
        draw_bbox: bool = True,
        draw_skeleton: bool = True,
        draw_status_bar: bool = True,
    ) -> None:
        self._writer: cv2.VideoWriter | None = None
        self._draw_bbox = draw_bbox
        self._draw_skeleton = draw_skeleton
        self._draw_status_bar = draw_status_bar

        if output_path is not None:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(p), fourcc, fps, frame_size,
            )
            logger.info("Video writer opened: %s", p)

    # ------------------------------------------------------------------
    # Frame annotation
    # ------------------------------------------------------------------

    def annotate(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        tracks: dict[int, Track],
        track_manager: TrackManager,
        anchor_bank: AnchorBank | None,
        frame_idx: int,
        *,
        camera_id: str = "",
    ) -> np.ndarray:
        vis = frame.copy()
        scale = self._scale(vis)

        for det in detections:
            track = (
                tracks.get(det.tracker_id)
                if det.tracker_id is not None
                else None
            )

            if self._draw_bbox:
                self._draw_detection(vis, det, track, track_manager, scale)
            elif self._draw_skeleton and det.keypoints is not None:
                color = (
                    _id_to_color(track.global_id)
                    if track and track.global_id else _NO_ID_COLOR
                )
                _draw_skeleton(vis, det.keypoints, color, scale)

        if self._draw_status_bar:
            self._draw_status(
                vis, frame_idx, track_manager, anchor_bank, scale,
                camera_id=camera_id,
            )

        return vis

    def write_frame(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("Video writer closed.")

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scale(frame: np.ndarray) -> float:
        return min(frame.shape[0], frame.shape[1]) / 720

    def _draw_detection(
        self,
        frame: np.ndarray,
        det: Detection,
        track: Track | None,
        tm: TrackManager,
        scale: float,
    ) -> None:
        if track is not None:
            color = (
                _id_to_color(track.global_id)
                if track.global_id else _NO_ID_COLOR
            )
            label_id = track.global_id or "?"
            label = f"{label_id} [{track.state.value}]"

            if track.state == TrackState.PENDING:
                report = tm.get_quality_report(track.tracker_id)
                if report and report.failed_at:
                    label += f" fail:{report.failed_at}"

            match = tm.get_match_result(track.tracker_id)
            if match and match.matched and track.state == TrackState.ACTIVE:
                label += f" sim:{match.similarity:.2f}"
        else:
            color = _NO_ID_COLOR
            label = "no-track"

        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        thickness = max(int(2 * scale), 1)
        font_scale = 0.55 * scale
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        text_size = cv2.getTextSize(label, font, font_scale, 1)[0]
        cv2.rectangle(
            frame,
            (x1, y1 - text_size[1] - 8),
            (x1 + text_size[0] + 4, y1),
            color, -1,
        )
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            font, font_scale, (0, 0, 0), max(int(scale), 1),
        )

        if self._draw_skeleton and det.keypoints is not None:
            skeleton_color = tuple(max(c - 40, 0) for c in color)
            _draw_skeleton(frame, det.keypoints, skeleton_color, scale)

    @staticmethod
    def _draw_status(
        frame: np.ndarray,
        frame_idx: int,
        tm: TrackManager,
        anchor_bank: AnchorBank | None,
        scale: float,
        *,
        camera_id: str = "",
    ) -> None:
        h, w = frame.shape[:2]
        bar_h = int(32 * scale)
        font_scale = 0.5 * scale
        font = cv2.FONT_HERSHEY_SIMPLEX

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        persons = anchor_bank.size if anchor_bank else "N/A"
        cam_label = f"Cam: {camera_id}  |  " if camera_id else ""
        text = (
            f"{cam_label}"
            f"Frame: {frame_idx}  |  "
            f"Active: {tm.active_count}  |  "
            f"Pending: {tm.pending_count}  |  "
            f"Persons: {persons}"
        )
        cv2.putText(
            frame, text,
            (int(10 * scale), h - int(10 * scale)),
            font, font_scale, (255, 255, 255), max(int(scale), 1),
        )


# ---------------------------------------------------------------------------
# Skeleton drawing (module-level)
# ---------------------------------------------------------------------------

def _draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    color: tuple[int, int, int],
    scale: float,
) -> None:
    thickness = max(int(2 * scale), 1)
    radius = max(int(3 * scale), 1)

    for idx in range(keypoints.shape[0]):
        x, y, conf = keypoints[idx]
        if conf < KP_CONF_THRESHOLD:
            continue
        cv2.circle(frame, (int(x), int(y)), radius, color, -1)

    for i, j in SKELETON_CONNECTIONS:
        if keypoints[i, 2] < KP_CONF_THRESHOLD:
            continue
        if keypoints[j, 2] < KP_CONF_THRESHOLD:
            continue
        pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
        pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
        cv2.line(frame, pt1, pt2, color, thickness)
