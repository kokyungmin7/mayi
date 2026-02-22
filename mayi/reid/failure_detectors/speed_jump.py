from __future__ import annotations

import math

from mayi.models.types import AnchorEntry, Detection, FailureReport, MatchResult
from mayi.reid.failure_detectors.base import FailureDetector


class SpeedJumpDetector(FailureDetector):
    """Flags matches where the person would have had to move at an
    impossible speed to get from their last-known position to the
    current detection.

    Only applies within the same camera.
    """

    def __init__(self, max_speed_px_per_frame: float = 200) -> None:
        self._max_speed = max_speed_px_per_frame

    @property
    def name(self) -> str:
        return "speed_jump"

    def check(
        self,
        match: MatchResult,
        detection: Detection,
        anchor: AnchorEntry,
    ) -> FailureReport:
        if not match.matched:
            return FailureReport(detected=False)

        if detection.camera_id != anchor.camera_id:
            return FailureReport(detected=False)

        anchor_cx = (anchor.bbox[0] + anchor.bbox[2]) / 2
        anchor_cy = (anchor.bbox[1] + anchor.bbox[3]) / 2
        det_cx, det_cy = detection.center

        distance = math.sqrt(
            (det_cx - anchor_cx) ** 2 + (det_cy - anchor_cy) ** 2
        )
        frames_elapsed = max(abs(detection.frame_idx - anchor.frame_idx), 1)
        speed = distance / frames_elapsed

        if speed > self._max_speed:
            return FailureReport(
                detected=True,
                detector_name=self.name,
                reason=(
                    f"Impossible speed: {speed:.1f} px/frame "
                    f"(max={self._max_speed})"
                ),
                details={
                    "speed": speed,
                    "distance": distance,
                    "frames_elapsed": frames_elapsed,
                },
            )
        return FailureReport(detected=False)

    @classmethod
    def from_config(cls, config: dict) -> SpeedJumpDetector:
        return cls(
            max_speed_px_per_frame=config.get("max_speed_px_per_frame", 200),
        )
