from __future__ import annotations

import numpy as np

from mayi.models.types import FailureReport


class ConsistencyMonitor:
    """Periodically verifies that an active track still matches its
    assigned anchor embedding.

    Used by TrackManager — not a FailureDetector subclass because it
    operates on active tracks rather than on new match results.
    """

    def __init__(
        self,
        check_interval: int = 60,
        threshold: float = 0.5,
    ) -> None:
        self._interval = check_interval
        self._threshold = threshold

    @property
    def check_interval(self) -> int:
        return self._interval

    def should_check(self, frame_counter: int) -> bool:
        return frame_counter >= self._interval

    def check(
        self,
        current_embedding: np.ndarray,
        anchor_embedding: np.ndarray,
    ) -> FailureReport:
        sim = float(np.dot(current_embedding, anchor_embedding))
        if sim < self._threshold:
            return FailureReport(
                detected=True,
                detector_name="consistency_monitor",
                reason=(
                    f"Consistency drop: sim={sim:.3f} "
                    f"< threshold={self._threshold}"
                ),
                details={"similarity": sim, "threshold": self._threshold},
            )
        return FailureReport(detected=False, details={"similarity": sim})

    @classmethod
    def from_config(cls, config: dict) -> ConsistencyMonitor:
        return cls(
            check_interval=config.get("consistency_check_interval", 60),
            threshold=config.get("consistency_threshold", 0.5),
        )
