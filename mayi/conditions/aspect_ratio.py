from __future__ import annotations

import numpy as np

from mayi.conditions.base import Condition
from mayi.models.types import ConditionResult, Detection


class AspectRatioCondition(Condition):
    """Reject detections whose width/height ratio falls outside the
    expected range for a standing/walking person.

    Typical human bounding boxes have a ratio (width / height) between
    0.25 (narrow, arms at sides) and 0.65 (wide, arms spread or
    carrying objects).
    """

    def __init__(
        self,
        min_ratio: float = 0.25,
        max_ratio: float = 0.65,
        priority: int = 1,
    ) -> None:
        self._min_ratio = min_ratio
        self._max_ratio = max_ratio
        self._priority = priority

    @property
    def name(self) -> str:
        return "aspect_ratio"

    @property
    def priority(self) -> int:
        return self._priority

    def check(self, detection: Detection, frame: np.ndarray) -> ConditionResult:
        if detection.height == 0:
            return ConditionResult(passed=False, score=0.0, reason="zero height")

        ratio = detection.width / detection.height

        if ratio < self._min_ratio or ratio > self._max_ratio:
            return ConditionResult(
                passed=False,
                score=0.0,
                reason=f"aspect ratio {ratio:.2f} out of [{self._min_ratio}, {self._max_ratio}]",
            )

        mid = (self._min_ratio + self._max_ratio) / 2
        half_range = (self._max_ratio - self._min_ratio) / 2
        score = 1.0 - abs(ratio - mid) / half_range

        return ConditionResult(passed=True, score=max(score, 0.0))

    @classmethod
    def from_config(cls, config: dict) -> AspectRatioCondition:
        return cls(
            min_ratio=config.get("min_ratio", 0.25),
            max_ratio=config.get("max_ratio", 0.65),
            priority=config.get("priority", 1),
        )
