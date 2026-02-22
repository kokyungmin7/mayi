from __future__ import annotations

import numpy as np

from mayi.conditions.base import Condition
from mayi.models.types import ConditionResult, Detection


class BBoxSizeCondition(Condition):
    """Reject detections whose bounding box is too small.

    Score scales linearly from 0 at the minimum threshold to 1 at
    twice the threshold.
    """

    def __init__(self, min_width: float = 64, min_height: float = 128, priority: int = 0) -> None:
        self._min_width = min_width
        self._min_height = min_height
        self._priority = priority

    @property
    def name(self) -> str:
        return "bbox_size"

    @property
    def priority(self) -> int:
        return self._priority

    def check(self, detection: Detection, frame: np.ndarray) -> ConditionResult:
        w, h = detection.width, detection.height

        if w < self._min_width or h < self._min_height:
            return ConditionResult(
                passed=False,
                score=0.0,
                reason=f"bbox too small: {w:.0f}x{h:.0f} (min {self._min_width}x{self._min_height})",
            )

        w_score = min(w / (self._min_width * 2), 1.0)
        h_score = min(h / (self._min_height * 2), 1.0)
        score = (w_score + h_score) / 2

        return ConditionResult(passed=True, score=score)

    @classmethod
    def from_config(cls, config: dict) -> BBoxSizeCondition:
        return cls(
            min_width=config.get("min_width", 64),
            min_height=config.get("min_height", 128),
            priority=config.get("priority", 0),
        )
