from __future__ import annotations

import numpy as np

from mayi.conditions.base import Condition
from mayi.models.types import KEYPOINT_NAME_TO_INDEX, ConditionResult, Detection

_DEFAULT_REQUIRED = ["left_knee", "right_knee", "left_ankle", "right_ankle"]
_DEFAULT_VISIBILITY_THRESHOLD = 0.5


class PoseCompletenessCondition(Condition):
    """Reject detections that lack the required lower-body keypoints.

    YOLO pose outputs 17 COCO keypoints, each with (x, y, visibility).
    This condition checks whether at least ``min_visible`` of the
    ``required_keypoints`` have a visibility score above the threshold.
    """

    def __init__(
        self,
        required_keypoints: list[str] | None = None,
        min_visible: int = 3,
        visibility_threshold: float = _DEFAULT_VISIBILITY_THRESHOLD,
        priority: int = 2,
    ) -> None:
        self._required_names = required_keypoints or _DEFAULT_REQUIRED
        self._required_indices = [KEYPOINT_NAME_TO_INDEX[k] for k in self._required_names]
        self._min_visible = min_visible
        self._visibility_threshold = visibility_threshold
        self._priority = priority

    @property
    def name(self) -> str:
        return "pose_completeness"

    @property
    def priority(self) -> int:
        return self._priority

    def check(self, detection: Detection, frame: np.ndarray) -> ConditionResult:
        if detection.keypoints is None:
            return ConditionResult(
                passed=False,
                score=0.0,
                reason="no keypoints available",
            )

        visible_count = 0
        for idx in self._required_indices:
            if detection.keypoints[idx, 2] >= self._visibility_threshold:
                visible_count += 1

        if visible_count < self._min_visible:
            return ConditionResult(
                passed=False,
                score=0.0,
                reason=f"visible keypoints {visible_count}/{self._min_visible} "
                f"(required: {self._required_names})",
            )

        score = visible_count / len(self._required_indices)
        return ConditionResult(passed=True, score=score)

    @classmethod
    def from_config(cls, config: dict) -> PoseCompletenessCondition:
        return cls(
            required_keypoints=config.get("required_keypoints", _DEFAULT_REQUIRED),
            min_visible=config.get("min_visible", 3),
            visibility_threshold=config.get("visibility_threshold", _DEFAULT_VISIBILITY_THRESHOLD),
            priority=config.get("priority", 2),
        )
