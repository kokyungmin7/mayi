from __future__ import annotations

from typing import TYPE_CHECKING

from mayi.conditions.aspect_ratio import AspectRatioCondition
from mayi.conditions.bbox_size import BBoxSizeCondition
from mayi.conditions.pose_completeness import PoseCompletenessCondition

if TYPE_CHECKING:
    from mayi.conditions.base import Condition

CONDITION_REGISTRY: dict[str, type[Condition]] = {
    "bbox_size": BBoxSizeCondition,
    "aspect_ratio": AspectRatioCondition,
    "pose_completeness": PoseCompletenessCondition,
}


def build_condition_pipeline(config: dict) -> "ConditionPipeline":
    """Build a ConditionPipeline from the ``conditions`` section of the
    YAML config.

    Example config::

        conditions:
          enabled: [bbox_size, aspect_ratio, pose_completeness]
          bbox_size:
            min_width: 64
            min_height: 128
            priority: 0
    """
    from mayi.conditions.base import ConditionPipeline

    pipeline = ConditionPipeline()
    enabled = config.get("enabled", [])

    for name in enabled:
        if name not in CONDITION_REGISTRY:
            raise ValueError(
                f"Unknown condition '{name}'. "
                f"Available: {list(CONDITION_REGISTRY.keys())}"
            )
        condition_cls = CONDITION_REGISTRY[name]
        condition_config = config.get(name, {})
        pipeline.add(condition_cls.from_config(condition_config))

    return pipeline
