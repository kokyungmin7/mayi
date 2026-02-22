from __future__ import annotations

from mayi.reid.failure_detectors.base import FailureDetector, FailureDetectorPipeline
from mayi.reid.failure_detectors.consistency_monitor import ConsistencyMonitor
from mayi.reid.failure_detectors.embedding_collapse import EmbeddingCollapseDetector
from mayi.reid.failure_detectors.speed_jump import SpeedJumpDetector

DETECTOR_REGISTRY: dict[str, type[FailureDetector]] = {
    "embedding_collapse": EmbeddingCollapseDetector,
    "speed_jump": SpeedJumpDetector,
}


def build_failure_detector_pipeline(config: dict) -> FailureDetectorPipeline:
    pipeline = FailureDetectorPipeline()
    enabled = config.get("enabled", [])

    for name in enabled:
        cls = DETECTOR_REGISTRY.get(name)
        if cls is None:
            continue
        detector_config = config.get(name, {})
        pipeline.add(cls.from_config(detector_config))

    return pipeline


__all__ = [
    "FailureDetector",
    "FailureDetectorPipeline",
    "ConsistencyMonitor",
    "EmbeddingCollapseDetector",
    "SpeedJumpDetector",
    "build_failure_detector_pipeline",
]
