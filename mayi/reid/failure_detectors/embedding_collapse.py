from __future__ import annotations

from mayi.models.types import AnchorEntry, Detection, FailureReport, MatchResult
from mayi.reid.failure_detectors.base import FailureDetector


class EmbeddingCollapseDetector(FailureDetector):
    """Flags matches whose similarity is above the match threshold
    but below a stricter confidence threshold — indicating a shaky
    match that should be verified by VLM.
    """

    def __init__(self, confidence_threshold: float = 0.85) -> None:
        self._confidence_threshold = confidence_threshold

    @property
    def name(self) -> str:
        return "embedding_collapse"

    def check(
        self,
        match: MatchResult,
        detection: Detection,
        anchor: AnchorEntry,
    ) -> FailureReport:
        if not match.matched:
            return FailureReport(detected=False)

        if match.similarity < self._confidence_threshold:
            return FailureReport(
                detected=True,
                detector_name=self.name,
                reason=(
                    f"Low-confidence match: sim={match.similarity:.3f} "
                    f"< confidence={self._confidence_threshold}"
                ),
                details={
                    "similarity": match.similarity,
                    "confidence_threshold": self._confidence_threshold,
                },
            )
        return FailureReport(detected=False)

    @classmethod
    def from_config(cls, config: dict) -> EmbeddingCollapseDetector:
        return cls(
            confidence_threshold=config.get("confidence_threshold", 0.85),
        )
