from __future__ import annotations

from abc import ABC, abstractmethod

from mayi.models.types import AnchorEntry, Detection, FailureReport, MatchResult


class FailureDetector(ABC):
    """Base class for Re-ID match validators.

    Each detector examines a successful Re-ID match and determines
    whether the result is trustworthy.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def check(
        self,
        match: MatchResult,
        detection: Detection,
        anchor: AnchorEntry,
    ) -> FailureReport: ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> FailureDetector: ...


class FailureDetectorPipeline:
    """Runs all registered failure detectors against a match result."""

    def __init__(self) -> None:
        self._detectors: list[FailureDetector] = []

    def add(self, detector: FailureDetector) -> FailureDetectorPipeline:
        self._detectors.append(detector)
        return self

    def check_all(
        self,
        match: MatchResult,
        detection: Detection,
        anchor: AnchorEntry,
    ) -> list[FailureReport]:
        return [
            report
            for d in self._detectors
            if (report := d.check(match, detection, anchor)).detected
        ]

    def __len__(self) -> int:
        return len(self._detectors)
