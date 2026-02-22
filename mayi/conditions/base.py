from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from mayi.models.types import ConditionResult, Detection, QualityReport


class Condition(ABC):
    """Base class for all quality-gate conditions.

    Subclasses must implement ``name``, ``priority``, ``check``, and
    ``from_config``.  Lower ``priority`` values run first so that cheap
    checks short-circuit before expensive ones.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def priority(self) -> int: ...

    @abstractmethod
    def check(self, detection: Detection, frame: np.ndarray) -> ConditionResult:
        """Evaluate the condition against a single detection.

        Returns a ``ConditionResult`` with ``passed`` and a quality
        ``score`` (0.0–1.0).
        """
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> Condition:
        """Instantiate from the condition-specific section of the YAML
        config."""
        ...


class ConditionPipeline:
    """Runs conditions in priority order with short-circuit on failure."""

    def __init__(self) -> None:
        self._conditions: list[Condition] = []

    def add(self, condition: Condition) -> ConditionPipeline:
        self._conditions.append(condition)
        self._conditions.sort(key=lambda c: c.priority)
        return self

    def remove(self, name: str) -> ConditionPipeline:
        self._conditions = [c for c in self._conditions if c.name != name]
        return self

    def evaluate(self, detection: Detection, frame: np.ndarray) -> QualityReport:
        results: dict[str, ConditionResult] = {}

        for cond in self._conditions:
            result = cond.check(detection, frame)
            results[cond.name] = result
            if not result.passed:
                return QualityReport(
                    passed=False,
                    overall_score=0.0,
                    results=results,
                    failed_at=cond.name,
                )

        if not results:
            return QualityReport(passed=True, overall_score=1.0, results=results)

        overall = sum(r.score for r in results.values()) / len(results)
        return QualityReport(passed=True, overall_score=overall, results=results)

    @property
    def condition_names(self) -> list[str]:
        return [c.name for c in self._conditions]

    def __len__(self) -> int:
        return len(self._conditions)
