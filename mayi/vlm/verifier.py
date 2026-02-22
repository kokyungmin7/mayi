from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

from mayi.vlm.analyzer import VLMAnalyzer, _parse_json

logger = logging.getLogger(__name__)

_VERIFY_PROMPT = """\
Compare these two person images. Are they the same person?
Consider: clothing, body shape, hair, accessories.
Return ONLY a JSON object:
{"same_person": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}
Do not output anything except the JSON object."""


class VLMVerifier:
    """Uses VLM to verify whether two person crops show the same person.

    Shares the underlying model with VLMAnalyzer (pass the same instance).
    """

    def __init__(
        self,
        analyzer: VLMAnalyzer,
        max_retries: int = 2,
    ) -> None:
        self._analyzer = analyzer
        self._max_retries = max_retries

    def verify(
        self,
        crop1_bgr: np.ndarray,
        crop2_bgr: np.ndarray,
    ) -> tuple[bool, float]:
        """Compare two BGR person crops.

        Returns (is_same_person, confidence).
        """
        if crop1_bgr is None or crop2_bgr is None:
            return False, 0.0

        pil1 = Image.fromarray(cv2.cvtColor(crop1_bgr, cv2.COLOR_BGR2RGB))
        pil2 = Image.fromarray(cv2.cvtColor(crop2_bgr, cv2.COLOR_BGR2RGB))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil1},
                    {"type": "image", "image": pil2},
                    {"type": "text", "text": _VERIFY_PROMPT},
                ],
            }
        ]

        for attempt in range(self._max_retries + 1):
            try:
                raw = self._analyzer._run_inference(messages)
                result = _parse_json(raw)

                if result is None:
                    logger.warning(
                        "VLM verify: unparseable response (attempt %d): %s",
                        attempt + 1, raw[:200],
                    )
                    continue

                is_same = bool(result.get("same_person", False))
                confidence = float(result.get("confidence", 0.0))
                reason = result.get("reason", "")
                logger.info(
                    "[VLM-VERIFY] same=%s, confidence=%.2f, reason='%s'",
                    is_same, confidence, reason,
                )
                return is_same, confidence

            except Exception:
                logger.exception(
                    "VLM verification failed (attempt %d)", attempt + 1,
                )

        return False, 0.0

    @classmethod
    def from_config(cls, config: dict, analyzer: VLMAnalyzer) -> VLMVerifier:
        return cls(
            analyzer=analyzer,
            max_retries=config.get("max_retries", 2),
        )
