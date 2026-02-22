from __future__ import annotations

import logging
from typing import Protocol

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Protocol for embedding extractors — swap to ONNX/TensorRT impl."""

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray | None: ...


class ReIDEmbedder:
    """Extracts person re-identification embeddings using
    SigLIP2 (MarketaJu/siglip2-person-description-reid).

    Preprocessing and inference are separated for future ONNX / TensorRT
    conversion: override ``infer()`` to plug in a different backend while
    keeping the same ``preprocess()`` / ``postprocess()`` logic.
    """

    FALLBACK_PROCESSOR = "google/siglip-base-patch16-224"

    def __init__(
        self,
        model_name: str = "MarketaJu/siglip2-person-description-reid",
        device: str | None = None,
    ) -> None:
        if device is None:
            device = self._detect_device()
        self._device = device

        logger.info("Loading Re-ID model '%s' on %s ...", model_name, device)

        self._model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True,
        )
        self._model.to(self._device)
        self._model.eval()

        self._processor = self._resolve_processor(model_name)
        logger.info("Re-ID model loaded.")

    # ------------------------------------------------------------------
    # Processor resolution (follows reference/reid_module.py strategy)
    # ------------------------------------------------------------------

    def _resolve_processor(self, model_name: str):
        if getattr(self._model, "processor", None) is not None:
            logger.info("Using processor attached to model object.")
            return self._model.processor

        if getattr(self._model, "image_processor", None) is not None:
            logger.info("Using image_processor attached to model object.")
            return self._model.image_processor

        config = getattr(self._model, "config", None)
        if config is not None:
            vision_cfg = getattr(config, "vision_config", None)
            if vision_cfg is not None:
                model_type = getattr(vision_cfg, "model_type", "")
                if "siglip" in model_type.lower():
                    logger.info(
                        "Detected SigLIP vision config — loading base "
                        "processor from '%s'.",
                        self.FALLBACK_PROCESSOR,
                    )
                    return AutoImageProcessor.from_pretrained(
                        self.FALLBACK_PROCESSOR,
                    )

        logger.info(
            "Using fallback processor from '%s'.", self.FALLBACK_PROCESSOR,
        )
        return AutoImageProcessor.from_pretrained(self.FALLBACK_PROCESSOR)

    # ------------------------------------------------------------------
    # Three-stage pipeline: preprocess → infer → postprocess
    # ------------------------------------------------------------------

    def preprocess(self, crop_bgr: np.ndarray) -> dict[str, torch.Tensor]:
        """BGR crop → model-ready tensor dict (device-placed)."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=rgb, return_tensors="pt")
        return {k: v.to(self._device) for k, v in inputs.items()}

    def infer(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run model forward pass. Replace this for ONNX / TensorRT."""
        with torch.no_grad():
            embeds = self._model.get_image_features(**inputs)

            if hasattr(embeds, "pooler_output") and embeds.pooler_output is not None:
                embeds = embeds.pooler_output
            elif hasattr(embeds, "last_hidden_state"):
                embeds = embeds.last_hidden_state
            elif hasattr(embeds, "image_embeds"):
                embeds = embeds.image_embeds

        return embeds

    @staticmethod
    def postprocess(embeds: torch.Tensor) -> np.ndarray:
        """Tensor → L2-normalised float32 numpy vector."""
        embedding = embeds.cpu().numpy()[0].astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        return embedding

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Full pipeline: preprocess → infer → postprocess."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        try:
            inputs = self.preprocess(crop_bgr)
            embeds = self.infer(inputs)
            return self.postprocess(embeds)
        except Exception:
            logger.exception("Embedding extraction failed")
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @classmethod
    def from_config(cls, config: dict) -> ReIDEmbedder:
        return cls(
            model_name=config.get(
                "model", "MarketaJu/siglip2-person-description-reid",
            ),
            device=config.get("device"),
        )
