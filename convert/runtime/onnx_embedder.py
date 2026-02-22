"""ONNX Runtime implementation of ReIDEmbedder.

This module provides a drop-in replacement for ReIDEmbedder that uses
ONNX Runtime for inference instead of PyTorch.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ONNXReIDEmbedder:
    """ONNX Runtime implementation of person re-identification embedder.

    This class implements the Embedder protocol and can be used as a drop-in
    replacement for ReIDEmbedder. It provides the same three-stage pipeline
    (preprocess → infer → postprocess) but uses ONNX Runtime for inference.
    """

    FALLBACK_PROCESSOR = "google/siglip-base-patch16-224"

    def __init__(
        self,
        onnx_path: str | Path,
        device: str = "cuda",
        provider_options: dict | None = None,
    ) -> None:
        """Initialize ONNX Re-ID embedder.

        Args:
            onnx_path: Path to ONNX model file
            device: Device to run on ("cuda" or "cpu")
            provider_options: Optional provider-specific options
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX inference. "
                "Install with: pip install onnxruntime or onnxruntime-gpu"
            )

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        # Configure execution providers
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        logger.info(
            "Loading ONNX Re-ID model from %s (device=%s)...",
            onnx_path,
            device,
        )

        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_options,
            providers=providers,
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(
            "ONNX model loaded (input=%s, output=%s)",
            self.input_name,
            self.output_name,
        )

        # Load processor for preprocessing
        self._processor = self._load_processor()

    def _load_processor(self):
        """Load image processor for preprocessing."""
        try:
            from transformers import AutoImageProcessor

            logger.info("Loading processor from %s", self.FALLBACK_PROCESSOR)
            return AutoImageProcessor.from_pretrained(self.FALLBACK_PROCESSOR)
        except ImportError:
            logger.warning(
                "transformers not available, using manual preprocessing"
            )
            return None

    def preprocess(self, crop_bgr: np.ndarray) -> dict[str, np.ndarray]:
        """BGR crop → model-ready numpy array.

        Args:
            crop_bgr: BGR image crop (H, W, 3)

        Returns:
            Dictionary with "pixel_values" key
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        if self._processor is not None:
            # Use HuggingFace processor
            inputs = self._processor(images=rgb, return_tensors="np")
            return inputs
        else:
            # Manual preprocessing (resize + normalize)
            # SigLIP expects 224x224 images normalized to [-1, 1]
            resized = cv2.resize(rgb, (224, 224))
            # Normalize to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            # Normalize to [-1, 1] (ImageNet-like normalization)
            mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            normalized = (normalized - mean) / std
            # Convert to NCHW format
            pixel_values = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
            return {"pixel_values": pixel_values}

    def infer(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        """Run ONNX Runtime inference.

        Args:
            inputs: Dictionary with "pixel_values" key

        Returns:
            Image embeddings (1, embedding_dim)
        """
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: inputs["pixel_values"]},
        )
        return outputs[0]

    @staticmethod
    def postprocess(embeds: np.ndarray) -> np.ndarray:
        """Tensor → L2-normalised float32 numpy vector.

        Args:
            embeds: Raw embeddings (1, embedding_dim)

        Returns:
            L2-normalized embedding vector (embedding_dim,)
        """
        embedding = embeds[0].astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        return embedding

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Full pipeline: preprocess → infer → postprocess.

        This method satisfies the Embedder protocol.

        Args:
            crop_bgr: BGR image crop (H, W, 3)

        Returns:
            L2-normalized embedding vector, or None on error
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        try:
            inputs = self.preprocess(crop_bgr)
            embeds = self.infer(inputs)
            return self.postprocess(embeds)
        except Exception:
            logger.exception("ONNX embedding extraction failed")
            return None

    @classmethod
    def from_config(cls, config: dict) -> ONNXReIDEmbedder:
        """Create embedder from configuration dictionary.

        Args:
            config: Configuration with "onnx_model_path" and optional "device"

        Returns:
            ONNXReIDEmbedder instance
        """
        return cls(
            onnx_path=config["onnx_model_path"],
            device=config.get("device", "cuda"),
        )
