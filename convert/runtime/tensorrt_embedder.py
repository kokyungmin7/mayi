"""TensorRT implementation of ReIDEmbedder.

This module provides a drop-in replacement for ReIDEmbedder that uses
TensorRT for inference instead of PyTorch.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TensorRTReIDEmbedder:
    """TensorRT implementation of person re-identification embedder.

    This class implements the Embedder protocol and can be used as a drop-in
    replacement for ReIDEmbedder. It provides the same three-stage pipeline
    (preprocess → infer → postprocess) but uses TensorRT for inference.
    """

    FALLBACK_PROCESSOR = "google/siglip-base-patch16-224"

    def __init__(
        self,
        engine_path: str | Path,
        max_batch_size: int = 8,
    ) -> None:
        """Initialize TensorRT Re-ID embedder.

        Args:
            engine_path: Path to TensorRT engine file
            max_batch_size: Maximum batch size for the engine
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError:
            raise ImportError(
                "TensorRT and PyCUDA are required for TensorRT inference. "
                "Install with: pip install tensorrt pycuda"
            )

        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        logger.info("Loading TensorRT Re-ID engine from %s...", engine_path)

        # Load TensorRT engine
        self.trt = trt
        self.cuda = cuda

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load engine from {engine_path}")

        self.context = self.engine.create_execution_context()

        # Get input/output bindings
        self.input_idx = 0
        self.output_idx = 1
        self.input_shape = self.engine.get_binding_shape(self.input_idx)
        self.output_shape = self.engine.get_binding_shape(self.output_idx)

        logger.info(
            "TensorRT engine loaded (input=%s, output=%s)",
            self.input_shape,
            self.output_shape,
        )

        # Allocate device memory
        self.max_batch_size = max_batch_size
        self.d_input = cuda.mem_alloc(
            max_batch_size * 3 * 224 * 224 * np.dtype(np.float32).itemsize
        )
        # Output size depends on embedding dimension
        embedding_dim = self.output_shape[-1]
        self.d_output = cuda.mem_alloc(
            max_batch_size * embedding_dim * np.dtype(np.float32).itemsize
        )

        self.stream = cuda.Stream()

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
            resized = cv2.resize(rgb, (224, 224))
            normalized = resized.astype(np.float32) / 255.0
            mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            normalized = (normalized - mean) / std
            pixel_values = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
            return {"pixel_values": pixel_values}

    def infer(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        """Run TensorRT inference.

        Args:
            inputs: Dictionary with "pixel_values" key

        Returns:
            Image embeddings (1, embedding_dim)
        """
        pixel_values = inputs["pixel_values"].astype(np.float32)
        batch_size = pixel_values.shape[0]

        # Copy input to device
        self.cuda.memcpy_htod_async(
            self.d_input, pixel_values.ravel(), self.stream
        )

        # Set dynamic batch size
        self.context.set_binding_shape(self.input_idx, pixel_values.shape)

        # Run inference
        bindings = [int(self.d_input), int(self.d_output)]
        self.context.execute_async_v2(
            bindings=bindings,
            stream_handle=self.stream.handle,
        )

        # Copy output from device
        embedding_dim = self.output_shape[-1]
        output = np.empty((batch_size, embedding_dim), dtype=np.float32)
        self.cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()

        return output

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
            logger.exception("TensorRT embedding extraction failed")
            return None

    @classmethod
    def from_config(cls, config: dict) -> TensorRTReIDEmbedder:
        """Create embedder from configuration dictionary.

        Args:
            config: Configuration with "tensorrt_engine_path" and optional "max_batch_size"

        Returns:
            TensorRTReIDEmbedder instance
        """
        return cls(
            engine_path=config["tensorrt_engine_path"],
            max_batch_size=config.get("max_batch_size", 8),
        )
