"""SigLIP2 Re-ID model converter for ONNX and TensorRT."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from convert.core.onnx_utils import optimize_onnx_model, verify_onnx_model
from convert.core.tensorrt_utils import convert_onnx_to_tensorrt

logger = logging.getLogger(__name__)


class SigLIPConverter:
    """Converter for SigLIP2 Re-ID model to ONNX/TensorRT.

    This converter exports only the vision encoder portion of SigLIP2,
    which is used for extracting person re-identification embeddings.
    """

    DEFAULT_MODEL = "MarketaJu/siglip2-person-description-reid"
    FALLBACK_PROCESSOR = "google/siglip-base-patch16-224"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
    ) -> None:
        """Initialize converter.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on ("cpu", "cuda", "mps")
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy-load the model and processor."""
        if self._model is not None:
            return

        logger.info("Loading model '%s' on %s...", self.model_name, self.device)
        self._model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self._model.to(self.device)
        self._model.eval()

        # Load processor (same logic as ReIDEmbedder)
        if hasattr(self._model, "processor") and self._model.processor is not None:
            self._processor = self._model.processor
        elif hasattr(self._model, "image_processor") and self._model.image_processor is not None:
            self._processor = self._model.image_processor
        else:
            logger.info("Using fallback processor: %s", self.FALLBACK_PROCESSOR)
            self._processor = AutoImageProcessor.from_pretrained(
                self.FALLBACK_PROCESSOR
            )

        logger.info("Model loaded successfully")

    def convert_to_onnx(
        self,
        output_path: str | Path = "models/siglip2-reid.onnx",
        opset_version: int = 17,
        optimize: bool = True,
        dynamic_batch: bool = True,
    ) -> Path:
        """Convert SigLIP2 model to ONNX format.

        Args:
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
            optimize: Whether to optimize the ONNX graph
            dynamic_batch: Enable dynamic batch size

        Returns:
            Path to converted ONNX model
        """
        self._load_model()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Converting to ONNX (opset=%d)...", opset_version)

        # Create dummy input (SigLIP expects 224x224 RGB images)
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

        # Dynamic axes for batch dimension
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                "pixel_values": {0: "batch_size"},
                "image_embeds": {0: "batch_size"},
            }

        # Wrapper to extract only vision features
        class VisionModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, pixel_values):
                outputs = self.model.get_image_features(pixel_values=pixel_values)

                # Handle different output formats
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    return outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state"):
                    return outputs.last_hidden_state
                elif hasattr(outputs, "image_embeds"):
                    return outputs.image_embeds
                elif isinstance(outputs, torch.Tensor):
                    return outputs
                else:
                    raise ValueError(f"Unexpected output type: {type(outputs)}")

        wrapped_model = VisionModelWrapper(self._model)
        wrapped_model.eval()

        # Export to ONNX
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes=dynamic_axes,
        )

        logger.info("ONNX export complete: %s", output_path)

        # Verify model
        try:
            metadata = verify_onnx_model(output_path)
            logger.info("ONNX model verified: %s", metadata)
        except Exception as e:
            logger.warning("ONNX verification failed: %s", e)

        # Optimize if requested
        if optimize:
            optimized_path = output_path.parent / f"{output_path.stem}-optimized.onnx"
            try:
                optimize_onnx_model(output_path, optimized_path)
                logger.info("Using optimized model: %s", optimized_path)
                return optimized_path
            except Exception as e:
                logger.warning("ONNX optimization failed: %s", e)
                logger.info("Using unoptimized model: %s", output_path)

        return output_path

    def convert_to_tensorrt(
        self,
        onnx_path: str | Path,
        output_path: str | Path = "models/siglip2-reid.engine",
        fp16: bool = True,
        workspace_gb: int = 4,
        min_batch: int = 1,
        opt_batch: int = 1,
        max_batch: int = 8,
    ) -> Path:
        """Convert ONNX model to TensorRT engine.

        Args:
            onnx_path: Path to ONNX model
            output_path: Path to save TensorRT engine
            fp16: Enable FP16 precision
            workspace_gb: GPU workspace size in GB
            min_batch: Minimum batch size
            opt_batch: Optimal batch size
            max_batch: Maximum batch size

        Returns:
            Path to TensorRT engine
        """
        onnx_path = Path(onnx_path)
        output_path = Path(output_path)

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        logger.info("Converting ONNX to TensorRT...")
        engine_path = convert_onnx_to_tensorrt(
            onnx_path=onnx_path,
            engine_path=output_path,
            fp16=fp16,
            workspace_gb=workspace_gb,
            min_batch=min_batch,
            opt_batch=opt_batch,
            max_batch=max_batch,
            input_name="pixel_values",
            input_shape=(3, 224, 224),
        )

        logger.info("TensorRT conversion complete: %s", engine_path)
        return engine_path

    def validate_conversion(
        self,
        onnx_path: str | Path,
        num_samples: int = 10,
        cosine_threshold: float = 0.99,
        mse_threshold: float = 1e-4,
    ) -> dict:
        """Validate ONNX model outputs against PyTorch.

        Args:
            onnx_path: Path to ONNX model
            num_samples: Number of random samples to test
            cosine_threshold: Minimum cosine similarity
            mse_threshold: Maximum MSE error

        Returns:
            Validation results dictionary
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for validation. "
                "Install with: pip install onnxruntime"
            )

        self._load_model()
        onnx_path = Path(onnx_path)

        logger.info("Validating ONNX model against PyTorch...")

        # Load ONNX model
        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )

        results = {
            "passed": 0,
            "failed": 0,
            "cosine_similarities": [],
            "mse_errors": [],
        }

        # Test with random inputs
        for i in range(num_samples):
            # Generate random input
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

            # PyTorch inference
            with torch.no_grad():
                pt_output = self._model.get_image_features(pixel_values=dummy_input)
                if hasattr(pt_output, "pooler_output") and pt_output.pooler_output is not None:
                    pt_output = pt_output.pooler_output
                elif hasattr(pt_output, "last_hidden_state"):
                    pt_output = pt_output.last_hidden_state
                elif hasattr(pt_output, "image_embeds"):
                    pt_output = pt_output.image_embeds
                pt_output = pt_output.cpu().numpy()

            # ONNX inference
            onnx_output = session.run(
                None,
                {"pixel_values": dummy_input.cpu().numpy()},
            )[0]

            # Compare outputs
            # Normalize embeddings
            pt_norm = pt_output[0] / (np.linalg.norm(pt_output[0]) + 1e-8)
            onnx_norm = onnx_output[0] / (np.linalg.norm(onnx_output[0]) + 1e-8)

            cosine_sim = float(np.dot(pt_norm, onnx_norm))
            mse = float(np.mean((pt_norm - onnx_norm) ** 2))

            results["cosine_similarities"].append(cosine_sim)
            results["mse_errors"].append(mse)

            if cosine_sim >= cosine_threshold and mse <= mse_threshold:
                results["passed"] += 1
            else:
                results["failed"] += 1
                logger.warning(
                    "Sample %d failed: cosine=%.4f (threshold=%.4f), "
                    "mse=%.2e (threshold=%.2e)",
                    i,
                    cosine_sim,
                    cosine_threshold,
                    mse,
                    mse_threshold,
                )

        # Summary statistics
        results["mean_cosine_similarity"] = float(
            np.mean(results["cosine_similarities"])
        )
        results["min_cosine_similarity"] = float(
            np.min(results["cosine_similarities"])
        )
        results["mean_mse"] = float(np.mean(results["mse_errors"]))
        results["max_mse"] = float(np.max(results["mse_errors"]))

        logger.info(
            "Validation complete: %d/%d passed (%.1f%%)",
            results["passed"],
            num_samples,
            100.0 * results["passed"] / num_samples,
        )
        logger.info(
            "Mean cosine similarity: %.6f (min: %.6f)",
            results["mean_cosine_similarity"],
            results["min_cosine_similarity"],
        )
        logger.info(
            "Mean MSE: %.2e (max: %.2e)",
            results["mean_mse"],
            results["max_mse"],
        )

        return results
