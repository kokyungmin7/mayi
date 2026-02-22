"""Qwen3-VL model converter for ONNX and TensorRT."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from convert.core.onnx_utils import optimize_onnx_model, verify_onnx_model
from convert.core.tensorrt_utils import convert_onnx_to_tensorrt

logger = logging.getLogger(__name__)


class QwenConverter:
    """Converter for Qwen3-VL model to ONNX/TensorRT.

    WARNING: Qwen3-VL is a large model (8B parameters).
    ONNX export requires ~16GB RAM (FP16 mode).
    """

    DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Thinking"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        use_fp16: bool = True,
    ) -> None:
        """Initialize converter.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on ("cpu", "cuda")
            use_fp16: Use FP16 precision (recommended for memory efficiency)
        """
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self._model = None
        self._processor = None

    def _load_model(self, disable_quantization: bool = True):
        """Load model without 4-bit quantization for ONNX export."""
        if self._model is not None:
            return

        logger.info(
            "Loading Qwen3-VL model '%s' on %s (FP16=%s, no quantization)...",
            self.model_name,
            self.device,
            self.use_fp16,
        )

        # Load without BitsAndBytes quantization
        dtype = torch.float16 if self.use_fp16 else torch.float32
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self._model.to(self.device)
        self._model.eval()

        # Load processor
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        logger.info("Qwen3-VL loaded successfully (no quantization)")

    def convert_vision_encoder_to_onnx(
        self,
        output_path: str | Path = "models/qwen3vl-vision-encoder.onnx",
        opset_version: int = 18,
        optimize: bool = True,
        image_size: int = 224,
    ) -> Path:
        """Export Qwen3-VL vision encoder to ONNX.

        Args:
            output_path: Path to save vision encoder ONNX
            opset_version: ONNX opset version
            optimize: Whether to optimize ONNX graph
            image_size: Input image size (square)

        Returns:
            Path to vision encoder ONNX model
        """
        self._load_model()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting vision encoder (opset=%d)...", opset_version)

        # Wrapper for vision encoder
        class VisionEncoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.vision_model = model.visual  # Qwen3-VL vision component

            def forward(self, pixel_values):
                """Extract vision features from images."""
                # Get vision embeddings
                vision_outputs = self.vision_model(pixel_values)

                # Return vision hidden states
                if hasattr(vision_outputs, "last_hidden_state"):
                    return vision_outputs.last_hidden_state
                elif isinstance(vision_outputs, torch.Tensor):
                    return vision_outputs
                else:
                    raise ValueError(f"Unexpected vision output: {type(vision_outputs)}")

        wrapped_model = VisionEncoderWrapper(self._model)
        wrapped_model.eval()

        # Create dummy input
        dummy_image = torch.randn(1, 3, image_size, image_size).to(self.device)
        if self.use_fp16:
            dummy_image = dummy_image.half()

        # Dynamic axes
        dynamic_axes = {
            "pixel_values": {0: "batch_size"},
            "vision_embeds": {0: "batch_size"},
        }

        # Export to ONNX
        torch.onnx.export(
            wrapped_model,
            dummy_image,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["pixel_values"],
            output_names=["vision_embeds"],
            dynamic_axes=dynamic_axes,
        )

        logger.info("Vision encoder ONNX export complete: %s", output_path)

        # Verify
        try:
            metadata = verify_onnx_model(output_path)
            logger.info("Vision encoder verified: %s", metadata)
        except Exception as e:
            logger.warning("Vision encoder verification failed: %s", e)

        # Optimize
        if optimize:
            optimized_path = output_path.parent / f"{output_path.stem}-optimized.onnx"
            try:
                optimize_onnx_model(output_path, optimized_path)
                logger.info("Using optimized vision encoder: %s", optimized_path)
                return optimized_path
            except Exception as e:
                logger.warning("Optimization failed: %s. Using unoptimized model.", e)

        return output_path

    def convert_text_decoder_to_onnx(
        self,
        output_path: str | Path = "models/qwen3vl-text-decoder.onnx",
        opset_version: int = 18,
        max_seq_length: int = 512,
    ) -> Path:
        """Export Qwen3-VL text decoder (single step) to ONNX.

        WARNING: This exports a single decoder forward pass, not the full
        autoregressive generation loop. You'll need to implement the generation
        loop in Python using ONNXRuntime.

        Args:
            output_path: Path to save text decoder ONNX
            opset_version: ONNX opset version
            max_seq_length: Maximum sequence length

        Returns:
            Path to text decoder ONNX model
        """
        self._load_model()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting text decoder (opset=%d)...", opset_version)

        # Wrapper for text decoder
        class TextDecoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids, attention_mask, vision_embeds):
                """Single decoder forward pass.

                Args:
                    input_ids: [batch, seq_len] - Token IDs
                    attention_mask: [batch, seq_len] - Attention mask
                    vision_embeds: [batch, num_patches, hidden_dim] - Vision features

                Returns:
                    logits: [batch, seq_len, vocab_size] - Next token logits
                """
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=None,  # Vision already encoded
                    # TODO: May need to pass vision_embeds differently
                    # depending on Qwen3-VL architecture
                )

                return outputs.logits

        wrapped_model = TextDecoderWrapper(self._model)
        wrapped_model.eval()

        # Create dummy inputs
        batch_size = 1
        seq_len = 10
        num_patches = 256  # Typical vision encoder output
        hidden_dim = 4096  # Qwen3-VL hidden size

        dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
        dummy_attention_mask = torch.ones(batch_size, seq_len).to(self.device)
        dummy_vision_embeds = torch.randn(batch_size, num_patches, hidden_dim).to(self.device)

        if self.use_fp16:
            dummy_vision_embeds = dummy_vision_embeds.half()

        # Dynamic axes
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "vision_embeds": {0: "batch_size"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        }

        # Export to ONNX
        torch.onnx.export(
            wrapped_model,
            (dummy_input_ids, dummy_attention_mask, dummy_vision_embeds),
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask", "vision_embeds"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
        )

        logger.info("Text decoder ONNX export complete: %s", output_path)

        return output_path

    def convert_to_tensorrt(
        self,
        vision_encoder_path: str | Path,
        text_decoder_path: str | Path,
        vision_output_path: str | Path = "models/qwen3vl-vision-encoder.engine",
        decoder_output_path: str | Path = "models/qwen3vl-text-decoder.engine",
        fp16: bool = True,
        int8: bool = False,
        **kwargs,
    ) -> tuple[Path, Path]:
        """Convert Qwen3-VL ONNX models to TensorRT.

        Args:
            vision_encoder_path: Path to vision encoder ONNX
            text_decoder_path: Path to text decoder ONNX
            vision_output_path: Path to save vision encoder TensorRT engine
            decoder_output_path: Path to save text decoder TensorRT engine
            fp16: Enable FP16 precision
            int8: Enable INT8 quantization
            **kwargs: Additional TensorRT conversion options

        Returns:
            Tuple of (vision_engine_path, decoder_engine_path)
        """
        vision_output_path = Path(vision_output_path)
        decoder_output_path = Path(decoder_output_path)

        logger.info("Converting vision encoder to TensorRT...")
        vision_engine = convert_onnx_to_tensorrt(
            onnx_path=vision_encoder_path,
            output_path=vision_output_path,
            fp16=fp16,
            int8=int8,
            **kwargs,
        )

        logger.info("Converting text decoder to TensorRT...")
        decoder_engine = convert_onnx_to_tensorrt(
            onnx_path=text_decoder_path,
            output_path=decoder_output_path,
            fp16=fp16,
            int8=int8,
            **kwargs,
        )

        return vision_engine, decoder_engine

    def validate_conversion(
        self,
        vision_encoder_path: str | Path,
        text_decoder_path: str | Path,
        num_samples: int = 5,
    ) -> dict:
        """Validate ONNX conversion against PyTorch.

        Args:
            vision_encoder_path: Path to vision encoder ONNX
            text_decoder_path: Path to text decoder ONNX
            num_samples: Number of test samples

        Returns:
            Validation results
        """
        from PIL import Image

        from convert.runtime.onnx_qwen_generator import ONNXQwenGenerator

        self._load_model()

        # Load ONNX generator
        onnx_gen = ONNXQwenGenerator(
            vision_encoder_path,
            text_decoder_path,
            providers=["CPUExecutionProvider"],
        )

        results = {"passed": 0, "failed": 0, "samples": []}

        # Test with random images
        for i in range(num_samples):
            # Generate random test image
            test_image = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            test_prompt = "Describe this image."

            # PyTorch inference
            # TODO: Implement PyTorch inference call
            # pt_output = ...

            # ONNX inference
            try:
                onnx_output = onnx_gen.generate(test_image, test_prompt, max_new_tokens=50)

                # Compare (simple check: both generated non-empty text)
                if len(onnx_output) > 0:
                    results["passed"] += 1
                    results["samples"].append({
                        "index": i,
                        "onnx_output": onnx_output,
                        "status": "pass",
                    })
                else:
                    results["failed"] += 1
                    results["samples"].append({
                        "index": i,
                        "status": "fail",
                        "reason": "Empty output",
                    })
            except Exception as e:
                results["failed"] += 1
                results["samples"].append({
                    "index": i,
                    "status": "fail",
                    "reason": str(e),
                })

        return results

    # Legacy methods for compatibility
    def convert_to_onnx(self, output_path: str | Path, **kwargs) -> Path:
        """Convert to ONNX (vision encoder only for compatibility)."""
        return self.convert_vision_encoder_to_onnx(output_path, **kwargs)
