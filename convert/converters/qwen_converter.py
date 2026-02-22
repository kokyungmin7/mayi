"""Qwen3-VL model converter (placeholder - deferred implementation)."""

from __future__ import annotations

from pathlib import Path


class QwenConverter:
    """Converter for Qwen3-VL model to ONNX/TensorRT.

    NOTE: This converter is currently a placeholder. Qwen3-VL conversion
    is deferred due to:
    - Large model size (8B parameters)
    - Complex multi-modal architecture
    - Already uses 4-bit quantization
    - Infrequent usage (VLM analysis is not real-time)

    Recommendation: Keep Qwen3-VL in PyTorch for now.

    If conversion becomes necessary:
    - Use ONNX export with sequence length constraints
    - Consider INT8 quantization for TensorRT
    - May need to split encoder-decoder components
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Thinking") -> None:
        self.model_name = model_name

    def convert_to_onnx(self, output_path: str | Path, **kwargs) -> Path:
        raise NotImplementedError(
            "Qwen3-VL ONNX conversion is not yet implemented. "
            "Please continue using the PyTorch version. "
            "See convert/converters/qwen_converter.py for details."
        )

    def convert_to_tensorrt(
        self,
        onnx_path: str | Path,
        output_path: str | Path,
        **kwargs,
    ) -> Path:
        raise NotImplementedError(
            "Qwen3-VL TensorRT conversion is not yet implemented. "
            "Please continue using the PyTorch version. "
            "See convert/converters/qwen_converter.py for details."
        )

    def validate_conversion(
        self,
        original_path: str | Path,
        converted_path: str | Path,
        **kwargs,
    ) -> bool:
        raise NotImplementedError(
            "Qwen3-VL validation is not yet implemented."
        )
