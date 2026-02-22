"""Base protocols and interfaces for model conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ModelConverter(Protocol):
    """Protocol for model converters to ONNX/TensorRT."""

    def convert_to_onnx(
        self,
        output_path: str | Path,
        **kwargs,
    ) -> Path:
        """Convert model to ONNX format.

        Args:
            output_path: Path to save the ONNX model
            **kwargs: Converter-specific parameters

        Returns:
            Path to the converted ONNX model
        """
        ...

    def convert_to_tensorrt(
        self,
        onnx_path: str | Path,
        output_path: str | Path,
        **kwargs,
    ) -> Path:
        """Convert ONNX model to TensorRT engine.

        Args:
            onnx_path: Path to the ONNX model
            output_path: Path to save the TensorRT engine
            **kwargs: TensorRT-specific parameters

        Returns:
            Path to the TensorRT engine
        """
        ...

    def validate_conversion(
        self,
        original_path: str | Path,
        converted_path: str | Path,
        **kwargs,
    ) -> bool:
        """Validate that converted model outputs match original.

        Args:
            original_path: Path to original model
            converted_path: Path to converted model
            **kwargs: Validation parameters

        Returns:
            True if validation passes
        """
        ...
