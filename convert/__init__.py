"""MAYI Model Conversion System

This package provides utilities for converting MAYI models to ONNX and TensorRT
formats for optimized inference performance.

Supported models:
- SigLIP2 Re-ID (MarketaJu/siglip2-person-description-reid)
- YOLO26-Pose (models/yolo26n-pose.pt)

Usage:
    from convert.converters import SigLIPConverter, YOLOPoseConverter
    from convert.runtime import ONNXReIDEmbedder, TensorRTReIDEmbedder

    # Convert models
    converter = SigLIPConverter()
    onnx_path = converter.convert_to_onnx()

    # Use in pipeline
    embedder = ONNXReIDEmbedder(onnx_path)
"""

__version__ = "0.1.0"
