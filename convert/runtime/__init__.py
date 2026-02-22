"""Runtime implementations for ONNX and TensorRT inference."""

from convert.runtime.onnx_embedder import ONNXReIDEmbedder
from convert.runtime.onnx_qwen_generator import ONNXQwenGenerator

__all__ = ["ONNXReIDEmbedder", "ONNXQwenGenerator"]

# TensorRT embedder requires tensorrt package
try:
    from convert.runtime.tensorrt_embedder import TensorRTReIDEmbedder

    __all__.append("TensorRTReIDEmbedder")
except ImportError:
    pass
