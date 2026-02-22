"""Runtime implementations for ONNX and TensorRT inference."""

from convert.runtime.onnx_embedder import ONNXReIDEmbedder

__all__ = ["ONNXReIDEmbedder"]

# TensorRT embedder requires tensorrt package
try:
    from convert.runtime.tensorrt_embedder import TensorRTReIDEmbedder

    __all__.append("TensorRTReIDEmbedder")
except ImportError:
    pass
