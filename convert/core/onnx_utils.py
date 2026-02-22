"""ONNX conversion and optimization utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import onnx

logger = logging.getLogger(__name__)


def optimize_onnx_model_generic(
    model_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Generic ONNX optimization using onnx.optimizer (not transformer-specific).

    More reliable for vision models like SigLIP that aren't well-suited
    for onnxruntime.transformers.optimizer.

    Args:
        model_path: Path to ONNX model
        output_path: Path to save optimized model

    Returns:
        Path to optimized model
    """
    from onnx import optimizer as onnx_optimizer

    logger.info("Optimizing ONNX model (generic): %s", model_path)

    model = onnx.load(str(model_path))

    # Apply generic optimization passes
    passes = [
        "eliminate_identity",
        "eliminate_nop_transpose",
        "eliminate_nop_pad",
        "eliminate_unused_initializer",
        "fuse_bn_into_conv",
        "fuse_consecutive_transposes",
        "fuse_transpose_into_gemm",
    ]

    optimized_model = onnx_optimizer.optimize(model, passes)

    # Save
    if output_path is None:
        output_path = model_path.parent / f"{model_path.stem}-optimized.onnx"

    onnx.save(optimized_model, str(output_path))
    logger.info("Generic optimization complete: %s", output_path)

    return output_path


def optimize_onnx_model(
    model_path: str | Path,
    output_path: str | Path | None = None,
    optimization_level: int = 2,
) -> Path:
    """Optimize ONNX model graph.

    Args:
        model_path: Path to ONNX model
        output_path: Path to save optimized model (default: overwrite original)
        optimization_level: Optimization level (0-2, higher = more aggressive)

    Returns:
        Path to optimized model
    """
    model_path = Path(model_path)
    if output_path is None:
        output_path = model_path.parent / f"{model_path.stem}-optimized.onnx"
    else:
        output_path = Path(output_path)

    logger.info("Loading ONNX model from %s", model_path)
    model = onnx.load(str(model_path))

    # Basic shape inference
    logger.info("Running shape inference...")
    model = onnx.shape_inference.infer_shapes(model)

    # Optimize graph with fallback strategy
    logger.info("Optimizing ONNX graph (level=%d)...", optimization_level)

    # Try 1: Transformer-specific optimizer (best for NLP models)
    try:
        from onnxruntime.transformers import optimizer

        optimized = optimizer.optimize_model(
            str(model_path),
            model_type="bert",  # Generic transformer optimization
            num_heads=0,  # Auto-detect
            hidden_size=0,  # Auto-detect
            optimization_options=None,
        )
        optimized.save_model_to_file(str(output_path))
        logger.info("Optimized model saved to %s", output_path)
        return output_path

    except ImportError:
        logger.warning(
            "onnxruntime.transformers not available, "
            "trying generic ONNX optimizer..."
        )

    except (ValueError, RuntimeError) as e:
        # Handle opset mismatch, graph errors, etc.
        logger.warning(
            "Transformer optimizer failed: %s. "
            "Trying generic ONNX optimizer...",
            e,
        )

    # Try 2: Generic ONNX optimizer (fallback for vision models)
    try:
        return optimize_onnx_model_generic(model_path, output_path)

    except Exception as e:
        logger.warning(
            "Generic optimization also failed: %s. "
            "Returning unoptimized model.",
            e,
        )
        # Fallback: just save with shape inference
        onnx.save(model, str(output_path))
        logger.info("Model with shape inference saved to %s", output_path)
        return output_path


def verify_onnx_model(model_path: str | Path) -> dict:
    """Verify ONNX model structure and get metadata.

    Args:
        model_path: Path to ONNX model

    Returns:
        Dictionary with model metadata
    """
    model_path = Path(model_path)
    model = onnx.load(str(model_path))

    # Check model validity
    onnx.checker.check_model(model)

    # Extract metadata
    metadata = {
        "opset_version": model.opset_import[0].version,
        "inputs": [],
        "outputs": [],
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
    }

    # Input shapes and types
    for inp in model.graph.input:
        shape = [
            dim.dim_value if dim.dim_value > 0 else dim.dim_param
            for dim in inp.type.tensor_type.shape.dim
        ]
        metadata["inputs"].append(
            {
                "name": inp.name,
                "shape": shape,
                "dtype": inp.type.tensor_type.elem_type,
            }
        )

    # Output shapes and types
    for out in model.graph.output:
        shape = [
            dim.dim_value if dim.dim_value > 0 else dim.dim_param
            for dim in out.type.tensor_type.shape.dim
        ]
        metadata["outputs"].append(
            {
                "name": out.name,
                "shape": shape,
                "dtype": out.type.tensor_type.elem_type,
            }
        )

    return metadata
