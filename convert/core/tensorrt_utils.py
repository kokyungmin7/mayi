"""TensorRT conversion utilities."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def convert_onnx_to_tensorrt(
    onnx_path: str | Path,
    engine_path: str | Path,
    fp16: bool = True,
    int8: bool = False,
    workspace_gb: int = 4,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 8,
) -> Path:
    """Convert ONNX model to TensorRT engine.

    Supports models with dynamic axes (e.g., variable num_patches for vision models).
    Automatically detects all inputs and creates optimization profiles for dynamic shapes.

    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        fp16: Enable FP16 precision
        int8: Enable INT8 precision (requires calibration)
        workspace_gb: GPU workspace size in GB
        min_batch: Minimum batch size for dynamic batching
        opt_batch: Optimal batch size for dynamic batching
        max_batch: Maximum batch size for dynamic batching

    Returns:
        Path to TensorRT engine
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError(
            "TensorRT is not installed. "
            "Install with: pip install tensorrt"
        )

    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)

    logger.info("Converting ONNX to TensorRT: %s -> %s", onnx_path, engine_path)

    # Create TensorRT builder
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)

    # Parse ONNX model
    logger.info("Parsing ONNX model...")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = []
            for i in range(parser.num_errors):
                errors.append(str(parser.get_error(i)))
            raise RuntimeError(f"ONNX parsing failed:\n" + "\n".join(errors))

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30)
    )

    if fp16:
        logger.info("Enabling FP16 precision")
        config.set_flag(trt.BuilderFlag.FP16)

    if int8:
        logger.info("Enabling INT8 precision (requires calibration)")
        config.set_flag(trt.BuilderFlag.INT8)
        # Note: Calibration would be needed here for INT8

    # Auto-detect all inputs and create optimization profile
    num_inputs = network.num_inputs
    logger.info(f"Detected {num_inputs} input(s)")

    profile = builder.create_optimization_profile()

    for i in range(num_inputs):
        input_tensor = network.get_input(i)
        input_name = input_tensor.name
        input_shape = tuple(input_tensor.shape)

        logger.info(f"  Input {i}: {input_name}, shape={input_shape}")

        # Handle dynamic shapes
        # For vision models: hidden_states has dynamic num_patches, grid_thw is fixed
        has_dynamic_dim = any(dim == -1 for dim in input_shape)

        if has_dynamic_dim:
            # Replace -1 with actual batch/patch sizes
            # Assume first dynamic dimension is num_patches or batch_size
            min_shape = []
            opt_shape = []
            max_shape = []

            for dim in input_shape:
                if dim == -1:
                    # Dynamic dimension: use provided batch sizes
                    min_shape.append(min_batch)
                    opt_shape.append(opt_batch)
                    max_shape.append(max_batch)
                else:
                    # Fixed dimension
                    min_shape.append(dim)
                    opt_shape.append(dim)
                    max_shape.append(dim)

            min_shape = tuple(min_shape)
            opt_shape = tuple(opt_shape)
            max_shape = tuple(max_shape)

            logger.info(
                f"    Dynamic shape profile: min={min_shape}, opt={opt_shape}, max={max_shape}"
            )

            profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
        else:
            # Fixed shape input (e.g., grid_thw [1, 3])
            logger.info(f"    Fixed shape: {input_shape}")
            profile.set_shape(input_name, min=input_shape, opt=input_shape, max=input_shape)

    config.add_optimization_profile(profile)

    # Build engine
    logger.info("Building TensorRT engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed")

    # Save engine
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    logger.info("TensorRT engine saved to %s", engine_path)
    return engine_path


def verify_tensorrt_engine(engine_path: str | Path) -> dict:
    """Verify TensorRT engine and get metadata.

    Args:
        engine_path: Path to TensorRT engine

    Returns:
        Dictionary with engine metadata
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError(
            "TensorRT is not installed. "
            "Install with: pip install tensorrt"
        )

    engine_path = Path(engine_path)

    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError(f"Failed to load engine from {engine_path}")

    metadata = {
        "num_bindings": engine.num_bindings,
        "num_optimization_profiles": engine.num_optimization_profiles,
        "max_batch_size": engine.max_batch_size,
        "device_memory_size": engine.device_memory_size,
        "bindings": [],
    }

    for i in range(engine.num_bindings):
        binding_name = engine.get_binding_name(i)
        binding_shape = engine.get_binding_shape(i)
        binding_dtype = engine.get_binding_dtype(i)
        is_input = engine.binding_is_input(i)

        metadata["bindings"].append(
            {
                "name": binding_name,
                "shape": tuple(binding_shape),
                "dtype": str(binding_dtype),
                "is_input": is_input,
            }
        )

    return metadata
