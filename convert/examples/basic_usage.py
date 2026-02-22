"""Basic usage examples for the MAYI conversion system."""

import cv2
import numpy as np

# Example 1: Converting SigLIP2 to ONNX
def example_convert_siglip():
    """Convert SigLIP2 model to ONNX."""
    from convert.converters import SigLIPConverter

    print("Converting SigLIP2 to ONNX...")
    converter = SigLIPConverter()

    # Convert to ONNX
    onnx_path = converter.convert_to_onnx(
        output_path="models/siglip2-reid.onnx",
        optimize=True,
    )
    print(f"ONNX model saved to: {onnx_path}")

    # Validate conversion
    results = converter.validate_conversion(onnx_path, num_samples=10)
    print(f"Validation: {results['passed']}/10 passed")
    print(f"Mean cosine similarity: {results['mean_cosine_similarity']:.6f}")


# Example 2: Converting to TensorRT
def example_convert_tensorrt():
    """Convert ONNX to TensorRT engine."""
    from convert.converters import SigLIPConverter

    print("Converting ONNX to TensorRT...")
    converter = SigLIPConverter()

    # Convert to TensorRT
    engine_path = converter.convert_to_tensorrt(
        onnx_path="models/siglip2-reid-optimized.onnx",
        output_path="models/siglip2-reid.engine",
        fp16=True,
        workspace_gb=4,
    )
    print(f"TensorRT engine saved to: {engine_path}")


# Example 3: Using ONNX embedder
def example_use_onnx_embedder():
    """Use ONNX embedder for inference."""
    from convert.runtime import ONNXReIDEmbedder

    print("Loading ONNX embedder...")
    embedder = ONNXReIDEmbedder(
        onnx_path="models/siglip2-reid-optimized.onnx",
        device="cuda",
    )

    # Load test image
    crop = cv2.imread("test_person.jpg")
    if crop is None:
        # Generate random image for testing
        crop = np.random.randint(0, 256, (256, 128, 3), dtype=np.uint8)

    # Extract embedding
    embedding = embedder.extract(crop)
    if embedding is not None:
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    else:
        print("Extraction failed")


# Example 4: Using TensorRT embedder
def example_use_tensorrt_embedder():
    """Use TensorRT embedder for inference."""
    try:
        from convert.runtime import TensorRTReIDEmbedder
    except ImportError:
        print("TensorRT not available. Install with: pip install tensorrt pycuda")
        return

    print("Loading TensorRT embedder...")
    embedder = TensorRTReIDEmbedder(
        engine_path="models/siglip2-reid.engine",
        max_batch_size=8,
    )

    # Load test image
    crop = np.random.randint(0, 256, (256, 128, 3), dtype=np.uint8)

    # Extract embedding
    embedding = embedder.extract(crop)
    if embedding is not None:
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")


# Example 5: Validating model outputs
def example_validate_models():
    """Validate that converted model matches PyTorch."""
    from mayi.reid.embedder import ReIDEmbedder
    from convert.runtime import ONNXReIDEmbedder
    from convert.verification import EmbeddingValidator

    print("Loading models...")
    pytorch_model = ReIDEmbedder()
    onnx_model = ONNXReIDEmbedder(onnx_path="models/siglip2-reid-optimized.onnx")

    # Generate test images
    validator = EmbeddingValidator()
    test_images = validator.generate_random_images(num_images=20)

    print("Validating outputs...")
    results = validator.validate_outputs(pytorch_model, onnx_model, test_images)

    print(f"\nValidation Results:")
    print(f"  Passed: {results['passed']}/{len(test_images)}")
    print(f"  Mean cosine similarity: {results['mean_cosine_similarity']:.6f}")
    print(f"  Mean MSE: {results['mean_mse']:.2e}")


# Example 6: Benchmarking performance
def example_benchmark_models():
    """Benchmark PyTorch vs ONNX vs TensorRT."""
    from mayi.reid.embedder import ReIDEmbedder
    from convert.runtime import ONNXReIDEmbedder
    from convert.verification import EmbeddingValidator, InferenceBenchmark

    print("Loading models...")
    pytorch_model = ReIDEmbedder()
    onnx_model = ONNXReIDEmbedder(onnx_path="models/siglip2-reid-optimized.onnx")

    # Try loading TensorRT if available
    models = {"pytorch": pytorch_model, "onnx": onnx_model}
    try:
        from convert.runtime import TensorRTReIDEmbedder

        trt_model = TensorRTReIDEmbedder(engine_path="models/siglip2-reid.engine")
        models["tensorrt"] = trt_model
    except ImportError:
        print("TensorRT not available, skipping TensorRT benchmark")

    # Generate test images
    validator = EmbeddingValidator()
    test_images = validator.generate_random_images(num_images=50)

    # Run benchmark
    print("Running benchmark...")
    benchmarker = InferenceBenchmark(warmup=10, runs=100)
    results = benchmarker.compare_models(models, test_images)

    # Print results table
    print("\nBenchmark Results:")
    print(benchmarker.format_results_table(results))


# Example 7: Converting YOLO
def example_convert_yolo():
    """Convert YOLO model to ONNX."""
    from convert.converters import YOLOPoseConverter

    print("Converting YOLO to ONNX...")
    converter = YOLOPoseConverter(model_path="models/yolo26n-pose.pt")

    # Convert to ONNX
    onnx_path = converter.convert_to_onnx(
        output_path="models/yolo26n-pose.onnx",
        imgsz=640,
        dynamic=True,
    )
    print(f"ONNX model saved to: {onnx_path}")

    # Validate
    results = converter.validate_conversion(onnx_path, num_samples=5)
    print(f"Validation: {results['passed']}/5 passed")


# Example 8: Using from configuration
def example_from_config():
    """Load embedder from configuration dictionary."""
    from mayi.reid.embedder import ReIDEmbedder

    # PyTorch backend (default)
    config_pytorch = {
        "backend": "pytorch",
        "model": "MarketaJu/siglip2-person-description-reid",
        "device": "cuda",
    }
    embedder = ReIDEmbedder.from_config(config_pytorch)
    print(f"Loaded embedder: {type(embedder).__name__}")

    # ONNX backend
    config_onnx = {
        "backend": "onnx",
        "onnx_model_path": "models/siglip2-reid-optimized.onnx",
        "device": "cuda",
    }
    embedder = ReIDEmbedder.from_config(config_onnx)
    print(f"Loaded embedder: {type(embedder).__name__}")

    # TensorRT backend
    config_trt = {
        "backend": "tensorrt",
        "tensorrt_engine_path": "models/siglip2-reid.engine",
        "max_batch_size": 8,
    }
    try:
        embedder = ReIDEmbedder.from_config(config_trt)
        print(f"Loaded embedder: {type(embedder).__name__}")
    except ImportError:
        print("TensorRT not available")


if __name__ == "__main__":
    import sys

    examples = {
        "1": ("Convert SigLIP2 to ONNX", example_convert_siglip),
        "2": ("Convert ONNX to TensorRT", example_convert_tensorrt),
        "3": ("Use ONNX embedder", example_use_onnx_embedder),
        "4": ("Use TensorRT embedder", example_use_tensorrt_embedder),
        "5": ("Validate models", example_validate_models),
        "6": ("Benchmark models", example_benchmark_models),
        "7": ("Convert YOLO", example_convert_yolo),
        "8": ("Load from config", example_from_config),
    }

    print("MAYI Conversion System - Examples")
    print("=" * 50)
    for key, (desc, _) in examples.items():
        print(f"{key}. {desc}")
    print("=" * 50)

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nSelect example (1-8): ").strip()

    if choice in examples:
        _, func = examples[choice]
        print(f"\nRunning: {examples[choice][0]}")
        print("-" * 50)
        func()
    else:
        print("Invalid choice")
