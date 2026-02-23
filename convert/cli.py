"""Command-line interface for model conversion."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """MAYI Model Conversion Tool

    Convert MAYI models to ONNX and TensorRT formats for optimized inference.
    """
    pass


@cli.command()
@click.option(
    "--model",
    type=click.Choice(["siglip", "yolo", "qwen"]),
    required=True,
    help="Model to convert",
)
@click.option(
    "--format",
    type=click.Choice(["onnx", "tensorrt", "both"]),
    required=True,
    help="Target format",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Configuration file (YAML)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="models/converted",
    help="Output directory for converted models",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    help="Path to source model (for YOLO)",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "cpu"]),
    default="auto",
    help="Device to use for model loading (auto=detect CUDA)",
)
def convert(model, format, config, output_dir, model_path, device):
    """Convert models to ONNX/TensorRT."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config if provided
    config_data = {}
    if config:
        with open(config, "r") as f:
            config_data = yaml.safe_load(f)

    if model == "siglip":
        from convert.converters import SigLIPConverter

        converter = SigLIPConverter()

        if format in ["onnx", "both"]:
            logger.info("Converting SigLIP to ONNX...")
            onnx_path = output_dir / "siglip2-reid.onnx"
            onnx_config = config_data.get("onnx", {})
            result_path = converter.convert_to_onnx(
                output_path=onnx_path,
                **onnx_config,
            )
            click.echo(f"✓ ONNX model saved to: {result_path}")

        if format in ["tensorrt", "both"]:
            logger.info("Converting SigLIP to TensorRT...")
            # Need ONNX model first
            if format == "tensorrt":
                onnx_path = output_dir / "siglip2-reid-optimized.onnx"
                if not onnx_path.exists():
                    onnx_path = output_dir / "siglip2-reid.onnx"
                if not onnx_path.exists():
                    click.echo(
                        "Error: ONNX model not found. Convert to ONNX first.",
                        err=True,
                    )
                    sys.exit(1)
            else:
                # Format is "both", use the ONNX we just created
                pass

            engine_path = output_dir / "siglip2-reid.engine"
            trt_config = config_data.get("tensorrt", {})
            result_path = converter.convert_to_tensorrt(
                onnx_path=onnx_path if format == "tensorrt" else result_path,
                output_path=engine_path,
                **trt_config,
            )
            click.echo(f"✓ TensorRT engine saved to: {result_path}")

    elif model == "yolo":
        from convert.converters import YOLOPoseConverter

        if model_path is None:
            model_path = "models/yolo26n-pose.pt"

        if not Path(model_path).exists():
            click.echo(f"Error: YOLO model not found: {model_path}", err=True)
            sys.exit(1)

        converter = YOLOPoseConverter(model_path=model_path)

        if format in ["onnx", "both"]:
            logger.info("Converting YOLO to ONNX...")
            onnx_path = output_dir / "yolo26n-pose.onnx"
            onnx_config = config_data.get("onnx", {})
            result_path = converter.convert_to_onnx(
                output_path=onnx_path,
                **onnx_config,
            )
            click.echo(f"✓ ONNX model saved to: {result_path}")

        if format in ["tensorrt", "both"]:
            logger.info("Converting YOLO to TensorRT...")
            engine_path = output_dir / "yolo26n-pose.engine"
            trt_config = config_data.get("tensorrt", {})
            result_path = converter.convert_to_tensorrt(
                output_path=engine_path,
                **trt_config,
            )
            click.echo(f"✓ TensorRT engine saved to: {result_path}")

    elif model == "qwen":
        from convert.converters import QwenConverter

        # Map CLI device option to converter parameter
        device_param = None if device == "auto" else device

        converter = QwenConverter(
            use_fp16=True,
            device=device_param,
        )

        # Log device being used
        logger.info(f"Using device: {converter.device}")

        if format in ["onnx", "both"]:
            logger.info("Converting Qwen3-VL to ONNX (vision encoder + text decoder)...")

            # Export vision encoder
            vision_path = output_dir / "qwen3vl-vision-encoder.onnx"
            onnx_config = config_data.get("onnx", {})
            vision_result = converter.convert_vision_encoder_to_onnx(
                output_path=vision_path,
                opset_version=onnx_config.get("opset_version", 18),
                optimize=onnx_config.get("optimize", True),
            )
            click.echo(f"✓ Vision encoder saved to: {vision_result}")

            # Export text decoder
            decoder_path = output_dir / "qwen3vl-text-decoder.onnx"
            decoder_result = converter.convert_text_decoder_to_onnx(
                output_path=decoder_path,
                opset_version=onnx_config.get("opset_version", 18),
                max_seq_length=onnx_config.get("max_seq_length", 512),
            )
            click.echo(f"✓ Text decoder saved to: {decoder_result}")

        if format in ["tensorrt", "both"]:
            logger.info("Converting Qwen3-VL to TensorRT...")

            # Need ONNX models first
            if format == "tensorrt":
                vision_onnx = output_dir / "qwen3vl-vision-encoder-optimized.onnx"
                if not vision_onnx.exists():
                    vision_onnx = output_dir / "qwen3vl-vision-encoder.onnx"
                decoder_onnx = output_dir / "qwen3vl-text-decoder.onnx"

                if not vision_onnx.exists() or not decoder_onnx.exists():
                    click.echo(
                        "Error: ONNX models not found. Convert to ONNX first.",
                        err=True,
                    )
                    sys.exit(1)
            else:
                # Format is "both", use the ONNX we just created
                vision_onnx = vision_result
                decoder_onnx = decoder_result

            # Convert to TensorRT
            vision_engine_path = output_dir / "qwen3vl-vision-encoder.engine"
            decoder_engine_path = output_dir / "qwen3vl-text-decoder.engine"
            trt_config = config_data.get("tensorrt", {})

            vision_engine, decoder_engine = converter.convert_to_tensorrt(
                vision_encoder_path=vision_onnx,
                text_decoder_path=decoder_onnx,
                vision_output_path=vision_engine_path,
                decoder_output_path=decoder_engine_path,
                fp16=trt_config.get("fp16", True),
                int8=trt_config.get("int8", False),
            )
            click.echo(f"✓ Vision encoder TensorRT engine saved to: {vision_engine}")
            click.echo(f"✓ Text decoder TensorRT engine saved to: {decoder_engine}")


@cli.command()
@click.option(
    "--model",
    type=click.Choice(["siglip", "yolo"]),
    required=True,
    help="Model to verify",
)
@click.option(
    "--onnx-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to ONNX model",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    help="Path to original model (for YOLO)",
)
@click.option(
    "--num-samples",
    type=int,
    default=10,
    help="Number of test samples",
)
def verify(model, onnx_path, model_path, num_samples):
    """Verify converted model outputs."""
    if model == "siglip":
        from convert.converters import SigLIPConverter

        converter = SigLIPConverter()
        results = converter.validate_conversion(
            onnx_path=onnx_path,
            num_samples=num_samples,
        )

        click.echo(f"\nValidation Results:")
        click.echo(f"  Passed: {results['passed']}/{num_samples}")
        click.echo(f"  Mean cosine similarity: {results['mean_cosine_similarity']:.6f}")
        click.echo(f"  Mean MSE: {results['mean_mse']:.2e}")

        if results["passed"] == num_samples:
            click.echo("\n✓ All tests passed!")
        else:
            click.echo(f"\n✗ {results['failed']} tests failed", err=True)
            sys.exit(1)

    elif model == "yolo":
        from convert.converters import YOLOPoseConverter

        if model_path is None:
            model_path = "models/yolo26n-pose.pt"

        converter = YOLOPoseConverter(model_path=model_path)
        results = converter.validate_conversion(
            onnx_path=onnx_path,
            num_samples=num_samples,
        )

        click.echo(f"\nValidation Results:")
        click.echo(f"  Passed: {results['passed']}/{num_samples}")
        click.echo(f"  Mean detection match: {results['mean_detection_match']*100:.1f}%")

        if results["passed"] == num_samples:
            click.echo("\n✓ All tests passed!")
        else:
            click.echo(f"\n✗ {results['failed']} tests failed", err=True)
            sys.exit(1)


@cli.command()
@click.option(
    "--model",
    type=click.Choice(["siglip"]),
    required=True,
    help="Model to benchmark",
)
@click.option(
    "--backends",
    multiple=True,
    default=["pytorch", "onnx"],
    help="Backends to compare",
)
@click.option(
    "--pytorch-model",
    type=str,
    default="MarketaJu/siglip2-person-description-reid",
    help="PyTorch model name",
)
@click.option(
    "--onnx-path",
    type=click.Path(exists=True),
    help="Path to ONNX model",
)
@click.option(
    "--tensorrt-path",
    type=click.Path(exists=True),
    help="Path to TensorRT engine",
)
@click.option(
    "--num-samples",
    type=int,
    default=100,
    help="Number of benchmark samples",
)
@click.option(
    "--warmup",
    type=int,
    default=10,
    help="Number of warmup iterations",
)
def benchmark(
    model,
    backends,
    pytorch_model,
    onnx_path,
    tensorrt_path,
    num_samples,
    warmup,
):
    """Benchmark model performance."""
    if model == "siglip":
        from convert.verification import EmbeddingValidator, InferenceBenchmark

        # Generate test images
        validator = EmbeddingValidator()
        test_images = validator.generate_random_images(num_images=50)

        # Load models
        models = {}

        if "pytorch" in backends:
            from mayi.reid.embedder import ReIDEmbedder

            logger.info("Loading PyTorch model...")
            models["pytorch"] = ReIDEmbedder(model_name=pytorch_model)

        if "onnx" in backends:
            if onnx_path is None:
                click.echo("Error: --onnx-path required for ONNX backend", err=True)
                sys.exit(1)

            from convert.runtime import ONNXReIDEmbedder

            logger.info("Loading ONNX model...")
            models["onnx"] = ONNXReIDEmbedder(onnx_path=onnx_path)

        if "tensorrt" in backends:
            if tensorrt_path is None:
                click.echo(
                    "Error: --tensorrt-path required for TensorRT backend",
                    err=True,
                )
                sys.exit(1)

            from convert.runtime import TensorRTReIDEmbedder

            logger.info("Loading TensorRT model...")
            models["tensorrt"] = TensorRTReIDEmbedder(engine_path=tensorrt_path)

        # Run benchmark
        benchmarker = InferenceBenchmark(warmup=warmup, runs=num_samples)
        results = benchmarker.compare_models(models, test_images)

        # Print results table
        click.echo("\n" + benchmarker.format_results_table(results))


if __name__ == "__main__":
    cli()
