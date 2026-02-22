"""Performance benchmarking utilities for model inference."""

from __future__ import annotations

import logging
import random
import time

import numpy as np

logger = logging.getLogger(__name__)


class InferenceBenchmark:
    """Benchmarks inference performance of embedding models."""

    def __init__(self, warmup: int = 10, runs: int = 100) -> None:
        """Initialize benchmark.

        Args:
            warmup: Number of warmup iterations
            runs: Number of benchmark iterations
        """
        self.warmup = warmup
        self.runs = runs

    def benchmark_model(
        self,
        embedder,
        test_images: list[np.ndarray],
        batch_sizes: list[int] | None = None,
    ) -> dict:
        """Measure latency and throughput of an embedder.

        Args:
            embedder: Embedder to benchmark (PyTorch, ONNX, or TensorRT)
            test_images: List of test images
            batch_sizes: List of batch sizes to test (default: [1])

        Returns:
            Dictionary with benchmark results
        """
        if batch_sizes is None:
            batch_sizes = [1]

        logger.info(
            "Benchmarking model (warmup=%d, runs=%d)...",
            self.warmup,
            self.runs,
        )

        results = {}

        for batch_size in batch_sizes:
            logger.info("Benchmarking batch_size=%d", batch_size)

            # Warmup
            for _ in range(self.warmup):
                img = random.choice(test_images)
                embedder.extract(img)

            # Benchmark
            latencies = []
            for _ in range(self.runs):
                img = random.choice(test_images)
                start = time.perf_counter()
                embedder.extract(img)
                latencies.append(time.perf_counter() - start)

            # Compute statistics
            latencies = np.array(latencies)
            results[f"batch_{batch_size}"] = {
                "mean_latency_ms": float(np.mean(latencies) * 1000),
                "std_latency_ms": float(np.std(latencies) * 1000),
                "min_latency_ms": float(np.min(latencies) * 1000),
                "max_latency_ms": float(np.max(latencies) * 1000),
                "p50_latency_ms": float(np.percentile(latencies, 50) * 1000),
                "p95_latency_ms": float(np.percentile(latencies, 95) * 1000),
                "p99_latency_ms": float(np.percentile(latencies, 99) * 1000),
                "throughput_fps": float(1.0 / np.mean(latencies)),
            }

            logger.info(
                "Batch %d: mean=%.2f ms, p50=%.2f ms, p95=%.2f ms, "
                "throughput=%.1f FPS",
                batch_size,
                results[f"batch_{batch_size}"]["mean_latency_ms"],
                results[f"batch_{batch_size}"]["p50_latency_ms"],
                results[f"batch_{batch_size}"]["p95_latency_ms"],
                results[f"batch_{batch_size}"]["throughput_fps"],
            )

        return results

    def compare_models(
        self,
        models: dict[str, any],
        test_images: list[np.ndarray],
        batch_sizes: list[int] | None = None,
    ) -> dict:
        """Compare performance of multiple models.

        Args:
            models: Dictionary mapping model names to embedder instances
            test_images: List of test images
            batch_sizes: List of batch sizes to test

        Returns:
            Dictionary with comparison results
        """
        logger.info("Comparing %d models...", len(models))

        results = {}
        for name, model in models.items():
            logger.info("Benchmarking %s...", name)
            results[name] = self.benchmark_model(model, test_images, batch_sizes)

        # Compute speedups relative to first model (baseline)
        baseline_name = list(models.keys())[0]
        baseline_results = results[baseline_name]

        for batch_size in (batch_sizes or [1]):
            batch_key = f"batch_{batch_size}"
            baseline_latency = baseline_results[batch_key]["mean_latency_ms"]

            logger.info("\n=== Batch size %d ===", batch_size)
            for name in models.keys():
                model_latency = results[name][batch_key]["mean_latency_ms"]
                speedup = baseline_latency / model_latency
                results[name][batch_key]["speedup"] = float(speedup)

                logger.info(
                    "%s: %.2f ms (%.2fx speedup)",
                    name,
                    model_latency,
                    speedup,
                )

        return results

    def format_results_table(self, results: dict) -> str:
        """Format benchmark results as a markdown table.

        Args:
            results: Results from compare_models()

        Returns:
            Markdown-formatted table string
        """
        lines = []
        lines.append("| Model | Batch | Mean (ms) | P50 (ms) | P95 (ms) | Throughput (FPS) | Speedup |")
        lines.append("|-------|-------|-----------|----------|----------|------------------|---------|")

        for model_name, model_results in results.items():
            for batch_key, metrics in model_results.items():
                if not batch_key.startswith("batch_"):
                    continue

                batch_size = batch_key.split("_")[1]
                speedup = metrics.get("speedup", 1.0)

                lines.append(
                    f"| {model_name} | {batch_size} | "
                    f"{metrics['mean_latency_ms']:.2f} | "
                    f"{metrics['p50_latency_ms']:.2f} | "
                    f"{metrics['p95_latency_ms']:.2f} | "
                    f"{metrics['throughput_fps']:.1f} | "
                    f"{speedup:.2f}x |"
                )

        return "\n".join(lines)
