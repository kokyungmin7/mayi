"""Model validation utilities to compare outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingValidator:
    """Validates that converted model outputs match original PyTorch model."""

    def __init__(
        self,
        cosine_threshold: float = 0.99,
        mse_threshold: float = 1e-4,
    ) -> None:
        """Initialize validator.

        Args:
            cosine_threshold: Minimum acceptable cosine similarity
            mse_threshold: Maximum acceptable MSE error
        """
        self.cosine_threshold = cosine_threshold
        self.mse_threshold = mse_threshold

    def validate_outputs(
        self,
        pytorch_embedder,
        converted_embedder,
        test_images: list[np.ndarray],
    ) -> dict:
        """Compare embeddings from PyTorch and converted models.

        Args:
            pytorch_embedder: Original PyTorch embedder
            converted_embedder: Converted embedder (ONNX or TensorRT)
            test_images: List of test images (BGR format)

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating embeddings on %d test images...", len(test_images))

        results = {
            "passed": 0,
            "failed": 0,
            "cosine_similarities": [],
            "mse_errors": [],
            "failed_indices": [],
        }

        for i, img in enumerate(test_images):
            # Extract embeddings
            baseline = pytorch_embedder.extract(img)
            converted = converted_embedder.extract(img)

            if baseline is None or converted is None:
                logger.warning("Sample %d: Extraction failed", i)
                results["failed"] += 1
                results["failed_indices"].append(i)
                continue

            # Cosine similarity (should be > threshold)
            cos_sim = float(np.dot(baseline, converted))
            results["cosine_similarities"].append(cos_sim)

            # MSE (should be < threshold)
            mse = float(np.mean((baseline - converted) ** 2))
            results["mse_errors"].append(mse)

            if cos_sim >= self.cosine_threshold and mse <= self.mse_threshold:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["failed_indices"].append(i)
                logger.warning(
                    "Sample %d failed: cosine=%.6f (threshold=%.6f), "
                    "mse=%.2e (threshold=%.2e)",
                    i,
                    cos_sim,
                    self.cosine_threshold,
                    mse,
                    self.mse_threshold,
                )

        # Compute summary statistics
        if results["cosine_similarities"]:
            results["mean_cosine_similarity"] = float(
                np.mean(results["cosine_similarities"])
            )
            results["min_cosine_similarity"] = float(
                np.min(results["cosine_similarities"])
            )
            results["max_cosine_similarity"] = float(
                np.max(results["cosine_similarities"])
            )

        if results["mse_errors"]:
            results["mean_mse"] = float(np.mean(results["mse_errors"]))
            results["max_mse"] = float(np.max(results["mse_errors"]))
            results["min_mse"] = float(np.min(results["mse_errors"]))

        # Overall pass rate
        total = len(test_images)
        results["pass_rate"] = results["passed"] / total if total > 0 else 0.0

        logger.info(
            "Validation complete: %d/%d passed (%.1f%%)",
            results["passed"],
            total,
            100.0 * results["pass_rate"],
        )

        if results["cosine_similarities"]:
            logger.info(
                "Cosine similarity: mean=%.6f, min=%.6f, max=%.6f",
                results["mean_cosine_similarity"],
                results["min_cosine_similarity"],
                results["max_cosine_similarity"],
            )

        if results["mse_errors"]:
            logger.info(
                "MSE: mean=%.2e, min=%.2e, max=%.2e",
                results["mean_mse"],
                results["min_mse"],
                results["max_mse"],
            )

        return results

    def load_test_images(
        self,
        data_dir: str | Path,
        num_images: int = 50,
    ) -> list[np.ndarray]:
        """Load test images from a directory.

        Args:
            data_dir: Directory containing test images
            num_images: Maximum number of images to load

        Returns:
            List of BGR images
        """
        import cv2

        data_dir = Path(data_dir)
        image_paths = []

        # Find image files
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_paths.extend(data_dir.rglob(ext))

        # Limit number of images
        image_paths = image_paths[:num_images]

        logger.info("Loading %d test images from %s", len(image_paths), data_dir)

        images = []
        for path in image_paths:
            img = cv2.imread(str(path))
            if img is not None:
                images.append(img)

        logger.info("Loaded %d images", len(images))
        return images

    def generate_random_images(
        self,
        num_images: int = 50,
        size: tuple[int, int] = (256, 128),
    ) -> list[np.ndarray]:
        """Generate random test images.

        Args:
            num_images: Number of images to generate
            size: Image size (height, width)

        Returns:
            List of random BGR images
        """
        logger.info("Generating %d random test images", num_images)

        images = []
        for _ in range(num_images):
            img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
            images.append(img)

        return images
