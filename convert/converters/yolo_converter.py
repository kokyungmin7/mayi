"""YOLO26-Pose model converter for ONNX and TensorRT."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class YOLOPoseConverter:
    """Converter for YOLO26-Pose model to ONNX/TensorRT.

    Uses Ultralytics built-in export functionality.
    Note: The BotSORT tracker will remain in Python - only detection
    and pose estimation are converted.
    """

    DEFAULT_MODEL_PATH = "models/yolo26n-pose.pt"

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        """Initialize converter.

        Args:
            model_path: Path to YOLO .pt model file
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")

    def convert_to_onnx(
        self,
        output_path: str | Path | None = None,
        imgsz: int = 640,
        simplify: bool = True,
        dynamic: bool = True,
        opset: int = 17,
    ) -> Path:
        """Convert YOLO model to ONNX format.

        Args:
            output_path: Output path (default: same dir as model with .onnx extension)
            imgsz: Input image size
            simplify: Simplify ONNX model
            dynamic: Enable dynamic batch and image size
            opset: ONNX opset version

        Returns:
            Path to converted ONNX model
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics is required. Install with: pip install ultralytics"
            )

        logger.info("Loading YOLO model from %s", self.model_path)
        model = YOLO(str(self.model_path))

        if output_path is None:
            output_path = self.model_path.parent / f"{self.model_path.stem}.onnx"
        else:
            output_path = Path(output_path)

        logger.info("Exporting to ONNX (imgsz=%d, dynamic=%s)...", imgsz, dynamic)

        # Export using Ultralytics built-in export
        exported_path = model.export(
            format="onnx",
            imgsz=imgsz,
            simplify=simplify,
            dynamic=dynamic,
            opset=opset,
        )

        # Ultralytics returns string path, convert to Path
        exported_path = Path(exported_path)

        # Move to desired output path if different
        if exported_path != output_path:
            exported_path.rename(output_path)
            logger.info("ONNX model saved to %s", output_path)
        else:
            logger.info("ONNX export complete: %s", exported_path)

        return output_path

    def convert_to_tensorrt(
        self,
        output_path: str | Path | None = None,
        imgsz: int = 640,
        half: bool = True,
        workspace: int = 4,
        dynamic: bool = True,
    ) -> Path:
        """Convert YOLO model directly to TensorRT engine.

        Args:
            output_path: Output path (default: same dir as model with .engine extension)
            imgsz: Input image size
            half: Use FP16 precision
            workspace: GPU workspace size in GB
            dynamic: Enable dynamic batch and image size

        Returns:
            Path to TensorRT engine
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics is required. Install with: pip install ultralytics"
            )

        logger.info("Loading YOLO model from %s", self.model_path)
        model = YOLO(str(self.model_path))

        if output_path is None:
            output_path = self.model_path.parent / f"{self.model_path.stem}.engine"
        else:
            output_path = Path(output_path)

        logger.info(
            "Exporting to TensorRT (imgsz=%d, half=%s, workspace=%dGB)...",
            imgsz,
            half,
            workspace,
        )

        # Export using Ultralytics built-in export
        exported_path = model.export(
            format="engine",  # TensorRT
            imgsz=imgsz,
            half=half,
            workspace=workspace,
            dynamic=dynamic,
        )

        # Ultralytics returns string path, convert to Path
        exported_path = Path(exported_path)

        # Move to desired output path if different
        if exported_path != output_path:
            exported_path.rename(output_path)
            logger.info("TensorRT engine saved to %s", output_path)
        else:
            logger.info("TensorRT export complete: %s", exported_path)

        return output_path

    def validate_conversion(
        self,
        onnx_path: str | Path,
        num_samples: int = 10,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> dict:
        """Validate ONNX model outputs against PyTorch.

        Args:
            onnx_path: Path to ONNX model
            num_samples: Number of random samples to test
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for matching detections

        Returns:
            Validation results dictionary
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics is required. Install with: pip install ultralytics"
            )

        import numpy as np

        onnx_path = Path(onnx_path)

        logger.info("Validating ONNX model against PyTorch...")

        # Load both models
        pt_model = YOLO(str(self.model_path))
        onnx_model = YOLO(str(onnx_path))

        results = {
            "passed": 0,
            "failed": 0,
            "detection_matches": [],
            "keypoint_differences": [],
        }

        # Test with random inputs
        for i in range(num_samples):
            # Generate random image
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

            # PyTorch inference
            pt_results = pt_model.predict(
                dummy_image,
                conf=conf_threshold,
                verbose=False,
            )[0]

            # ONNX inference
            onnx_results = onnx_model.predict(
                dummy_image,
                conf=conf_threshold,
                verbose=False,
            )[0]

            # Compare number of detections
            pt_boxes = pt_results.boxes
            onnx_boxes = onnx_results.boxes

            pt_count = len(pt_boxes) if pt_boxes is not None else 0
            onnx_count = len(onnx_boxes) if onnx_boxes is not None else 0

            match_rate = min(pt_count, onnx_count) / max(pt_count, onnx_count, 1)
            results["detection_matches"].append(match_rate)

            # Compare keypoints if available
            if pt_results.keypoints is not None and onnx_results.keypoints is not None:
                pt_kps = pt_results.keypoints.data.cpu().numpy()
                onnx_kps = onnx_results.keypoints.data.cpu().numpy()

                if pt_kps.shape == onnx_kps.shape and pt_kps.size > 0:
                    kp_diff = float(np.mean(np.abs(pt_kps - onnx_kps)))
                    results["keypoint_differences"].append(kp_diff)

            # Consider passed if detection match rate > 80%
            if match_rate >= 0.8:
                results["passed"] += 1
            else:
                results["failed"] += 1
                logger.warning(
                    "Sample %d: PT detections=%d, ONNX detections=%d (match=%.2f%%)",
                    i,
                    pt_count,
                    onnx_count,
                    match_rate * 100,
                )

        # Summary statistics
        results["mean_detection_match"] = float(
            np.mean(results["detection_matches"])
        )
        if results["keypoint_differences"]:
            results["mean_keypoint_diff"] = float(
                np.mean(results["keypoint_differences"])
            )

        logger.info(
            "Validation complete: %d/%d passed (%.1f%%)",
            results["passed"],
            num_samples,
            100.0 * results["passed"] / num_samples,
        )
        logger.info(
            "Mean detection match rate: %.2f%%",
            results["mean_detection_match"] * 100,
        )

        return results
