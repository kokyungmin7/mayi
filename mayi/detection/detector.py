from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from mayi.models.types import Detection


class PersonDetector:
    """Wraps YOLO26-pose for person detection, tracking, and pose
    keypoint extraction."""

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.5,
        tracker: str = "botsort",
    ) -> None:
        self._model = YOLO(str(model_path))
        self._confidence = confidence
        self._tracker = tracker if tracker.endswith(".yaml") else f"{tracker}.yaml"

    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int,
        camera_id: str,
    ) -> list[Detection]:
        results = self._model.track(
            frame,
            persist=True,
            conf=self._confidence,
            tracker=self._tracker,
            verbose=False,
        )

        detections: list[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy_all = boxes.xyxy.cpu().numpy()
            conf_all = boxes.conf.cpu().numpy()
            ids_all = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

            kps_all = None
            if result.keypoints is not None and result.keypoints.data.numel() > 0:
                kps_all = result.keypoints.data.cpu().numpy()  # (N, 17, 3)

            for i in range(len(boxes)):
                bbox = (
                    float(xyxy_all[i, 0]),
                    float(xyxy_all[i, 1]),
                    float(xyxy_all[i, 2]),
                    float(xyxy_all[i, 3]),
                )
                tracker_id = int(ids_all[i]) if ids_all is not None else None
                keypoints = kps_all[i] if kps_all is not None else None  # (17, 3)

                detections.append(
                    Detection(
                        bbox=bbox,
                        confidence=float(conf_all[i]),
                        tracker_id=tracker_id,
                        keypoints=keypoints,
                        frame_idx=frame_idx,
                        camera_id=camera_id,
                    )
                )

        return detections

    @classmethod
    def from_config(cls, config: dict) -> PersonDetector:
        model_path = config.get("model_path", "models/yolo26n-pose.pt")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        return cls(
            model_path=model_path,
            confidence=config.get("confidence", 0.5),
            tracker=config.get("tracker", "botsort"),
        )
