from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum

import numpy as np


# ---------------------------------------------------------------------------
# COCO Pose Keypoints (YOLO pose output order)
# ---------------------------------------------------------------------------

class KeypointIndex(IntEnum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


KEYPOINT_NAME_TO_INDEX: dict[str, int] = {kp.name.lower(): kp.value for kp in KeypointIndex}


# ---------------------------------------------------------------------------
# Detection (YOLO output per person)
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """YOLO26 detection + tracking + pose output for a single person."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (absolute pixels)
    confidence: float
    tracker_id: int | None
    keypoints: np.ndarray | None  # shape (17, 3) — x, y, visibility_conf
    frame_idx: int
    camera_id: str

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    @property
    def area(self) -> float:
        return self.width * self.height

    def crop_from_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = max(0, int(self.bbox[0]))
        y1 = max(0, int(self.bbox[1]))
        x2 = min(w, int(self.bbox[2]))
        y2 = min(h, int(self.bbox[3]))
        return frame[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Track State Machine
# ---------------------------------------------------------------------------

class TrackState(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    SUSPICIOUS = "suspicious"
    REASSIGNED = "reassigned"
    LOST = "lost"
    RECOVERED = "recovered"
    EXPIRED = "expired"
    DEAD = "dead"


@dataclass
class Track:
    """Runtime state of a single tracker ID within a camera."""

    tracker_id: int
    camera_id: str
    state: TrackState
    global_id: str | None = None

    first_frame_idx: int = 0
    last_frame_idx: int = 0
    pending_frame_count: int = 0
    lost_frame_count: int = 0

    last_bbox: tuple[float, float, float, float] | None = None
    last_center: tuple[float, float] | None = None
    consistency_frame_counter: int = 0


# ---------------------------------------------------------------------------
# Track History (for ID Mapper)
# ---------------------------------------------------------------------------

@dataclass
class TrackSegment:
    """One contiguous segment of a person visible in a camera."""

    camera_id: str
    tracker_id: int
    global_id: str
    start_frame: int
    end_frame: int


# ---------------------------------------------------------------------------
# Condition System
# ---------------------------------------------------------------------------

@dataclass
class ConditionResult:
    passed: bool
    score: float  # 0.0 ~ 1.0
    reason: str | None = None


@dataclass
class QualityReport:
    passed: bool
    overall_score: float  # 0.0 ~ 1.0  (mean of individual scores)
    results: dict[str, ConditionResult] = field(default_factory=dict)
    failed_at: str | None = None  # name of the condition that caused short-circuit


# ---------------------------------------------------------------------------
# Person Metadata (VLM output)
# ---------------------------------------------------------------------------

@dataclass
class PersonMetadata:
    gender: str | None = None
    top_color: str | None = None
    bottom_color: str | None = None
    top_type: str | None = None
    bottom_type: str | None = None


# ---------------------------------------------------------------------------
# Anchor Bank Entry
# ---------------------------------------------------------------------------

@dataclass
class AnchorEntry:
    """One anchor per global person ID."""

    global_id: str
    crop_image: np.ndarray  # BGR crop of the person
    embeddings: list[np.ndarray] = field(default_factory=list)
    representative_embedding: np.ndarray | None = None  # mean of embeddings
    quality_score: float = 0.0
    metadata: PersonMetadata | None = None

    camera_id: str = ""
    frame_idx: int = 0
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    timestamp: float = 0.0

    def add_embedding(self, embedding: np.ndarray) -> None:
        self.embeddings.append(embedding)
        self.representative_embedding = np.mean(self.embeddings, axis=0)


# ---------------------------------------------------------------------------
# Re-ID Matching
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    matched: bool
    global_id: str | None = None
    similarity: float = 0.0
    candidate_similarities: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Failure Detection
# ---------------------------------------------------------------------------

@dataclass
class FailureReport:
    detected: bool
    detector_name: str = ""
    reason: str | None = None
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Track Events (for JSON output)
# ---------------------------------------------------------------------------

@dataclass
class TrackEvent:
    event_type: str  # reid_new, reid_match, reid_failure, track_lost, ...
    frame_idx: int
    tracker_id: int
    global_id: str | None = None
    details: dict = field(default_factory=dict)
