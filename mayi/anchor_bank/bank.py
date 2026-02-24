from __future__ import annotations

import logging
from dataclasses import asdict

import numpy as np

from mayi.models.types import AnchorEntry, MatchResult, PersonMetadata

logger = logging.getLogger(__name__)


class AnchorBank:
    """Stores one anchor per global person ID.

    Each anchor holds:
    - A single best-quality BGR crop image
    - A list of embeddings (averaged into a representative vector)
    - VLM metadata (populated later)
    - Last-seen tracking info for speed-jump detection
    """

    def __init__(self) -> None:
        self._entries: dict[str, AnchorEntry] = {}
        self._last_seen: dict[str, tuple[str, tuple[float, float], int]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        global_id: str,
        crop_image: np.ndarray,
        embedding: np.ndarray,
        quality_score: float,
        camera_id: str,
        frame_idx: int,
        bbox: tuple[float, float, float, float],
        timestamp: float = 0.0,
    ) -> AnchorEntry:
        entry = AnchorEntry(
            global_id=global_id,
            crop_image=crop_image.copy(),
            quality_score=quality_score,
            camera_id=camera_id,
            frame_idx=frame_idx,
            bbox=bbox,
            timestamp=timestamp,
        )
        entry.add_embedding(embedding)
        self._entries[global_id] = entry

        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self._last_seen[global_id] = (camera_id, (cx, cy), frame_idx)

        logger.info(
            "[ANCHOR] Registered %s (quality=%.2f, frame=%d, cam=%s)",
            global_id, quality_score, frame_idx, camera_id,
        )
        return entry

    # ------------------------------------------------------------------
    # Search (cosine similarity — embeddings are L2-normalised)
    # ------------------------------------------------------------------

    def search(self, embedding: np.ndarray, threshold: float) -> MatchResult:
        if not self._entries:
            return MatchResult(matched=False)

        similarities: dict[str, float] = {}
        best_id: str | None = None
        best_sim: float = -1.0

        for gid, entry in self._entries.items():
            if entry.representative_embedding is None:
                continue
            sim = float(np.dot(embedding, entry.representative_embedding))
            similarities[gid] = sim
            if sim > best_sim:
                best_sim = sim
                best_id = gid

        if best_id is not None and best_sim >= threshold:
            return MatchResult(
                matched=True,
                global_id=best_id,
                similarity=best_sim,
                candidate_similarities=similarities,
            )
        return MatchResult(
            matched=False,
            similarity=best_sim if best_sim > -1 else 0.0,
            candidate_similarities=similarities,
        )

    def search_among(
        self,
        embedding: np.ndarray,
        global_ids: list[str],
        threshold: float,
    ) -> MatchResult:
        """Search for a match only among the specified global IDs."""
        if not global_ids:
            return MatchResult(matched=False)

        similarities: dict[str, float] = {}
        best_id: str | None = None
        best_sim: float = -1.0

        for gid in global_ids:
            entry = self._entries.get(gid)
            if entry is None or entry.representative_embedding is None:
                continue
            sim = float(np.dot(embedding, entry.representative_embedding))
            similarities[gid] = sim
            if sim > best_sim:
                best_sim = sim
                best_id = gid

        if best_id is not None and best_sim >= threshold:
            return MatchResult(
                matched=True,
                global_id=best_id,
                similarity=best_sim,
                candidate_similarities=similarities,
            )
        return MatchResult(
            matched=False,
            similarity=best_sim if best_sim > -1 else 0.0,
            candidate_similarities=similarities,
        )

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def add_embedding(self, global_id: str, embedding: np.ndarray) -> None:
        entry = self._entries.get(global_id)
        if entry is not None:
            entry.add_embedding(embedding)
            logger.debug(
                "[ANCHOR] %s embedding updated (total=%d)",
                global_id, len(entry.embeddings),
            )

    def update_crop_if_better(
        self,
        global_id: str,
        crop_image: np.ndarray,
        quality_score: float,
        camera_id: str,
        frame_idx: int,
        bbox: tuple[float, float, float, float],
    ) -> bool:
        entry = self._entries.get(global_id)
        if entry is None:
            return False
        if quality_score > entry.quality_score:
            entry.crop_image = crop_image.copy()
            entry.quality_score = quality_score
            entry.camera_id = camera_id
            entry.frame_idx = frame_idx
            entry.bbox = bbox
            entry.metadata = None
            logger.info(
                "[ANCHOR] %s crop upgraded (%.2f -> %.2f, frame=%d)",
                global_id, entry.quality_score, quality_score, frame_idx,
            )
            return True
        return False

    def update_last_seen(
        self,
        global_id: str,
        camera_id: str,
        center: tuple[float, float],
        frame_idx: int,
    ) -> None:
        self._last_seen[global_id] = (camera_id, center, frame_idx)

    def set_metadata(
        self,
        global_id: str,
        metadata: PersonMetadata,
    ) -> None:
        entry = self._entries.get(global_id)
        if entry is not None:
            entry.metadata = metadata
            logger.info("[ANCHOR] %s metadata updated", global_id)

    # ------------------------------------------------------------------
    # Metadata filtering
    # ------------------------------------------------------------------

    def filter_by_metadata(self, **filters: str | None) -> list[str]:
        """Return global_ids whose metadata matches the given filters.

        None filter values are ignored.  Only entries with populated
        metadata are considered.
        """
        result: list[str] = []
        for gid, entry in self._entries.items():
            if entry.metadata is None:
                continue
            meta_dict = asdict(entry.metadata)
            matches = True
            for field, value in filters.items():
                if value is None:
                    continue
                entry_val = meta_dict.get(field)
                if entry_val is not None and entry_val != value:
                    matches = False
                    break
            if matches:
                result.append(gid)
        return result

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, global_id: str) -> AnchorEntry | None:
        return self._entries.get(global_id)

    def get_last_seen(
        self, global_id: str,
    ) -> tuple[str, tuple[float, float], int] | None:
        return self._last_seen.get(global_id)

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def all_ids(self) -> list[str]:
        return list(self._entries.keys())

    def ids_needing_metadata(self) -> list[str]:
        """Return global_ids that have no VLM metadata yet."""
        return [
            gid for gid, entry in self._entries.items()
            if entry.metadata is None
        ]
