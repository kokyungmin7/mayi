from __future__ import annotations

import logging

import numpy as np

from mayi.anchor_bank.bank import AnchorBank
from mayi.conditions.base import ConditionPipeline
from mayi.models.types import (
    Detection,
    MatchResult,
    PersonMetadata,
    QualityReport,
    Track,
    TrackEvent,
    TrackState,
)
from mayi.reid.embedder import ReIDEmbedder
from mayi.reid.failure_detectors.base import FailureDetectorPipeline
from mayi.reid.failure_detectors.consistency_monitor import ConsistencyMonitor
from mayi.tracking.id_mapper import IDMapper
from mayi.vlm.analyzer import VLMAnalyzer
from mayi.vlm.verifier import VLMVerifier

logger = logging.getLogger(__name__)


class TrackManager:
    """Manages the lifecycle state machine for all tracked persons.

    Integrates:
      - Re-ID (SigLIP2) for identity resolution
      - Failure detectors (embedding collapse / speed jump)
      - Consistency monitor for active tracks
      - VLM (Qwen3-VL) for metadata extraction and verification
    """

    def __init__(
        self,
        condition_pipeline: ConditionPipeline,
        id_mapper: IDMapper,
        config: dict,
        *,
        embedder: ReIDEmbedder | None = None,
        anchor_bank: AnchorBank | None = None,
        reid_config: dict | None = None,
        failure_pipeline: FailureDetectorPipeline | None = None,
        consistency_monitor: ConsistencyMonitor | None = None,
        vlm_analyzer: VLMAnalyzer | None = None,
        vlm_verifier: VLMVerifier | None = None,
    ) -> None:
        self._conditions = condition_pipeline
        self._id_mapper = id_mapper
        self._embedder = embedder
        self._anchor_bank = anchor_bank

        self._max_pending_frames: int = config.get("max_pending_frames", 90)
        self._max_lost_frames: int = config.get("max_lost_frames", 150)

        rc = reid_config or {}
        self._similarity_threshold: float = rc.get("similarity_threshold", 0.5)
        self._confidence_threshold: float = rc.get("confidence_threshold", 0.85)
        self._metadata_rematch_threshold: float = rc.get(
            "metadata_rematch_threshold", 0.3,
        )
        self._metadata_rematch_mode: str = rc.get(
            "metadata_rematch_mode", "embedding",
        )
        self._metadata_rematch_max_vlm: int = rc.get(
            "metadata_rematch_max_vlm_candidates", 3,
        )
        self._metadata_filter_fields: list[str] = rc.get(
            "metadata_filter_fields", ["gender", "top_color", "bottom_color"],
        )

        self._failure_pipeline = failure_pipeline
        self._consistency_monitor = consistency_monitor
        self._vlm_analyzer = vlm_analyzer
        self._vlm_verifier = vlm_verifier

        self._tracks: dict[int, Track] = {}
        self._quality_reports: dict[int, QualityReport] = {}
        self._match_results: dict[int, MatchResult] = {}
        self._events: list[TrackEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self,
        detections: list[Detection],
        frame: np.ndarray,
        frame_idx: int,
    ) -> dict[int, Track]:
        current_ids = {
            d.tracker_id for d in detections if d.tracker_id is not None
        }

        self._mark_missing_tracks(current_ids, frame_idx)

        for det in detections:
            if det.tracker_id is None:
                continue

            tid = det.tracker_id

            if tid in self._tracks:
                self._handle_existing(self._tracks[tid], det, frame, frame_idx)
            else:
                track = Track(
                    tracker_id=tid,
                    camera_id=det.camera_id,
                    state=TrackState.PENDING,
                    first_frame_idx=frame_idx,
                    last_frame_idx=frame_idx,
                    last_bbox=det.bbox,
                    last_center=det.center,
                )
                self._tracks[tid] = track
                self._handle_pending(track, det, frame, frame_idx)

        return self._tracks

    def get_quality_report(self, tracker_id: int) -> QualityReport | None:
        return self._quality_reports.get(tracker_id)

    def get_match_result(self, tracker_id: int) -> MatchResult | None:
        return self._match_results.get(tracker_id)

    def drain_events(self) -> list[TrackEvent]:
        events = self._events
        self._events = []
        return events

    @property
    def active_count(self) -> int:
        return sum(
            1 for t in self._tracks.values() if t.state == TrackState.ACTIVE
        )

    @property
    def pending_count(self) -> int:
        return sum(
            1 for t in self._tracks.values() if t.state == TrackState.PENDING
        )

    # ------------------------------------------------------------------
    # Mark missing tracks
    # ------------------------------------------------------------------

    def _mark_missing_tracks(
        self, current_ids: set[int], frame_idx: int,
    ) -> None:
        for tid, track in list(self._tracks.items()):
            if tid in current_ids:
                continue

            if track.state in (
                TrackState.ACTIVE,
                TrackState.PENDING,
                TrackState.SUSPICIOUS,
            ):
                self._transition(track, TrackState.LOST)
                track.lost_frame_count = 0
                self._emit(
                    "track_lost", frame_idx, tid, track.global_id,
                )
                logger.info(
                    "[TRACK] tid=%d (%s) -> LOST at frame %d",
                    tid, track.global_id or "?", frame_idx,
                )

            elif track.state == TrackState.LOST:
                track.lost_frame_count += 1
                if track.lost_frame_count > self._max_lost_frames:
                    self._transition(track, TrackState.DEAD)
                    self._emit(
                        "track_dead", frame_idx, tid, track.global_id,
                        {"lost_frames": track.lost_frame_count},
                    )
                    logger.info(
                        "[TRACK] tid=%d (%s) -> DEAD at frame %d "
                        "(lost for %d frames)",
                        tid, track.global_id or "?", frame_idx,
                        track.lost_frame_count,
                    )

    # ------------------------------------------------------------------
    # Dispatch by current state
    # ------------------------------------------------------------------

    def _handle_existing(
        self,
        track: Track,
        det: Detection,
        frame: np.ndarray,
        frame_idx: int,
    ) -> None:
        track.last_frame_idx = frame_idx
        track.last_bbox = det.bbox
        track.last_center = det.center

        if track.state == TrackState.PENDING:
            self._handle_pending(track, det, frame, frame_idx)

        elif track.state == TrackState.ACTIVE:
            self._id_mapper.update_end_frame(
                det.camera_id, track.tracker_id, frame_idx,
            )
            if self._anchor_bank and track.global_id:
                self._anchor_bank.update_last_seen(
                    track.global_id, det.camera_id,
                    det.center, frame_idx,
                )
            self._handle_consistency_check(track, det, frame, frame_idx)

        elif track.state in (TrackState.LOST, TrackState.RECOVERED):
            track.lost_frame_count = 0
            self._handle_recovered(track, det, frame, frame_idx)

        elif track.state in (TrackState.EXPIRED, TrackState.DEAD):
            track.state = TrackState.PENDING
            track.pending_frame_count = 0
            track.global_id = None
            self._handle_pending(track, det, frame, frame_idx)

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_pending(
        self,
        track: Track,
        det: Detection,
        frame: np.ndarray,
        frame_idx: int,
    ) -> None:
        report = self._conditions.evaluate(det, frame)
        self._quality_reports[track.tracker_id] = report
        track.pending_frame_count += 1

        if report.passed:
            global_id = self._resolve_identity(track, det, frame, report)
            track.global_id = global_id
            self._id_mapper.register(
                det.camera_id, track.tracker_id, global_id, frame_idx,
            )
            self._transition(track, TrackState.ACTIVE)
            track.consistency_frame_counter = 0

        elif track.pending_frame_count > self._max_pending_frames:
            self._transition(track, TrackState.EXPIRED)
            logger.info(
                "[TRACK] tid=%d -> EXPIRED after %d pending frames",
                track.tracker_id, track.pending_frame_count,
            )

    def _handle_recovered(
        self,
        track: Track,
        det: Detection,
        frame: np.ndarray,
        frame_idx: int,
    ) -> None:
        self._transition(track, TrackState.RECOVERED)

        if track.global_id is None:
            track.state = TrackState.PENDING
            track.pending_frame_count = 0
            self._handle_pending(track, det, frame, frame_idx)
            return

        if self._embedder is None or self._anchor_bank is None:
            self._transition(track, TrackState.ACTIVE)
            self._id_mapper.update_end_frame(
                det.camera_id, track.tracker_id, frame_idx,
            )
            return

        report = self._conditions.evaluate(det, frame)
        if not report.passed:
            self._transition(track, TrackState.ACTIVE)
            self._id_mapper.update_end_frame(
                det.camera_id, track.tracker_id, frame_idx,
            )
            logger.info(
                "[RE-ID] tid=%d RECOVERED as %s (quality too low, "
                "trusting tracker, frame=%d)",
                track.tracker_id, track.global_id, frame_idx,
            )
            return

        crop = det.crop_from_frame(frame)
        embedding = self._embedder.extract(crop)

        if embedding is None:
            self._transition(track, TrackState.ACTIVE)
            logger.info(
                "[RE-ID] tid=%d RECOVERED as %s (embedding failed, "
                "trusting tracker, frame=%d)",
                track.tracker_id, track.global_id, frame_idx,
            )
            return

        anchor = self._anchor_bank.get(track.global_id)

        if anchor is None or anchor.representative_embedding is None:
            self._transition(track, TrackState.ACTIVE)
            return

        sim = float(np.dot(embedding, anchor.representative_embedding))

        if sim >= self._similarity_threshold:
            self._anchor_bank.add_embedding(track.global_id, embedding)
            self._transition(track, TrackState.ACTIVE)
            self._id_mapper.update_end_frame(
                det.camera_id, track.tracker_id, frame_idx,
            )
            self._match_results[track.tracker_id] = MatchResult(
                matched=True, global_id=track.global_id, similarity=sim,
            )
            self._emit(
                "reid_recovery", frame_idx, track.tracker_id,
                track.global_id, {"similarity": round(sim, 3)},
            )
            logger.info(
                "[RE-ID] tid=%d RECOVERED as %s CONFIRMED "
                "(sim=%.3f, frame=%d)",
                track.tracker_id, track.global_id, sim, frame_idx,
            )
            track.consistency_frame_counter = 0
        else:
            old_gid = track.global_id
            self._emit(
                "reid_recovery_failed", frame_idx, track.tracker_id,
                old_gid, {"similarity": round(sim, 3)},
            )
            logger.warning(
                "[RE-ID] tid=%d recovery FAILED for %s (sim=%.3f < %.2f, "
                "frame=%d). Re-identifying...",
                track.tracker_id, old_gid, sim,
                self._similarity_threshold, frame_idx,
            )
            self._id_mapper.unregister(det.camera_id, track.tracker_id)
            track.global_id = None
            track.state = TrackState.PENDING
            track.pending_frame_count = 0
            self._resolve_and_activate(track, det, frame, report, frame_idx)

    # ------------------------------------------------------------------
    # Consistency check (for ACTIVE tracks)
    # ------------------------------------------------------------------

    def _handle_consistency_check(
        self,
        track: Track,
        det: Detection,
        frame: np.ndarray,
        frame_idx: int,
    ) -> None:
        if self._consistency_monitor is None:
            return
        if self._embedder is None or self._anchor_bank is None:
            return
        if track.global_id is None:
            return

        track.consistency_frame_counter += 1
        if not self._consistency_monitor.should_check(
            track.consistency_frame_counter,
        ):
            return

        track.consistency_frame_counter = 0

        report = self._conditions.evaluate(det, frame)
        if not report.passed:
            return

        crop = det.crop_from_frame(frame)
        embedding = self._embedder.extract(crop)
        if embedding is None:
            return

        anchor = self._anchor_bank.get(track.global_id)
        if anchor is None or anchor.representative_embedding is None:
            return

        failure = self._consistency_monitor.check(
            embedding, anchor.representative_embedding,
        )

        if failure.detected:
            sim = failure.details.get("similarity", 0.0)
            self._emit(
                "consistency_fail", frame_idx, track.tracker_id,
                track.global_id,
                {"similarity": round(sim, 3), "reason": failure.reason},
            )
            logger.warning(
                "[CONSISTENCY] tid=%d (%s) similarity=%.3f at frame %d "
                "-> SUSPICIOUS",
                track.tracker_id, track.global_id, sim, frame_idx,
            )

            if self._vlm_verifier and anchor.crop_image is not None:
                is_same, conf = self._vlm_verifier.verify(
                    crop, anchor.crop_image,
                )
                self._emit(
                    "vlm_verify", frame_idx, track.tracker_id,
                    track.global_id,
                    {"same_person": is_same, "confidence": round(conf, 3)},
                )
                if is_same:
                    self._anchor_bank.add_embedding(
                        track.global_id, embedding,
                    )
                    logger.info(
                        "[VLM] tid=%d (%s) VLM confirmed same person "
                        "(conf=%.2f, frame=%d)",
                        track.tracker_id, track.global_id, conf, frame_idx,
                    )
                    return

            self._transition(track, TrackState.SUSPICIOUS)
            old_gid = track.global_id
            self._id_mapper.unregister(det.camera_id, track.tracker_id)
            track.global_id = None
            track.state = TrackState.PENDING
            track.pending_frame_count = 0
            logger.info(
                "[CONSISTENCY] tid=%d unassigned from %s, re-identifying...",
                track.tracker_id, old_gid,
            )
            self._resolve_and_activate(track, det, frame, report, frame_idx)

    # ------------------------------------------------------------------
    # Re-ID resolution
    # ------------------------------------------------------------------

    def _resolve_identity(
        self,
        track: Track,
        det: Detection,
        frame: np.ndarray,
        report: QualityReport,
    ) -> str:
        if self._embedder is None or self._anchor_bank is None:
            gid = self._id_mapper.create_new_id()
            self._emit("reid_new", det.frame_idx, track.tracker_id, gid)
            logger.info(
                "[TRACK] tid=%d -> ACTIVE as %s (no Re-ID, quality=%.2f, "
                "frame=%d)",
                track.tracker_id, gid, report.overall_score, det.frame_idx,
            )
            return gid

        crop = det.crop_from_frame(frame)
        embedding = self._embedder.extract(crop)

        if embedding is None:
            gid = self._id_mapper.create_new_id()
            self._emit("reid_new", det.frame_idx, track.tracker_id, gid)
            logger.warning(
                "[RE-ID] tid=%d embedding extraction failed, assigning "
                "new %s (frame=%d)",
                track.tracker_id, gid, det.frame_idx,
            )
            return gid

        match = self._anchor_bank.search(embedding, self._similarity_threshold)
        self._match_results[track.tracker_id] = match

        if match.candidate_similarities:
            sims_str = ", ".join(
                f"{k}={v:.3f}" for k, v in
                sorted(
                    match.candidate_similarities.items(),
                    key=lambda x: x[1], reverse=True,
                )[:5]
            )
            logger.info(
                "[RE-ID] tid=%d embedding compared — candidates: [%s]",
                track.tracker_id, sims_str,
            )

        if match.matched:
            anchor = self._anchor_bank.get(match.global_id)
            failures = self._check_match_failures(match, det, anchor)

            if failures:
                verified = self._vlm_verify_match(
                    crop, anchor, match, det, track,
                )
                if not verified:
                    gid = self._id_mapper.create_new_id()
                    self._anchor_bank.register(
                        global_id=gid, crop_image=crop,
                        embedding=embedding,
                        quality_score=report.overall_score,
                        camera_id=det.camera_id,
                        frame_idx=det.frame_idx,
                        bbox=det.bbox,
                    )
                    self._emit(
                        "reid_new", det.frame_idx, track.tracker_id, gid,
                        {"reason": "failure_detected"},
                    )
                    self._request_vlm_metadata(gid)
                    logger.info(
                        "[RE-ID] tid=%d -> NEW %s (failure detected, "
                        "VLM rejected, frame=%d)",
                        track.tracker_id, gid, det.frame_idx,
                    )
                    return gid

            self._anchor_bank.add_embedding(match.global_id, embedding)
            self._anchor_bank.update_crop_if_better(
                match.global_id, crop, report.overall_score,
                det.camera_id, det.frame_idx, det.bbox,
            )
            self._emit(
                "reid_match", det.frame_idx, track.tracker_id,
                match.global_id, {"similarity": round(match.similarity, 3)},
            )
            logger.info(
                "[RE-ID] tid=%d -> MATCHED %s (sim=%.3f, frame=%d)",
                track.tracker_id, match.global_id,
                match.similarity, det.frame_idx,
            )
            return match.global_id
        else:
            best_info = (
                f"best={match.similarity:.3f}"
                if match.candidate_similarities else "no candidates"
            )
            logger.info(
                "[RE-ID] tid=%d embedding below threshold "
                "(%s < %.2f) — trying metadata re-match (frame=%d)",
                track.tracker_id, best_info,
                self._similarity_threshold, det.frame_idx,
            )

            rematch_gid, metadata = self._try_metadata_rematch(
                embedding, crop, det,
            )

            if rematch_gid is not None:
                self._anchor_bank.add_embedding(rematch_gid, embedding)
                self._anchor_bank.update_crop_if_better(
                    rematch_gid, crop, report.overall_score,
                    det.camera_id, det.frame_idx, det.bbox,
                )
                logger.info(
                    "[RE-ID] tid=%d -> MATCHED %s via metadata re-match "
                    "(frame=%d)",
                    track.tracker_id, rematch_gid, det.frame_idx,
                )
                return rematch_gid

            gid = self._id_mapper.create_new_id()
            self._anchor_bank.register(
                global_id=gid, crop_image=crop,
                embedding=embedding,
                quality_score=report.overall_score,
                camera_id=det.camera_id,
                frame_idx=det.frame_idx,
                bbox=det.bbox,
            )
            self._emit("reid_new", det.frame_idx, track.tracker_id, gid)

            if metadata is not None:
                self._anchor_bank.set_metadata(gid, metadata)
                logger.info(
                    "[RE-ID] tid=%d -> NEW %s (embedding %s < %.2f, "
                    "metadata re-match also failed, "
                    "pre-extracted metadata set, frame=%d)",
                    track.tracker_id, gid, best_info,
                    self._similarity_threshold, det.frame_idx,
                )
            else:
                self._request_vlm_metadata(gid)
                logger.info(
                    "[RE-ID] tid=%d -> NEW %s (%s < threshold %.2f, "
                    "frame=%d)",
                    track.tracker_id, gid, best_info,
                    self._similarity_threshold, det.frame_idx,
                )
            return gid

    def _resolve_and_activate(
        self,
        track: Track,
        det: Detection,
        frame: np.ndarray,
        report: QualityReport,
        frame_idx: int,
    ) -> None:
        global_id = self._resolve_identity(track, det, frame, report)
        track.global_id = global_id
        self._id_mapper.register(
            det.camera_id, track.tracker_id, global_id, frame_idx,
        )
        self._transition(track, TrackState.ACTIVE)
        track.consistency_frame_counter = 0

    # ------------------------------------------------------------------
    # Failure detector checks
    # ------------------------------------------------------------------

    def _check_match_failures(
        self,
        match: MatchResult,
        det: Detection,
        anchor,
    ) -> list:
        if self._failure_pipeline is None or anchor is None:
            return []

        failures = self._failure_pipeline.check_all(match, det, anchor)
        for f in failures:
            self._emit(
                "reid_failure", det.frame_idx,
                det.tracker_id or 0, match.global_id,
                {
                    "detector": f.detector_name,
                    "reason": f.reason,
                    **f.details,
                },
            )
            logger.warning(
                "[FAILURE] %s: %s (tid=%d, frame=%d)",
                f.detector_name, f.reason,
                det.tracker_id or 0, det.frame_idx,
            )
        return failures

    def _vlm_verify_match(
        self,
        crop: np.ndarray,
        anchor,
        match: MatchResult,
        det: Detection,
        track: Track,
    ) -> bool:
        """If VLM is available, verify whether the match is legitimate."""
        if self._vlm_verifier is None or anchor is None:
            return True  # no VLM → accept the match
        if anchor.crop_image is None:
            return True

        is_same, conf = self._vlm_verifier.verify(crop, anchor.crop_image)
        self._emit(
            "vlm_verify", det.frame_idx, track.tracker_id,
            match.global_id,
            {"same_person": is_same, "confidence": round(conf, 3)},
        )
        if is_same:
            logger.info(
                "[VLM] tid=%d match to %s CONFIRMED (conf=%.2f, frame=%d)",
                track.tracker_id, match.global_id, conf, det.frame_idx,
            )
        else:
            logger.warning(
                "[VLM] tid=%d match to %s REJECTED (conf=%.2f, frame=%d)",
                track.tracker_id, match.global_id, conf, det.frame_idx,
            )
        return is_same

    # ------------------------------------------------------------------
    # Metadata-based re-matching
    # ------------------------------------------------------------------

    def _try_metadata_rematch(
        self,
        embedding: np.ndarray,
        crop: np.ndarray,
        det: Detection,
    ) -> tuple[str | None, PersonMetadata | None]:
        """Attempt re-match via VLM metadata when embedding search fails.

        1. Extract metadata from the current crop via VLM.
        2. Filter anchor_bank for anchors with matching appearance.
        3. Re-compare embedding against filtered candidates with a
           lower threshold.

        Returns ``(matched_global_id, extracted_metadata)``.
        Both are ``None`` when VLM is unavailable.
        ``matched_global_id`` is ``None`` if no candidate passes.
        ``extracted_metadata`` is returned even on match failure so the
        caller can set it on a newly created anchor without a redundant
        VLM call.
        """
        if self._vlm_analyzer is None or self._anchor_bank is None:
            return None, None

        tid = det.tracker_id or 0

        # --- Step 1: Extract metadata for current person ---
        try:
            raw = self._vlm_analyzer.analyze_person(crop)
        except Exception:
            logger.exception(
                "[METADATA-REMATCH] tid=%d VLM inference failed (frame=%d)",
                tid, det.frame_idx,
            )
            return None, None

        if raw is None:
            logger.info(
                "[METADATA-REMATCH] tid=%d VLM returned None — "
                "skipping metadata re-match (frame=%d)",
                tid, det.frame_idx,
            )
            return None, None

        metadata = PersonMetadata(
            gender=raw.get("gender"),
            top_color=raw.get("top_color"),
            bottom_color=raw.get("bottom_color"),
            top_type=raw.get("top_type"),
            bottom_type=raw.get("bottom_type"),
        )
        logger.info(
            "[METADATA-REMATCH] tid=%d metadata extracted: "
            "gender=%s, top=%s %s, bottom=%s %s (frame=%d)",
            tid,
            metadata.gender, metadata.top_color, metadata.top_type,
            metadata.bottom_color, metadata.bottom_type,
            det.frame_idx,
        )

        # --- Step 2: Build filter and query anchor bank ---
        filter_kwargs: dict[str, str] = {}
        for field in self._metadata_filter_fields:
            value = getattr(metadata, field, None)
            if value is not None:
                filter_kwargs[field] = value

        if not filter_kwargs:
            logger.info(
                "[METADATA-REMATCH] tid=%d no usable metadata fields "
                "for filtering — skipping (frame=%d)",
                tid, det.frame_idx,
            )
            return None, metadata

        logger.info(
            "[METADATA-REMATCH] tid=%d filtering anchor bank by: %s",
            tid, filter_kwargs,
        )

        candidates = self._anchor_bank.filter_by_metadata(**filter_kwargs)

        if not candidates:
            logger.info(
                "[METADATA-REMATCH] tid=%d no anchors with matching "
                "metadata (frame=%d)",
                tid, det.frame_idx,
            )
            return None, metadata

        logger.info(
            "[METADATA-REMATCH] tid=%d found %d candidate(s): %s",
            tid, len(candidates), candidates,
        )

        # --- Step 3: Re-match among filtered candidates ---
        use_vlm = (
            self._metadata_rematch_mode == "vlm"
            and self._vlm_verifier is not None
        )

        if use_vlm:
            return self._metadata_rematch_vlm(
                crop, embedding, candidates, filter_kwargs,
                metadata, tid, det,
            )

        return self._metadata_rematch_embedding(
            embedding, candidates, filter_kwargs,
            metadata, tid, det,
        )

    # ------------------------------------------------------------------

    def _metadata_rematch_embedding(
        self,
        embedding: np.ndarray,
        candidates: list[str],
        filter_kwargs: dict[str, str],
        metadata: PersonMetadata,
        tid: int,
        det: Detection,
    ) -> tuple[str | None, PersonMetadata | None]:
        """Step 3 — embedding mode: re-compare with lower threshold."""
        rematch = self._anchor_bank.search_among(
            embedding, candidates, self._metadata_rematch_threshold,
        )

        if rematch.candidate_similarities:
            sims_str = ", ".join(
                f"{k}={v:.3f}" for k, v in
                sorted(
                    rematch.candidate_similarities.items(),
                    key=lambda x: x[1], reverse=True,
                )
            )
            logger.info(
                "[METADATA-REMATCH] tid=%d similarities among "
                "filtered: [%s]",
                tid, sims_str,
            )

        if rematch.matched:
            self._match_results[tid] = rematch
            self._emit(
                "reid_metadata_rematch", det.frame_idx, tid,
                rematch.global_id,
                {
                    "mode": "embedding",
                    "similarity": round(rematch.similarity, 3),
                    "filter": filter_kwargs,
                    "candidates": len(candidates),
                },
            )
            logger.info(
                "[METADATA-REMATCH] tid=%d -> MATCHED %s via embedding "
                "re-match (sim=%.3f, threshold=%.2f, frame=%d)",
                tid, rematch.global_id, rematch.similarity,
                self._metadata_rematch_threshold, det.frame_idx,
            )
            return rematch.global_id, metadata

        logger.info(
            "[METADATA-REMATCH] tid=%d no match among %d filtered "
            "candidate(s) (best=%.3f < threshold=%.2f, frame=%d)",
            tid, len(candidates), rematch.similarity,
            self._metadata_rematch_threshold, det.frame_idx,
        )
        return None, metadata

    def _metadata_rematch_vlm(
        self,
        crop: np.ndarray,
        embedding: np.ndarray,
        candidates: list[str],
        filter_kwargs: dict[str, str],
        metadata: PersonMetadata,
        tid: int,
        det: Detection,
    ) -> tuple[str | None, PersonMetadata | None]:
        """Step 3 — VLM mode: visually compare crops instead of embeddings."""
        ranked = self._anchor_bank.search_among(
            embedding, candidates, threshold=-1.0,
        )
        sorted_ids = sorted(
            ranked.candidate_similarities,
            key=ranked.candidate_similarities.get,
            reverse=True,
        )[:self._metadata_rematch_max_vlm]

        if ranked.candidate_similarities:
            sims_str = ", ".join(
                f"{k}={v:.3f}"
                for k in sorted_ids
                if (v := ranked.candidate_similarities.get(k)) is not None
            )
            logger.info(
                "[METADATA-REMATCH-VLM] tid=%d top-%d candidates by "
                "embedding: [%s]",
                tid, self._metadata_rematch_max_vlm, sims_str,
            )

        for rank, gid in enumerate(sorted_ids, 1):
            anchor = self._anchor_bank.get(gid)
            if anchor is None or anchor.crop_image is None:
                logger.info(
                    "[METADATA-REMATCH-VLM] tid=%d skipping %s "
                    "(no crop available)",
                    tid, gid,
                )
                continue

            sim = ranked.candidate_similarities.get(gid, 0.0)
            logger.info(
                "[METADATA-REMATCH-VLM] tid=%d verifying vs %s "
                "(%d/%d, embed_sim=%.3f, frame=%d)",
                tid, gid, rank, len(sorted_ids), sim, det.frame_idx,
            )

            is_same, conf = self._vlm_verifier.verify(crop, anchor.crop_image)

            self._emit(
                "vlm_metadata_rematch_verify", det.frame_idx, tid, gid,
                {
                    "same_person": is_same,
                    "confidence": round(conf, 3),
                    "embedding_similarity": round(sim, 3),
                    "rank": rank,
                },
            )

            if is_same:
                logger.info(
                    "[METADATA-REMATCH-VLM] tid=%d -> MATCHED %s "
                    "(VLM confirmed, conf=%.2f, embed_sim=%.3f, "
                    "frame=%d)",
                    tid, gid, conf, sim, det.frame_idx,
                )
                self._match_results[tid] = MatchResult(
                    matched=True, global_id=gid, similarity=sim,
                    candidate_similarities=ranked.candidate_similarities,
                )
                self._emit(
                    "reid_metadata_rematch", det.frame_idx, tid, gid,
                    {
                        "mode": "vlm",
                        "vlm_confidence": round(conf, 3),
                        "embedding_similarity": round(sim, 3),
                        "filter": filter_kwargs,
                        "candidates": len(candidates),
                        "vlm_attempts": rank,
                    },
                )
                return gid, metadata

            logger.info(
                "[METADATA-REMATCH-VLM] tid=%d vs %s REJECTED "
                "(conf=%.2f, frame=%d)",
                tid, gid, conf, det.frame_idx,
            )

        logger.info(
            "[METADATA-REMATCH-VLM] tid=%d no match after VLM "
            "verification of %d candidate(s) (frame=%d)",
            tid, len(sorted_ids), det.frame_idx,
        )
        return None, metadata

    # ------------------------------------------------------------------
    # VLM metadata extraction
    # ------------------------------------------------------------------

    def _request_vlm_metadata(self, global_id: str) -> None:
        """Extract VLM metadata for a newly registered anchor."""
        if self._vlm_analyzer is None or self._anchor_bank is None:
            return

        anchor = self._anchor_bank.get(global_id)
        if anchor is None or anchor.crop_image is None:
            return

        try:
            raw = self._vlm_analyzer.analyze_person(anchor.crop_image)
            if raw is not None:
                metadata = PersonMetadata(
                    gender=raw.get("gender"),
                    top_color=raw.get("top_color"),
                    bottom_color=raw.get("bottom_color"),
                    top_type=raw.get("top_type"),
                    bottom_type=raw.get("bottom_type"),
                )
                self._anchor_bank.set_metadata(global_id, metadata)
                logger.info(
                    "[VLM] Metadata extracted for %s: %s",
                    global_id, raw,
                )
            else:
                logger.warning(
                    "[VLM] Metadata extraction returned None for %s "
                    "(JSON parsing failed)",
                    global_id,
                )
        except Exception:
            logger.exception(
                "[VLM] Metadata extraction failed for %s", global_id,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emit(
        self,
        event_type: str,
        frame_idx: int,
        tracker_id: int,
        global_id: str | None = None,
        details: dict | None = None,
    ) -> None:
        self._events.append(TrackEvent(
            event_type=event_type,
            frame_idx=frame_idx,
            tracker_id=tracker_id,
            global_id=global_id,
            details=details or {},
        ))

    @staticmethod
    def _transition(track: Track, new_state: TrackState) -> None:
        track.state = new_state
