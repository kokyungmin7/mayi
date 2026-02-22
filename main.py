from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import yaml

from mayi.anchor_bank.bank import AnchorBank
from mayi.conditions import build_condition_pipeline
from mayi.detection.detector import PersonDetector
from mayi.output.json_writer import JSONWriter
from mayi.output.video_writer import VideoWriter
from mayi.reid.embedder import ReIDEmbedder
from mayi.reid.failure_detectors import (
    ConsistencyMonitor,
    build_failure_detector_pipeline,
)
from mayi.tracking.id_mapper import IDMapper
from mayi.tracking.track_manager import TrackManager
from mayi.vlm.analyzer import VLMAnalyzer
from mayi.vlm.verifier import VLMVerifier


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MAYI — Person Re-ID + VLM verification pipeline",
    )

    p.add_argument("--video", type=str, required=True, help="Input video path")
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--camera-id", type=str, default="cam0")

    vis = p.add_argument_group("visualisation / output")
    vis.add_argument(
        "--show", action="store_true",
        help="Display annotated video in a window (requires display)",
    )
    vis.add_argument(
        "--output-video", type=str, default=None,
        help="Save annotated result video to this path",
    )
    vis.add_argument(
        "--output-json", type=str, default=None,
        help="Save tracking results as JSON",
    )

    feat = p.add_argument_group("feature toggles")
    feat.add_argument(
        "--no-reid", action="store_true",
        help="Disable Re-ID (tracking only)",
    )
    feat.add_argument(
        "--no-vlm", action="store_true",
        help="Disable VLM (no metadata extraction / verification)",
    )

    return p


def main() -> None:
    args = build_argparser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("mayi")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # --- Components ---
    detector = PersonDetector.from_config(config["detection"])
    conditions = build_condition_pipeline(config["conditions"])
    id_mapper = IDMapper()

    embedder: ReIDEmbedder | None = None
    anchor_bank: AnchorBank | None = None
    failure_pipeline = None
    consistency_monitor = None
    vlm_analyzer: VLMAnalyzer | None = None
    vlm_verifier: VLMVerifier | None = None

    if not args.no_reid:
        embedder = ReIDEmbedder.from_config(config["reid"])
        anchor_bank = AnchorBank()

        fd_cfg = config.get("reid", {}).get("failure_detectors", {})
        failure_pipeline = build_failure_detector_pipeline(fd_cfg)

        consistency_monitor = ConsistencyMonitor.from_config(
            config.get("tracking", {}),
        )

    if not args.no_vlm and not args.no_reid:
        vlm_cfg = config.get("vlm", {})
        vlm_analyzer = VLMAnalyzer.from_config(vlm_cfg)
        vlm_verifier = VLMVerifier.from_config(vlm_cfg, vlm_analyzer)

    track_manager = TrackManager(
        condition_pipeline=conditions,
        id_mapper=id_mapper,
        config=config.get("tracking", {}),
        embedder=embedder,
        anchor_bank=anchor_bank,
        reid_config=config.get("reid"),
        failure_pipeline=failure_pipeline,
        consistency_monitor=consistency_monitor,
        vlm_analyzer=vlm_analyzer,
        vlm_verifier=vlm_verifier,
    )

    json_writer = JSONWriter()

    # --- Video input ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Video writer ---
    video_writer: VideoWriter | None = None
    if args.output_video or args.show:
        video_writer = VideoWriter(
            output_path=args.output_video,
            fps=fps,
            frame_size=(w, h),
            draw_bbox=config.get("output", {}).get("draw_bbox", True),
            draw_skeleton=True,
            draw_status_bar=config.get("output", {}).get("draw_status", True),
        )

    reid_status = "ON" if embedder else "OFF"
    vlm_status = "ON" if vlm_analyzer else "OFF"
    log.info(
        "Processing %s (%dx%d, %.1f fps, %d frames, "
        "Re-ID=%s, VLM=%s)",
        args.video, w, h, fps, total_frames, reid_status, vlm_status,
    )

    frame_idx = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame, frame_idx, args.camera_id)
        tracks = track_manager.process_frame(detections, frame, frame_idx)

        events = track_manager.drain_events()
        json_writer.add_events(events)

        if video_writer is not None:
            vis = video_writer.annotate(
                frame, detections, tracks,
                track_manager, anchor_bank, frame_idx,
            )
            video_writer.write_frame(vis)

            if args.show:
                cv2.imshow("MAYI - Person Re-ID", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if frame_idx > 0 and frame_idx % 500 == 0:
            elapsed = time.time() - t_start
            speed = frame_idx / elapsed if elapsed > 0 else 0
            persons = anchor_bank.size if anchor_bank else 0
            log.info(
                "Progress: %d/%d frames (%.1f fps), %d persons",
                frame_idx, total_frames, speed, persons,
            )

        frame_idx += 1

    cap.release()
    if video_writer is not None:
        video_writer.release()
    if args.show:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start

    # --- Save JSON ---
    if args.output_json:
        json_writer.write(
            args.output_json,
            video_path=args.video,
            camera_id=args.camera_id,
            fps=fps,
            total_frames=frame_idx,
            anchor_bank=anchor_bank,
            id_mapper=id_mapper,
        )

    # --- Summary ---
    log.info(
        "Done. %d frames in %.1fs (%.1f fps).",
        frame_idx, elapsed, frame_idx / elapsed if elapsed > 0 else 0,
    )
    if anchor_bank:
        log.info("%d unique persons identified.", anchor_bank.size)
        for gid in anchor_bank.all_ids:
            segs = id_mapper.get_history(gid)
            frames_str = ", ".join(
                f"tid={s.tracker_id}[{s.start_frame}-{s.end_frame}]"
                for s in segs
            )
            meta = ""
            entry = anchor_bank.get(gid)
            if entry and entry.metadata:
                meta = (
                    f" | {entry.metadata.gender}, "
                    f"{entry.metadata.top_color} {entry.metadata.top_type}, "
                    f"{entry.metadata.bottom_color} {entry.metadata.bottom_type}"
                )
            log.info("  %s: %s%s", gid, frames_str, meta)


if __name__ == "__main__":
    main()
