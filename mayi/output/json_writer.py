from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from mayi.anchor_bank.bank import AnchorBank
from mayi.models.types import TrackEvent
from mayi.tracking.id_mapper import IDMapper

logger = logging.getLogger(__name__)


class JSONWriter:
    """Writes structured tracking results to a JSON file."""

    def __init__(self) -> None:
        self._events: list[TrackEvent] = []

    def add_event(self, event: TrackEvent) -> None:
        self._events.append(event)

    def add_events(self, events: list[TrackEvent]) -> None:
        self._events.extend(events)

    def write(
        self,
        path: str | Path,
        *,
        video_path: str = "",
        camera_id: str = "cam0",
        fps: float = 30.0,
        total_frames: int = 0,
        anchor_bank: AnchorBank | None = None,
        id_mapper: IDMapper | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        result: dict = {
            "video": video_path,
            "camera_id": camera_id,
            "fps": fps,
            "total_frames": total_frames,
        }

        if anchor_bank is not None:
            persons = []
            for gid in anchor_bank.all_ids:
                entry = anchor_bank.get(gid)
                person: dict = {"global_id": gid}

                if entry and entry.metadata:
                    person["metadata"] = asdict(entry.metadata)
                else:
                    person["metadata"] = None

                if id_mapper is not None:
                    segments = id_mapper.get_history(gid)
                    person["track_segments"] = [
                        {
                            "camera_id": s.camera_id,
                            "tracker_id": s.tracker_id,
                            "start_frame": s.start_frame,
                            "end_frame": s.end_frame,
                        }
                        for s in segments
                    ]
                persons.append(person)
            result["persons"] = persons

        result["events"] = [
            {
                "type": e.event_type,
                "frame": e.frame_idx,
                "tracker_id": e.tracker_id,
                "global_id": e.global_id,
                **e.details,
            }
            for e in self._events
        ]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(
            "Tracking results written to %s (%d persons, %d events)",
            path, len(result.get("persons", [])), len(result["events"]),
        )
