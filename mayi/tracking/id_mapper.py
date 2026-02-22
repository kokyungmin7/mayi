from __future__ import annotations

from mayi.models.types import TrackSegment


class IDMapper:
    """Manages tracker_id <-> global_id mappings across cameras.

    Each camera has its own tracker_id namespace.  The IDMapper
    translates those local IDs into a unified global person ID.
    """

    def __init__(self) -> None:
        self._map: dict[str, dict[int, str]] = {}  # camera_id -> {tracker_id -> global_id}
        self._history: dict[str, list[TrackSegment]] = {}  # global_id -> segments
        self._next_id: int = 1

    def get(self, camera_id: str, tracker_id: int) -> str | None:
        return self._map.get(camera_id, {}).get(tracker_id)

    def register(
        self,
        camera_id: str,
        tracker_id: int,
        global_id: str,
        start_frame: int = 0,
    ) -> None:
        if camera_id not in self._map:
            self._map[camera_id] = {}
        self._map[camera_id][tracker_id] = global_id

        if global_id not in self._history:
            self._history[global_id] = []
        self._history[global_id].append(
            TrackSegment(
                camera_id=camera_id,
                tracker_id=tracker_id,
                global_id=global_id,
                start_frame=start_frame,
                end_frame=start_frame,
            )
        )

    def unregister(self, camera_id: str, tracker_id: int) -> None:
        cam_map = self._map.get(camera_id)
        if cam_map and tracker_id in cam_map:
            del cam_map[tracker_id]

    def update_end_frame(self, camera_id: str, tracker_id: int, frame_idx: int) -> None:
        global_id = self.get(camera_id, tracker_id)
        if global_id is None:
            return
        segments = self._history.get(global_id, [])
        for seg in reversed(segments):
            if seg.camera_id == camera_id and seg.tracker_id == tracker_id:
                seg.end_frame = frame_idx
                break

    def create_new_id(self) -> str:
        gid = f"P-{self._next_id:03d}"
        self._next_id += 1
        return gid

    def get_history(self, global_id: str) -> list[TrackSegment]:
        return self._history.get(global_id, [])

    @property
    def all_global_ids(self) -> set[str]:
        return set(self._history.keys())
