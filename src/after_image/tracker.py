from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

BUFFER_LENGTH = 5
MAX_PEOPLE = 5


@dataclass
class Track:
    id: int
    last_center: tuple[float, float]
    frames: deque = field(default_factory=lambda: deque(maxlen=BUFFER_LENGTH))


ShouldRecord = Callable[[Track | None, float, float], bool]


def make_should_record_by_distance(threshold: float = 80) -> ShouldRecord:
    def predicate(track: Track | None, cx: float, cy: float) -> bool:
        if track is None:
            return True
        dx = cx - track.last_center[0]
        dy = cy - track.last_center[1]
        return (dx * dx + dy * dy) ** 0.5 >= threshold

    return predicate


def should_record_always(track: Track | None, cx: float, cy: float) -> bool:
    return True


SHOULD_RECORD: dict[str, ShouldRecord] = {
    'distance': make_should_record_by_distance(),
    'always': should_record_always,
}


class TrackManager:
    def __init__(
        self,
        max_people: int = MAX_PEOPLE,
        should_record: ShouldRecord | None = None,
    ):
        self.max_people = max_people
        if should_record is None:
            should_record = make_should_record_by_distance()
        self.should_record = should_record
        self.people: list[Track] = []

    def find(self, tid: int) -> Track | None:
        for track in self.people:
            if track.id == tid:
                return track
        return None

    def update(
        self, tid: int, cx: float, cy: float, frame: np.ndarray, frame_count: int
    ) -> None:
        track = self.find(tid)
        if not self.should_record(track, cx, cy):
            return
        if track is None:
            track = Track(id=tid, last_center=(cx, cy))
            self.people.append(track)
            while len(self.people) > self.max_people:
                self.people.pop(0)
        track.frames.append((frame_count, frame.copy()))
        track.last_center = (cx, cy)

    def __iter__(self):
        return iter(self.people)

    def __len__(self) -> int:
        return len(self.people)
