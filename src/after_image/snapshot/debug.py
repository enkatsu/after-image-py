from collections.abc import Iterable

import cv2
import numpy as np

from ..tracker import BUFFER_LENGTH, MAX_PEOPLE, Track, TrackManager


def annotate(
    display: np.ndarray,
    manager: TrackManager,
    detections: Iterable[tuple[int, np.ndarray]],
) -> None:
    for tid, box in detections:
        x1, y1, x2, y2 = box.astype(int)
        track = manager.find(tid)
        buf_len = len(track.frames) if track else 0
        maxlen = track.frames.maxlen if track else BUFFER_LENGTH
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            display,
            f'ID {tid} ({buf_len}/{maxlen})',
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )


def render_buffer_grid(
    people: Iterable[Track],
    thumb_width: int = 240,
    thumb_height: int = 135,
) -> np.ndarray:
    grid = np.zeros(
        (MAX_PEOPLE * thumb_height, BUFFER_LENGTH * thumb_width, 3),
        dtype=np.uint8,
    )
    for r, track in enumerate(people):
        y0 = r * thumb_height
        for c, (_, img) in enumerate(track.frames):
            x0 = c * thumb_width
            grid[y0 : y0 + thumb_height, x0 : x0 + thumb_width] = cv2.resize(
                img, (thumb_width, thumb_height)
            )
        cv2.putText(
            grid,
            f'ID {track.id}',
            (4, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return grid


def list_cameras(n: int = 5) -> None:
    for i in range(n):
        print(i, cv2.VideoCapture(i).isOpened())
