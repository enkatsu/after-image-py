"""データモッシュ済み mp4 をループ再生する。差し替えは次のループの頭で行う。"""

import threading
from pathlib import Path

import cv2
import numpy as np


class MoshedPlayer:
    """単一の mp4 をループ再生し、新しいパスを受け取ったら次のループ頭で切り替える。

    update_source() はワーカースレッドから呼ばれる前提でロックで保護する。
    next_frame() はメインスレッド専用。
    """

    def __init__(self):
        self._next_path: Path | None = None
        self._current_path: Path | None = None
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()

    @property
    def current_path(self) -> Path | None:
        with self._lock:
            return self._current_path

    def update_source(self, path: Path) -> None:
        """次に再生する mp4 を予約する。現在再生中のループは最後まで完走する。"""
        with self._lock:
            self._next_path = path

    def is_ready(self) -> bool:
        with self._lock:
            return self._current_path is not None or self._next_path is not None

    def next_frame(self) -> np.ndarray | None:
        """次の表示フレームを返す。再生対象がまだ無ければ None。"""
        if self._cap is None and not self._open_next_or_current():
            return None

        assert self._cap is not None
        ok, frame = self._cap.read()
        if ok:
            return frame

        self._cap.release()
        self._cap = None
        if not self._open_next_or_current():
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def _open_next_or_current(self) -> bool:
        with self._lock:
            path = self._next_path or self._current_path
            self._next_path = None
        if path is None:
            return False
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False
        self._cap = cap
        with self._lock:
            self._current_path = path
        return True

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
