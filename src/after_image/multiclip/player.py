"""複数の mp4 を並列にループ再生し、各クリップから 1 フレームずつ取り出す。"""

import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from ..tracker import Track


class MulticlipPlayer:
    """ストックされたクリップを並列ループ再生する。

    set_clips() でストックを差し替える。既に開いているキャプチャは維持し、
    新しく追加されたものだけ開く（差し替え時の頭出し挙動を最小化）。
    """

    def __init__(self):
        self._captures: dict[Path, cv2.VideoCapture] = {}
        self._order: list[Path] = []
        self._lock = threading.Lock()

    def is_ready(self) -> bool:
        with self._lock:
            return bool(self._captures)

    def set_clips(self, paths: list[Path]) -> None:
        """再生対象のクリップを更新する。消えたものは閉じ、新規は開く。"""
        with self._lock:
            wanted = list(paths)
            wanted_set = set(wanted)
            for p in list(self._captures):
                if p not in wanted_set:
                    self._captures[p].release()
                    del self._captures[p]
            for p in wanted:
                if p not in self._captures:
                    cap = cv2.VideoCapture(str(p))
                    if cap.isOpened():
                        self._captures[p] = cap
            self._order = wanted

    def read_frames(self) -> list[np.ndarray]:
        """各クリップから次の 1 フレームを取り出す。EOF に達したものは先頭に戻る。"""
        frames: list[np.ndarray] = []
        with self._lock:
            for p in self._order:
                cap = self._captures.get(p)
                if cap is None:
                    continue
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                if ok:
                    frames.append(frame)
        return frames

    def close(self) -> None:
        with self._lock:
            for cap in self._captures.values():
                cap.release()
            self._captures.clear()
            self._order.clear()


def build_shim_tracks(clip_frames: list[np.ndarray]) -> list[Track]:
    """既存の effects.RENDERERS に渡すためのダミー Track を作る。

    各クリップの「現在再生位置の 1 フレーム」を 1 個の Track として扱うことで、
    effects.collect_records() が時系列順に並べられる形にする。
    """
    return [
        Track(
            id=i,
            last_center=(0.0, 0.0),
            frames=deque([(i, frame)]),
        )
        for i, frame in enumerate(clip_frames)
    ]
