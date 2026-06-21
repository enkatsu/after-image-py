"""バックグラウンドで build_moshed を回し、完成 mp4 のパスをコールバックで通知する。

ビルド中に新しい request が来たら最新だけ覚えておき（latest-wins）、終わり次第続けて
処理する。古い moshed_NNNN.mp4 はディスクに残しすぎないよう keep_recent 個だけ残す。
"""

import sys
import threading
from collections.abc import Callable
from pathlib import Path

from .pipeline import build_moshed


class MoshWorker:
    def __init__(
        self,
        output_dir: Path,
        fps: int,
        on_ready: Callable[[Path], None],
        keep_recent: int = 3,
        delta: int = 0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.on_ready = on_ready
        self.keep_recent = keep_recent
        self.delta = delta

        self._lock = threading.Lock()
        self._pending_clips: list[Path] | None = None
        self._event = threading.Event()
        self._stop = False
        self._thread = threading.Thread(
            target=self._run, name='mosh-worker', daemon=True
        )
        self._build_index = 0
        self._outputs: list[Path] = []

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop = True
        self._event.set()
        self._thread.join(timeout=timeout)

    def request(self, clip_paths: list[Path]) -> None:
        """最新のクリップ群でビルドをキックする。生成中なら次の素材として上書き。"""
        with self._lock:
            self._pending_clips = list(clip_paths)
        self._event.set()

    def _run(self) -> None:
        while not self._stop:
            self._event.wait()
            self._event.clear()
            if self._stop:
                break

            with self._lock:
                clips = self._pending_clips
                self._pending_clips = None
            if not clips:
                continue

            try:
                self._build_index += 1
                output = (
                    self.output_dir / f'moshed_{self._build_index:04d}.mp4'
                )
                build_moshed(clips, output, fps=self.fps, delta=self.delta)
                self._outputs.append(output)
                self._prune_old()
                self.on_ready(output)
            except Exception as e:
                print(f'mosh build failed: {e}', file=sys.stderr)

    def _prune_old(self) -> None:
        while len(self._outputs) > self.keep_recent:
            old = self._outputs.pop(0)
            old.unlink(missing_ok=True)
