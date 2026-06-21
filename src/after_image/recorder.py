from collections import deque
from pathlib import Path

import cv2
import numpy as np


class SceneRecorder:
    """人が映っている間を 1 シーンとして mp4 を録画し、FIFO でストックする。

    feed() を毎フレーム呼ぶ。person_present が True になったら新規クリップを開き、
    False が end_silence_frames 連続したら閉じる。max_seconds を超えたら強制的に
    閉じて次のシーンに切り替える。min_frames 未満のクリップは破棄する。
    """

    def __init__(
        self,
        output_dir: Path,
        width: int,
        height: int,
        fps: float,
        max_stock: int = 5,
        max_seconds: float = 5.0,
        end_silence_frames: int = 15,
        min_frames: int = 30,
        fourcc: str = 'mp4v',
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.max_stock = max_stock
        self.max_frames = max(1, int(max_seconds * fps))
        self.end_silence_frames = end_silence_frames
        self.min_frames = min_frames
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)

        self.writer: cv2.VideoWriter | None = None
        self.current_path: Path | None = None
        self.current_frame_count = 0
        self.silence_count = 0
        self.clip_index = 0
        self.stock: deque[Path] = deque()

    @property
    def is_recording(self) -> bool:
        return self.writer is not None

    def feed(self, frame: np.ndarray, person_present: bool) -> Path | None:
        """1 フレームを処理する。クリップが完成した瞬間にそのパスを返す。

        破棄された短すぎるクリップでは None を返す。
        """
        if self.writer is None:
            if person_present:
                self._open_new()
                self._write(frame)
            return None

        if person_present:
            self.silence_count = 0
        else:
            self.silence_count += 1

        self._write(frame)

        if self.current_frame_count >= self.max_frames:
            return self._close()
        if self.silence_count >= self.end_silence_frames:
            return self._close()
        return None

    def flush(self) -> Path | None:
        """終了時に呼ぶ。録画中であれば閉じてクリップを確定する。"""
        if self.writer is None:
            return None
        return self._close()

    def _open_new(self) -> None:
        self.clip_index += 1
        path = self.output_dir / f'clip_{self.clip_index:04d}.mp4'
        writer = cv2.VideoWriter(
            str(path), self.fourcc, self.fps, (self.width, self.height)
        )
        if not writer.isOpened():
            raise RuntimeError(f'failed to open VideoWriter for {path}')
        self.writer = writer
        self.current_path = path
        self.current_frame_count = 0
        self.silence_count = 0

    def _write(self, frame: np.ndarray) -> None:
        assert self.writer is not None
        self.writer.write(frame)
        self.current_frame_count += 1

    def _close(self) -> Path | None:
        assert self.writer is not None and self.current_path is not None
        self.writer.release()
        path = self.current_path
        frames = self.current_frame_count
        self.writer = None
        self.current_path = None
        self.current_frame_count = 0
        self.silence_count = 0

        if frames < self.min_frames:
            path.unlink(missing_ok=True)
            return None

        while len(self.stock) >= self.max_stock:
            old = self.stock.popleft()
            old.unlink(missing_ok=True)
        self.stock.append(path)
        return path
