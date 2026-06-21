"""I-frame 削除と P-frame 複製によるデータモッシュ処理。

tiberiuiancu/datamoshing の mosh.py (Unlicense) を関数化したもの。
詳細は LICENSES/datamoshing-UNLICENSE.txt を参照。
"""

import subprocess
import tempfile
from pathlib import Path


FRAME_START = bytes.fromhex('30306463')
IFRAME_MARKER = bytes.fromhex('0001B0')
PFRAME_MARKER = bytes.fromhex('0001B6')


def mosh(
    input_video: Path,
    output_video: Path,
    start_frame: int = 0,
    end_frame: int = -1,
    fps: int = 30,
    delta: int = 0,
    work_dir: Path | None = None,
) -> None:
    """input_video の指定区間にデータモッシュを適用し output_video に書き出す。

    delta=0 のとき I-frame 削除モード、delta>0 のとき P-frame 複製モード。
    work_dir に AVI 中間ファイルを書き出す（未指定なら一時ディレクトリ）。
    """
    input_video = Path(input_video)
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        dir=str(work_dir) if work_dir else None
    ) as td:
        td_path = Path(td)
        input_avi = td_path / 'datamoshing_input.avi'
        output_avi = td_path / 'datamoshing_output.avi'

        subprocess.run(
            [
                'ffmpeg', '-loglevel', 'error', '-y',
                '-i', str(input_video),
                '-crf', '0', '-pix_fmt', 'yuv420p',
                '-bf', '0', '-b', '10000k',
                '-r', str(fps),
                str(input_avi),
            ],
            check=True,
        )

        in_bytes = input_avi.read_bytes()
        frames = in_bytes.split(FRAME_START)

        with output_avi.open('wb') as out:
            out.write(frames[0])
            body = frames[1:]

            n_video_frames = sum(
                1 for f in body if f[5:8] in (IFRAME_MARKER, PFRAME_MARKER)
            )
            actual_end = n_video_frames if end_frame < 0 else end_frame

            if delta:
                _write_delta_repeat(out, body, start_frame, actual_end, delta)
            else:
                _write_iframe_removal(out, body, start_frame, actual_end)

        subprocess.run(
            [
                'ffmpeg', '-loglevel', 'error', '-y',
                '-i', str(output_avi),
                '-crf', '18', '-pix_fmt', 'yuv420p',
                '-vcodec', 'libx264', '-acodec', 'aac',
                '-b', '10000k', '-r', str(fps),
                str(output_video),
            ],
            check=True,
        )


def _write_frame(out, frame: bytes) -> None:
    out.write(FRAME_START + frame)


def _write_iframe_removal(
    out, frames: list[bytes], start: int, end: int
) -> None:
    for index, frame in enumerate(frames):
        if index < start or end < index or frame[5:8] != IFRAME_MARKER:
            out.write(FRAME_START + frame)


def _write_delta_repeat(
    out, frames: list[bytes], start: int, end: int, n_repeat: int
) -> None:
    if n_repeat > end - start:
        raise ValueError('not enough frames to repeat')

    repeat_frames: list[bytes] = []
    repeat_index = 0
    for index, frame in enumerate(frames):
        marker = frame[5:8]
        if marker not in (IFRAME_MARKER, PFRAME_MARKER) or not (
            start <= index < end
        ):
            _write_frame(out, frame)
            continue

        if len(repeat_frames) < n_repeat and marker != IFRAME_MARKER:
            repeat_frames.append(frame)
            _write_frame(out, frame)
        elif len(repeat_frames) == n_repeat:
            _write_frame(out, repeat_frames[repeat_index])
            repeat_index = (repeat_index + 1) % n_repeat
        else:
            _write_frame(out, frame)
