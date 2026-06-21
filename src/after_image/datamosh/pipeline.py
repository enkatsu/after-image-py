"""複数の mp4 クリップを結合し、データモッシュを適用して 1 本の mp4 を出力する。"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from .mosh import mosh

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / '.env')

CLIPS_DIR = Path(os.environ.get('CLIPS_DIR', str(PROJECT_ROOT / 'clips')))
MOSHED_DIR = Path(os.environ.get('MOSHED_DIR', str(PROJECT_ROOT / 'moshed')))
MOSHED_FPS = int(float(os.environ.get('MOSHED_FPS', '30')))
MOSH_DELTA = int(os.environ.get('MOSH_DELTA', '0'))


def probe_frame_count(video: Path, fps: int) -> int:
    """ffprobe で動画の長さを取得し、指定 fps でのフレーム数に換算する。

    AVI 再エンコード後のフレーム数を想定するため fps を掛けて推定する。
    """
    result = subprocess.run(
        [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video),
        ],
        capture_output=True, text=True, check=True,
    )
    duration = float(result.stdout.strip())
    return int(round(duration * fps))


def concat_clips(
    clip_paths: list[Path], output: Path, work_dir: Path
) -> None:
    """ffmpeg concat demuxer で複数 mp4 を結合する。再エンコードはしない。"""
    listing = work_dir / 'concat_list.txt'
    listing.write_text(
        ''.join(f"file '{p.resolve()}'\n" for p in clip_paths)
    )
    subprocess.run(
        [
            'ffmpeg', '-loglevel', 'error', '-y',
            '-f', 'concat', '-safe', '0',
            '-i', str(listing),
            '-c', 'copy',
            str(output),
        ],
        check=True,
    )


def build_moshed(
    clip_paths: list[Path],
    output_path: Path,
    fps: int = 30,
    work_dir: Path | None = None,
    delta: int = 0,
) -> Path:
    """クリップ群を時系列で結合し、データモッシュを適用した mp4 を作る。

    delta=0 のとき I-frame 除去：全クリップを結合してから 2 本目以降の I-frame を
    削除する（シーン境目で前クリップの絵が後クリップの動きで歪む）。
    delta>0 のとき P-frame 複製：各クリップを個別に delta mosh してから結合する
    （各シーンの冒頭の動きがそのシーン内で繰り返し再生される）。
    """
    clips = [Path(p) for p in clip_paths]
    if not clips:
        raise ValueError('no clips provided')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        dir=str(work_dir) if work_dir else None
    ) as td:
        td_path = Path(td)

        if delta > 0:
            _build_delta_per_clip(clips, output_path, fps, delta, td_path)
        else:
            _build_iframe_removal_concat(clips, output_path, fps, td_path)

    return output_path


def _build_iframe_removal_concat(
    clips: list[Path], output_path: Path, fps: int, td_path: Path
) -> None:
    if len(clips) == 1:
        input_video = clips[0]
        start_frame = 1
    else:
        combined = td_path / 'combined.mp4'
        concat_clips(clips, combined, td_path)
        input_video = combined
        start_frame = probe_frame_count(clips[0], fps)

    mosh(
        input_video=input_video,
        output_video=output_path,
        start_frame=start_frame,
        fps=fps,
        delta=0,
        work_dir=td_path,
    )


def _build_delta_per_clip(
    clips: list[Path],
    output_path: Path,
    fps: int,
    delta: int,
    td_path: Path,
) -> None:
    moshed_clips: list[Path] = []
    for i, clip in enumerate(clips):
        moshed_i = td_path / f'moshed_{i:03d}.mp4'
        mosh(
            input_video=clip,
            output_video=moshed_i,
            start_frame=0,
            fps=fps,
            delta=delta,
            work_dir=td_path,
        )
        moshed_clips.append(moshed_i)

    if len(moshed_clips) == 1:
        shutil.move(str(moshed_clips[0]), str(output_path))
    else:
        concat_clips(moshed_clips, output_path, td_path)


def main() -> None:
    """CLIエントリ。CLIPS_DIR 内の clip_*.mp4 を全部結合して moshed.mp4 を出す。"""
    clip_paths = sorted(CLIPS_DIR.glob('clip_*.mp4'))
    if not clip_paths:
        print(f'no clips found in {CLIPS_DIR}')
        return

    output = MOSHED_DIR / 'moshed.mp4'
    mode = f'delta={MOSH_DELTA}' if MOSH_DELTA > 0 else 'iframe-removal'
    print(f'moshing {len(clip_paths)} clips ({mode}) -> {output}')
    for p in clip_paths:
        print(f'  - {p.name}')
    build_moshed(clip_paths, output, fps=MOSHED_FPS, delta=MOSH_DELTA)
    print(f'done: {output}')


if __name__ == '__main__':
    main()
