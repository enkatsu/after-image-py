from collections.abc import Callable, Iterable

import cv2
import numpy as np

from .tracker import Track


Renderer = Callable[[np.ndarray, Iterable[Track]], np.ndarray]


def collect_records(people: Iterable[Track]) -> list[tuple[int, np.ndarray]]:
    """全人物の残像フレームを重複なく集めて時系列順に並べる。"""
    seen: set[int] = set()
    records: list[tuple[int, np.ndarray]] = []
    for track in people:
        for ts, img in track.frames:
            if ts in seen:
                continue
            seen.add(ts)
            records.append((ts, img))
    records.sort(key=lambda x: x[0])
    return records


def multiply_blend(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """乗算合成。白い部分は透過、黒い部分は重ねるほど濃くなる。"""
    return cv2.multiply(a, b, scale=1 / 255.0)


def multiple_exposure(
    current: np.ndarray, people: Iterable[Track], current_alpha: float = 0.1
) -> np.ndarray:
    """多重露光。残像を時系列に重ね、最後に現在フレームを薄く混ぜる。"""
    records = collect_records(people)
    n = len(records)
    if n == 0:
        return current.copy()

    display = np.zeros_like(current)
    for i, (_, img) in enumerate(records):
        t = ((i / (n - 1)) ** 2.5) * 0.5 if n > 1 else 0.5
        display = cv2.addWeighted(display, 1.0 - t, img, t, 0)

    return cv2.addWeighted(display, 1.0 - current_alpha, current, current_alpha, 0)


def denoise(src: np.ndarray) -> np.ndarray:
    """エッジ検出前のノイズ低減。輪郭を保ったまま平滑化する。"""
    return cv2.bilateralFilter(src, d=5, sigmaColor=50, sigmaSpace=50)


def to_line_art(src: np.ndarray) -> np.ndarray:
    """Sobel フィルタで階調のある線画に変換する（白背景に黒線）。"""
    gray = cv2.cvtColor(denoise(src), cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.clip(cv2.magnitude(gx, gy), 0, 255).astype(np.uint8)
    line = 255 - mag
    return cv2.cvtColor(line, cv2.COLOR_GRAY2BGR)


def to_plotter_line(src: np.ndarray) -> np.ndarray:
    """Canny で 1px のシャープな線に変換する（プロッター描画風）。"""
    gray = cv2.cvtColor(denoise(src), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 80)
    line = 255 - edges
    return cv2.cvtColor(line, cv2.COLOR_GRAY2BGR)


def line_overlay(
    current: np.ndarray,
    people: Iterable[Track],
    line_fn: Callable[[np.ndarray], np.ndarray],
    exponent: float,
) -> np.ndarray:
    """線画化した残像を白背景に乗算合成で重ねる。exponent が小さいほど古い残像も濃く残る。"""
    records = collect_records(people)
    n = len(records)
    result = np.full_like(current, 255)
    white = np.full_like(current, 255)

    for i, (_, img) in enumerate(records):
        t = ((i / (n - 1)) ** exponent) if n > 1 else 1.0
        faded = cv2.addWeighted(line_fn(img), t, white, 1.0 - t, 0)
        result = multiply_blend(result, faded)

    return multiply_blend(result, line_fn(current))


def sketch_overlay(current: np.ndarray, people: Iterable[Track]) -> np.ndarray:
    """鉛筆スケッチ風の残像表現。古い残像はかなり薄くなる。"""
    return line_overlay(current, people, to_line_art, exponent=2.5)


def plotter_overlay(current: np.ndarray, people: Iterable[Track]) -> np.ndarray:
    """プロッター描画風の残像表現。古い残像も線形にフェードして残す。"""
    return line_overlay(current, people, to_plotter_line, exponent=1.0)


RENDERERS: dict[str, Renderer] = {
    'multi': multiple_exposure,
    'sketch': sketch_overlay,
    'plotter': plotter_overlay,
}
