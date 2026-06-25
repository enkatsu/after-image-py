import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / '.env')

import cv2
from ultralytics import YOLO

from ..model import default_model_path
from ..recorder import SceneRecorder
from .player import MoshedPlayer
from .worker import MoshWorker

PERSON_CLASS_ID = 0
CAMERA_INDEX = int(os.environ.get('CAMERA_INDEX', '1'))
FRAME_WIDTH = int(os.environ.get('FRAME_WIDTH', '1280'))
FRAME_HEIGHT = int(os.environ.get('FRAME_HEIGHT', '720'))
MODEL_PATH = os.environ.get('MODEL_PATH', default_model_path(PROJECT_ROOT))
CLIPS_DIR = Path(os.environ.get('CLIPS_DIR', str(PROJECT_ROOT / 'clips')))
MOSHED_DIR = Path(os.environ.get('MOSHED_DIR', str(PROJECT_ROOT / 'moshed')))

MAX_STOCK = int(os.environ.get('MAX_STOCK', '5'))
SCENE_MAX_SECONDS = float(os.environ.get('SCENE_MAX_SECONDS', '5.0'))
SCENE_END_SILENCE_FRAMES = int(os.environ.get('SCENE_END_SILENCE_FRAMES', '15'))
SCENE_MIN_FRAMES = int(os.environ.get('SCENE_MIN_FRAMES', '30'))
MOSH_DELTA = int(os.environ.get('MOSH_DELTA', '0'))

DEBUG = os.environ.get('DEBUG', '').lower() in ('1', 'true', 'yes')


def draw_debug(
    display, recorder: SceneRecorder, player: MoshedPlayer, fps: float
) -> None:
    mode = 'MOSHED' if player.is_ready() else 'LIVE'
    rec = 'REC' if recorder.is_recording else 'IDLE'
    rec_color = (0, 0, 255) if recorder.is_recording else (200, 200, 200)
    cv2.putText(
        display, f'{mode}', (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
    )
    cv2.putText(
        display, f'{rec}  stock={len(recorder.stock)}/{MAX_STOCK}', (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, rec_color, 2,
    )
    if recorder.is_recording:
        secs = recorder.current_frame_count / max(fps, 1.0)
        cv2.putText(
            display, f'{recorder.current_frame_count}f / {secs:.1f}s',
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 2,
        )


def main():
    model = YOLO(MODEL_PATH, task='detect')
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or FRAME_WIDTH
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or FRAME_HEIGHT
    print(f'camera: {actual_w}x{actual_h} @ {fps:.2f}fps')

    recorder = SceneRecorder(
        output_dir=CLIPS_DIR,
        width=actual_w,
        height=actual_h,
        fps=fps,
        max_stock=MAX_STOCK,
        max_seconds=SCENE_MAX_SECONDS,
        end_silence_frames=SCENE_END_SILENCE_FRAMES,
        min_frames=SCENE_MIN_FRAMES,
    )
    player = MoshedPlayer()
    worker = MoshWorker(
        output_dir=MOSHED_DIR,
        fps=int(round(fps)),
        on_ready=lambda p: (player.update_source(p), print(f'moshed ready: {p.name}')),
        delta=MOSH_DELTA,
    )
    mode = f'delta={MOSH_DELTA}' if MOSH_DELTA > 0 else 'iframe-removal'
    print(f'mosh mode: {mode}')
    worker.start()

    cv2.namedWindow('after-image-datamosh', cv2.WINDOW_NORMAL)
    if not DEBUG:
        cv2.setWindowProperty(
            'after-image-datamosh',
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.track(
                frame,
                persist=True,
                classes=[PERSON_CLASS_ID],
                verbose=False,
            )[0]
            person_present = (
                results.boxes.id is not None and len(results.boxes.id) > 0
            )

            finished = recorder.feed(frame, person_present)
            if finished is not None:
                print(
                    f'clip finished: {finished.name} '
                    f'(stock={len(recorder.stock)}/{MAX_STOCK})'
                )
                worker.request(list(recorder.stock))

            display = player.next_frame()
            if display is None:
                display = frame

            if DEBUG:
                display = display.copy()
                draw_debug(display, recorder, player, fps)

            cv2.imshow('after-image-datamosh', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        worker.stop()
        player.close()
        finished = recorder.flush()
        if finished is not None:
            print(f'clip finished (flush): {finished.name}')
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
