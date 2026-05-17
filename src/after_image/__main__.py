import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / '.env')

import cv2
import numpy as np
from ultralytics import YOLO

from .debug import annotate, list_cameras, render_buffer_grid
from .effects import RENDERERS
from .tracker import SHOULD_RECORD, TrackManager

PERSON_CLASS_ID = 0
CAMERA_INDEX = int(os.environ.get('CAMERA_INDEX', '1'))
FRAME_WIDTH = int(os.environ.get('FRAME_WIDTH', '1280'))
FRAME_HEIGHT = int(os.environ.get('FRAME_HEIGHT', '720'))
MODEL_PATH = os.environ.get(
    'MODEL_PATH', str(PROJECT_ROOT / 'models' / 'yolov8n.pt')
)

render = RENDERERS[os.environ.get('EFFECT', 'plotter')]
should_record = SHOULD_RECORD[os.environ.get('TRIGGER', 'distance')]
DEBUG = os.environ.get('DEBUG', '').lower() in ('1', 'true', 'yes')


def extract_detections(results) -> list[tuple[int, np.ndarray]]:
    if results.boxes.id is None:
        return []
    ids = results.boxes.id.cpu().numpy().astype(int)
    boxes = results.boxes.xyxy.cpu().numpy()
    return [(int(tid), box) for tid, box in zip(ids, boxes)]


def main():
    if DEBUG:
        list_cameras()

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    manager = TrackManager(should_record=should_record)
    frame_count = 0

    if not DEBUG:
        cv2.namedWindow('after-image-py', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            'after-image-py', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1

        results = model.track(
            frame,
            persist=True,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )[0]
        detections = extract_detections(results)

        for tid, box in detections:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            manager.update(tid, cx, cy, frame, frame_count)

        display = render(frame, manager)
        if DEBUG:
            annotate(display, manager, detections)
            cv2.imshow('buffer', render_buffer_grid(manager))

        cv2.imshow('after-image-py', display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
