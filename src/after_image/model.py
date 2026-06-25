from pathlib import Path

NCNN_DIR_NAME = 'yolov8n_ncnn_model'
PT_NAME = 'yolov8n.pt'


def default_model_path(project_root: Path) -> str:
    """Resolve the default YOLO model path.

    Prefer the exported NCNN model directory when present: NCNN runs inference
    on CPU without PyTorch's compute kernels, which avoids the
    "Illegal instruction" crash seen with the PyPI torch wheels on some
    aarch64 boards (e.g. Raspberry Pi). Fall back to the .pt weights
    (torch backend) when the NCNN model is not available.
    """
    ncnn = project_root / 'models' / NCNN_DIR_NAME
    if ncnn.is_dir():
        return str(ncnn)
    return str(project_root / 'models' / PT_NAME)
