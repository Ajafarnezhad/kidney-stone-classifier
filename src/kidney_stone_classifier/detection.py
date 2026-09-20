"""YOLOv8-based kidney stone detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml
from ultralytics import YOLO

logger = logging.getLogger(__name__)

CLASS_NAMES = ["stone"]


@dataclass
class DetectionConfig:
    """Expects a YOLO-format dataset: `data_root/images/{train,val}` and
    `data_root/labels/{train,val}`, mirroring what `ultralytics` requires.
    """

    data_root: Path
    epochs: int = 15
    img_size: int = 384
    batch_size: int = 16
    weights: str = "yolov8n.pt"
    save_dir: Path = Path("saved_models")
    run_name: str = "kidney_stone_yolo"


def write_dataset_yaml(config: DetectionConfig) -> Path:
    """Generate the Ultralytics dataset YAML that `YOLO.train` requires.

    `ultralytics` expects `data=` to be a YAML file describing `train`/`val`
    image directories and class names -- passing a bare folder there does
    not train on the intended data.
    """
    data_root = Path(config.data_root)
    yaml_path = data_root / "kidney_stone.yaml"
    payload = {
        "path": str(data_root),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    yaml_path.write_text(yaml.safe_dump(payload))
    logger.info("Wrote YOLO dataset config to %s", yaml_path)
    return yaml_path


def train_yolo(config: DetectionConfig) -> YOLO:
    dataset_yaml = write_dataset_yaml(config)
    model = YOLO(config.weights)
    model.train(
        data=str(dataset_yaml),
        epochs=config.epochs,
        imgsz=config.img_size,
        batch=config.batch_size,
        workers=4,
        amp=True,
        name=config.run_name,
        project=str(config.save_dir),
    )
    return model


def run_inference(model: YOLO, image_paths: list[str], output_dir: Path) -> list[Path]:
    """Run detection on a list of images and save annotated copies.

    Uses `ultralytics`' own `Results.save()` instead of hand-rolled OpenCV
    drawing code, which assumed a hardcoded `.JPG` (uppercase) extension for
    matching label files and broke on any other case or format.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for image_path in image_paths:
        results = model(image_path)
        for result in results:
            out_path = output_dir / Path(image_path).name
            result.save(filename=str(out_path))
            saved_paths.append(out_path)

    return saved_paths
