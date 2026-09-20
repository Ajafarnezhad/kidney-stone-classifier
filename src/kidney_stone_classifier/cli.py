"""Command-line entry point.

Usage:
    python -m kidney_stone_classifier.cli classify --data-path /path/to/dataset
    python -m kidney_stone_classifier.cli detect --data-root /path/to/yolo_dataset
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import KidneyStoneDataset, prepare_classification_dataframe, stratified_split
from .detection import DetectionConfig, run_inference, train_yolo
from .evaluate import evaluate_classification_model, plot_training_history
from .models import SUPPORTED_MODELS
from .train import train_one_model
from .transforms import build_eval_transforms, build_train_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_classification(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    df = prepare_classification_dataframe(args.data_path)
    train_df, val_df, test_df = stratified_split(
        df, val_size=args.val_size, test_size=args.test_size, seed=args.seed
    )
    logger.info("Dataset split: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(
        KidneyStoneDataset(train_df, build_train_transforms(args.img_size)), shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        KidneyStoneDataset(val_df, build_eval_transforms(args.img_size)), shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        KidneyStoneDataset(test_df, build_eval_transforms(args.img_size)), shuffle=False, **loader_kwargs
    )

    save_dir = Path(args.save_dir)
    for model_name in args.models:
        logger.info("Training %s...", model_name)
        model, history, best_path = train_one_model(
            model_name, train_loader, val_loader, device, args.epochs, args.lr, args.weight_decay, save_dir
        )
        logger.info("Saved best %s checkpoint to %s", model_name, best_path)
        plot_training_history(history, model_name, save_dir)
        evaluate_classification_model(model, test_loader, device, model_name, save_dir)


def run_detection(args: argparse.Namespace) -> None:
    config = DetectionConfig(
        data_root=Path(args.data_root),
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        save_dir=Path(args.save_dir),
    )
    model = train_yolo(config)
    if args.predict_images:
        saved = run_inference(model, args.predict_images, Path(args.save_dir) / "predictions")
        logger.info("Saved %d annotated predictions.", len(saved))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kidney stone classification and detection pipelines.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    classify = subparsers.add_parser("classify", help="Train and evaluate classification models.")
    classify.add_argument("--data-path", required=True, help="Folder containing 'Normal' and 'stone' subfolders.")
    classify.add_argument("--models", nargs="+", default=list(SUPPORTED_MODELS), choices=SUPPORTED_MODELS)
    classify.add_argument("--epochs", type=int, default=15)
    classify.add_argument("--batch-size", type=int, default=32)
    classify.add_argument("--img-size", type=int, default=384)
    classify.add_argument("--lr", type=float, default=1e-3)
    classify.add_argument("--weight-decay", type=float, default=1e-2)
    classify.add_argument("--val-size", type=float, default=0.15)
    classify.add_argument("--test-size", type=float, default=0.15)
    classify.add_argument("--num-workers", type=int, default=4)
    classify.add_argument("--save-dir", default="saved_models")
    classify.add_argument("--seed", type=int, default=42)
    classify.set_defaults(func=run_classification)

    detect = subparsers.add_parser("detect", help="Train a YOLOv8 kidney stone detector.")
    detect.add_argument("--data-root", required=True, help="YOLO-format dataset root (images/, labels/).")
    detect.add_argument("--epochs", type=int, default=15)
    detect.add_argument("--batch-size", type=int, default=16)
    detect.add_argument("--img-size", type=int, default=384)
    detect.add_argument("--save-dir", default="saved_models")
    detect.add_argument("--predict-images", nargs="*", default=[])
    detect.set_defaults(func=run_detection)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
