# Kidney Stone Classifier & Detector

Transfer-learning classification (EfficientNet-B3 / ResNet-101 / DenseNet-161)
and YOLOv8 object detection for kidney stone medical imaging.

A tested, modular Python package with a real CLI (`--epochs`, `--batch-size`,
`--img-size`, and more) for both the classification and detection pipelines.

## Features

- **Classification**: fine-tune EfficientNet-B3, ResNet-101, or DenseNet-161
  with a binary head, mixed-precision training, cosine LR scheduling, and
  early best-checkpoint saving.
- **Detection**: train a YOLOv8 model to localize stones, with a properly
  generated Ultralytics dataset YAML.
- **Evaluation**: classification report, confusion matrix, and training-curve
  plots saved to disk as artifacts.
- **Tested**: dataset scanning, stratified splitting, transform shapes, and
  YOLO config generation are covered by unit tests that don't require a GPU,
  a dataset download, or downloading pretrained weights.

## Design notes

- **Real CLI**: `kidney_stone_classifier.cli` exposes `--epochs`,
  `--batch-size`, `--img-size` and more for both `classify` and `detect`.
- **Valid YOLO training config**: `detection.py` generates the YAML file
  Ultralytics' `data=` argument requires (describing `train`/`val` image
  directories and class names), rather than pointing it at a bare image
  folder.
- **Stratified splitting**: `stratified_split` uses
  `sklearn.model_selection.train_test_split` with `stratify=`, guaranteeing
  minority-class examples land in every split.
- **Current AMP API**: uses `torch.autocast`/`torch.amp.GradScaler`, guarded
  so the same code runs on CPU without mixed precision.
- **No hardcoded paths**: the CLI takes `--data-path` / `--data-root`
  explicitly; dataset download is a separate opt-in script
  (`scripts/download_dataset.py`) so importing the package never triggers a
  network call.
- **Robust bounding-box drawing**: detection inference uses Ultralytics' own
  `Results.save()` instead of hand-rolled OpenCV code tied to a specific
  file-extension convention.
- **Saved artifacts**: evaluation plots are written to disk
  (`saved_models/<model>_confusion_matrix.png`, `..._history.png`) instead
  of only being shown inline.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Dataset

Classification expects a folder with `Normal/` and `stone/` subdirectories
of images. Detection expects a YOLO-format dataset root with
`images/{train,val}` and `labels/{train,val}`.

To fetch the source Kaggle dataset used for classification:

```bash
python scripts/download_dataset.py
```

(requires a Kaggle API token at `~/.kaggle/kaggle.json`).

## Usage

Train and evaluate classification models:

```bash
python -m kidney_stone_classifier.cli classify \
    --data-path /path/to/dataset \
    --models efficientnet resnet densenet \
    --epochs 15 --batch-size 32 --img-size 384
```

Train a YOLOv8 detector and optionally run inference on a few images:

```bash
python -m kidney_stone_classifier.cli detect \
    --data-root /path/to/yolo_dataset \
    --epochs 15 \
    --predict-images image1.jpg image2.jpg
```

Run the test suite:

```bash
pytest
```

## Project structure

```
├── scripts/download_dataset.py   # optional Kaggle download helper
├── src/kidney_stone_classifier/
│   ├── dataset.py       # dataframe building, stratified split, Dataset class
│   ├── transforms.py    # train/eval augmentation pipelines
│   ├── models.py         # backbone factory
│   ├── train.py          # training loop for one classification model
│   ├── evaluate.py       # test-set report + plots
│   ├── detection.py      # YOLO dataset YAML + train/inference wrappers
│   └── cli.py             # `classify` / `detect` subcommands
├── tests/
└── saved_models/           # generated at train time (gitignored)
```

## Possible extensions

- Grad-CAM visualizations for classification interpretability.
- Automated hyperparameter search (e.g. Optuna) across backbones.
- Multi-modal fusion with clinical metadata alongside imaging.

## License

MIT — see [LICENSE](LICENSE).

## Author

**Amirhossein Jafarnezhad** ([@Ajafarnezhad](https://github.com/Ajafarnezhad))
