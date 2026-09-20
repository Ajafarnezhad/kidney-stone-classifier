"""Classification dataset preparation and loading."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Normal", "stone"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def prepare_classification_dataframe(data_path: str | Path) -> pd.DataFrame:
    """Scan `data_path/<class_name>/` folders and list image files with labels.

    Filters by file extension (case-insensitive) so non-image files, such
    as stray `.txt` annotation leftovers, are never included.
    """
    data_path = Path(data_path)
    records = []

    for class_name in CLASS_NAMES:
        folder = data_path / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected class folder not found: {folder}")
        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({"file_path": str(image_path), "label": class_name})

    if not records:
        raise FileNotFoundError(f"No images found under {data_path} for classes {CLASS_NAMES}")

    df = pd.DataFrame(records)
    logger.info("Found %d images: %s", len(df), df["label"].value_counts().to_dict())
    return df


def stratified_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split that preserves class balance in each split.

    Guards against a validation or test split ending up with too few (or
    zero) examples of the minority class on an imbalanced dataset.
    """
    if not 0 < val_size + test_size < 1:
        raise ValueError("val_size + test_size must be between 0 and 1.")

    holdout_size = val_size + test_size
    train_df, holdout_df = train_test_split(
        df, test_size=holdout_size, stratify=df["label"], random_state=seed
    )
    relative_test_size = test_size / holdout_size
    val_df, test_df = train_test_split(
        holdout_df, test_size=relative_test_size, stratify=holdout_df["label"], random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class KidneyStoneDataset(Dataset):
    """PyTorch `Dataset` over a dataframe of `file_path`/`label` rows."""

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]
        image = Image.open(row["file_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, CLASS_TO_IDX[row["label"]]
