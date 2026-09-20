import pytest
from PIL import Image

from kidney_stone_classifier.dataset import (
    CLASS_TO_IDX,
    KidneyStoneDataset,
    prepare_classification_dataframe,
    stratified_split,
)


def _make_fake_dataset(root, n_normal=20, n_stone=10):
    for class_name, count in [("Normal", n_normal), ("stone", n_stone)]:
        folder = root / class_name
        folder.mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (8, 8), color=(i % 255, 0, 0)).save(folder / f"{i}.jpg")
        # A non-image file that must be ignored.
        (folder / "notes.txt").write_text("not an image")
    return root


def test_prepare_classification_dataframe_finds_images_and_ignores_non_images(tmp_path):
    _make_fake_dataset(tmp_path, n_normal=5, n_stone=3)

    df = prepare_classification_dataframe(tmp_path)

    assert len(df) == 8
    assert set(df["label"]) == {"Normal", "stone"}
    assert (df["label"] == "Normal").sum() == 5
    assert (df["label"] == "stone").sum() == 3


def test_prepare_classification_dataframe_raises_on_missing_class_folder(tmp_path):
    (tmp_path / "Normal").mkdir()

    with pytest.raises(FileNotFoundError):
        prepare_classification_dataframe(tmp_path)


def test_stratified_split_preserves_class_balance(tmp_path):
    _make_fake_dataset(tmp_path, n_normal=40, n_stone=20)
    df = prepare_classification_dataframe(tmp_path)

    train_df, val_df, test_df = stratified_split(df, val_size=0.2, test_size=0.2, seed=42)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    for split in (train_df, val_df, test_df):
        ratio = (split["label"] == "stone").mean()
        assert ratio == pytest.approx(1 / 3, abs=0.1)


def test_kidney_stone_dataset_returns_tensor_and_correct_label(tmp_path):
    _make_fake_dataset(tmp_path, n_normal=2, n_stone=2)
    df = prepare_classification_dataframe(tmp_path)

    dataset = KidneyStoneDataset(df, transform=None)
    image, label = dataset[0]

    assert label in CLASS_TO_IDX.values()
    assert image.size == (8, 8)
