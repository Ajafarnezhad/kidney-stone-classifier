"""Download the Kaggle kidney stone dataset and print its local path.

Requires a configured Kaggle API token (`~/.kaggle/kaggle.json`).
Kept separate from the library code so importing `kidney_stone_classifier`
never triggers a network download as a side effect.

Usage:
    python scripts/download_dataset.py
"""

import kagglehub

DATASET = "imtkaggleteam/kidney-stone-classification-and-object-detection"


def main() -> None:
    path = kagglehub.dataset_download(DATASET)
    print(f"Dataset downloaded to: {path}")


if __name__ == "__main__":
    main()
