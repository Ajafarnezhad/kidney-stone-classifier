from pathlib import Path

import yaml

from kidney_stone_classifier.detection import CLASS_NAMES, DetectionConfig, write_dataset_yaml


def test_write_dataset_yaml_produces_valid_ultralytics_config(tmp_path):
    config = DetectionConfig(data_root=tmp_path)

    yaml_path = write_dataset_yaml(config)

    assert yaml_path == tmp_path / "kidney_stone.yaml"
    payload = yaml.safe_load(yaml_path.read_text())

    assert payload["path"] == str(tmp_path)
    assert payload["train"] == "images/train"
    assert payload["val"] == "images/val"
    assert payload["names"] == {i: name for i, name in enumerate(CLASS_NAMES)}
