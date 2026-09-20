from PIL import Image

from kidney_stone_classifier.transforms import build_eval_transforms, build_train_transforms


def test_eval_transform_produces_correct_tensor_shape():
    transform = build_eval_transforms(img_size=64)
    image = Image.new("RGB", (100, 50))

    tensor = transform(image)

    assert tensor.shape == (3, 64, 64)


def test_train_transform_produces_correct_tensor_shape():
    transform = build_train_transforms(img_size=64)
    image = Image.new("RGB", (100, 50))

    tensor = transform(image)

    assert tensor.shape == (3, 64, 64)
