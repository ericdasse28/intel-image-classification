"""Module that facilitates interactions with
the training and testing datasets."""

import os
from pathlib import Path

import numpy as np

from intel_image_classification.image_helpers import Image, collect_images


def get_data(data_path: Path):
    images = get_images(data_path)
    X, y = get_features_and_labels(images)
    X = normalize(X)

    return X, y


def get_images(data_path: Path):
    images = []
    category_folders = os.listdir(data_path)
    for category_folder in category_folders:
        images.extend(collect_images(f"{data_path}/{category_folder}"))

    return images


def get_features_and_labels(images: list[Image]):
    X_train = []
    y_train = []

    for image in images:
        X_train.append(image.content)
        y_train.append(get_label_index(image.category))

    return np.array(X_train), np.array(y_train)


def normalize(X: np.ndarray):
    # To save memory, we perform normalization in place
    with np.nditer(X, flags=["multi_index"], op_flags=["readwrite"]) as it:
        for x in it:
            x[...] = float(x) / 255.0

    return X


def get_label_index(label: str) -> int:
    nature_labels_dict = {
        "buildings": 0,
        "forest": 1,
        "glacier": 2,
        "mountain": 3,
        "sea": 4,
        "street": 5,
    }

    return nature_labels_dict[label]
