"""Preprocess Intel image classification dataset."""

import glob
import os
from pathlib import Path

import cv2
from matplotlib import pyplot as plt


def get_training_data_path():
    current_folder = Path(__file__).parent
    project_root = current_folder.parent

    return project_root / "data/raw/seg_train"


def get_prepared_data_dir():
    current_folder = Path(__file__).parent
    project_root = current_folder.parent

    return project_root / "data/prepared"


def shrink_dataset_images(new_size: tuple):
    """Preprocess training dataset."""

    train_path = get_training_data_path()

    for folder in os.listdir(f"{train_path}/seg_train"):
        files = glob.glob(pathname=f"{train_path}/seg_train/{folder}/*.jpg")

        for file in files:
            image = plt.imread(file)
            resized_image = cv2.resize(image, new_size)

            file_name = os.path.basename(file)
            prepared_image_dir = get_prepared_data_dir() / f"train/{folder}"
            os.makedirs(prepared_image_dir, exist_ok=True)
            prepared_image_path = prepared_image_dir / file_name
            plt.imsave(prepared_image_path, resized_image)


def main():
    shrink_dataset_images(new_size=(100, 100))
