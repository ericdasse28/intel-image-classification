"""Preprocess Intel image classification dataset."""

import glob
import os

import cv2
from matplotlib import pyplot as plt

from intel_image_classification.dataset_paths import (
    get_prepared_train_data_dir,
    get_raw_training_data_dir,
)


def shrink_dataset_images(new_size: tuple):
    """Preprocess training dataset."""

    raw_train_data_dir = get_raw_training_data_dir()

    for class_folder in os.listdir(f"{raw_train_data_dir}/seg_train"):
        image_files = glob.glob(
            pathname=f"{raw_train_data_dir}/seg_train/{class_folder}/*.jpg"
        )
        prepared_image_dir = get_prepared_train_data_dir() / class_folder
        os.makedirs(prepared_image_dir, exist_ok=True)

        for image_file in image_files:
            image = plt.imread(image_file)
            resized_image = cv2.resize(image, new_size)

            image_file_name = os.path.basename(image_file)
            prepared_image_path = prepared_image_dir / image_file_name
            plt.imsave(prepared_image_path, resized_image)


def main():
    shrink_dataset_images(new_size=(100, 100))
