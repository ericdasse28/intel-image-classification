"""Preprocess Intel image classification dataset."""

import argparse
import os
from pathlib import Path

import cv2
from loguru import logger
from matplotlib import pyplot as plt

from intel_image_classification.image_helpers import Image, collect_images


def preprocess_images(images: list[Image]) -> list[Image]:
    # Shrinking the images can help saving time during training
    shrinked_images = [
        resize_image(
            image,
            new_size=(100, 100),
        )
        for image in images
    ]

    return shrinked_images


def resize_image(image: Image, new_size: tuple) -> Image:
    resized_image = Image(
        image.name, cv2.resize(image.content, new_size), image.category
    )
    return resized_image


def save_preprocessed_images(images: list[Image], save_dir: os.PathLike):
    for image in images:
        preprocessed_image_dir = Path(save_dir) / image.category
        # Create folder if it doesn't exist
        os.makedirs(preprocessed_image_dir, exist_ok=True)

        preprocessed_image_path = preprocessed_image_dir / image.name
        plt.imsave(preprocessed_image_path, image.content)


def generate_prepared_data(raw_path: Path, prepared_path: Path):
    category_folders = os.listdir(raw_path)
    for category_folder in category_folders:
        logger.info(f"Preprocessing {category_folder} images...")
        images = collect_images(f"{raw_path}/{category_folder}")
        preprocessed_images = preprocess_images(images)
        save_preprocessed_images(
            preprocessed_images,
            prepared_path,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", "-d")
    parser.add_argument(
        "--prepared-dataset-path",
        "-o",
    )

    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)
    prepared_dataset_path = Path(args.prepared_dataset_path)

    generate_prepared_data(
        raw_path=dataset_path / "seg_train/seg_train",
        prepared_path=prepared_dataset_path / "train",
    )
    generate_prepared_data(
        raw_path=dataset_path / "seg_test/seg_test",
        prepared_path=prepared_dataset_path / "test",
    )

    logger.success("Images preprocessed!")
