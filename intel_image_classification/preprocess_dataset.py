"""Preprocess Intel image classification dataset."""

from dataclasses import dataclass
import glob
import os
import argparse
from pathlib import Path

import cv2
from matplotlib import pyplot as plt

from loguru import logger


@dataclass
class Image:
    name: str
    content: cv2.typing.MatLike
    category: str


def collect_images(category_folder):
    """Collect images from category folder"""
    image_files = glob.glob(pathname=f"{category_folder}/*.jpg")

    return [
        Image(
            name=os.path.basename(image_file),
            content=plt.imread(image_file),
            category=os.path.basename(category_folder),
        )
        for image_file in image_files
    ]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", "-d")
    parser.add_argument(
        "--preproc-dataset-path",
        "-o",
    )

    args = parser.parse_args()
    dataset_path = args.dataset_path
    preprocessed_dataset_path = args.preproc_dataset_path

    category_folders = os.listdir(dataset_path)
    for category_folder in category_folders:
        logger.info(f"Preprocessing {category_folder} images...")
        images = collect_images(f"{dataset_path}/{category_folder}")
        preprocessed_images = preprocess_images(images)
        save_preprocessed_images(
            preprocessed_images,
            preprocessed_dataset_path,
        )

    logger.success("Images preprocessed!")
