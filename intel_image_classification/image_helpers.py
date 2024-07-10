from dataclasses import dataclass
import glob
import os

import cv2
from matplotlib import pyplot as plt


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
