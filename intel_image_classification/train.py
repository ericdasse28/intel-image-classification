"""Train the model."""

import argparse
import os

import joblib
import numpy as np
import tensorflow as tf
from keras import layers
from loguru import logger

from intel_image_classification.image_helpers import Image, collect_images


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


def get_features_and_labels(images: list[Image]):
    X_train = []
    y_train = []

    for image in images:
        X_train.append(image.content)
        y_train.append(get_label_index(image.category))

    return np.array(X_train), np.array(y_train)


def normalize_data(X: np.ndarray):
    with np.nditer(X, flags=["multi_index"], op_flags=["readwrite"]) as it:
        for x in it:
            x[...] = float(x) / 255.0

    return X


def train(X_train, y_train):
    num_classes = 6
    model = tf.keras.Sequential(
        [
            layers.Conv2D(
                filters=32, kernel_size=(3, 3), activation="relu"
            ),  # Convolutional layer
            layers.MaxPooling2D(2, 2),  # Pooling layer
            layers.Dropout(0.25),  # Dropout layer
            layers.Conv2D(
                filters=64, kernel_size=(3, 3), activation="relu"
            ),  # Convolutional layer
            layers.MaxPooling2D(2, 2),  # Pooling layer
            layers.Dropout(0.25),  # Dropout layer
            layers.Conv2D(
                filters=128, kernel_size=(3, 3), activation="relu"
            ),  # Convolutional layer
            layers.MaxPooling2D(2, 2),  # Pooling layer
            layers.Dropout(0.25),  # Dropout layer
            layers.Flatten(),  # Flatten layer
            layers.Dense(128, activation="relu"),  # Fully connected layer
            layers.Dropout(0.5),  # Dropout layer
            layers.Dense(num_classes, activation="softmax"),  # Output layer
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=2,
        validation_split=0.2,
    )

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", "-d")
    parser.add_argument("--model-save-path", "-o")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    model_save_path = args.model_save_path

    logger.info(f"Loading dataset from {dataset_path}...")
    images = []
    category_folders = os.listdir(dataset_path)
    for category_folder in category_folders:
        logger.info(f"Collecting images from {category_folder}")
        images.extend(collect_images(f"{dataset_path}/{category_folder}"))

    logger.info("Getting training data...")
    X_train, y_train = get_features_and_labels(images)
    logger.info("Normalizing training data...")
    X_train = normalize_data(X_train)

    logger.info("Training the model...")
    model = train(X_train, y_train)

    logger.info(f"Saving model to {model_save_path}")
    joblib.dump(model, model_save_path)
