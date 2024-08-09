"""Train the model."""

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import yaml
from keras import Sequential, layers, losses
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
    # To save memory, we perform normalization in place
    with np.nditer(X, flags=["multi_index"], op_flags=["readwrite"]) as it:
        for x in it:
            x[...] = float(x) / 255.0

    return X


def get_training_data(images: list[Image]):
    X_train, y_train = get_features_and_labels(images)
    X_train = normalize_data(X_train)

    return X_train, y_train


def train(X_train, y_train, *, batch_size, validation_split, epochs):
    model = Sequential(
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
            layers.Dense(6, activation="softmax"),  # Output layer
        ]
    )

    model.compile(
        optimizer="adam",
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
    )

    return model


def get_training_images(dataset_path):
    images = []
    category_folders = os.listdir(dataset_path)
    for category_folder in category_folders:
        logger.info(f"Collecting images from {category_folder}")
        images.extend(collect_images(f"{dataset_path}/{category_folder}"))
    return images


def get_training_params():
    params_path = Path(__file__).parent.parent / "params.yaml"
    with open(params_path) as params_file:
        training_params = yaml.safe_load(params_file)["train"]

    return training_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", "-d")
    parser.add_argument("--model-save-path", "-o")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    model_save_path = args.model_save_path

    logger.info(f"Loading dataset from {dataset_path}...")
    images = get_training_images(dataset_path)

    logger.info("Getting training data...")
    X_train, y_train = get_training_data(images)

    logger.info("Training the model...")
    training_params = get_training_params()
    model = train(X_train, y_train, **training_params)

    logger.info(f"Saving model to {model_save_path}")
    joblib.dump(model, model_save_path)
