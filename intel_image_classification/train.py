"""Train the model."""

import joblib

import argparse
import os

import tensorflow as tf
from keras import layers
import numpy as np
from intel_image_classification.image_helpers import Image, collect_images


def get_training_data(images: list[Image]):
    X_train = []
    y_train = []

    for image in images:
        X_train.append(image.content)
        y_train.append(image.category)

    return np.array(X_train), np.array(y_train)


def normalize_data(X: np.ndarray):
    return X.astype(np.float32) / 255.0


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

    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", "-d")
    parser.add_argument("--model-save-path", "-o")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    model_save_path = args.model_save_path

    images = []
    category_folders = os.listdir(dataset_path)
    for category_folder in category_folders:
        images.extend(collect_images(f"{dataset_path}/{category_folder}"))

    X_train, y_train = get_training_data(images)
    X_train = normalize_data(X_train)
    model = train(X_train, y_train)
    joblib.dump(model, model_save_path)
