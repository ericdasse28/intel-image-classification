import argparse
from pathlib import Path

import joblib
from loguru import logger

from dvclive import Live
from intel_image_classification.dataset import get_data


def get_testing_data():
    test_data_path = Path(__file__).parent.parent / "data/prepared/test"
    return get_data(test_data_path)


def load_model(model_path: Path):
    model = joblib.load(model_path)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    args = parser.parse_args()

    model_path = args.model_path

    logger.info("Model evaluation...")
    X_test, y_test = get_testing_data()
    model = load_model(model_path)
    with Live(resume=True) as live:
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        live.log_metric("test/accuracy", test_accuracy, plot=False)
        live.log_metric("test/loss", test_loss, plot=False)
