import argparse
import json
from pathlib import Path
from pkgutil import get_data

import joblib


def get_testing_data():
    test_data_path = Path(__file__).parent.parent / "data/prepared/test"
    return get_data(test_data_path)


def load_model(model_path: Path):
    model = joblib.load(model_path)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--metrics-save-path")
    args = parser.parse_args()

    model_path = args.model_path
    metrics_save_path = args.metrics_save_path

    X_test, y_test = get_testing_data()
    model = load_model(model_path)
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    metrics = {"accuracy": test_accuracy, "loss": test_loss}

    json.dump(metrics, metrics_save_path)
