"""Helper functions to retrieve datasets path."""

from pathlib import Path


def _get_project_root():
    current_folder = Path(__file__).parent
    project_root = current_folder.parent
    return project_root


def get_raw_training_data_dir():
    project_root = _get_project_root()

    return project_root / "data/raw/seg_train"


def get_prepared_data_dir():
    project_root = _get_project_root()

    return project_root / "data/prepared"
