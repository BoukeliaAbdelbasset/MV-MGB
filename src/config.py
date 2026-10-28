"""
Global configuration, random seeds, and dataset path discovery for MV-MGB.
"""
import os
import random
import numpy as np

GLOBAL_SEED = 42
N_JOBS = int(os.environ.get("N_JOBS", 4))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DATA_DIR = os.path.join(REPO_ROOT, "datasets")

DATASET_PATHS = {
    "dataset1": {
        "train": os.path.join(LOCAL_DATA_DIR, "dataset1/train.fasta"),
        "test": os.path.join(LOCAL_DATA_DIR, "dataset1/test.fasta"),
        "type": "fasta"
    },
    "dataset1-nd": {
        "train": os.path.join(LOCAL_DATA_DIR, "dataset1-nd/train.fasta"),
        "test": os.path.join(LOCAL_DATA_DIR, "dataset1-nd/test.fasta"),
        "type": "fasta"
    },
    "dataset2": {
        "train": os.path.join(LOCAL_DATA_DIR, "dataset2/train.fasta"),
        "test": os.path.join(LOCAL_DATA_DIR, "dataset2/test.fasta"),
        "type": "fasta"
    },
    "41": {
        "train": os.path.join(LOCAL_DATA_DIR, "41/train_data.csv"),
        "test": os.path.join(LOCAL_DATA_DIR, "41/test_data.csv"),
        "val": os.path.join(LOCAL_DATA_DIR, "41/val_data.csv"),
        "type": "csv"
    },
    "177": {
        "train": os.path.join(LOCAL_DATA_DIR, "177/train_data.csv"),
        "test": os.path.join(LOCAL_DATA_DIR, "177/test_data.csv"),
        "val": os.path.join(LOCAL_DATA_DIR, "177/val_data.csv"),
        "type": "csv"
    },
    "nrc": {
        "train": os.path.join(LOCAL_DATA_DIR, "nRC-public/dataset_nRC_train.fasta"),
        "test": os.path.join(LOCAL_DATA_DIR, "nRC-public/dataset_nRC_test.fasta"),
        "type": "fasta"
    },
    "demo": {
        "test": os.path.join(LOCAL_DATA_DIR, "demo/demo_transcripts.fasta"),
        "type": "fasta"
    }
}


def set_seed(seed: int = GLOBAL_SEED) -> None:
    """Enforce deterministic behavior across Python, NumPy, and Scikit-Learn."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
