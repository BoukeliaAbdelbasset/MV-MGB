"""
Test zero sequence contamination between train and test partitions.
"""
import hashlib
from src.data_loader import load_dataset


def test_zero_dataset1_train_test_overlap():
    seq_tr, _, seq_te, _ = load_dataset("dataset1")
    hashes_tr = {hashlib.md5(s.encode("utf-8")).hexdigest() for s in seq_tr}
    hashes_te = {hashlib.md5(s.encode("utf-8")).hexdigest() for s in seq_te}
    overlap = hashes_tr.intersection(hashes_te)
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping sequences between train and test!"
