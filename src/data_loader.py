"""
Dataset loader for FASTA and CSV RNA sequences.
"""
import os
import pandas as pd
from Bio import SeqIO
from .canonicalization import canonicalize_rna
from .config import DATASET_PATHS


def load_fasta_sequences(path: str) -> tuple[list[str], list[str], list[str]]:
    """
    Parse a FASTA file and return (ids, canonicalized_sequences, labels).
    Assumes standard format where label is the last space-delimited token or tag in the header.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"FASTA file not found: {path}")
        
    records = list(SeqIO.parse(path, "fasta"))
    ids = [r.id for r in records]
    seqs = [canonicalize_rna(str(r.seq)) for r in records]
    
    # Try extracting class label from header description
    labels = []
    for r in records:
        parts = r.description.strip().split()
        if len(parts) > 1 and not parts[-1].startswith(">"):
            labels.append(parts[-1].replace("[Class:", "").replace("]", "").strip())
        elif len(parts) == 1 and not parts[0].startswith(">"):
            labels.append(parts[0].replace("[Class:", "").replace("]", "").strip())
        else:
            labels.append("unknown")
    return ids, seqs, labels


def load_csv_sequences(path: str) -> tuple[list[str], list[str], list[str]]:
    """
    Parse a CSV file and return (ids, canonicalized_sequences, labels).
    Detects sequence and class columns automatically.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
        
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    seq_col = cols.get("sequence", cols.get("seq", "Sequence"))
    class_col = cols.get("class", cols.get("label", cols.get("family", "Class")))
    
    ids = [f"seq_{i+1}" for i in range(len(df))]
    seqs = [canonicalize_rna(str(s)) for s in df[seq_col]]
    labels = [str(c).strip() for c in df[class_col]]
    return ids, seqs, labels


def load_dataset(name: str = "dataset1", max_train: int = None, max_test: int = None, seed: int = 42):
    """
    Load train and test splits for a benchmark dataset.
    Supports both FASTA (dataset1, dataset1-nd, dataset2, nrc, demo) and CSV (41, 177).
    """
    if name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_PATHS.keys())}")
        
    cfg = DATASET_PATHS[name]
    dtype = cfg.get("type", "fasta")
    loader_fn = load_csv_sequences if dtype == "csv" else load_fasta_sequences
    
    tr_path = cfg.get("train")
    te_path = cfg.get("test")
    
    seq_tr, y_tr = [], []
    if tr_path and os.path.exists(tr_path):
        _, seq_tr, y_tr = loader_fn(tr_path)
        if max_train is not None and len(seq_tr) > max_train:
            import numpy as np
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(seq_tr), size=max_train, replace=False)
            seq_tr = [seq_tr[i] for i in idx]
            y_tr = [y_tr[i] for i in idx]
            
    seq_te, y_te = [], []
    if te_path and os.path.exists(te_path):
        _, seq_te, y_te = loader_fn(te_path)
        if max_test is not None and len(seq_te) > max_test:
            import numpy as np
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(seq_te), size=max_test, replace=False)
            seq_te = [seq_te[i] for i in idx]
            y_te = [y_te[i] for i in idx]
            
    return seq_tr, y_tr, seq_te, y_te
