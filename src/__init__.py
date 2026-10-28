"""
MV-MGB: Multi-View Meta-Gradient Boosting for Non-Coding RNA Classification.
"""
from .canonicalization import canonicalize_rna, reverse_complement
from .data_loader import load_dataset, load_fasta_sequences
from .pipeline import MVMGBClassifier
from .models.decision_rule import apply_chow_triage, TriageResult

__version__ = "1.0.0"
__author__ = "Rabah Lebsir and Abdelbasset Boukelia"
__all__ = [
    "MVMGBClassifier",
    "canonicalize_rna",
    "reverse_complement",
    "load_dataset",
    "load_fasta_sequences",
    "apply_chow_triage",
    "TriageResult"
]
