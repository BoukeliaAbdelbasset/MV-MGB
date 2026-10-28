"""
Unit tests for the four complementary biophysical views.
"""
import pytest
import numpy as np
from src.canonicalization import canonicalize_rna
from src.views.view2_natural_vectors import NaturalVectorEncoder
from src.views.view3_structural_fingerprints import BSFEncoder
from src.views.view4_stacking_profiles import StackingPropensityEncoder


SAMPLE_SEQS = [
    "GGGCCCGUAGCUCAGUGGUAGAGCGCUCGCUUUGCAUGUGAGAGGUACCGGGAUCGAUACCCGGCGGGUCCA",
    "UGAGGUAGUAGGUUGUAUAGUU",
    "ACGUACGUACGUACGUACGUACGUACGU"
]


def test_canonicalization():
    seq = "atgcn nragcu"
    clean = canonicalize_rna(seq)
    assert "T" not in clean
    assert " " not in clean
    assert all(c in "ACGUN" for c in clean)


def test_view2_natural_vectors():
    enc = NaturalVectorEncoder()
    X = enc.encode_dataset(SAMPLE_SEQS)
    assert X.shape == (3, 21)
    assert not np.isnan(X).any()


def test_view3_biophysical_structural_fingerprints():
    enc = BSFEncoder()
    X = enc.encode_dataset(SAMPLE_SEQS)
    assert X.shape == (3, 22)
    assert not np.isnan(X).any()


def test_view4_stacking_profiles():
    enc = StackingPropensityEncoder()
    X = enc.encode_dataset(SAMPLE_SEQS)
    assert X.shape == (3, 60)
    assert not np.isnan(X).any()
