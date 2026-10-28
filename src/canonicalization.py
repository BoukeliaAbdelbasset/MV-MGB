"""
Sequence canonicalization and hashing utilities.
"""
import re
import hashlib

_COMPLEMENT = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}


def canonicalize_rna(seq: str) -> str:
    """
    Standardize RNA sequences:
      1. Convert to uppercase
      2. Map Thymine (T) to Uracil (U)
      3. Map any non-canonical character outside {A, C, G, U} to 'N'
    """
    if not isinstance(seq, str):
        seq = str(seq)
    seq = seq.upper().replace('T', 'U')
    seq = re.sub(r'[^ACGU]', 'N', seq)
    return seq


def reverse_complement(seq: str) -> str:
    """Return reverse-complement of an RNA sequence."""
    seq = canonicalize_rna(seq)
    return ''.join(_COMPLEMENT.get(base, 'N') for base in reversed(seq))


def sequence_hash(seq: str, canonicalize: bool = True) -> str:
    """Compute deterministic SHA-256 hash of sequence."""
    if canonicalize:
        seq = canonicalize_rna(seq)
    return hashlib.sha256(seq.encode('utf-8')).hexdigest()


def sequence_canonical_hash_pair(seq: str) -> tuple[str, str]:
    """Compute hashes for both forward and reverse-complement orientations."""
    fwd = canonicalize_rna(seq)
    rev = reverse_complement(fwd)
    return sequence_hash(fwd, canonicalize=False), sequence_hash(rev, canonicalize=False)
