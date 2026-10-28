"""
View 2: Natural Vectors / Topological Moments (21D).
Verified against Deng et al. (2011) moment normalisation.
"""
import numpy as np
from ..canonicalization import canonicalize_rna


class NaturalVectorEncoder:
    """
    Computes 21-dimensional Natural Vector representation:
      For each nucleotide c in {A, C, G, U}:
        - Normalized frequency: n_c / L
        - Normalized mean position: mu_c / L
        - Normalized central moments D_m^(c) for m in {2, 3, 4}:
            D_m^(c) = sum_{j=1}^{n_c} (s_j^(c) - mu_c)^m / (n_c^(m-1) * L^(m-1))
      - Total sequence length: L
      
    Total dimensionality: 4 * 5 + 1 = 21 dimensions.
    """
    def __init__(self, max_moment: int = 4):
        self.max_moment = max_moment
        self.chars = ['A', 'C', 'G', 'U']

    def encode(self, seq: str) -> np.ndarray:
        seq = canonicalize_rna(seq)
        L = len(seq)
        if L == 0:
            return np.zeros(len(self.chars) * (self.max_moment + 1) + 1, dtype=np.float32)

        vec = []
        for c in self.chars:
            pos = np.array([i + 1 for i, ch in enumerate(seq) if ch == c], dtype=np.float64)
            nc = len(pos)
            if nc == 0:
                vec.extend([0.0] * (self.max_moment + 1))
                continue

            # Normalized count and mean position
            vec.append(nc / float(L))
            mu = np.mean(pos)
            vec.append(mu / float(L))

            # Deng et al. (2011) normalized central moments m = 2, 3, ..., max_moment
            for m in range(2, self.max_moment + 1):
                denom = (float(nc) ** (m - 1)) * (float(L) ** (m - 1))
                val = np.sum((pos - mu) ** m) / denom if denom > 0 else 0.0
                vec.append(val)

        # 21st feature: Sequence length
        vec.append(float(L))
        return np.array(vec, dtype=np.float32)

    def encode_dataset(self, sequences: list[str]) -> np.ndarray:
        return np.stack([self.encode(s) for s in sequences])
