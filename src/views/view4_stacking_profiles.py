"""
View 4: Nearest-Neighbour Stacking-Propensity Profiles (60D).
Five orthogonal channels projected onto a 12-basis cubic B-spline system.
"""
import numpy as np
from scipy.interpolate import BSpline
from ..canonicalization import canonicalize_rna


def build_bspline_basis(n_points: int, n_basis: int = 12, degree: int = 3) -> np.ndarray:
    """
    Construct uniform cubic B-spline basis matrix B in R^(n_points x n_basis).
    Knots are clamped at [0, 1] with (n_basis - degree - 1) interior knots.
    """
    t = np.linspace(0.0, 1.0, max(n_points, 4))
    n_interior = n_basis - degree - 1
    interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    knots = np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])
    
    B = np.zeros((len(t), n_basis), dtype=np.float64)
    for k in range(n_basis):
        c = np.zeros(n_basis)
        c[k] = 1.0
        spl = BSpline(knots, c, degree, extrapolate=False)
        B[:, k] = np.nan_to_num(spl(t))
    return B[:n_points]


class StackingPropensityEncoder:
    """
    Projects raw RNA transcripts into a 60-dimensional functional feature space
    via 5 orthogonal biophysical channels and a 12-basis cubic B-spline:
      Ch1: GC indicator (0.0 / 1.0)
      Ch2: Purine-Pyrimidine Z-curve coordinate (+1.0 for A,G; -1.0 for C,U)
      Ch3: Amino-Keto Z-curve coordinate (+1.0 for A,C; -1.0 for G,U)
      Ch4: Turner nearest-neighbour dinucleotide stacking free energy (Delta G°37, kcal/mol)
      Ch5: Shift-tolerant antiparallel base-pairing propensity Psi_pair (delta in {-2..+2})
    """
    # SantaLucia (2004) / Xia et al. (1998) Turner RNA nearest-neighbour Delta G°37 (kcal/mol)
    _TURNER_DELTA_G = {
        'AA': -0.9, 'AC': -2.2, 'AG': -2.1, 'AU': -1.1,
        'CA': -2.1, 'CC': -3.3, 'CG': -2.4, 'CU': -2.1,
        'GA': -2.4, 'GC': -3.4, 'GG': -3.3, 'GU': -2.2,
        'UA': -1.3, 'UC': -2.4, 'UG': -2.1, 'UU': -0.9,
    }
    _COMPATIBLE_PAIRS = {
        ('A', 'U'), ('U', 'A'),
        ('G', 'C'), ('C', 'G'),
        ('G', 'U'), ('U', 'G')  # Wobble pairing
    }

    def __init__(self, n_basis: int = 12, degree: int = 3, ridge_lambda: float = 1e-6):
        self.n_basis = n_basis
        self.degree = degree
        self.ridge_lambda = ridge_lambda

    def _build_signal_matrix(self, seq: str) -> np.ndarray:
        L = len(seq)
        ch1 = np.array([1.0 if c in 'GC' else 0.0 for c in seq], dtype=np.float64)
        ch2 = np.array([1.0 if c in 'AG' else (-1.0 if c in 'CU' else 0.0) for c in seq], dtype=np.float64)
        ch3 = np.array([1.0 if c in 'AC' else (-1.0 if c in 'GU' else 0.0) for c in seq], dtype=np.float64)

        # Ch4: Turner Stacking Delta G
        ch4 = np.zeros(L, dtype=np.float64)
        for i in range(L - 1):
            ch4[i] = self._TURNER_DELTA_G.get(seq[i:i + 2], 0.0)
        if L >= 2:
            ch4[L - 1] = ch4[L - 2]

        # Ch5: Shift-tolerant antiparallel pairing propensity Psi_pair
        ch5 = np.zeros(L, dtype=np.float64)
        for i in range(L):
            target_j = L - 1 - i
            if target_j > i:
                match_cnt = 0.0
                for delta in range(-2, 3):
                    jd = target_j + delta
                    if 0 <= jd < L and jd != i:
                        if (seq[i], seq[jd]) in self._COMPATIBLE_PAIRS:
                            match_cnt += 1.0
                ch5[i] = match_cnt / 5.0
        return np.stack([ch1, ch2, ch3, ch4, ch5], axis=1)

    def encode(self, seq: str) -> np.ndarray:
        seq = canonicalize_rna(seq)
        L = len(seq)
        if L == 0:
            return np.zeros(self.n_basis * 5, dtype=np.float32)

        S = self._build_signal_matrix(seq)
        # Pad short sequences if length is less than 4 for cubic spline
        if S.shape[0] < 4:
            S = np.pad(S, ((0, 4 - S.shape[0]), (0, 0)), mode='edge')

        B = build_bspline_basis(S.shape[0], n_basis=self.n_basis, degree=self.degree)
        reg = self.ridge_lambda * np.eye(self.n_basis)
        coef = np.linalg.solve(B.T @ B + reg, B.T @ S)
        return coef.flatten(order='F').astype(np.float32)

    def encode_dataset(self, sequences: list[str]) -> np.ndarray:
        return np.stack([self.encode(s) for s in sequences])
