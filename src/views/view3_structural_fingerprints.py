"""
View 3: Biophysical Structural Fingerprints (BSF, 22D).
Domain-specific structural, thermodynamic proxy, and consensus RNA motifs.
"""
import re
import numpy as np
from ..canonicalization import canonicalize_rna


class BSFEncoder:
    """
    Computes 22-dimensional biophysical and structural fingerprint features.
    
    Feature indices (0 to 21):
      0: Shift-tolerant hairpin complementarity score (shifts delta in {-3..+3})
      1-5: GC content across 5 contiguous sequence quintiles
      6: Sequence length L
      7: log(1 + L)
      8: Box C motif (RUGAUGA) in 5' region
      9: Box D motif (CUGA) in 3' region
      10: Box C motif in internal region
      11: Box D' motif (CUGA) in internal region
      12: H/ACA consensus (ANANNA) in internal region
      13: 3' tail ACA indicator (last 10 nt)
      14: 3' tail AGA indicator (last 10 nt)
      15: Global H/ACA consensus count
      16: AU-rich element (ARE: AUUUA) density
      17: Poly-pyrimidine tract (YYYYYYY) density
      18: GNRA tetraloop density
      19: Palindrome fraction (k=4)
      20: Palindrome fraction (k=6)
      21: Global GC content
    """
    _IUPAC = {
        'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U',
        'R': '[AG]', 'Y': '[CU]', 'S': '[GC]', 'W': '[AU]', 'K': '[GU]',
        'M': '[AC]', 'B': '[CGU]', 'D': '[AGU]', 'H': '[ACU]',
        'V': '[ACG]', 'N': '[ACGUN]',
    }
    _COMPLEMENT = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}

    def _compile_motif(self, motif: str) -> re.Pattern:
        pattern = ''.join(self._IUPAC.get(c, c) for c in canonicalize_rna(motif))
        return re.compile(pattern)

    def __init__(self):
        self._pat_box_c = self._compile_motif('RUGAUGA')
        self._pat_box_d = self._compile_motif('CUGA')
        self._pat_haca = self._compile_motif('ANANNA')
        self._pat_are = self._compile_motif('AUUUA')
        self._pat_polypy = self._compile_motif('YYYYYYY')
        self._pat_gnra = self._compile_motif('GNRA')

    def _hairpin_score(self, seq: str) -> float:
        L = len(seq)
        if L < 12:
            return 0.0
        m = L // 2
        arm5 = seq[:m]
        arm3 = seq[L - m:]
        arm3_rc = ''.join(self._COMPLEMENT.get(b, 'N') for b in reversed(arm3))

        best = 0.0
        for delta in range(-3, 4):
            if delta >= 0:
                s5 = arm5[delta:]
                s3 = arm3_rc[:len(s5)]
            else:
                s5 = arm5[:len(arm3_rc) + delta]
                s3 = arm3_rc[-delta:]
            overlap = min(len(s5), len(s3))
            if overlap > 0:
                score = sum(1.0 for a, b in zip(s5[:overlap], s3[:overlap]) if a == b and a != 'N') / float(m)
                if score > best:
                    best = score
        return float(best)

    def _palindrome_fraction(self, seq: str, k: int) -> float:
        L = len(seq)
        n = L - k + 1
        if n <= 0:
            return 0.0
        cnt = 0
        for i in range(n):
            kmer = seq[i:i + k]
            kmer_rc = ''.join(self._COMPLEMENT.get(b, 'N') for b in reversed(kmer))
            if kmer == kmer_rc and 'N' not in kmer:
                cnt += 1
        return float(cnt) / float(n)

    def encode(self, seq: str) -> np.ndarray:
        seq = canonicalize_rna(seq)
        L = len(seq)
        if L == 0:
            return np.zeros(22, dtype=np.float32)

        # 0: Shift-tolerant hairpin
        hp = self._hairpin_score(seq)

        # 1-5: GC content across 5 quintiles
        step = max(1, L // 5)
        gc_quintiles = []
        for q in range(5):
            sub = seq[q * step:(q + 1) * step] if q < 4 else seq[q * step:]
            gc_val = (sub.count('G') + sub.count('C')) / float(max(len(sub), 1))
            gc_quintiles.append(gc_val)

        # 6, 7: Length metrics
        f_len = float(L)
        f_loglen = float(np.log1p(L))

        # Regions for motifs
        w5 = max(15, int(L * 0.25))
        w3_start = max(0, L - w5)
        reg_5p = seq[:w5]
        reg_3p = seq[w3_start:]
        reg_mid = seq[w5:w3_start] if w3_start > w5 else seq

        # 8-11: Box C/D motifs
        box_c_5p = float(len(self._pat_box_c.findall(reg_5p)))
        box_d_3p = float(len(self._pat_box_d.findall(reg_3p)))
        box_c_mid = float(len(self._pat_box_c.findall(reg_mid)))
        box_d_mid = float(len(self._pat_box_d.findall(reg_mid)))

        # 12-15: H/ACA motifs
        haca_mid = float(len(self._pat_haca.findall(reg_mid)))
        tail10 = seq[max(0, L - 10):]
        tail_aca = 1.0 if 'ACA' in tail10 else 0.0
        tail_aga = 1.0 if 'AGA' in tail10 else 0.0
        haca_glob = float(len(self._pat_haca.findall(seq)))

        # 16-18: Densities
        are_dens = float(len(self._pat_are.findall(seq))) / float(L)
        polypy_dens = float(len(self._pat_polypy.findall(seq))) / float(L)
        gnra_dens = float(len(self._pat_gnra.findall(seq))) / float(L)

        # 19-21: Palindromes and Global GC
        pal4 = self._palindrome_fraction(seq, 4)
        pal6 = self._palindrome_fraction(seq, 6)
        gc_glob = (seq.count('G') + seq.count('C')) / float(L)

        feats = (
            [hp] + gc_quintiles + [f_len, f_loglen,
             box_c_5p, box_d_3p, box_c_mid, box_d_mid,
             haca_mid, tail_aca, tail_aga, haca_glob,
             are_dens, polypy_dens, gnra_dens,
             pal4, pal6, gc_glob]
        )
        return np.array(feats, dtype=np.float32)

    def encode_dataset(self, sequences: list[str]) -> np.ndarray:
        return np.stack([self.encode(s) for s in sequences])
