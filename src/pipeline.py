"""
High-Level End-to-End Pipeline API for MV-MGB.
Enables one-line training, serialization, inference, and diagnostic triage
on raw ribonucleotide sequences.
"""
from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from typing import Sequence, Union

from .canonicalization import canonicalize_rna
from .views.view2_natural_vectors import NaturalVectorEncoder
from .views.view3_structural_fingerprints import BSFEncoder
from .views.view4_stacking_profiles import StackingPropensityEncoder
from .models.stacking import MultiViewStacking
from .models.decision_rule import apply_chow_triage, TriageResult


class MVMGBClassifier:
    """
    Multi-View Meta-Gradient Boosting (MV-MGB) Classifier for Non-Coding RNA.
    
    Seamlessly encapsulates:
      - Primary transcript biophysical canonicalisation
      - View 1: Sublinear-scaled oligonucleotide TF-IDF spectra (35,000D)
      - View 2: Topological natural vector moments (21D)
      - View 3: Biophysical structural fingerprints & loop heuristics (22D)
      - View 4: Continuous Turner stacking thermodynamics B-splines (60D)
      - Tier-1 Base Estimators & Stratified Cross-Validation Probability Stacking
      - Tier-2 Histogram Gradient Boosted Decision Tree Meta-Learner
      - Chow-Type Bayesian 3-Way Diagnostic Triage Rule
    """

    def __init__(
        self,
        n_folds: int = 5,
        test_inference_mode: str = "refit",
        random_state: int = 42,
        append_raw_features: bool = False
    ):
        self.n_folds = n_folds
        self.test_inference_mode = test_inference_mode
        self.random_state = random_state
        self.append_raw_features = append_raw_features
        
        # Encoders
        self.nv_encoder = NaturalVectorEncoder()
        self.bsf_encoder = BSFEncoder()
        self.bsp_encoder = StackingPropensityEncoder()
        
        # Stacking model
        self.stacker = MultiViewStacking(
            n_folds=self.n_folds,
            test_inference_mode=self.test_inference_mode,
            append_raw_features=self.append_raw_features,
            random_state=self.random_state
        )
        self.classes_ = None
        self.is_fitted_ = False

    def extract_features(self, sequences: Sequence[str]) -> dict[str, Union[list[str], np.ndarray]]:
        """
        Canonicalize raw RNA sequences and extract parallel multi-view feature representations.
        
        Parameters
        ----------
        sequences : list or sequence of str
            Raw RNA/DNA sequence strings.
            
        Returns
        -------
        dict with keys: 'view1_tfidf', 'view2_nv', 'view3_bsf', 'view4_bsp'
        """
        clean_seqs = [canonicalize_rna(s) for s in sequences]
        
        X_nv = self.nv_encoder.encode_dataset(clean_seqs)
        X_bsf = self.bsf_encoder.encode_dataset(clean_seqs)
        X_bsp = self.bsp_encoder.encode_dataset(clean_seqs)
        
        return {
            'view1_tfidf': clean_seqs,
            'view2_nv': X_nv,
            'view3_bsf': X_bsf,
            'view4_bsp': X_bsp
        }

    def fit(self, sequences: Sequence[str], labels: Sequence[str]) -> MVMGBClassifier:
        """
        Fit the multi-view stacking pipeline on training sequences and labels.
        
        Parameters
        ----------
        sequences : list of str
            Primary RNA sequences.
        labels : list of str
            Ground-truth functional class labels.
        """
        views = self.extract_features(sequences)
        self.stacker.fit(views, labels)
        self.classes_ = np.array(self.stacker.classes_)
        self.is_fitted_ = True
        return self

    def predict(self, sequences: Sequence[str]) -> np.ndarray:
        """
        Predict functional class labels for query RNA sequences.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted yet. Call fit() or load().")
        views = self.extract_features(sequences)
        return self.stacker.predict(views)

    def predict_proba(self, sequences: Sequence[str]) -> np.ndarray:
        """
        Predict posterior class probabilities on the probability simplex.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted yet. Call fit() or load().")
        views = self.extract_features(sequences)
        return self.stacker.predict_proba(views)

    def predict_triage(
        self,
        sequences: Sequence[str],
        tau_review: float = 0.85,
        tau_reject: float = 0.50
    ) -> list[TriageResult]:
        """
        Perform Chow-type 3-way diagnostic triage:
          - ACCEPT: High confidence automated call (p_max >= tau_review)
          - HUMAN_REVIEW: Ambiguous call routed to expert review (tau_reject <= p_max < tau_review)
          - REJECT_NOVEL: Putative novel/uncharacterized family or OOD sequence (p_max < tau_reject)
        """
        probs = self.predict_proba(sequences)
        return apply_chow_triage(
            probs,
            self.classes_,
            tau_review=tau_review,
            tau_reject=tau_reject
        )

    def predict_dataframe(
        self,
        sequences: Sequence[str],
        seq_ids: Sequence[str] = None,
        tau_review: float = 0.85,
        tau_reject: float = 0.50
    ) -> pd.DataFrame:
        """
        Generate a comprehensive pandas DataFrame containing sequence IDs, predicted class,
        confidence, triage action, and per-class probability columns.
        """
        if seq_ids is None:
            seq_ids = [f"seq_{i+1:04d}" for i in range(len(sequences))]
            
        probs = self.predict_proba(sequences)
        triage = apply_chow_triage(probs, self.classes_, tau_review, tau_reject)
        
        rows = []
        for sid, seq, res in zip(seq_ids, sequences, triage):
            row = {
                "Sequence_ID": sid,
                "Length": len(seq),
                "Predicted_Class": res.predicted_class,
                "Confidence": res.confidence,
                "Triage_Action": res.action
            }
            row.update(res.probabilities)
            rows.append(row)
            
        return pd.DataFrame(rows)

    def save(self, filepath: str) -> None:
        """Serialize the complete fitted model to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath, compress=3)
        print(f"[OK] Saved MV-MGB pipeline to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> MVMGBClassifier:
        """Load a serialized MV-MGB pipeline from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        model = joblib.load(filepath)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is of type {type(model)}, expected {cls}.")
        return model
