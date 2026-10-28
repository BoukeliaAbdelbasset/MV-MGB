"""
Chow-Type Bayesian 3-Way Diagnostic Triage Rule.
Enables automated acceptance, human expert laboratory review,
and out-of-distribution novel family rejection.
"""
from typing import NamedTuple
import numpy as np


class TriageResult(NamedTuple):
    predicted_class: str
    confidence: float
    action: str  # 'ACCEPT', 'HUMAN_REVIEW', or 'REJECT_NOVEL'
    probabilities: dict[str, float]


def apply_chow_triage(
    probabilities: np.ndarray,
    classes: list[str],
    tau_review: float = 0.85,
    tau_reject: float = 0.50
) -> list[TriageResult]:
    """
    Apply Chow-type Bayesian 3-way triage to multi-class prediction probabilities.

    Parameters
    ----------
    probabilities : np.ndarray of shape (N, K)
        Predicted posterior class probabilities for N sequences over K classes.
    classes : list of str
        List of K class names corresponding to columns of probabilities.
    tau_review : float, default=0.85
        Acceptance threshold: if p_max >= tau_review, accept automated call.
    tau_reject : float, default=0.50
        Rejection threshold: if p_max < tau_reject, flag as novel / out-of-distribution.
        Otherwise (tau_reject <= p_max < tau_review), route to secondary expert review.

    Returns
    -------
    list of TriageResult
    """
    results = []
    classes = np.array(classes)
    
    for row in probabilities:
        top_idx = int(np.argmax(row))
        p_max = float(row[top_idx])
        pred_class = str(classes[top_idx])
        
        if p_max >= tau_review:
            action = "ACCEPT"
        elif p_max < tau_reject:
            action = "REJECT_NOVEL"
        else:
            action = "HUMAN_REVIEW"
            
        prob_dict = {str(c): float(p) for c, p in zip(classes, row)}
        results.append(TriageResult(
            predicted_class=pred_class,
            confidence=p_max,
            action=action,
            probabilities=prob_dict
        ))
        
    return results
