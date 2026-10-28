"""
Unit tests for model loading, prediction, and triage.
"""
import os
import pytest
import numpy as np
from src.pipeline import MVMGBClassifier

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/mvmgb_dataset1.joblib")


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Pre-trained model not found")
def test_model_loading_and_prediction():
    model = MVMGBClassifier.load(MODEL_PATH)
    assert model.is_fitted_
    assert len(model.classes_) == 13

    # tRNA sequence
    trna = "GGGCCCGUAGCUCAGUGGUAGAGCGCUCGCUUUGCAUGUGAGAGGUACCGGGAUCGAUACCCGGCGGGUCCA"
    pred = model.predict([trna])
    assert pred[0] == "tRNA"

    # Probabilities
    probs = model.predict_proba([trna])
    assert probs.shape == (1, 13)
    assert np.isclose(probs.sum(), 1.0, atol=1e-4)

    # Triage
    triage = model.predict_triage([trna])
    assert len(triage) == 1
    assert triage[0].action in ["ACCEPT", "HUMAN_REVIEW", "REJECT_NOVEL"]
    assert triage[0].predicted_class == "tRNA"
