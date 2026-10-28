#!/usr/bin/env python3
"""
Evaluate MV-MGB on Benchmark Corpora with In-Depth Diagnostics.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, matthews_corrcoef

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.pipeline import MVMGBClassifier
from src.data_loader import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate MV-MGB on Benchmark Corpora")
    parser.add_argument("--dataset", type=str, default="dataset1", 
                        choices=["dataset1", "dataset1-nd", "dataset2", "41", "177", "nrc"],
                        help="Benchmark dataset to evaluate.")
    parser.add_argument("--model", type=str, default=os.path.join(REPO_ROOT, "models/mvmgb_dataset1.joblib"),
                        help="Path to serialized trained model.")
    parser.add_argument("--max-test", type=int, default=None,
                        help="Max test sequences to evaluate.")
    parser.add_argument("--tau-review", type=float, default=0.85,
                        help="Chow triage review threshold.")
    parser.add_argument("--tau-reject", type=float, default=0.50,
                        help="Chow triage reject threshold.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Optional CSV output file for predictions.")
    args = parser.parse_args()

    print(f"Loading benchmark test set: {args.dataset}...")
    _, _, seq_te, y_te = load_dataset(args.dataset, max_test=args.max_test)
    print(f"Loaded {len(seq_te)} test transcripts across {len(set(y_te))} classes.")

    print(f"Loading pre-trained model: {args.model}...")
    model = MVMGBClassifier.load(args.model)

    print("Executing multi-view inference...")
    y_pred = model.predict(seq_te)
    probs = model.predict_proba(seq_te)

    acc = accuracy_score(y_te, y_pred)
    mcc = matthews_corrcoef(y_te, y_pred)

    print("\n" + "=" * 65)
    print(f"  Benchmark: {args.dataset.upper()} | Overall Accuracy: {acc*100:.2f}% | MCC: {mcc:.4f}")
    print("=" * 65)
    print("\nDetailed Per-Class Performance:")
    print(classification_report(y_te, y_pred, digits=4))

    # Triage summary
    triage = model.predict_triage(seq_te, tau_review=args.tau_review, tau_reject=args.tau_reject)
    actions = [t.action for t in triage]
    print(f"Bayesian Chow Triage Distribution (tau_review={args.tau_review:.2f}, tau_reject={args.tau_reject:.2f}):")
    for act in ["ACCEPT", "HUMAN_REVIEW", "REJECT_NOVEL"]:
        count = actions.count(act)
        print(f"  {act:<14}: {count:5d} ({count/len(actions)*100:5.2f}%)")

    if args.output:
        df_out = pd.DataFrame({
            "True_Label": y_te,
            "Predicted_Label": y_pred,
            "Confidence": [t.confidence for t in triage],
            "Triage_Action": actions
        })
        df_out.to_csv(args.output, index=False)
        print(f"\n[OK] Benchmark evaluation results saved to: {args.output}")


if __name__ == "__main__":
    main()
