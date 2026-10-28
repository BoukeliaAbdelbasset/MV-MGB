#!/usr/bin/env python3
"""
Train MV-MGB Model on Custom FASTA Datasets.

Usage:
  python scripts/train_mvmgb.py --dataset dataset1 --output models/mvmgb_trained.joblib
  python scripts/train_mvmgb.py --train my_train.fasta --test my_test.fasta --output models/custom.joblib
"""
import argparse
import os
import sys
import time
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.pipeline import MVMGBClassifier
from src.data_loader import load_dataset, load_fasta_sequences, load_csv_sequences


def main():
    parser = argparse.ArgumentParser(
        description="Train MV-MGB on RNA Datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, default="dataset1",
                        choices=["dataset1", "dataset1-nd", "dataset2", "41", "177", "nrc"],
                        help="Benchmark dataset name.")
    parser.add_argument("--train", type=str, default=None, help="Custom train FASTA/CSV path.")
    parser.add_argument("--test", type=str, default=None, help="Custom test FASTA/CSV path.")
    parser.add_argument("--output", "-o", type=str, 
                        default=os.path.join(REPO_ROOT, "models/mvmgb_custom.joblib"),
                        help="Output path for serialized model.")
    parser.add_argument("--folds", type=int, default=5, help="Number of Tier-1 CV folds.")
    parser.add_argument("--max-train", type=int, default=None, help="Max training samples for rapid testing.")
    parser.add_argument("--max-test", type=int, default=None, help="Max test samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed.")
    
    args = parser.parse_args()

    print("=" * 70)
    print("  MV-MGB Model Training Pipeline")
    print("=" * 70)
    
    if args.train and args.test:
        print(f"Loading custom datasets:\n  Train: {args.train}\n  Test:  {args.test}")
        load_fn = load_fasta_sequences if not args.train.endswith(".csv") else load_csv_sequences
        _, seq_tr, y_tr = load_fn(args.train)
        load_fn_te = load_fasta_sequences if not args.test.endswith(".csv") else load_csv_sequences
        _, seq_te, y_te = load_fn_te(args.test)
        if args.max_train and len(seq_tr) > args.max_train:
            seq_tr, y_tr = seq_tr[:args.max_train], y_tr[:args.max_train]
        if args.max_test and len(seq_te) > args.max_test:
            seq_te, y_te = seq_te[:args.max_test], y_te[:args.max_test]
    else:
        print(f"Loading benchmark dataset: {args.dataset}")
        seq_tr, y_tr, seq_te, y_te = load_dataset(
            args.dataset, 
            max_train=args.max_train, 
            max_test=args.max_test, 
            seed=args.seed
        )
        
    print(f"[OK] Loaded {len(seq_tr)} training sequences, {len(seq_te)} test sequences.")
    print(f"Classes ({len(set(y_tr))}): {sorted(set(y_tr))}")

    # Initialize model
    model = MVMGBClassifier(
        n_folds=args.folds,
        test_inference_mode="refit",
        random_state=args.seed
    )

    print("\nStarting multi-view feature extraction and Tier-1/Tier-2 training...")
    t0 = time.time()
    model.fit(seq_tr, y_tr)
    fit_time = time.time() - t0
    print(f"[OK] Training completed in {fit_time:.2f}s.")

    # Save model
    model.save(args.output)

    # Evaluate
    print("\nEvaluating model on test split...")
    t0 = time.time()
    pred_te = model.predict(seq_te)
    eval_time = time.time() - t0

    acc = accuracy_score(y_te, pred_te)
    f1_w = f1_score(y_te, pred_te, average="weighted", zero_division=0)
    f1_m = f1_score(y_te, pred_te, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_te, pred_te)

    print("\n" + "=" * 50)
    print("           Test Benchmark Results")
    print("=" * 50)
    print(f"  Accuracy:      {acc*100:6.2f}%")
    print(f"  Weighted F1:   {f1_w:6.4f}")
    print(f"  Macro F1:      {f1_m:6.4f}")
    print(f"  Matthews CC:   {mcc:6.4f}")
    print(f"  Test Throughput: {len(seq_te)/eval_time:.1f} seqs/s ({eval_time/len(seq_te)*1000:.2f} ms/seq)")
    print("=" * 50)


if __name__ == "__main__":
    main()
