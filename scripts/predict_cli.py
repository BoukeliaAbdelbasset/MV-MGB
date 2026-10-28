#!/usr/bin/env python3
"""
Command-Line Interface (CLI) for MV-MGB Non-Coding RNA Classification.

Usage:
  # Predict on a FASTA file:
  python scripts/predict_cli.py --fasta datasets/demo/demo_transcripts.fasta --output results.csv

  # Predict on a raw single sequence:
  python scripts/predict_cli.py --sequence "GGGCCCGUAGCUCAGUGGUAGAGCGCUCGCUUUGCAUGUGAGAGGUACCGGGAUCGAUACCCGGCGGGUCCA"
"""
import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.pipeline import MVMGBClassifier
from src.data_loader import load_fasta_sequences, load_csv_sequences


def main():
    parser = argparse.ArgumentParser(
        description="MV-MGB: High-Throughput Non-Coding RNA Classification and Triage CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--fasta", "-f", type=str, help="Path to input FASTA file.")
    group.add_argument("--sequence", "-s", type=str, help="Single RNA sequence string.")
    group.add_argument("--csv", "-c", type=str, help="Path to input CSV file containing RNA sequences.")
    group.add_argument("--demo", action="store_true", help="Run inference on bundled demo transcripts.")
    
    parser.add_argument("--model", "-m", type=str, 
                        default=os.path.join(REPO_ROOT, "models/mvmgb_dataset1.joblib"),
                        help="Path to pre-trained MV-MGB model (.joblib).")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Optional CSV output file path for batch results.")
    parser.add_argument("--tau-review", type=float, default=0.85,
                        help="Bayesian triage threshold for automated acceptance.")
    parser.add_argument("--tau-reject", type=float, default=0.50,
                        help="Bayesian triage threshold for novel/OOD rejection.")
    
    args = parser.parse_args()

    print("=" * 75)
    print("  MV-MGB: Multi-View Meta-Gradient Boosting for ncRNA Classification")
    print("=" * 75)
    
    # Default to demo if nothing provided
    if not (args.fasta or args.sequence or args.csv or args.demo):
        print("No input provided. Defaulting to --demo mode.")
        args.demo = True

    # 1. Load model
    if not os.path.exists(args.model):
        print(f"[ERROR] Pre-trained model not found at: {args.model}")
        print("Please train a model first using scripts/train_mvmgb.py or run_training.sh.")
        sys.exit(1)
        
    print(f"Loading model from: {args.model} ...")
    t0 = time.time()
    model = MVMGBClassifier.load(args.model)
    print(f"[OK] Model loaded in {time.time() - t0:.2f}s ({len(model.classes_)} classes registered).")
    
    # 2. Gather sequences
    if args.sequence:
        seq_ids = ["query_seq_1"]
        seqs = [args.sequence.strip().upper().replace("T", "U")]
    elif args.demo:
        demo_path = os.path.join(REPO_ROOT, "datasets/demo/demo_transcripts.fasta")
        print(f"Loading demo transcripts from: {demo_path} ...")
        seq_ids, seqs, _ = load_fasta_sequences(demo_path)
        print(f"[OK] Parsed {len(seqs)} demo transcripts.")
    elif args.csv:
        print(f"Reading CSV file: {args.csv} ...")
        seq_ids, seqs, _ = load_csv_sequences(args.csv)
        print(f"[OK] Parsed {len(seqs)} sequences from CSV.")
    else:
        print(f"Reading FASTA file: {args.fasta} ...")
        seq_ids, seqs, _ = load_fasta_sequences(args.fasta)
        print(f"[OK] Parsed {len(seqs)} sequences.")

    # 3. Perform Inference
    print(f"Running multi-view extraction and meta-gradient boosting on {len(seqs)} transcript(s)...")
    t0 = time.time()
    df_results = model.predict_dataframe(
        sequences=seqs,
        seq_ids=seq_ids,
        tau_review=args.tau_review,
        tau_reject=args.tau_reject
    )
    elapsed = time.time() - t0
    rate = len(seqs) / elapsed if elapsed > 0 else 0
    print(f"[OK] Inference completed in {elapsed:.3f}s ({rate:.1f} seqs/sec).")
    
    # 4. Display or Save Results
    print("\n" + "-" * 75)
    print(f"{'Sequence ID':<22} {'Length':<8} {'Predicted Class':<16} {'Confidence':<12} {'Triage Action':<14}")
    print("-" * 75)
    for _, row in df_results.head(15).iterrows():
        print(f"{str(row['Sequence_ID']):<22} {row['Length']:<8} {str(row['Predicted_Class']):<16} {row['Confidence']:<12.4f} {str(row['Triage_Action']):<14}")
    
    if len(df_results) > 15:
        print(f"... and {len(df_results) - 15} more sequences.")
    print("-" * 75)
    
    # Action summary
    counts = df_results['Triage_Action'].value_counts()
    print("\nTriage Decision Breakdown:")
    for action, count in counts.items():
        pct = (count / len(df_results)) * 100
        print(f"  [{action:<14}] : {count:4d} sequences ({pct:5.1f}%)")
        
    if args.output:
        df_results.to_csv(args.output, index=False)
        print(f"\n[OK] Full predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
