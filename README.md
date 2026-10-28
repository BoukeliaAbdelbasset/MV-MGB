<div align="center">

# MV-MGB: Multi-View Meta-Gradient Boosting for Non-Coding RNA Classification

[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![ESWA Journal](https://img.shields.io/badge/Target-Expert%20Systems%20with%20Applications-orange.svg)](https://www.sciencedirect.com/journal/expert-systems-with-applications)
[![Tests Passing](https://img.shields.io/badge/tests-6%20passed-brightgreen.svg)]()
[![Hardware: CPU-Native](https://img.shields.io/badge/hardware-100%25%20CPU--Native%20(No%20GPU)-blueviolet.svg)]()
[![Accuracy](https://img.shields.io/badge/Dataset%201%20Accuracy-95.65%25-success.svg)]()
[![GitHub Repo](https://img.shields.io/badge/GitHub-BoukeliaAbdelbasset%2FMV--MGB-181717?logo=github)](https://github.com/BoukeliaAbdelbasset/MV-MGB)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.23724491-blue.svg)](https://zenodo.org/records/23724491)

**An Explainable Multi-View Expert Decision Support System for Alignment-Free Non-Coding RNA Classification Using Biophysical and Geometric Feature Spaces**

*Rabah Lebsir<sup>1</sup> and Abdelbasset Boukelia<sup>2,*</sup>*  
<sup>1</sup>Department of Computer Science, Université 8 Mai 1945 Guelma, Algeria  
<sup>2</sup>Centre de Recherche en Biotechnologie (CRBt), Constantine, Algeria  
<sup>*</sup>Corresponding authors: `rabah.lebsir@univ-guelma.dz` | `a.boukelia@crbt.dz`

</div>

---

## 🌟 Highlights
- **State-of-the-Art Accuracy:** Achieves **95.65% accuracy** (95% BCa CI [94.81%, 96.45%]), **0.9564 weighted F1**, and **0.9528 MCC** on the canonical 13-class benchmark, outperforming published deep foundation models (RiNALMo, ncrna-deep) and alignment algorithms.
- **Strict Linear Complexity $\mathcal{O}(L)$:** Amortized batch inference throughput of **1,162 sequences/second** (0.86 ms/seq) on standard consumer CPU hardware—no expensive GPUs required.
- **Biophysically Grounded:** View 4 nearest-neighbour Turner stacking free energies correlate with full ViennaRNA partition function base-pairing probabilities ($r = +0.3089$, $p < 10^{-300}$).
- **Two-Tier Out-of-Fold Simplex Stacking:** Specialized Tier-1 base estimators project each distinct feature space into out-of-fold probability simplices ($\Delta^{12}$), preventing sparse sequence $k$-mers from dominating compact biophysical vectors and eliminating data leakage.
- **Chow-Type Bayesian Diagnostic Triage:** Formulates an optimal 3-way triage rule with rejection option: automated acceptance ($p_{\max} \ge 0.85$), secondary laboratory review ($0.50 \le p_{\max} < 0.85$), and novel transcript / decoy rejection ($p_{\max} < 0.50$).

---

## 🏗️ System Architecture

```
                    Primary Ribonucleotide Transcript S ∈ {A, C, G, U}^L
                                           │
                                  [Canonicalisation] (T -> U, Degenerate Imputation, Hash Isolation)
                                           │
           ┌───────────────────────┬───────┴───────────────────────┬──────────────────────┐
           ▼                       ▼                               ▼                      ▼
    [ View 1: Motif ]      [ View 2: Topology ]          [ View 3: Structure ]   [ View 4: Stacking ]
  Sublinear TF-IDF 3-8    Natural Vector Moments        Biophysical Structural    Continuous Turner
   k-mers (35,000D)          (21 Dimensions)             Fingerprints (22D)      B-Splines (60D)
           │                       │                               │                      │
           ▼                       ▼                               ▼                      ▼
   [ Tier-1 Estimator ]    [ Tier-1 Estimator ]          [ Tier-1 Estimator ]    [ Tier-1 Estimator ]
   L2-Logistic Regress.    HistGradientBoosting              Random Forest          MLP (128x64)
           │                       │                               │                      │
           └───────────────────────┴───────┬───────────────────────┴──────────────────────┘
                                           ▼
                 Concatenated Out-of-Fold Probability Simplex Z ∈ Δ^51
                                           │
                                           ▼
                 [ Tier-2 Meta-Learner: Histogram Gradient Boosted Trees ]
                                           │
                                           ▼
                     Calibrated Posterior Class Probabilities P ∈ Δ^12
                                           │
                                           ▼
                 [ Chow-Type Bayesian 3-Way Diagnostic Decision Triage ]
                       /                   |                   \
            p_max >= 0.85       0.50 <= p_max < 0.85        p_max < 0.50
                 ▼                         ▼                      ▼
            [ ACCEPT ]              [ HUMAN_REVIEW ]       [ REJECT_NOVEL ]
       Automated Laboratory       Routed to Secondary        Novel Family or
         Decision Output              Expert Review           Genomic Decoy
```

---

## 📁 Repository Structure

```
MVBDSource/
├── README.md                          # Repository overview and documentation
├── TUTORIAL.md                        # Comprehensive user guide and tutorials
├── tutorial_mvmgb_inference.ipynb     # Interactive Jupyter Notebook Tutorial (pre-rendered)
├── LICENSE                            # Apache 2.0 Open Source License
├── requirements.txt                   # Pinned Python dependencies
├── environment.yml                    # Conda environment definition
├── pyproject.toml                     # Modern package metadata
├── setup.py                           # Setup script for pip install -e .
├── Makefile                           # Automated build and test commands
├── run_training.sh                    # Bash script: Step 1 (Model Training)
├── run_predict.sh                     # Bash script: Step 2 (Inference & Bayesian Triage)
├── run_evaluate.sh                    # Bash script: Step 3 (Benchmark Evaluation)
├── run_all.sh                         # Bash script: Master End-to-End Pipeline
├── models/                            # Pre-trained models and metadata
│   ├── mvmgb_dataset1.joblib          # Serialized production pipeline (126 MB)
│   └── model_metadata.json            # Model parameters and registered metrics
├── datasets/                          # Benchmark and demonstration datasets
│   ├── dataset1/                      # Creux 13-class canonical benchmark (FASTA, dot, pkl)
│   ├── dataset1-nd/                   # No-degenerate sequence benchmark split
│   ├── dataset2/                      # Extended 13-class benchmark (31k train / 13k test)
│   ├── 41/                            # Rfam 41-family benchmark (166k train / 16k test CSV)
│   ├── 177/                           # Rfam 177 open-set screening (219k train / 29k test CSV)
│   ├── nRC-public/                    # Comparative nRC benchmark corpus
│   ├── demo/                          # 14 representative demo transcripts
│   └── README.md                      # Comprehensive dataset specifications
├── src/                               # Core Python library
│   ├── __init__.py
│   ├── pipeline.py                    # High-level MVMGBClassifier API
│   ├── canonicalization.py            # Sequence standardization and MD5 hashing
│   ├── data_loader.py                 # Fast FASTA/CSV parsing utilities
│   ├── config.py                      # Deterministic seeds and path discovery
│   ├── views/                         # Four feature space encoders
│   │   ├── view1_tfidf.py             # View 1: Sublinear TF-IDF k-mers
│   │   ├── view2_natural_vectors.py   # View 2: 21D Natural Vectors
│   │   ├── view3_structural_fingerprints.py # View 3: 22D Biophysical Fingerprints
│   │   └── view4_stacking_profiles.py # View 4: 60D Turner Stacking B-Splines
│   └── models/
│       ├── stacking.py                # MultiViewStacking engine (Tier 1 + Tier 2)
│       └── decision_rule.py           # Chow Bayesian 3-way triage logic
├── scripts/
│   ├── predict_cli.py                 # CLI inference tool for FASTA/CSV screening
│   ├── train_mvmgb.py                 # Model training from custom FASTA/CSV
│   ├── evaluate_benchmark.py          # Benchmark evaluation and diagnostics
│   ├── run_training.sh                # Executable training runner
│   ├── run_predict.sh                 # Executable prediction runner
│   ├── run_evaluate.sh                # Executable evaluation runner
│   └── run_all.sh                     # Executable master pipeline runner
└── tests/                             # Unit tests and leak-free verification
    ├── test_views.py                  # Encoder shape and validity tests
    ├── test_inference.py              # Model loading and prediction tests
    └── test_leakage.py                # MD5 hash disjointness test
```

---

## ⚡ Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/BoukeliaAbdelbasset/MV-MGB.git
cd MV-MGB

# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Predict on a Raw Sequence in Python (3 Lines)
```python
from src.pipeline import MVMGBClassifier

model = MVMGBClassifier.load("models/mvmgb_dataset1.joblib")
seq = "GGGCCCGUAGCUCAGUGGUAGAGCGCUCGCUUUGCAUGUGAGAGGUACCGGGAUCGAUACCCGGCGGGUCCA"

pred_class = model.predict([seq])[0]
print(f"Predicted Family: {pred_class}")
# Output: Predicted Family: tRNA
```

### 3. Screen FASTA Files via CLI or Bash Script
```bash
# Using the dedicated bash script:
./run_predict.sh --demo

# Or screening any custom FASTA / CSV:
./run_predict.sh --fasta datasets/demo/demo_transcripts.fasta --output predictions.csv

# Or directly in Python:
python scripts/predict_cli.py --fasta datasets/demo/demo_transcripts.fasta --output predictions.csv
```

---

## 🚀 One-Command Pipeline Bash Scripts

MV-MGB includes ready-to-run bash scripts for each stage of the research pipeline:

```bash
# 1. Master All-in-One Pipeline (End-to-End Execution: tests -> train -> eval -> predict)
./run_all.sh --quick               # Fast end-to-end verification (< 30s)
./run_all.sh                       # Standard end-to-end pipeline run

# 2. Training on Benchmark or Custom Data
./run_training.sh --dataset dataset1                 # Train on Creux Dataset 1
./run_training.sh --dataset dataset2                 # Train on Dataset 2
./run_training.sh --dataset 41 --max-train 5000      # Train on Rfam 41
./run_training.sh --train custom_tr.fasta --test custom_te.fasta -o models/custom.joblib

# 3. High-Throughput Inference & Bayesian Chow Triage
./run_predict.sh --demo                              # Instant demo inference
./run_predict.sh --sequence "GGGCCCGUAGCUCAGUGGU..." # Predict single sequence
./run_predict.sh --fasta my_sequences.fasta -o out.csv # Batch FASTA screening
./run_predict.sh --csv my_sequences.csv -o out.csv   # Batch CSV screening

# 4. Held-Out Benchmark Evaluation & Diagnostics
./run_evaluate.sh --dataset dataset1                 # Evaluate on Dataset 1
./run_evaluate.sh --dataset dataset2                 # Evaluate on Dataset 2
./run_evaluate.sh --dataset 41 --max-test 1000       # Evaluate on Rfam 41
```

---

## 📓 Interactive Jupyter Notebook Tutorial

Launch the pre-rendered interactive tutorial:
```bash
jupyter notebook tutorial_mvmgb_inference.ipynb
```
The notebook guides you through:
1. Loading the pre-trained pipeline in memory.
2. Extracting and visualizing feature vectors from all four biophysical views.
3. Multi-class probability simplex inspection with Matplotlib bar plots.
4. Testing the Chow-type Bayesian triage rule on canonical vs. decoy sequences.
5. High-throughput batch FASTA processing.

---

## 🧪 Verification & Tests

Run the test suite to verify code integrity and zero train-test sequence overlap:
```bash
PYTHONPATH=. pytest tests/ -v
```
All 6 unit tests pass out-of-the-box in under 4 seconds.

---

## 📊 Benchmark Summary

| Functional Class | Full Description | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: |
| `5S_rRNA` | 5S Ribosomal RNA | 0.9855 | 0.9784 | **0.9819** |
| `5_8S_rRNA` | 5.8S Ribosomal RNA | 0.9817 | 0.9907 | **0.9862** |
| `CD-box` | C/D-box small nucleolar RNA | 0.9419 | 0.9257 | **0.9337** |
| `HACA-box` | H/ACA-box small nucleolar RNA | 0.9038 | 0.8868 | **0.8952** |
| `IRES` | Internal Ribosome Entry Site | 0.9630 | 0.9571 | **0.9600** |
| `Intron_gpI` | Group I Catalytic Intron | 0.9737 | 0.9867 | **0.9801** |
| `Intron_gpII` | Group II Catalytic Intron | 0.9583 | 0.9787 | **0.9684** |
| `leader` | Ribosomal RNA Leader | 0.9259 | 0.9346 | **0.9302** |
| `miRNA` | MicroRNA Precursor | 0.9461 | 0.9507 | **0.9484** |
| `riboswitch` | Regulatory Riboswitch | 0.9733 | 0.9865 | **0.9799** |
| `ribozyme` | Catalytic Ribozyme | 0.9608 | 0.9608 | **0.9608** |
| `scaRNA` | Small Cajal Body-Specific RNA | 0.9286 | 0.9155 | **0.9220** |
| `tRNA` | Transfer RNA | 0.9944 | 0.9944 | **0.9944** |
| **Overall** | **13 Families (N = 2,253)** | **0.9567** | **0.9565** | **0.9564** |

---

## 📜 Citation

If you use MV-MGB in your research, please cite:

```bibtex
@article{lebsir2026mvmgb,
  title={An Explainable Multi-View Expert Decision Support System for Alignment-Free Non-Coding RNA Classification Using Biophysical and Geometric Feature Spaces},
  author={Lebsir, Rabah and Boukelia, Abdelbasset}
  
}
```

---

## 📄 License
This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.
# MV-MGB
