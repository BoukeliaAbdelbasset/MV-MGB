# MV-MGB User Guide & Tutorial
*An Explainable Multi-View Expert Decision Support System for Alignment-Free Non-Coding RNA Classification*

**Authors:** Rabah Lebsir (`rabah.lebsir@univ-guelma.dz`) and Abdelbasset Boukelia (`a.boukelia@crbt.dz`)  
**Affiliations:** Université 8 Mai 1945 Guelma & Centre de Recherche en Biotechnologie (CRBt), Constantine, Algeria  
**Repository:** [https://github.com/BoukeliaAbdelbasset/MV-MGB](https://github.com/BoukeliaAbdelbasset/MV-MGB)  
**Zenodo Record:** [https://zenodo.org/records/23724491](https://zenodo.org/records/23724491)  
**Target Journal:** *Expert Systems with Applications* (ESWA), Elsevier Science  

---

## Table of Contents
1. [Overview](#1-overview)
2. [Installation & Setup](#2-installation--setup)
3. [Quick Start (30-Second Example)](#3-quick-start-30-second-example)
4. [Using the Pre-Trained Model](#4-using-the-pre-trained-model)
5. [The Four Biophysical Feature Spaces](#5-the-four-biophysical-feature-spaces)
6. [Chow-Type Bayesian Diagnostic Triage](#6-chow-type-bayesian-diagnostic-triage)
7. [Command-Line Interface (CLI)](#7-command-line-interface-cli)
8. [Training on Custom FASTA Corpora](#8-training-on-custom-fasta-corpora)
9. [Interactive Jupyter Notebook Tutorial](#9-interactive-jupyter-notebook-tutorial)
10. [Reproducing Benchmark Metrics](#10-reproducing-benchmark-metrics)

---

## 1. Overview
Non-coding RNA (ncRNA) functional annotation is foundational for precision medicine, molecular biology, and transcriptomics. Existing algorithms frequently rely on slow structural alignment or multi-GPU transformer foundation models that create computational bottlenecks and uncalibrated predictions.

**MV-MGB** (Multi-View Meta-Gradient Boosting) is a lightweight, CPU-native expert decision support system that operates in strictly linear $\mathcal{O}(L)$ time. It maps primary sequences across four complementary biophysical and geometric spaces:
- **View 1:** Sublinear TF-IDF Oligonucleotide Spectra (35,000D)
- **View 2:** Topological Natural Vector Moments (21D)
- **View 3:** Biophysical Structural Fingerprints (22D)
- **View 4:** Continuous Turner Thermodynamic Stacking B-Splines (60D)

Tier-1 base estimators project these spaces into out-of-fold probability simplices ($\Delta^{12}$), which a Tier-2 Histogram Gradient Boosted Decision Tree (HistGBDT) integrates without data leakage, reaching **95.65% accuracy** on canonical benchmarks.

---

## 2. Installation & Setup

### Prerequisites
- Python $\ge$ 3.9 (tested up to Python 3.13)
- Linux / macOS / Windows WSL2

### Standard Installation
```bash
# Clone or navigate to the standalone directory
cd MVBDSource

# Create a clean virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install package in editable mode
pip install -e .
```

### Conda Environment (Alternative)
```bash
conda env create -f environment.yml
conda activate mvmgb
```

---

## 3. Quick Start (30-Second Example)

Classify an RNA sequence in three lines of Python:

```python
from src.pipeline import MVMGBClassifier

# 1. Load the pre-trained model
model = MVMGBClassifier.load("models/mvmgb_dataset1.joblib")

# 2. Query sequence (e.g., Transfer RNA)
rna_seq = "GGGCCCGUAGCUCAGUGGUAGAGCGCUCGCUUUGCAUGUGAGAGGUACCGGGAUCGAUACCCGGCGGGUCCA"

# 3. Predict functional class & confidence
pred_class = model.predict([rna_seq])[0]
confidence = model.predict_proba([rna_seq]).max()

print(f"Predicted Family: {pred_class} (Confidence: {confidence*100:.2f}%)")
# Output: Predicted Family: tRNA (Confidence: 100.00%)
```

---

## 4. Using the Pre-Trained Model

The pre-trained model is serialized at:
```
models/mvmgb_dataset1.joblib
```
Trained on 6,320 canonical non-coding transcripts across 13 functional ncRNA families:
1. `5S_rRNA` (5S Ribosomal RNA)
2. `5_8S_rRNA` (5.8S Ribosomal RNA)
3. `CD-box` (C/D-box small nucleolar RNA)
4. `HACA-box` (H/ACA-box small nucleolar RNA)
5. `IRES` (Internal Ribosome Entry Site)
6. `Intron_gpI` (Group I Catalytic Intron)
7. `Intron_gpII` (Group II Catalytic Intron)
8. `leader` (Ribosomal RNA Leader)
9. `miRNA` (MicroRNA Precursor)
10. `riboswitch` (Regulatory Riboswitch)
11. `ribozyme` (Catalytic Ribozyme)
12. `scaRNA` (Small Cajal Body-Specific RNA)
13. `tRNA` (Transfer RNA)

Model metadata and registered metrics are documented in `models/model_metadata.json`.

---

## 5. The Four Biophysical Feature Spaces

You can extract and inspect any individual feature view using the underlying encoder modules:

```python
from src.views.view2_natural_vectors import NaturalVectorEncoder
from src.views.view3_structural_fingerprints import BSFEncoder
from src.views.view4_stacking_profiles import StackingPropensityEncoder

seqs = ["GGGCCCGUAGCUCAGUGGUAGAGCGCUCGCUUUGCAUGUGAGAGGUACCGGGAUCGAUACCCGGCGGGUCCA"]

# View 2: Topological Moments (21 dimensions)
nv_encoder = NaturalVectorEncoder()
X_nv = nv_encoder.encode_dataset(seqs)  # shape: (1, 21)

# View 3: Biophysical Structural Fingerprints (22 dimensions)
bsf_encoder = BSFEncoder()
X_bsf = bsf_encoder.encode_dataset(seqs)  # shape: (1, 22)

# View 4: Turner Stacking B-Splines (60 dimensions)
bsp_encoder = StackingPropensityEncoder()
X_bsp = bsp_encoder.encode_dataset(seqs)  # shape: (1, 60)
```

---

## 6. Chow-Type Bayesian Diagnostic Triage

To prevent catastrophic misclassification in laboratory or clinical pipelines, MV-MGB implements a Chow-type Bayesian triage rule with two tunable decision boundaries:

```python
# Perform triage
triage = model.predict_triage(
    [rna_seq],
    tau_review=0.85,  # Acceptance threshold
    tau_reject=0.50   # OOD / Novel rejection threshold
)[0]

print(f"Action:     {triage.action}")
print(f"Prediction: {triage.predicted_class}")
print(f"Confidence: {triage.confidence:.4f}")
```

### Decision Rules:
| Action | Condition | Interpretation |
| :--- | :--- | :--- |
| **`ACCEPT`** | $p_{\max} \ge 0.85$ | High-confidence automated call; direct experimental routing. |
| **`HUMAN_REVIEW`** | $0.50 \le p_{\max} < 0.85$ | Ambiguous transcript; routed to secondary laboratory inspection. |
| **`REJECT_NOVEL`** | $p_{\max} < 0.50$ | Uncharacterized family or genomic decoy; rejected from closed taxonomy. |

---

## 7. Command-Line Interface (CLI)

MV-MGB includes a high-throughput CLI tool (`scripts/predict_cli.py`):

### Classify a Single Transcript:
```bash
python scripts/predict_cli.py --sequence "GGGCCCGUAGCUCAGUGGUAGAGCGCUCGCUUUGCAUGUGAGAGGUACCGGGAUCGAUACCCGGCGGGUCCA"
```

### High-Throughput Batch FASTA Screening:
```bash
python scripts/predict_cli.py \
  --fasta datasets/demo/demo_transcripts.fasta \
  --output predictions.csv \
  --tau-review 0.85 \
  --tau-reject 0.50
```

The resulting `predictions.csv` includes:
- `Sequence_ID`
- `Length`
- `Predicted_Class`
- `Confidence`
- `Triage_Action`
- All 13 posterior class probabilities (`5S_rRNA`, ..., `tRNA`)

---

## 8. Training on Custom FASTA Corpora

You can train MV-MGB from scratch on any custom FASTA dataset:

```bash
python scripts/train_mvmgb.py \
  --train datasets/dataset1/train.fasta \
  --test datasets/dataset1/test.fasta \
  --output models/my_custom_model.joblib \
  --folds 5 \
  --seed 42
```

---

## 9. Interactive Jupyter Notebook Tutorial

Launch the interactive Jupyter notebook for step-by-step visual exploration:

```bash
jupyter notebook tutorial_mvmgb_inference.ipynb
```
The notebook includes:
- Interactive sequence ingestion & canonicalisation
- Feature vector inspection and Matplotlib visualizations
- Probability simplex bar charts
- Triage demonstrations on known sequences vs. synthetic decoys
- Batch FASTA evaluation

---

## 10. Reproducing Benchmark Metrics

To reproduce the benchmark performance reported in the manuscript:

```bash
# Run full evaluation suite
python scripts/evaluate_benchmark.py --dataset dataset1

# Run unit and leak-free validation tests
PYTHONPATH=. pytest tests/ -v
```

---

## Citation
If you use MV-MGB or these models in your research, please cite:

```bibtex
@article{lebsir2026mvmgb,
  title={An Explainable Multi-View Expert Decision Support System for Alignment-Free Non-Coding RNA Classification Using Biophysical and Geometric Feature Spaces},
  author={Lebsir, Rabah and Boukelia, Abdelbasset},
  journal={Expert Systems with Applications},
  year={2026},
  publisher={Elsevier}
}
```
