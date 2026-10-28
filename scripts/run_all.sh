#!/usr/bin/env bash
# ==============================================================================
# MV-MGB: Multi-View Meta-Gradient Boosting for Non-Coding RNA Classification
# Master End-to-End Pipeline Script ("All-in-One")
# ==============================================================================
set -euo pipefail

# ANSI color codes
BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
MAGENTA="\033[0;35m"
NC="\033[0m"

# Resolve repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

print_banner() {
    echo -e "${BLUE}==============================================================================${NC}"
    echo -e "${BOLD}${MAGENTA}   MV-MGB: Master End-to-End Pipeline Execution                              ${NC}"
    echo -e "${BOLD}${CYAN}   Multi-View Meta-Gradient Boosting for Non-Coding RNA Classification        ${NC}"
    echo -e "${BLUE}==============================================================================${NC}"
    echo -e " Authors: Rabah Lebsir & Abdelbasset Boukelia"
    echo -e " Repository Root: ${SCRIPT_DIR}"
    echo -e " Date / Time: $(date)"
    echo -e "${BLUE}==============================================================================${NC}"
}

print_help() {
    echo -e "Usage: ./run_all.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --quick                   Fast demonstration run (uses subsampling for rapid <30s execution)"
    echo "  --full                    Full production run on complete datasets"
    echo "  -d, --dataset <name>      Target benchmark dataset (dataset1, dataset1-nd, dataset2, 41, 177, nrc)"
    echo "                            Default: dataset1"
    echo "  --retrain                 Force model retraining even if a serialized model already exists"
    echo "  --skip-tests              Skip running pytest verification"
    echo "  -o, --output-dir <path>   Directory to save pipeline outputs and logs (default: results/)"
    echo "  -h, --help                Display this help message and exit"
    echo ""
    echo "Examples:"
    echo "  ./run_all.sh --quick               # Quick end-to-end verification (< 30s)"
    echo "  ./run_all.sh                       # Standard end-to-end pipeline run"
    echo "  ./run_all.sh --dataset dataset2    # Run full pipeline on Dataset 2"
    echo ""
}

# Defaults
MODE="standard"
DATASET="dataset1"
FORCE_RETRAIN=false
SKIP_TESTS=false
OUTPUT_DIR="results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            MODE="quick"
            shift
            ;;
        --full)
            MODE="full"
            shift
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        --retrain)
            FORCE_RETRAIN=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            print_banner
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR] Unknown option: $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

print_banner

mkdir -p "${OUTPUT_DIR}"
mkdir -p models

MODEL_FILE="models/mvmgb_${DATASET}.joblib"
EVAL_RESULTS_CSV="${OUTPUT_DIR}/eval_${DATASET}.csv"
PRED_RESULTS_CSV="${OUTPUT_DIR}/predictions_demo.csv"

echo -e "${YELLOW}Execution Plan:${NC}"
echo -e "  Mode:              ${BOLD}${MODE^^}${NC}"
echo -e "  Target Dataset:    ${BOLD}${DATASET}${NC}"
echo -e "  Target Model:      ${GREEN}${MODEL_FILE}${NC}"
echo -e "  Output Directory:  ${OUTPUT_DIR}"
echo -e "  Force Retrain:     ${FORCE_RETRAIN}"
echo ""

# ------------------------------------------------------------------------------
# STEP 1: Environment & Dependency Check
# ------------------------------------------------------------------------------
echo -e "${CYAN}==============================================================================${NC}"
echo -e "${BOLD}${CYAN}[STEP 1/5] Checking Python Environment & Dependencies...${NC}"
echo -e "${CYAN}==============================================================================${NC}"

python3 -c "
import sys
import numpy, scipy, pandas, sklearn, Bio, joblib
print(f'  Python Version:     {sys.version.split()[0]}')
print(f'  NumPy:              {numpy.__version__}')
print(f'  SciPy:              {scipy.__version__}')
print(f'  Pandas:             {pandas.__version__}')
print(f'  Scikit-Learn:       {sklearn.__version__}')
print(f'  Biopython:          {Bio.__version__}')
print(f'  Joblib:             {joblib.__version__}')
"
echo -e "${GREEN}[OK] All core scientific libraries are verified.${NC}"
echo ""

# Check dataset presence
echo -e "Verifying local datasets in ${SCRIPT_DIR}/datasets:"
for d in dataset1 dataset1-nd dataset2 41 177 nRC-public demo; do
    if [[ -d "datasets/${d}" ]]; then
        echo -e "  [x] datasets/${d}"
    else
        echo -e "  ${YELLOW}[!] datasets/${d} not found${NC}"
    fi
done
echo ""

# ------------------------------------------------------------------------------
# STEP 2: Zero-Leakage & View Encoders Unit Testing
# ------------------------------------------------------------------------------
if [[ "${SKIP_TESTS}" = false ]]; then
    echo -e "${CYAN}==============================================================================${NC}"
    echo -e "${BOLD}${CYAN}[STEP 2/5] Running Automated Unit & Data-Leakage Tests (pytest)...${NC}"
    echo -e "${CYAN}==============================================================================${NC}"
    pytest -v tests/
    echo -e "${GREEN}[OK] All unit tests passed without leakage or feature corruption.${NC}"
    echo ""
else
    echo -e "${YELLOW}[STEP 2/5] Skipping unit tests (--skip-tests active).${NC}"
    echo ""
fi

# ------------------------------------------------------------------------------
# STEP 3: Model Training / Verification
# ------------------------------------------------------------------------------
echo -e "${CYAN}==============================================================================${NC}"
echo -e "${BOLD}${CYAN}[STEP 3/5] Model Training / Verification Step...${NC}"
echo -e "${CYAN}==============================================================================${NC}"

if [[ -f "${MODEL_FILE}" && "${FORCE_RETRAIN}" = false ]]; then
    echo -e "${GREEN}[INFO] Pre-trained model found at: ${MODEL_FILE}.${NC}"
    echo -e "Skipping redundant training. (Use --retrain to force re-computation)."
else
    echo -e "${YELLOW}Training MV-MGB on ${DATASET}...${NC}"
    TRAIN_ARGS=(--dataset "${DATASET}" --output "${MODEL_FILE}")
    if [[ "${MODE}" = "quick" ]]; then
        echo -e "${YELLOW}[Quick Mode] Limiting to 300 train / 100 test samples for fast verification.${NC}"
        TRAIN_ARGS+=(--max-train 300 --max-test 100)
    fi
    ./run_training.sh "${TRAIN_ARGS[@]}"
fi
echo ""

# ------------------------------------------------------------------------------
# STEP 4: Held-Out Test Benchmark Evaluation
# ------------------------------------------------------------------------------
echo -e "${CYAN}==============================================================================${NC}"
echo -e "${BOLD}${CYAN}[STEP 4/5] Evaluating Model on Held-Out Benchmark Test Set...${NC}"
echo -e "${CYAN}==============================================================================${NC}"

EVAL_ARGS=(--dataset "${DATASET}" --model "${MODEL_FILE}" --output "${EVAL_RESULTS_CSV}")
if [[ "${MODE}" = "quick" ]]; then
    EVAL_ARGS+=(--max-test 100)
fi

./run_evaluate.sh "${EVAL_ARGS[@]}"
echo ""

# ------------------------------------------------------------------------------
# STEP 5: End-to-End Prediction & Diagnostic Triage
# ------------------------------------------------------------------------------
echo -e "${CYAN}==============================================================================${NC}"
echo -e "${BOLD}${CYAN}[STEP 5/5] Performing Diagnostic Inference & Bayesian Triage on Demo Data...${NC}"
echo -e "${CYAN}==============================================================================${NC}"

./run_predict.sh --demo --model "${MODEL_FILE}" --output "${PRED_RESULTS_CSV}"
echo ""

# ------------------------------------------------------------------------------
# PIPELINE COMPLETION SUMMARY
# ------------------------------------------------------------------------------
echo -e "${GREEN}==============================================================================${NC}"
echo -e "${BOLD}${GREEN}   MV-MGB FULL PIPELINE COMPLETED SUCCESSFULLY!                               ${NC}"
echo -e "${GREEN}==============================================================================${NC}"
echo -e "Artifacts generated:"
echo -e "  - Serialized Model:        ${BOLD}${MODEL_FILE}${NC}"
echo -e "  - Benchmark Evaluation:    ${BOLD}${EVAL_RESULTS_CSV}${NC}"
echo -e "  - Prediction & Triage:     ${BOLD}${PRED_RESULTS_CSV}${NC}"
echo -e "${GREEN}==============================================================================${NC}"
