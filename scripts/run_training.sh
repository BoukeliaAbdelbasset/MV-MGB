#!/usr/bin/env bash
# ==============================================================================
# MV-MGB: Multi-View Meta-Gradient Boosting for Non-Coding RNA Classification
# Step 1: Model Training Pipeline Script
# ==============================================================================
set -euo pipefail

# ANSI color codes
BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
NC="\033[0m"

# Resolve repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

print_banner() {
    echo -e "${BLUE}==============================================================================${NC}"
    echo -e "${BOLD}${CYAN}   MV-MGB: Training Pipeline Script                                           ${NC}"
    echo -e "${BLUE}==============================================================================${NC}"
}

print_help() {
    echo -e "Usage: ./run_training.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --dataset <name>      Benchmark dataset name (dataset1, dataset1-nd, dataset2, 41, 177, nrc)"
    echo "                            Default: dataset1"
    echo "  -t, --train <path>        Custom training file (.fasta or .csv)"
    echo "  -e, --test <path>         Custom test evaluation file (.fasta or .csv)"
    echo "  -o, --output <path>       Output path for serialized .joblib model"
    echo "                            Default: models/mvmgb_<dataset>.joblib"
    echo "  -f, --folds <int>         Number of Tier-1 cross-validation folds (default: 5)"
    echo "  -m, --max-train <int>     Maximum number of training sequences (subsampling for quick runs)"
    echo "  -n, --max-test <int>      Maximum number of test sequences (subsampling for quick runs)"
    echo "  -s, --seed <int>          Random seed for deterministic initialization (default: 42)"
    echo "  -h, --help                Display this help message and exit"
    echo ""
    echo "Examples:"
    echo "  ./run_training.sh --dataset dataset1"
    echo "  ./run_training.sh --dataset dataset2 --folds 5"
    echo "  ./run_training.sh --dataset 41 --max-train 5000"
    echo "  ./run_training.sh --train custom_train.fasta --test custom_test.fasta --output models/custom.joblib"
    echo ""
}

# Default arguments
DATASET="dataset1"
TRAIN_FILE=""
TEST_FILE=""
OUTPUT_MODEL=""
FOLDS=5
MAX_TRAIN=""
MAX_TEST=""
SEED=42

# Parse CLI options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -t|--train)
            TRAIN_FILE="$2"
            shift 2
            ;;
        -e|--test)
            TEST_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_MODEL="$2"
            shift 2
            ;;
        -f|--folds)
            FOLDS="$2"
            shift 2
            ;;
        -m|--max-train)
            MAX_TRAIN="$2"
            shift 2
            ;;
        -n|--max-test)
            MAX_TEST="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            print_banner
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR] Unknown argument: $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

print_banner

# Determine default output model path if not set
if [[ -z "${OUTPUT_MODEL}" ]]; then
    if [[ -n "${TRAIN_FILE}" ]]; then
        OUTPUT_MODEL="models/mvmgb_custom.joblib"
    else
        OUTPUT_MODEL="models/mvmgb_${DATASET}.joblib"
    fi
fi

# Ensure output directory exists
mkdir -p "$(dirname "${OUTPUT_MODEL}")"

# Verify Python environment
PYTHON_BIN="$(which python3 || which python || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
    echo -e "${RED}[ERROR] Python 3 executable not found in PATH.${NC}"
    exit 1
fi

echo -e "${YELLOW}Configuration:${NC}"
if [[ -n "${TRAIN_FILE}" ]]; then
    echo -e "  Mode:              Custom Files"
    echo -e "  Train File:        ${TRAIN_FILE}"
    echo -e "  Test File:         ${TEST_FILE}"
else
    echo -e "  Dataset:           ${BOLD}${DATASET}${NC}"
fi
echo -e "  Output Model:      ${GREEN}${OUTPUT_MODEL}${NC}"
echo -e "  Tier-1 CV Folds:   ${FOLDS}"
echo -e "  Random Seed:       ${SEED}"
[[ -n "${MAX_TRAIN}" ]] && echo -e "  Max Train Limit:   ${MAX_TRAIN}"
[[ -n "${MAX_TEST}" ]] && echo -e "  Max Test Limit:    ${MAX_TEST}"
echo ""

# Build Python execution command
CMD=(python3 scripts/train_mvmgb.py --output "${OUTPUT_MODEL}" --folds "${FOLDS}" --seed "${SEED}")

if [[ -n "${TRAIN_FILE}" ]]; then
    if [[ ! -f "${TRAIN_FILE}" ]]; then
        echo -e "${RED}[ERROR] Training file not found: ${TRAIN_FILE}${NC}"
        exit 1
    fi
    CMD+=(--train "${TRAIN_FILE}")
    if [[ -n "${TEST_FILE}" ]]; then
        CMD+=(--test "${TEST_FILE}")
    fi
else
    CMD+=(--dataset "${DATASET}")
fi

if [[ -n "${MAX_TRAIN}" ]]; then
    CMD+=(--max-train "${MAX_TRAIN}")
fi

if [[ -n "${MAX_TEST}" ]]; then
    CMD+=(--max-test "${MAX_TEST}")
fi

echo -e "${CYAN}Executing: ${CMD[*]}${NC}"
echo ""
"${CMD[@]}"

echo ""
echo -e "${GREEN}==============================================================================${NC}"
echo -e "${BOLD}${GREEN}[SUCCESS] MV-MGB model training and benchmark evaluation completed!${NC}"
echo -e "Model artifact saved at: ${BOLD}${OUTPUT_MODEL}${NC}"
echo -e "${GREEN}==============================================================================${NC}"
