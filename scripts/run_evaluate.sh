#!/usr/bin/env bash
# ==============================================================================
# MV-MGB: Multi-View Meta-Gradient Boosting for Non-Coding RNA Classification
# Step 3: Test Benchmark Evaluation Script
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
    echo -e "${BOLD}${CYAN}   MV-MGB: Benchmark Evaluation & Diagnostics Script                          ${NC}"
    echo -e "${BLUE}==============================================================================${NC}"
}

print_help() {
    echo -e "Usage: ./run_evaluate.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --dataset <name>      Benchmark dataset to evaluate (dataset1, dataset1-nd, dataset2, 41, 177, nrc)"
    echo "                            Default: dataset1"
    echo "  -m, --model <path>        Path to serialized model (.joblib)"
    echo "                            Default: models/mvmgb_<dataset>.joblib or models/mvmgb_dataset1.joblib"
    echo "  -n, --max-test <int>      Subsample maximum number of test sequences for quick verification"
    echo "  -o, --output <path>       Save detailed evaluation predictions to CSV file"
    echo "  --tau-review <float>      Bayesian Chow threshold for automated acceptance (default: 0.85)"
    echo "  --tau-reject <float>      Bayesian Chow threshold for novel/OOD rejection (default: 0.50)"
    echo "  -h, --help                Display this help message and exit"
    echo ""
    echo "Examples:"
    echo "  ./run_evaluate.sh --dataset dataset1"
    echo "  ./run_evaluate.sh --dataset dataset1-nd --output eval_dataset1_nd.csv"
    echo "  ./run_evaluate.sh --dataset dataset2"
    echo "  ./run_evaluate.sh --dataset 41 --max-test 1000"
    echo ""
}

# Defaults
DATASET="dataset1"
MODEL_PATH=""
MAX_TEST=""
OUTPUT_CSV=""
TAU_REVIEW="0.85"
TAU_REJECT="0.50"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -n|--max-test)
            MAX_TEST="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_CSV="$2"
            shift 2
            ;;
        --tau-review)
            TAU_REVIEW="$2"
            shift 2
            ;;
        --tau-reject)
            TAU_REJECT="$2"
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

# Auto-detect model if not provided
if [[ -z "${MODEL_PATH}" ]]; then
    if [[ -f "models/mvmgb_${DATASET}.joblib" ]]; then
        MODEL_PATH="models/mvmgb_${DATASET}.joblib"
    elif [[ -f "models/mvmgb_dataset1.joblib" ]]; then
        MODEL_PATH="models/mvmgb_dataset1.joblib"
    elif [[ -f "models/test_model.joblib" ]]; then
        MODEL_PATH="models/test_model.joblib"
    else
        FIRST_MODEL="$(find models/ -name "*.joblib" | head -n 1 || true)"
        if [[ -n "${FIRST_MODEL}" ]]; then
            MODEL_PATH="${FIRST_MODEL}"
        else
            MODEL_PATH="models/mvmgb_dataset1.joblib"
        fi
    fi
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo -e "${RED}[ERROR] Serialized model not found at: ${MODEL_PATH}${NC}"
    echo -e "Please train a model first using ./run_training.sh or specify an existing model path with --model."
    exit 1
fi

echo -e "${YELLOW}Evaluation Configuration:${NC}"
echo -e "  Benchmark Dataset: ${BOLD}${DATASET}${NC}"
echo -e "  Model Path:        ${GREEN}${MODEL_PATH}${NC}"
echo -e "  Review Threshold:  ${TAU_REVIEW}"
echo -e "  Reject Threshold:  ${TAU_REJECT}"
[[ -n "${MAX_TEST}" ]] && echo -e "  Max Test Limit:    ${MAX_TEST}"
[[ -n "${OUTPUT_CSV}" ]] && echo -e "  Output CSV:        ${OUTPUT_CSV}"
echo ""

CMD=(python3 scripts/evaluate_benchmark.py --dataset "${DATASET}" --model "${MODEL_PATH}" --tau-review "${TAU_REVIEW}" --tau-reject "${TAU_REJECT}")

if [[ -n "${MAX_TEST}" ]]; then
    CMD+=(--max-test "${MAX_TEST}")
fi

if [[ -n "${OUTPUT_CSV}" ]]; then
    mkdir -p "$(dirname "${OUTPUT_CSV}")"
    CMD+=(--output "${OUTPUT_CSV}")
fi

echo -e "${CYAN}Executing: ${CMD[*]}${NC}"
echo ""
"${CMD[@]}"

echo ""
echo -e "${GREEN}==============================================================================${NC}"
echo -e "${BOLD}${GREEN}[SUCCESS] Benchmark evaluation successfully completed!${NC}"
echo -e "${GREEN}==============================================================================${NC}"
