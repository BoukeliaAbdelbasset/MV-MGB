#!/usr/bin/env bash
# ==============================================================================
# MV-MGB: Multi-View Meta-Gradient Boosting for Non-Coding RNA Classification
# Step 2: Prediction & Diagnostic Triage Pipeline Script
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
    echo -e "${BOLD}${CYAN}   MV-MGB: High-Throughput Inference & Triage Script                          ${NC}"
    echo -e "${BLUE}==============================================================================${NC}"
}

print_help() {
    echo -e "Usage: ./run_predict.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -f, --fasta <path>        Path to input multi-FASTA file"
    echo "  -s, --sequence <str>      Single RNA sequence string (e.g. \"GGGCCCGUAGC...\")"
    echo "  -c, --csv <path>          Path to input CSV file containing RNA sequences"
    echo "  -d, --demo                Run inference on bundled demo transcripts (default if no input)"
    echo "  -m, --model <path>        Path to serialized model (.joblib)"
    echo "                            Default: models/mvmgb_dataset1.joblib"
    echo "  -o, --output <path>       Save predictions and triage decisions to CSV file"
    echo "  --tau-review <float>      Bayesian Chow threshold for automated acceptance (default: 0.85)"
    echo "  --tau-reject <float>      Bayesian Chow threshold for novel/OOD rejection (default: 0.50)"
    echo "  -h, --help                Display this help message and exit"
    echo ""
    echo "Examples:"
    echo "  ./run_predict.sh --demo"
    echo "  ./run_predict.sh --sequence \"GGGCCCGUAGCUCAGUGGUAGAGCGCUCGCUUUGCAUGUGAGAGGUACCGGGAUCGAUACCCGGCGGGUCCA\""
    echo "  ./run_predict.sh --fasta datasets/dataset1/test.fasta --output predictions.csv"
    echo "  ./run_predict.sh --csv datasets/41/test_data.csv --output results_41.csv"
    echo ""
}

# Defaults
INPUT_FASTA=""
INPUT_SEQ=""
INPUT_CSV=""
RUN_DEMO=false
MODEL_PATH=""
OUTPUT_CSV=""
TAU_REVIEW="0.85"
TAU_REJECT="0.50"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--fasta)
            INPUT_FASTA="$2"
            shift 2
            ;;
        -s|--sequence)
            INPUT_SEQ="$2"
            shift 2
            ;;
        -c|--csv)
            INPUT_CSV="$2"
            shift 2
            ;;
        -d|--demo)
            RUN_DEMO=true
            shift
            ;;
        -m|--model)
            MODEL_PATH="$2"
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

# Auto-detect default model if not provided
if [[ -z "${MODEL_PATH}" ]]; then
    if [[ -f "models/mvmgb_dataset1.joblib" ]]; then
        MODEL_PATH="models/mvmgb_dataset1.joblib"
    elif [[ -f "models/test_model.joblib" ]]; then
        MODEL_PATH="models/test_model.joblib"
    else
        # Find any joblib in models/
        FIRST_MODEL="$(find models/ -name "*.joblib" | head -n 1 || true)"
        if [[ -n "${FIRST_MODEL}" ]]; then
            MODEL_PATH="${FIRST_MODEL}"
        else
            MODEL_PATH="models/mvmgb_dataset1.joblib"
        fi
    fi
fi

# If model does not exist, alert user
if [[ ! -f "${MODEL_PATH}" ]]; then
    echo -e "${YELLOW}[WARNING] Model file not found at: ${MODEL_PATH}${NC}"
    echo -e "Training a base model first via ./run_training.sh or specifying an existing model with --model is required."
    echo -e "Would you like to train a quick demonstration model now? Run:"
    echo -e "  ${BOLD}./run_training.sh --dataset dataset1 --max-train 200${NC}"
    exit 1
fi

# If no input mode was chosen, default to demo
if [[ -z "${INPUT_FASTA}" && -z "${INPUT_SEQ}" && -z "${INPUT_CSV}" && "${RUN_DEMO}" = false ]]; then
    echo -e "${YELLOW}No specific input specified. Automatically activating --demo mode...${NC}"
    RUN_DEMO=true
fi

# Build CLI command
CMD=(python3 scripts/predict_cli.py --model "${MODEL_PATH}" --tau-review "${TAU_REVIEW}" --tau-reject "${TAU_REJECT}")

if [[ -n "${INPUT_SEQ}" ]]; then
    CMD+=(--sequence "${INPUT_SEQ}")
elif [[ "${RUN_DEMO}" = true ]]; then
    CMD+=(--demo)
elif [[ -n "${INPUT_CSV}" ]]; then
    if [[ ! -f "${INPUT_CSV}" ]]; then
        echo -e "${RED}[ERROR] Input CSV not found: ${INPUT_CSV}${NC}"
        exit 1
    fi
    CMD+=(--csv "${INPUT_CSV}")
elif [[ -n "${INPUT_FASTA}" ]]; then
    if [[ ! -f "${INPUT_FASTA}" ]]; then
        echo -e "${RED}[ERROR] Input FASTA not found: ${INPUT_FASTA}${NC}"
        exit 1
    fi
    CMD+=(--fasta "${INPUT_FASTA}")
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
echo -e "${BOLD}${GREEN}[SUCCESS] MV-MGB inference and Bayesian triage completed!${NC}"
if [[ -n "${OUTPUT_CSV}" ]]; then
    echo -e "Detailed results saved to: ${BOLD}${OUTPUT_CSV}${NC}"
fi
echo -e "${GREEN}==============================================================================${NC}"
