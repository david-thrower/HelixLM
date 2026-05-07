#!/usr/bin/env bash
# scripts/run_nas_round.sh
# Example runner for the 3-round HelixLM NAS protocol.
#
# Usage:
#   bash scripts/run_nas_round.sh screening 256 5
#   bash scripts/run_nas_round.sh validation 256 3
#   bash scripts/run_nas_round.sh final 256 1
#
# Environment:
#   HF_TOKEN        - HuggingFace token (for dataset streaming)
#   MLFLOW_TRACKING_URI - MLflow server (default: http://localhost:5000)
#   HELIXLM_PATH    - Path to helix_lm package (default: parent of scripts/)

set -euo pipefail

ROUND=${1:-screening}
SEQ_LEN=${2:-256}
N_JOBS=${3:-1}
STUDY_NAME=${4:-helixlm_nas}
OUTPUT_DIR=${5:-./nas_results}

# Validate round
if [[ "$ROUND" != "screening" && "$ROUND" != "validation" && "$ROUND" != "final" ]]; then
    echo "Error: round must be screening, validation, or final"
    exit 1
fi

# Set defaults
export HF_TOKEN=${HF_TOKEN:-}
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://localhost:5000}
export HELIXLM_PATH=${HELIXLM_PATH:-$(dirname "$0")/..}

echo "========================================"
echo "HelixLM NAS Runner"
echo "Round      : $ROUND"
echo "Seq Len    : $SEQ_LEN"
echo "Parallel   : $N_JOBS"
echo "Study Name : $STUDY_NAME"
echo "Output Dir : $OUTPUT_DIR"
echo "MLflow URI : $MLFLOW_TRACKING_URI"
echo "========================================"

# Round-specific overrides
if [[ "$ROUND" == "screening" ]]; then
    EXTRA_FLAGS=""
    echo "Mode: Fast elimination (20K samples, 2 epochs, many trials)"
elif [[ "$ROUND" == "validation" ]]; then
    EXTRA_FLAGS="--enqueue-top 15"
    echo "Mode: Confirm top configs (100K samples, 5 epochs)"
elif [[ "$ROUND" == "final" ]]; then
    EXTRA_FLAGS="--enqueue-top 3"
    echo "Mode: Convergence (full dataset, 10 epochs, single best config)"
fi

# Run
python "${HELIXLM_PATH}/scripts/nas_helixlm.py" \
    --round "$ROUND" \
    --seq-len "$SEQ_LEN" \
    --n-jobs "$N_JOBS" \
    --study-name "$STUDY_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --mlflow-uri "$MLFLOW_TRACKING_URI" \
    $EXTRA_FLAGS

echo "========================================"
echo "Round $ROUND complete."
echo "Results in: $OUTPUT_DIR/nas_${ROUND}_results.csv"
echo "========================================"
