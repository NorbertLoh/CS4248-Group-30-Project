#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# If not running inside a SLURM job, allocate an interactive job with srun
# and re-run this script inside the allocated step so GPUs are available.
if [ -z "${SLURM_JOB_ID:-}" ]; then
	echo "Not running inside SLURM job. Allocating resources with srun..."
	constraint="xgpe"
	# exec srun --unbuffered --gres=gpu:1 --constraint=${constraint} --mem=64G "$0" "$@ --time=120"
	# exec srun -p gpu --gpus=1 -w xgpi0 "$0" "$@"
	# exec srun -p gpu --gpus=1 -w xgpe8 --mem=64G "$0" "$@"
	exec srun --unbuffered --label --partition=gpu-long --time=3:00:00 --gres="gpu:a100-40:1"  --mem=64G "$0" "$@"

fi

# Use node-local scratch for temporary and cache files to avoid NFS .nfs* busy-file errors.
SCRATCH_ROOT="${SLURM_TMPDIR:-/tmp/$USER/cs4248}"
mkdir -p "$SCRATCH_ROOT" "$SCRATCH_ROOT/tmp" "$SCRATCH_ROOT/pip-cache" "$SCRATCH_ROOT/hf-cache"
export TMPDIR="$SCRATCH_ROOT/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export PIP_CACHE_DIR="$SCRATCH_ROOT/pip-cache"
export HF_HOME="$SCRATCH_ROOT/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$SCRATCH_ROOT/xdg-cache"

VENV_DIR=".venv"
PYBIN="$VENV_DIR/bin/python"
TARGET_SCRIPT="${TARGET_SCRIPT:-cara/classification/train_roberta.py}"

ROBERTA_DATA_FILE="${ROBERTA_DATA_FILE:-datapreparation/output/predictions_rationale_vllm_8B_all.jsonl}"
ROBERTA_OUTPUT_DIR="${ROBERTA_OUTPUT_DIR:-./metameme_roberta_model}"
ROBERTA_ARGS="${ROBERTA_ARGS:-}"

echo "Starting remote run at $(date)"

if [ ! -x "$PYBIN" ]; then
	echo "Virtualenv not ready at $PYBIN" >&2
	echo "Run ./remote_setup.sh first to create the environment and install dependencies." >&2
	exit 1
fi

echo "Running on host: $(hostname)"
if command -v nvidia-smi >/dev/null 2>&1; then
	nvidia-smi -L || true
fi

if [ ! -f "$TARGET_SCRIPT" ]; then
	echo "Target script not found: $TARGET_SCRIPT" >&2
	exit 1
fi

echo "RoBERTa data file: $ROBERTA_DATA_FILE"
echo "RoBERTa output dir: $ROBERTA_OUTPUT_DIR"

mkdir -p "$(dirname "$ROBERTA_OUTPUT_DIR")"

echo "Running RoBERTa training script: $PYBIN $TARGET_SCRIPT $ROBERTA_ARGS $*"
DATA_FILE="$ROBERTA_DATA_FILE" \
OUTPUT_DIR="$ROBERTA_OUTPUT_DIR" \
"$PYBIN" -u "$TARGET_SCRIPT" \
	$ROBERTA_ARGS "$@"

echo "Remote run finished at $(date)"
