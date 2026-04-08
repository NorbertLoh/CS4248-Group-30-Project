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
TARGET_SCRIPT="${TARGET_SCRIPT:-cara/ablation/roberta-clip-baseline.py}"

BASELINE_DATA_PATH="${BASELINE_DATA_PATH:-facebook-data/dev.jsonl}"
BASELINE_IMG_DIR="${BASELINE_IMG_DIR:-facebook-data/img}"
BASELINE_OUT_PATH="${BASELINE_OUT_PATH:-datapreparation/output/predictions_roberta_clip_baseline.jsonl}"
ROBERTA_MODEL_DIR="${ROBERTA_MODEL_DIR:-metameme_roberta_model}"
CLIP_MODEL_ID="${CLIP_MODEL_ID:-openai/clip-vit-base-patch32}"
TEXT_WEIGHT="${TEXT_WEIGHT:-0.7}"
IMAGE_WEIGHT="${IMAGE_WEIGHT:-0.3}"
THRESHOLD="${THRESHOLD:-0.5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BASELINE_ARGS="${BASELINE_ARGS:-}"

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

echo "Baseline data path: $BASELINE_DATA_PATH"
echo "Baseline image dir: $BASELINE_IMG_DIR"
echo "Baseline output path: $BASELINE_OUT_PATH"
echo "RoBERTa model dir: $ROBERTA_MODEL_DIR"
echo "CLIP model id: $CLIP_MODEL_ID"
echo "Weights: text=$TEXT_WEIGHT image=$IMAGE_WEIGHT"
echo "Threshold: $THRESHOLD"
echo "Batch size: $BATCH_SIZE"

mkdir -p "$(dirname "$BASELINE_OUT_PATH")"

echo "Running RoBERTa-CLIP baseline script: $PYBIN $TARGET_SCRIPT $BASELINE_ARGS $*"
BASELINE_DATA_PATH="$BASELINE_DATA_PATH" \
BASELINE_IMG_DIR="$BASELINE_IMG_DIR" \
BASELINE_OUT_PATH="$BASELINE_OUT_PATH" \
ROBERTA_MODEL_DIR="$ROBERTA_MODEL_DIR" \
CLIP_MODEL_ID="$CLIP_MODEL_ID" \
TEXT_WEIGHT="$TEXT_WEIGHT" \
IMAGE_WEIGHT="$IMAGE_WEIGHT" \
THRESHOLD="$THRESHOLD" \
BATCH_SIZE="$BATCH_SIZE" \
"$PYBIN" -u "$TARGET_SCRIPT" \
	$BASELINE_ARGS "$@"

echo "Remote run finished at $(date)"
