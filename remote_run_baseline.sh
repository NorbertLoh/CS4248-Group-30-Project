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
	exec srun --unbuffered --label --gres="gpu:a100-40:1"  --mem=64G "$0" "$@"

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
TARGET_SCRIPT="${TARGET_SCRIPT:-cara/baseline.py}"

BASELINE_DATA_PATH="${BASELINE_DATA_PATH:-facebook-data/dev.jsonl}"
BASELINE_OUT_PATH="${BASELINE_OUT_PATH:-datapreparation/output/predictions_baseline_vllm_8b_Instruct.jsonl}"
BASELINE_MODEL_ID="${BASELINE_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
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
echo "Baseline output path: $BASELINE_OUT_PATH"
echo "Model ID: $BASELINE_MODEL_ID"

mkdir -p "$(dirname "$BASELINE_OUT_PATH")"

echo "Running baseline inference script: $PYBIN $TARGET_SCRIPT $BASELINE_ARGS $*"
BASELINE_DATA_PATH="$BASELINE_DATA_PATH" \
BASELINE_OUT_PATH="$BASELINE_OUT_PATH" \
BASELINE_MODEL_ID="$BASELINE_MODEL_ID" \
"$PYBIN" -u "$TARGET_SCRIPT" \
	$BASELINE_ARGS "$@"

echo "Remote run finished at $(date)"
