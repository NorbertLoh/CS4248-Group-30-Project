#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# If not running inside a SLURM job, allocate an interactive job with srun
# and re-run this script inside the allocated step so GPUs are available.
if [ -z "${SLURM_JOB_ID:-}" ]; then
	echo "Not running inside SLURM job. Allocating resources with srun..."
	constraint="xgpe"
	exec srun --unbuffered --gres=gpu:1 --constraint=${constraint} --mem=16G "$0" "$@"
    # exec srun --unbuffered --label --gres="gpu:a100-80:1"  --mem=64G "$0" "$@"

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
TARGET_SCRIPT="${TARGET_SCRIPT:-cara/invidual-test/invidiual-baseline.py}"

if [ "${PYBIN#/}" = "$PYBIN" ]; then
	PYBIN="$(pwd)/$PYBIN"
fi

BASELINE_MODEL_ID="${BASELINE_MODEL_ID:-Qwen/Qwen3-VL-8B-Thinking}"
BASELINE_ARGS="${BASELINE_ARGS:-}"

echo "Starting individual baseline remote run at $(date)"

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

echo "Model ID: $BASELINE_MODEL_ID"
echo "Running individual baseline inference script: $PYBIN $TARGET_SCRIPT $BASELINE_ARGS $*"

TARGET_SCRIPT_DIR="$(dirname "$TARGET_SCRIPT")"
TARGET_SCRIPT_NAME="$(basename "$TARGET_SCRIPT")"

pushd "$TARGET_SCRIPT_DIR" >/dev/null
BASELINE_MODEL_ID="$BASELINE_MODEL_ID" \
"$PYBIN" -u "$TARGET_SCRIPT_NAME" \
	$BASELINE_ARGS "$@"
popd >/dev/null

echo "Individual baseline remote run finished at $(date)"
