#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# If not running inside a SLURM job, allocate an interactive job with srun
# and re-run this script inside the allocated step so GPUs are available.
if [ -z "${SLURM_JOB_ID:-}" ]; then
	echo "Not running inside SLURM job. Allocating resources with srun..."
	constraint="xgpe"
	# exec srun --gres=gpu:1 --constraint=${constraint} "$0" "$@"
	# exec srun -p gpu --gpus=1 -w xgpi0 "$0" "$@"
	# exec srun -p gpu --gpus=1 -w xgpe8 --mem=64G "$0" "$@"
	exec srun --unbuffered --label --gres="gpu:a100-40:1" --time=03:00:00 --mem=32G "$0" "$@"

fi

VENV_DIR=".venv"
PYBIN="$VENV_DIR/bin/python"
TARGET_SCRIPT="api-inference/qwen-rag.py"
QWEN_REPO_ID="Qwen/Qwen3-VL-2B-Instruct"
QWEN_MODEL_DIR="${QWEN_VL_MODEL_PATH:-$(pwd)/models/Qwen3-VL-2B-Instruct}"

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

if [ ! -f "$QWEN_MODEL_DIR/model.safetensors" ]; then
	echo "Qwen model not found at: $QWEN_MODEL_DIR"
	echo "Downloading $QWEN_REPO_ID to remote workspace..."
	mkdir -p "$QWEN_MODEL_DIR"
	"$PYBIN" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$QWEN_REPO_ID",
    local_dir=r"$QWEN_MODEL_DIR",
)
print("Model download complete")
PY
fi

export QWEN_VL_MODEL_PATH="$QWEN_MODEL_DIR"
echo "Using QWEN_VL_MODEL_PATH=$QWEN_VL_MODEL_PATH"

echo "Running inference script with: $PYBIN $TARGET_SCRIPT $@"
"$PYBIN" -u "$TARGET_SCRIPT" "$@"

echo "Remote run finished at $(date)"
