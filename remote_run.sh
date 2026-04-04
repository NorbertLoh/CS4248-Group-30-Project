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

VENV_DIR=".venv"
PYBIN="$VENV_DIR/bin/python"
# TARGET_SCRIPT="sft/sft.py"
# TARGET_SCRIPT="sft/export_gguf.py"
# TARGET_SCRIPT="sft/gguftest.py"
# TARGET_SCRIPT="sft/loratest.py"
TARGET_SCRIPT="inference/inference.py"
TARGET_ARGS="--kb_version no_context"
# TARGET_SCRIPT="inference/build_clip_rag.py"
# TARGET_ARGS="--version all"
QWEN_REPO_ID="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
QWEN_MODEL_DIR="${QWEN_UNSLOTH_4BIT_MODEL:-$(pwd)/models/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit}"

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

if [ ! -f "$QWEN_MODEL_DIR/config.json" ]; then
	echo "Unsloth 4-bit model not found locally at: $QWEN_MODEL_DIR"
	echo "Downloading $QWEN_REPO_ID to remote workspace..."
	mkdir -p "$QWEN_MODEL_DIR"
	"$PYBIN" - <<PY
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="$QWEN_REPO_ID",
    local_dir=r"$QWEN_MODEL_DIR",
    local_dir_use_symlinks=False,
    token=os.environ.get("HF_TOKEN"),
)
print("Model download complete")
PY
fi

TARGET_ARGS="$TARGET_ARGS"
echo "Using local Unsloth model dir: $QWEN_MODEL_DIR"

echo "Running inference script with: $PYBIN $TARGET_SCRIPT $TARGET_ARGS $@"
"$PYBIN" -u "$TARGET_SCRIPT" $TARGET_ARGS "$@"

echo "Remote run finished at $(date)"
