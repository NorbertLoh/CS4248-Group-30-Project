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
TARGET_SCRIPT="${TARGET_SCRIPT:-cara/qwen-rag-vllm-lora.py}"

STAGE2_DATA_PATH="${STAGE2_DATA_PATH:-datapreparation/output/facebook-samples-test-rationale-excluded.jsonl}"
STAGE2_OUT_PATH="${STAGE2_OUT_PATH:-datapreparation/output/predictions_stage2_vllm_lora.jsonl}"
UNSLOTH_QWEN3_VL_MODEL_ID="${UNSLOTH_QWEN3_VL_MODEL_ID:-Qwen/Qwen3-VL-8B-Thinking}"
USE_JUDGE_LORA="${USE_JUDGE_LORA:-1}"
JUDGE_LORA_PATH="${JUDGE_LORA_PATH:-judge-qwen3-lora}"
STAGE2_ARGS="${STAGE2_ARGS:-}"

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

echo "Stage 2 data path: $STAGE2_DATA_PATH"
echo "Stage 2 output path: $STAGE2_OUT_PATH"
echo "Model ID: $UNSLOTH_QWEN3_VL_MODEL_ID"
echo "Use judge LoRA: $USE_JUDGE_LORA"
echo "Judge LoRA path: $JUDGE_LORA_PATH"

mkdir -p "$(dirname "$STAGE2_OUT_PATH")"

echo "Running LoRA inference script: $PYBIN $TARGET_SCRIPT $STAGE2_ARGS $*"
STAGE2_DATA_PATH="$STAGE2_DATA_PATH" \
STAGE2_OUT_PATH="$STAGE2_OUT_PATH" \
UNSLOTH_QWEN3_VL_MODEL_ID="$UNSLOTH_QWEN3_VL_MODEL_ID" \
USE_JUDGE_LORA="$USE_JUDGE_LORA" \
JUDGE_LORA_PATH="$JUDGE_LORA_PATH" \
"$PYBIN" -u "$TARGET_SCRIPT" \
	$STAGE2_ARGS "$@"

echo "Remote run finished at $(date)"
