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
TARGET_SCRIPT="${TARGET_SCRIPT:-cara/ablation/simple-rag.py}"

STAGE2_DATA_PATH="${STAGE2_DATA_PATH:-facebook-data/dev.jsonl}"
STAGE2_OUT_PATH="${STAGE2_OUT_PATH:-datapreparation/output/predictions_simple_rag_qwen3vl8b_dev.jsonl}"
RAG_VLM_MODEL_ID="${RAG_VLM_MODEL_ID:-Qwen/Qwen3-VL-8B-Thinking}"
MEMECAP_DATA="${MEMECAP_DATA:-memecap-data/memes-trainval.json}"
RAG_TOP_K="${RAG_TOP_K:-3}"
RAG_SCORE_THRESHOLD="${RAG_SCORE_THRESHOLD:-0.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
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
echo "Model ID: $RAG_VLM_MODEL_ID"
echo "MemeCap data path: $MEMECAP_DATA"
echo "RAG top-k: $RAG_TOP_K"
echo "RAG score threshold: $RAG_SCORE_THRESHOLD"
echo "Max new tokens: $MAX_NEW_TOKENS"

mkdir -p "$(dirname "$STAGE2_OUT_PATH")"

echo "Running simple-rag inference script: $PYBIN $TARGET_SCRIPT $STAGE2_ARGS $*"
STAGE2_DATA_PATH="$STAGE2_DATA_PATH" \
STAGE2_OUT_PATH="$STAGE2_OUT_PATH" \
RAG_VLM_MODEL_ID="$RAG_VLM_MODEL_ID" \
MEMECAP_DATA="$MEMECAP_DATA" \
RAG_TOP_K="$RAG_TOP_K" \
RAG_SCORE_THRESHOLD="$RAG_SCORE_THRESHOLD" \
MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
"$PYBIN" -u "$TARGET_SCRIPT" \
	$STAGE2_ARGS "$@"

echo "Remote run finished at $(date)"
