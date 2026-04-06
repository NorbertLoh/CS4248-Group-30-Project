#!/usr/bin/env bash
set -euo pipefail

# Script to fetch artifacts produced by cara/sft/training.py
# Usage: ./get_predictions.sh

REMOTE_USER="${REMOTE_USER:-norbert}"
REMOTE_HOST="${REMOTE_HOST:-xlogin.comp.nus.edu.sg}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-CS4248/cs4248}"

REMOTE_ARTIFACTS=(
	"outputs/explainhm-judge-lora"
	"judge-qwen3-lora"
	# "judge-qwen3-lora-gguf"
)

REMOTE_ROOT="~/$REMOTE_BASE_DIR"
LOCAL_ROOT="$(cd "$(dirname "$0")" && pwd)"

for artifact in "${REMOTE_ARTIFACTS[@]}"; do
	REMOTE_PATH="$REMOTE_ROOT/$artifact"
	if ssh "$REMOTE_USER@$REMOTE_HOST" "test -e '$REMOTE_PATH'"; then
		echo "Fetching artifact from $REMOTE_HOST:$REMOTE_PATH -> $LOCAL_ROOT/"
		rsync -azP --relative "$REMOTE_USER@$REMOTE_HOST:$REMOTE_ROOT/./$artifact" "$LOCAL_ROOT/"
		echo "Done. Downloaded $artifact"
	else
		echo "Skipping missing artifact: $REMOTE_PATH"
	fi
done