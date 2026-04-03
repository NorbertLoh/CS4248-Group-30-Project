#!/usr/bin/env bash
set -euo pipefail

# Script to fetch predictions produced by hateful-captioning/hateful-captioning.py
# Usage: ./get_predictions.sh

REMOTE_USER="${REMOTE_USER:-norbert}"
REMOTE_HOST="${REMOTE_HOST:-xlogin.comp.nus.edu.sg}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-CS4248/cs4248}"

REMOTE_FILE="~/$REMOTE_BASE_DIR/hateful-captioning/captions_output2.jsonl"
LOCAL_DIR="./hateful-captioning"

mkdir -p "$LOCAL_DIR"

echo "Fetching predictions from $REMOTE_HOST:$REMOTE_FILE -> $LOCAL_DIR/"
rsync -azP "$REMOTE_USER@$REMOTE_HOST:$REMOTE_FILE" "$LOCAL_DIR/"

echo "Done. Downloaded captions_output1.jsonl"