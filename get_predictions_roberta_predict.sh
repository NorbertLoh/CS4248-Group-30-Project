#!/usr/bin/env bash
set -euo pipefail

# Script to fetch outputs produced by cara/sft/rationale_roberta.py
# Usage: ./get_predictions_roberta_predict.sh

REMOTE_USER="${REMOTE_USER:-norbert}"
REMOTE_HOST="${REMOTE_HOST:-xlogin.comp.nus.edu.sg}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-CS4248/cs4248}"

REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-datapreparation/output}"
REMOTE_OUTPUT_FILE="${REMOTE_OUTPUT_FILE:-final_roberta_predictions_dev.jsonl}"
REMOTE_DIR="~/$REMOTE_BASE_DIR/$REMOTE_OUTPUT_DIR"
REMOTE_FILE="$REMOTE_DIR/$REMOTE_OUTPUT_FILE"
LOCAL_BASE_DIR="$(cd "$(dirname "$0")" && pwd)/$REMOTE_OUTPUT_DIR"

mkdir -p "$LOCAL_BASE_DIR"

if [ -n "$REMOTE_OUTPUT_FILE" ]; then
	echo "Fetching output file from $REMOTE_HOST:$REMOTE_FILE -> $LOCAL_BASE_DIR/"
	rsync -azP "$REMOTE_USER@$REMOTE_HOST:$REMOTE_FILE" "$LOCAL_BASE_DIR/"
	echo "Done. Downloaded $REMOTE_OUTPUT_FILE"
else
	LOCAL_PARENT_DIR="$(dirname "$LOCAL_BASE_DIR")"
	mkdir -p "$LOCAL_PARENT_DIR"
	echo "Fetching output directory from $REMOTE_HOST:$REMOTE_DIR -> $LOCAL_PARENT_DIR/"
	rsync -azP "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR" "$LOCAL_PARENT_DIR/"
	echo "Done. Downloaded directory $REMOTE_OUTPUT_DIR"
fi