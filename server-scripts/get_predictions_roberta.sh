#!/usr/bin/env bash
set -euo pipefail

# Script to fetch outputs produced by cara/classification/train_roberta.py
# Usage: ./get_predictions_roberta.sh

REMOTE_USER="${REMOTE_USER:-norbert}"
REMOTE_HOST="${REMOTE_HOST:-xlogin.comp.nus.edu.sg}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-CS4248/cs4248}"

REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-metameme_roberta_model}"
REMOTE_DIR="~/$REMOTE_BASE_DIR/$REMOTE_OUTPUT_DIR"
LOCAL_BASE_DIR="$(cd "$(dirname "$0")" && pwd)/$REMOTE_OUTPUT_DIR"

mkdir -p "$LOCAL_BASE_DIR"

LOCAL_PARENT_DIR="$(dirname "$LOCAL_BASE_DIR")"
mkdir -p "$LOCAL_PARENT_DIR"
echo "Fetching model directory from $REMOTE_HOST:$REMOTE_DIR -> $LOCAL_PARENT_DIR/"
rsync -azP "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR" "$LOCAL_PARENT_DIR/"
echo "Done. Downloaded directory $REMOTE_OUTPUT_DIR"