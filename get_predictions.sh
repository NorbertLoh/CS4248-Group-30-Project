#!/bin/bash
# Script to fetch all files from remote output directory
# Usage: ./get_predictions.sh

REMOTE_USER="norbert"
REMOTE_HOST="xlogin.comp.nus.edu.sg"
REMOTE_BASE_DIR="CS4248/cs4248"  # Adjusted to match your folder structure

REMOTE_FOLDER="~/$REMOTE_BASE_DIR/datapreparation/output/"
LOCAL_FOLDER="./datapreparation/output/"

# Ensure local directory exists
mkdir -p "$LOCAL_FOLDER"

echo "Syncing all files from $REMOTE_HOST:$REMOTE_FOLDER to $LOCAL_FOLDER"

# -a: archive mode (preserves permissions/timestamps)
# -v: verbose
# -z: compress data during transfer (great for .jsonl text files)
# -P: show progress bar
rsync -avzP "$REMOTE_USER@$REMOTE_HOST:$REMOTE_FOLDER" "$LOCAL_FOLDER"