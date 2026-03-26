#!/usr/bin/env bash
set -euo pipefail

# Configuration
REMOTE_USER="norbert"
REMOTE_HOST="xlogin.comp.nus.edu.sg"
REMOTE_BASE_DIR="CS4248"  # remote base directory inside user's home

PROJECT_NAME="$(basename "$(pwd)")"
REMOTE_TARGET="$REMOTE_BASE_DIR/$PROJECT_NAME"

echo "Syncing project to ${REMOTE_USER}@${REMOTE_HOST}:~/${REMOTE_TARGET} (excluding local virtualenvs)"
# exclude common virtualenv folders to avoid copying heavy local envs
rsync -azP \
	--exclude '.git' \
	--exclude '__pycache__' \
	--exclude '*.pyc' \
	--exclude '.venv' \
	--exclude 'venv' \
	--exclude '.env' \
	--exclude 'models' \
	--exclude 'results' \
	. "${REMOTE_USER}@${REMOTE_HOST}:~/${REMOTE_TARGET}"

echo "Sync complete — not submitting any remote jobs (deploy-only mode)."
