#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR=".venv"
PYBIN="$VENV_DIR/bin/python"
TRANSFORMERS_PIN="transformers==4.35.2"

echo "Starting remote environment setup at $(date)"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

if [ ! -x "$PYBIN" ]; then
  echo "Virtualenv python missing at $PYBIN" >&2
  exit 1
fi

# Ensure pip exists in virtualenv
if ! "$PYBIN" -m pip --version >/dev/null 2>&1; then
  echo "pip missing in virtualenv; trying ensurepip"
  "$PYBIN" -m ensurepip --upgrade || true
fi
if ! "$PYBIN" -m pip --version >/dev/null 2>&1; then
  echo "pip still missing in virtualenv" >&2
  exit 1
fi

"$PYBIN" -m pip install --upgrade pip setuptools wheel

if [ -f requirements.txt ]; then
  echo "Installing requirements from requirements.txt"
  "$PYBIN" -m pip install -r requirements.txt
fi

# Ensure a working transformers install with pipeline API
set +e
"$PYBIN" - <<'PY'
import sys
import transformers
ok = hasattr(transformers, "pipeline")
print("transformers version:", getattr(transformers, "__version__", "unknown"))
print("pipeline available:", ok)
sys.exit(0 if ok else 2)
PY
rc=$?
set -e
if [ $rc -ne 0 ]; then
  echo "Fixing transformers with pinned reinstall: ${TRANSFORMERS_PIN}"
  "$PYBIN" -m pip install --upgrade --force-reinstall "${TRANSFORMERS_PIN}"
fi

# If GPU is visible, ensure CUDA-enabled torch build is present
if command -v nvidia-smi >/dev/null 2>&1; then
  set +e
  "$PYBIN" - <<'PY'
import sys
import torch
print("torch version:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
sys.exit(0 if torch.cuda.is_available() else 3)
PY
  trc=$?
  set -e
  if [ $trc -ne 0 ]; then
    echo "Installing CUDA torch wheel (cu121)"
    "$PYBIN" -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
  fi
fi

echo "Remote environment setup complete at $(date)"
