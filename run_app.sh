#!/usr/bin/env bash
# Start the TXM crack-detection app. Nothing to configure.
#
#   ./run_app.sh
#
# Creates a virtualenv on first run, installs dependencies, then serves the app
# at http://127.0.0.1:8800 -- drag images in, correct them, press Retrain.
#
# Deliberately the same shape as the SEM project's run_app.sh so the two behave
# identically: same venv-on-first-run, same requirements stamp, same "warn about
# the optional heavy dependency rather than failing" behaviour.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8800}"
VENV="${VENV:-.venv}"

if [ ! -d "$VENV" ]; then
  echo "==> creating virtualenv in $VENV (first run only)"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"

# Only reinstall when requirements change -- a stamp file keeps startup fast.
STAMP="$VENV/.req-stamp"
if [ ! -f "$STAMP" ] || ! cmp -s requirements.txt "$STAMP"; then
  echo "==> installing dependencies (this takes a few minutes the first time)"
  python3 -m pip install --quiet --upgrade pip
  python3 -m pip install --quiet -r requirements.txt
  cp requirements.txt "$STAMP"
fi

mkdir -p app_data/images app_data/models models dataset_cache paint/corrections

# Expand the compressed ground truth / corrections a distributed checkout ships
# with, and build the 17-feature stacks that were deliberately not shipped.
# Idempotent -- a no-op on every run after the first.
python3 code/unpack_package.py || echo "==> WARNING: unpack step failed; retrain validation may be unavailable"

# SAM is optional in the sense that the app still runs without it -- it falls back
# to the 17-feature model alone. Say so rather than crashing on first predict.
if ! python3 -c "import torch" 2>/dev/null; then
  echo "==> NOTE: PyTorch not installed, so SAM is unavailable."
  echo "    The app still runs on the 17-feature model alone"
  echo "    (mean IoU 0.744 vs 0.821 for the SAM ensemble)."
  echo "    To enable it:  pip install torch transformers"
fi

if [ ! -f models/pixel_sam_hybrid.joblib ] && [ ! -f models/pixel_hgb_final.joblib ]; then
  echo "==> WARNING: no model found in models/."
  echo "    The app will start but cannot predict until one is present."
fi

# KMP_DUPLICATE_LIB_OK: scikit-learn and torch each vendor an OpenMP runtime, and
# loading both in one process aborts on macOS without this.
export KMP_DUPLICATE_LIB_OK=TRUE
# Let the SAM pass use all of unified memory rather than a fraction of it.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"

echo "==> serving on http://127.0.0.1:$PORT"
echo "    drop images onto the window; press Ctrl-C to stop"
exec python3 app/server.py
