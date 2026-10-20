#!/usr/bin/env bash
# Build, test, and run every benchmark notebook on this machine.
#
#   ./run_all.sh            full run (about 10-30 minutes depending on the machine)
#   QUICK=1 ./run_all.sh    small data, a few minutes, for checking that everything works
#
# Executed notebooks and numbers go to notebooks/results/<this machine>/, so the
# committed notebooks are never overwritten. Commit that folder to share results.
set -euo pipefail
cd "$(dirname "$0")"

command -v cargo >/dev/null || { echo "Rust is required. Install it from https://rustup.rs, then run this again."; exit 1; }
PY=${PYTHON:-python3}
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ is required"'

[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/pip install -q "numpy>=2" scikit-learn maturin pytest jupyter nbconvert matplotlib pandas
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  .venv/bin/pip install -q mlx   # the GPU engine; Apple silicon only
fi

echo "== building the Rust engines for this CPU"
.venv/bin/maturin develop --release
echo "== parity tests against scikit-learn"
.venv/bin/python -m pytest -q

SLUG=$(.venv/bin/python -c 'import sys; sys.path.insert(0, "notebooks"); from bench_utils import machine_slug; print(machine_slug())')
OUT="notebooks/results/$SLUG/executed"
mkdir -p "$OUT"
echo "== running notebooks on $SLUG (QUICK=${QUICK:-0})"
for nb in notebooks/0*.ipynb; do
  echo "   $(basename "$nb")"
  .venv/bin/jupyter nbconvert --to notebook --execute "$nb" --output-dir "$OUT" \
    --ExecutePreprocessor.timeout=7200 --log-level=WARN
done
echo "== done. Numbers: notebooks/results/$SLUG/   Executed notebooks: $OUT/"
