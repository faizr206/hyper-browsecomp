#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ $# -ne 1 ]]; then
  echo "usage: bash run_eval.sh CONFIG.yaml" >&2
  exit 2
fi

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"
PYTHON_BIN="python"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
  export PATH="$PWD/.venv/bin:$PATH"
fi

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

"$PYTHON_BIN" -m hyper_browsecomp.runner "$1"
