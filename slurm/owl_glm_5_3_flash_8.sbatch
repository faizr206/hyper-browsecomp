#!/usr/bin/env bash
#SBATCH --job-name=hyperbc-glm53-8
#SBATCH --partition=ws-ia
#SBATCH --time=01:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=ws-l4-013
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

if [[ -n "${HYPERBC_REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd -- "$HYPERBC_REPO_ROOT" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run_eval.sh" ]]; then
  REPO_ROOT="$(cd -- "$SLURM_SUBMIT_DIR" && pwd)"
else
  echo "Submit from the repository root or export HYPERBC_REPO_ROOT." >&2
  exit 2
fi
cd "$REPO_ROOT"

CONFIG="configs/owl_full/openrouter_glm_5_3_flash_8.yaml"
if [[ ! -f "$CONFIG" || ! -f ".env" ]]; then
  echo "Missing $CONFIG or $REPO_ROOT/.env." >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not available on PATH." >&2
  exit 2
fi

export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$REPO_ROOT/.playwright-browsers}"

echo "SLURM job: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Config: ${CONFIG}"
echo "Samples: 1-8 (8 concurrent OWL workers)"

uv run bash run_eval.sh "$CONFIG"
