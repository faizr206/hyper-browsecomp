#!/usr/bin/env bash
#SBATCH --job-name=hyperbc-owl-retry
#SBATCH --array=0-21%2
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --nodelist=ws-l4-013
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

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

BASE_CONFIG="${BASE_CONFIG:-configs/owl_full/openrouter_gemini_flash.yaml}"
RETAINED_IDS_FILE="${RETAINED_IDS_FILE:-logs/no-internet-result/retained_ids.txt}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-20}"
PARALLEL_SAMPLES="${PARALLEL_SAMPLES:-2}"
OWL_TASK_TIMEOUT_SECONDS="${OWL_TASK_TIMEOUT_SECONDS:-2400}"
INSPECT_ATTEMPT_TIMEOUT="${INSPECT_ATTEMPT_TIMEOUT:-2460}"
OWL_MAX_MODEL_CALLS="${OWL_MAX_MODEL_CALLS:-250}"

if [[ -z "${RETRY_SOURCE_JOB_IDS:-}" ]]; then
  echo "RETRY_SOURCE_JOB_IDS must contain colon-separated source SLURM job IDs." >&2
  exit 2
fi
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" || -z "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  echo "This script must be submitted as a SLURM array job." >&2
  exit 2
fi
if ! [[ "$SAMPLES_PER_SHARD" =~ ^[1-9][0-9]*$ ]]; then
  echo "SAMPLES_PER_SHARD must be a positive integer." >&2
  exit 2
fi
if [[ ! -f "$BASE_CONFIG" || ! -f "$RETAINED_IDS_FILE" || ! -f ".env" ]]; then
  echo "Missing base config, retained-ID file, or repository .env." >&2
  exit 2
fi

RUN_ROOT="logs/owl/slurm/${SLURM_ARRAY_JOB_ID}"
SHARD_ROOT="${RUN_ROOT}/shard-${SLURM_ARRAY_TASK_ID}"
FAILED_IDS_FILE="${SHARD_ROOT}/failed_ids.txt"
SHARD_CONFIG="${SHARD_ROOT}/config.yaml"
SHARD_LOG_DIR="${SHARD_ROOT}/eval"
SHARD_TRACE_DIR="${SHARD_ROOT}/traces"
mkdir -p "$SHARD_LOG_DIR" "$SHARD_TRACE_DIR"

IFS=':' read -r -a SOURCE_JOB_IDS <<< "$RETRY_SOURCE_JOB_IDS"
COLLECT_ARGS=()
for source_job_id in "${SOURCE_JOB_IDS[@]}"; do
  if ! [[ "$source_job_id" =~ ^[0-9]+$ ]]; then
    echo "Invalid source SLURM job ID: $source_job_id" >&2
    exit 2
  fi
  COLLECT_ARGS+=(--run-root "logs/owl/slurm/${source_job_id}")
done

uv run python scripts/collect_failed_sample_ids.py \
  "${COLLECT_ARGS[@]}" \
  --retained-ids "$RETAINED_IDS_FILE" \
  --output "$FAILED_IDS_FILE"

TOTAL_SAMPLES="$(wc -l < "$FAILED_IDS_FILE" | tr -d ' ')"
SHARD_START=$((SLURM_ARRAY_TASK_ID * SAMPLES_PER_SHARD + 1))
if (( SHARD_START > TOTAL_SAMPLES )); then
  echo "Retry array index $SLURM_ARRAY_TASK_ID has no failed IDs; exiting successfully."
  exit 0
fi
SHARD_END=$((SHARD_START + SAMPLES_PER_SHARD - 1))
if (( SHARD_END > TOTAL_SAMPLES )); then
  SHARD_END="$TOTAL_SAMPLES"
fi

export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$REPO_ROOT/.playwright-browsers}"

uv run python scripts/make_slurm_shard_config.py \
  --base "$BASE_CONFIG" \
  --output "$SHARD_CONFIG" \
  --sample-ids-source "$FAILED_IDS_FILE" \
  --start "$SHARD_START" \
  --end "$SHARD_END" \
  --log-dir "$SHARD_LOG_DIR" \
  --trace-dir "$SHARD_TRACE_DIR" \
  --task-timeout "$OWL_TASK_TIMEOUT_SECONDS" \
  --attempt-timeout "$INSPECT_ATTEMPT_TIMEOUT" \
  --max-model-calls "$OWL_MAX_MODEL_CALLS" \
  --parallel "$PARALLEL_SAMPLES"

echo "Retry source jobs: ${RETRY_SOURCE_JOB_IDS}"
echo "Failed-ID positions: ${SHARD_START}-${SHARD_END} of ${TOTAL_SAMPLES}"
echo "Parallel samples: ${PARALLEL_SAMPLES}"
echo "OWL timeout: ${OWL_TASK_TIMEOUT_SECONDS}s"
echo "Inspect attempt timeout: ${INSPECT_ATTEMPT_TIMEOUT}s"
echo "OWL model-call limit: ${OWL_MAX_MODEL_CALLS}"
echo "Output directory: ${SHARD_ROOT}"

# Twenty retry samples create at most ten 2-worker waves. At 2,460 seconds per
# wave, the timeout-bound runtime is 6h50m, within the eight-hour allocation.
uv run bash run_eval.sh "$SHARD_CONFIG"
