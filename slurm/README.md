# Running the OWL benchmark on SLURM

The supplied array job reads the 423 IDs in
`logs/no-internet-result/retained_ids.txt` and splits them into seventeen
independent shards of at most 25 samples. Up to two shards run at a time
(`--array=0-16%2`), and each shard uses two sample workers. Therefore, at most
four OWL agents share the cluster's public IP at any time.

The checked-in launcher lets SLURM select compute nodes. Use `--nodelist` at
submission time only when a particular node is required.

With the default 40-minute OWL timeout, 41-minute Inspect attempt timeout, and
250-model-call limit, a 25-sample shard has 13 worker waves. Its timeout-based
upper estimate is 8 hours 53 minutes, and the SLURM allocation enforces a
10-hour hard limit.

## One-time setup on the shared filesystem

Use Python 3.11 or 3.12. Load site-specific modules for Python, FFmpeg, and
Node.js first if your cluster provides them, then install the locked runtimes:

```bash
uv sync --frozen --extra dev
uv sync --frozen --project owl_runtime
export PLAYWRIGHT_BROWSERS_PATH="$PWD/.playwright-browsers"
uv run --project owl_runtime playwright install chromium
```

Playwright Chromium also needs its Linux shared libraries. On a cluster where
compute images do not provide them, ask the administrator to install the
Playwright dependencies or run the job from an Apptainer image that contains
them. `playwright install --with-deps chromium` is suitable only where you have
the required package-manager privileges.

Create `.env` in the repository and restrict its permissions:

```bash
chmod 600 .env
```

At minimum, the default configuration requires `OPENROUTER_API_KEY` and
`HYPERBROWSECOMP_KEY`. Compute nodes must have outbound access to OpenRouter,
Hugging Face, DuckDuckGo, Wikipedia, and the websites/media being evaluated.

## Submit and monitor

Submit from the repository checkout:

```bash
sbatch slurm/owl_retained_array.sbatch
squeue -u "$USER"
```

For the GLM 5.3 Flash validation, submit the dedicated job. It runs dataset
problems 1–8 together as eight concurrent OWL workers on one `ws-ia` node:

```bash
sbatch slurm/owl_glm_5_3_flash_8.sbatch
```

SLURM executes a spooled copy of the batch script, so the launcher uses
`SLURM_SUBMIT_DIR` as the repository root. If submission must happen from
elsewhere, export the shared checkout explicitly:

```bash
sbatch --export=ALL,HYPERBC_REPO_ROOT=/shared/path/hyper-browsecomp \
  /shared/path/hyper-browsecomp/slurm/owl_retained_array.sbatch
```

Add site-specific `#SBATCH --partition` and `#SBATCH --account` directives if
your cluster requires them, or pass them to `sbatch` on the command line.

Results are isolated by array job and shard:

```text
logs/owl/slurm/<array-job-id>/shard-<index>/
  config.yaml
  sample_ids.txt
  eval/*.eval
  traces/*.json
  traces/*.log
```

Scheduler stdout and stderr are written in the submission directory as
`slurm-hyperbc-owl-<job>_<index>.out` and `.err`.

## Throughput choices

The default `%2` permits two SLURM shards at a time. Each shard runs two OWL
samples, giving four concurrent agents in total. This requires two allocations
of 8 CPUs and 20 GiB each and doubles the traffic from the cluster's public IP.

To return to the lower-risk two-agent configuration, submit with:

```bash
sbatch --array=0-16%1 slurm/owl_retained_array.sbatch
```

Use `%1` if the provider or websites begin returning rate-limit, CAPTCHA, or
access-denied responses.

The base config and timeouts can be overridden at submission:

```bash
sbatch --export=ALL,OWL_TASK_TIMEOUT_SECONDS=1200,INSPECT_ATTEMPT_TIMEOUT=1260 \
  slurm/owl_retained_array.sbatch
```

For GLM 5.3 Flash, run two exclusive one-node allocations with eight OWL
samples on each node (16 concurrent samples total):

```bash
sbatch --partition=ws-ia --array=0-16%2 --nodes=1 --exclusive \
  --nodelist=ws-l4-013,ws-l5-011 \
  --cpus-per-task=16 --mem=64G \
  --export=ALL,BASE_CONFIG=configs/owl_full/openrouter_glm_5_3_flash.yaml,PARALLEL_SAMPLES=8 \
  slurm/owl_retained_array.sbatch
```

The `%2` array throttle limits the run to two active allocations. Combined
with `--exclusive`, the two active shards occupy `ws-l4-013` and `ws-l5-011`,
one per node, with eight OWL workers on each. Both nodes currently expose 48
CPUs and 230 GB RAM; the job requests 16 CPUs and 64 GB per allocation.

The model-call guard can also be overridden:

```bash
sbatch --export=ALL,OWL_MAX_MODEL_CALLS=250 \
  slurm/owl_retained_array.sbatch
```

If a shard is interrupted, locate its partial `.eval` and resume that shard
with its generated config:

```bash
uv run bash run_eval.sh resume \
  logs/owl/slurm/<array-job-id>/shard-<index>/config.yaml \
  logs/owl/slurm/<array-job-id>/shard-<index>/eval/<partial-log>.eval
```

Submit the resume command as a small separate batch job rather than running it
on a login node. The runner retries only missing or errored samples from that
shard.

## Retry all errored samples

After one or more source arrays finish, retry only samples whose OWL traces
ended with `status: error`. The retry array preserves the retained-ID order,
deduplicates failures across source jobs, and uses shards of at most 20 samples
so 40-minute attempts remain within an eight-hour allocation.

```bash
sbatch \
  --dependency=afterany:202169:CONTINUATION_JOB_ID \
  --export=ALL,RETRY_SOURCE_JOB_IDS=202169:CONTINUATION_JOB_ID \
  slurm/owl_failed_retry_array.sbatch
```

The retry job runs at most two shards with two workers each. Empty array
elements exit successfully, so the fixed `0-21` range safely covers up to all
423 retained IDs. Keep retry results separate from the original Pass@1 result;
a merged score is a retry-assisted result.
