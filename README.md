# hyper-browsecomp

`hyper-browsecomp` is a compact Inspect AI harness for BrowseComp-style web research evals. It runs JSONL datasets from `data/`, gives the model web search and fetch tools, and scores answers with a BrowseComp-style judge.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Environment

The runner supports one generic env contract for the main model:

```bash
MODEL_PROVIDER=deepseek
MODEL_API_KEY=...
MODEL_BASE_URL=https://api.deepseek.com
```

Optional separate scorer credentials:

```bash
SCORER_PROVIDER=openrouter
SCORER_API_KEY=...
SCORER_BASE_URL=https://openrouter.ai/api/v1
```

Web backends:

```bash
EXA_API_KEY=...
FIRECRAWL_API_KEY=...
```

The runner maps the generic model env vars into Inspect's provider-specific `openai-api/<provider>/<model>` env names before invoking `inspect eval`.

## Run

Use the single shell entrypoint with any YAML config:

```bash
bash run_eval.sh configs/web.yaml
bash run_eval.sh configs/web_code.yaml
```

Each run writes `.eval` logs under `logs/` and then renames the newest run log to:

```text
{data_name}_{model_name}_{timestamp}_{id}.eval
```

## Config

Configs are flat YAML files. Example:

```yaml
provider: deepseek
model_name: deepseek-chat
scorer_provider: openrouter
scorer_model_name: openai/gpt-5.4-mini
data_path: data/dev.jsonl
# Optional: 1-based inclusive sample numbers. "1-2" evaluates samples 1 and 2.
sample_range: "1-2"
tool_profile: web
search_backend: exa
fetch_backend: firecrawl
max_steps: 12
inspect_max_samples_parallel: 1
inspect_model_max_retries: 1
inspect_attempt_timeout: 60
inspect_retry_on_error: 3
inspect_no_fail_on_error: true
inspect_continue_on_fail: true
no_sandbox: true
```

Key fields:

- `model` or `provider` + `model_name`
- `scorer_model` or `scorer_provider` + `scorer_model_name`
- `tool_profile`: `web` or `web_code`
- `search_backend`: `exa` or `firecrawl`
- `fetch_backend`: `exa` or `firecrawl`
- `data_path`, `sample_range`, `start_index`, `end_index`, `num_samples`
- `max_steps`
- `inspect_max_samples_parallel`, `inspect_model_max_retries`, `inspect_attempt_timeout`
- `inspect_retry_on_error`, `inspect_no_fail_on_error`, `inspect_continue_on_fail`
- `no_sandbox`

`sample_range` is the easiest way to run a subset. It uses 1-based inclusive sample
numbers, so `sample_range: "1-2"` runs dataset samples 1 and 2. The older
`start_index`, `end_index`, and `num_samples` fields are still available, but they
use Python slicing semantics.

By default the runner caps model API retries, retries sample errors 3 times,
records failed samples without failing the whole run, and continues after failed
samples. One bad sample will not stop the whole evaluation.

Removed from the old repo:

- OpenRouter-specific modes and adapters
- Browser/PDF/OCR/video tools
- dataset conversion scripts
- multiple wrapper scripts

## Tool behavior

- `web` exposes only `web_search` and `web_fetch`
- `web_code` adds `bash()` and `python()`

The agent prompt explicitly tells the model to use `web_search` and `web_fetch` for internet retrieval and only use `bash` or `python` when web tools are insufficient.
