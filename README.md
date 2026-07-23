# hyper-browsecomp

`hyper-browsecomp` is a compact Inspect AI harness for BrowseComp-style web research evals. It runs JSONL datasets from `data/`, gives the model web search and fetch tools, and scores answers with a BrowseComp-style judge.

## Install

```bash
uv sync --extra dev
```

This creates a local `.venv` and installs the project with its development
dependencies from `pyproject.toml` and `uv.lock`.

## Environment

Store API key values in `.env`:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
XAI_API_KEY=...
MISTRAL_API_KEY=...
PERPLEXITY_API_KEY=...
DEEPSEEK_API_KEY=...
OPENROUTER_API_KEY=...
DASHSCOPE_API_KEY=...
```

Each YAML config chooses which environment variable to read for the main model
and scorer:

```yaml
provider: openai
model_name: gpt-5.4-mini
model_api_key_env: OPENAI_API_KEY
model_base_url: https://api.openai.com/v1

scorer_provider: openrouter
scorer_model_name: openai/gpt-5.4-mini
scorer_api_key_env: OPENROUTER_API_KEY
scorer_base_url: https://openrouter.ai/api/v1
```

Native Inspect providers are used for `openai`, `anthropic`, `gemini`, `grok`,
`mistral`, and `perplexity`. `gemini` is accepted in YAML and resolved to
Inspect's `google/<model>` provider path. Other provider names, such as `qwen`,
continue to use Inspect's OpenAI-compatible `openai-api/<provider>/<model>` path.

Web backends:

```bash
EXA_API_KEY=...
FIRECRAWL_API_KEY=...
```

The runner maps the selected key values and YAML base URLs into Inspect's
provider-specific environment variables before invoking `inspect eval`. For
example, `model_api_key_env: OPENAI_API_KEY` with `provider: openai` sets
`OPENAI_API_KEY`; `provider: gemini` writes `GOOGLE_API_KEY` and
`GOOGLE_BASE_URL`; `provider: grok` writes `XAI_API_KEY` and `XAI_BASE_URL`.

## Run

Use the single shell entrypoint with any YAML config:

```bash
uv run bash run_eval.sh configs/dev_openai.yaml
uv run bash run_eval.sh configs/dev_anthropic.yaml
uv run bash run_eval.sh configs/dev_gemini.yaml
uv run bash run_eval.sh configs/dev_grok.yaml
uv run bash run_eval.sh configs/dev_mistral.yaml
uv run bash run_eval.sh configs/dev_perplexity.yaml
uv run bash run_eval.sh configs/dev_qwen.yaml
uv run bash run_eval.sh configs/web.yaml
uv run bash run_eval.sh configs/web_code.yaml
```

The provider smoke configs all run the single-question `data/dev.jsonl` dataset:

```text
configs/dev_openai.yaml      OPENAI_API_KEY      https://api.openai.com/v1
configs/dev_anthropic.yaml   ANTHROPIC_API_KEY   https://api.anthropic.com
configs/dev_gemini.yaml      GOOGLE_API_KEY      https://generativelanguage.googleapis.com
configs/dev_grok.yaml        XAI_API_KEY         api.x.ai
configs/dev_mistral.yaml     MISTRAL_API_KEY     https://api.mistral.ai
configs/dev_perplexity.yaml  PERPLEXITY_API_KEY  https://api.perplexity.ai
configs/dev_qwen.yaml        DASHSCOPE_API_KEY   https://dashscope-intl.aliyuncs.com/compatible-mode/v1
```

Each run writes `.eval` logs under `logs/` and then renames the newest run log to:

```text
{data_name}_{model_name}_{timestamp}_{id}.eval
```

If a run stops in the middle, resume it by passing the original config and the
partial `.eval` log:

```bash
uv run bash run_eval.sh resume configs/web.yaml logs/dev_gpt-5.4-mini_20260723T120000Z_abc123.eval
```

The resume command reads the original run's sample selection from the log, finds
samples that are missing from the log or ended with an error, and starts a new
run with `--sample-id` limited to those unfinished samples.

## Config

Configs are flat YAML files. Example:

```yaml
provider: deepseek
model_name: deepseek-chat
scorer_provider: openrouter
scorer_model_name: openai/gpt-5.4-mini
model_api_key_env: DEEPSEEK_API_KEY
model_base_url: https://api.deepseek.com
scorer_api_key_env: OPENROUTER_API_KEY
scorer_base_url: https://openrouter.ai/api/v1
data_path: data/dev.jsonl
# Optional: 1-based inclusive sample numbers. "1-2" evaluates samples 1 and 2.
sample_range: "1-2"
tool_profile: web
search_backend: internal
fetch_backend: none
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
- `model_api_key_env`, `model_base_url`
- `scorer_api_key_env`, `scorer_base_url`
- `tool_profile`: `web` or `web_code`
- `search_backend`: `internal`, `exa`, `firecrawl`, or `none`
- `fetch_backend`: `exa`, `firecrawl`, or `none`
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

- `search_backend: internal` uses Inspect's standard provider-native web search
  for `openai`, `anthropic`, `gemini`, `grok`, `mistral`, and `perplexity`
- `search_backend: none` disables search
- `fetch_backend: none` disables page fetching
- `web` exposes the configured web tools
- `web_code` adds `bash()` and `python()`

The agent prompt adapts to the configured search and fetch backends, so it does
not ask the model to call tools that are disabled.
