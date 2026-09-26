# HyperBrowseComp evaluation harness

`hyper-browsecomp` is an [Inspect AI](https://inspect.aisi.org.uk/) harness for
BrowseComp-style web research evaluations. It supports local JSONL fixtures and
the encrypted `afaji/HyperBrowseComp` Hugging Face dataset, with two execution
paths:

- Inspect ReAct agents using configurable search and fetch backends
- an isolated OWL/CAMEL Workforce with browser and multimodal tools

Both paths use the same dataset loader, BrowseComp-style judge, resumable runs,
and `.eval` output format.

## Quick start

```bash
uv sync --extra dev
cp .env.example .env
uv run pytest -q
```

This creates a local `.venv` and installs the project with its development
dependencies from `pyproject.toml` and `uv.lock`. Add only the credentials
required by your selected config to `.env`.

### Optional OWL runtime

The OWL harness has an isolated runtime because OWL/CAMEL and Inspect require
incompatible Pydantic versions. Install its locked dependencies and Chromium
once before an OWL run:

```bash
uv sync --project owl_runtime
uv run --project owl_runtime playwright install chromium
```

Local video decoding also needs FFmpeg on `PATH`. YouTube downloads use the
locked `yt-dlp-ejs` dependency and enable Node.js when `node` is on `PATH`.

The test suite covers configuration loading, dataset handling, runner command
construction, scoring-task setup, web tools, and the isolated OWL harness.

## Repository layout

- `configs/`: development, full-benchmark, backend, and model configurations
- `data/`: small smoke-test fixtures; the full encrypted dataset is loaded from
  Hugging Face at runtime
- `src/hyper_browsecomp/`: Inspect task, runner, scorer, and web-tool code
- `owl_runtime/`: separately locked OWL/CAMEL worker and multimodal utilities
- `scripts/`: result recovery and trace-report helpers
- `slurm/`: cluster launchers and operating notes
- `tests/`: offline unit tests

Generated logs, traces, temporary files, local environments, and downloaded
datasets are intentionally excluded from version control; see `.gitignore`.

## Credentials and datasets

Start from `.env.example` and fill in only the keys used by the selected model,
scorer, and web backends. The file is ignored by Git.

Each YAML config names the environment variables used by the main model and
scorer:

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

`minimax` uses the repository's direct MiniMax adapter for `MiniMax-M3`. It
supports native search and direct Exa in separate resumable 423-question
configs under `configs/internal_full/` and `configs/exa_full/`.

Web backends:

```bash
EXA_API_KEY=...
FIRECRAWL_API_KEY=...
```

Only set the backend key for backends selected by the YAML config. For example,
`search_backend: exa` and `fetch_backend: exa` need `EXA_API_KEY`, while
`fetch_backend: none` does not need `FIRECRAWL_API_KEY`.

For the full HyperBrowseComp dataset, put the hex-encoded AES-256-GCM master key
in `.env`:

```bash
HYPERBROWSECOMP_KEY=...
```

Then set the config dataset path to the repo id:

```yaml
data_path: afaji/HyperBrowseComp
```

The loader downloads the `test` split with `datasets.load_dataset()`, decrypts
`question` and `answer` in memory, and maps each row to the same Inspect sample
shape as the local JSONL datasets.

The runner maps the selected key values and YAML base URLs into Inspect's
provider-specific environment variables before invoking `inspect eval`. For
example, `model_api_key_env: OPENAI_API_KEY` with `provider: openai` sets
`OPENAI_API_KEY`; `provider: gemini` writes `GOOGLE_API_KEY` and
`GOOGLE_BASE_URL`; `provider: grok` writes `XAI_API_KEY` and `XAI_BASE_URL`.

## Running evaluations

All evaluations use the same entrypoint:

```bash
uv run bash run_eval.sh path/to/config.yaml
```

### ReAct examples

```bash
# One-question development fixtures
uv run bash run_eval.sh configs/exa_dev/dev_openai.yaml
uv run bash run_eval.sh configs/internal_search_dev/dev_openai.yaml

# Full encrypted dataset
uv run bash run_eval.sh configs/exa_full/openai.yaml

# Web tools plus bash and Python
uv run bash run_eval.sh configs/web_code.yaml
```

Development configs use `data/dev.jsonl`. Full configs load
`afaji/HyperBrowseComp` and require `HYPERBROWSECOMP_KEY`.

### OWL examples

| Purpose | Config |
| --- | --- |
| Gemini image smoke test | `configs/owl_dev/openrouter_gemini_flash.yaml` |
| Gemini video smoke test | `configs/owl_dev/openrouter_gemini_flash_video.yaml` |
| Gemini five-media validation | `configs/owl_dev/openrouter_gemini_flash_media.yaml` |
| Gemini full benchmark | `configs/owl_full/openrouter_gemini_flash.yaml` |

Run the smoke test, then the model's validation config, before starting a full
benchmark. Gemini can use OpenRouter's native YouTube route through Google AI
Studio; GLM uses the local download-and-frame route.

The five-media validation exercises YouTube, Bilibili, image, rendered-PDF,
and audio tools. To test those tools directly without the planner or judge:

```bash
uv run python owl_runtime/smoke.py
uv run python owl_runtime/smoke.py --cases pdf,audio
```

Direct smoke checks save reports under `logs/owl/media-smoke/`. They validate
media access and expected answers, not benchmark accuracy.


### Outputs and resuming

Each run writes `.eval` logs under `logs/` and then renames the newest run log to:

```text
{data_name}_{model_name}_{timestamp}_{id}.eval
```

If a run stops in the middle, resume it by passing the original config and the
partial `.eval` log:

```bash
uv run bash run_eval.sh resume configs/exa_full/openai.yaml logs/HyperBrowseComp_gpt-5.4-mini_20260723T120000Z_abc123.eval
```

The resume command reads the original run's sample selection from the log, finds
samples that are missing from the log or ended with an error, and starts a new
run with `--sample-id` limited to those unfinished samples.

## Configuration

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
harness: react
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

### Core settings

Key fields:

- `model` or `provider` + `model_name`
- `harness`: `react` (the original Inspect agent) or `owl`
- `model_args`: provider-specific model arguments passed through Inspect's `-M` option
- `scorer_model` or `scorer_provider` + `scorer_model_name`
- `model_api_key_env`, `model_base_url`
- `scorer_api_key_env`, `scorer_base_url`
- `tool_profile`: `web` or `web_code`
- `search_backend`: `internal`, `exa`, `firecrawl`, or `none`
- `fetch_backend`: `exa`, `firecrawl`, or `none`
- `data_path`, `sample_ids_path`, `sample_ids_file`, `sample_range`, `start_index`,
  `end_index`, `num_samples`
- `max_steps`
- `inspect_max_samples_parallel`, `inspect_model_max_retries`, `inspect_attempt_timeout`
- `inspect_retry_on_error`, `inspect_no_fail_on_error`, `inspect_continue_on_fail`,
  `inspect_ctl_server`
- `no_sandbox`

### Sample selection

`sample_ids_path` passes the listed IDs to Inspect's `--sample-id` selector;
blank lines and comment lines beginning with `#` are ignored. `sample_ids_file`
filters the loaded task dataset in the file's exact order and rejects blank or
duplicate lines. Use only one sample-selection mechanism in a config.

`sample_range` uses 1-based inclusive positions, so `"1-2"` selects the first
two samples. The older `start_index`, `end_index`, and `num_samples` fields use
Python slicing semantics.

### OWL settings

OWL-specific fields are `owl_model_name`, `owl_api_key_env`, `owl_base_url`,
`owl_headless`, `owl_multimodal`, `owl_browser_round_limit`,
`owl_task_timeout_seconds`, `owl_timeout_scale`, `owl_finalize_reserve_seconds`,
`owl_max_external_tool_calls`, `owl_max_model_calls`,
`owl_model_max_retries`, `owl_max_tokens`, and `owl_reasoning_effort`.
`owl_trace_dir` defaults to `logs/owl/traces` and stores a
readable OWL console trajectory plus structured Workforce events for every
sample; their paths are also recorded in the sample's `.eval` metadata. If the
first three model fields are omitted, the main model's name, key environment
variable, and base URL are reused. `owl_browser_round_limit` caps each OWL
visual-browser call's interaction rounds; unlike the ReAct harness's
`max_steps`, it is not a global task-turn limit. The recommended Pass@1 profile
uses a 1,200-second hard timeout, reserves the final 120 seconds for synthesis,
allows 50 external tool calls, uses 180 model calls as a runaway guard, and
allows one retry for a transient OWL model request. It sets
`inspect_retry_on_error: 0` so an errored sample is not rerun; Inspect's
`inspect_model_max_retries` is separate and applies to Inspect-managed calls
such as the judge, not the OWL worker's OpenRouter calls.

For lower-throughput models, `owl_timeout_scale` multiplies OWL's nested agent,
browser, and media-tool timeouts without changing the separate task deadline.
The GLM full configs use `3.5`, based on 140 versus 40 tokens per second, and
scale the task deadline and finalization reserve by the same ratio.

### Traces and reports

OWL traces are created as soon as each sample starts. Tail the newest `.log`
under `logs/owl/traces/` to watch model calls, tool calls, worker responses, and
browser activity live. Its matching `.json` reports `status: running` during
execution and refreshes partial token, model-call, tool, and media counters
after every worker progress event. It is finalized with the completion,
termination reason, tool counts, and structured
Workforce lifecycle events. It also records wall time, turns/model calls,
input/output/reasoning/total tokens, model latency, per-role totals, and a
per-call timing/token breakdown. The worker streams compact progress snapshots
to the parent so a hard timeout still retains partial token, model-call, tool,
and media statistics. Aggregate statistics and trace paths are also
copied into `metadata.owl_harness` in the `.eval`. Each model call includes
actual image/video/audio content-part counts and available provider usage
details. `media_events` distinguishes a download, submitted frames/pages/audio,
and a received response. Extracted video frames and rendered PDF pages are kept
in the adjacent `<trace-stem>.media/` directory. Native YouTube processing has
no locally extracted frames; look for a `video_url` input and provider-reported
video tokens instead. A correct judge answer alone does not prove media access.
Inline media base64 is redacted from the console trace.
If Inspect itself is interrupted before solver cleanup completes, the runner
marks any newly created sidecar as `evaluation_process_ended` instead of
leaving it permanently at `status: running`; the console trajectory remains
available, though in-memory counters may be unavailable for that abrupt case.

After an evaluation finishes, render its `.eval` plus OWL JSON sidecars as one
readable Markdown report:

```bash
uv run --extra dev python scripts/render_owl_trace_report.py \
  logs/owl/<run>.eval --output logs/owl/<run>.traces.md
```

The report contains a five-sample summary, questions, reference and model
answers, judge explanations, model/tool/token statistics, media evidence, a
compact workforce timeline, and links back to every raw trace.

### OWL parallelism and SLURM

`inspect_max_samples_parallel` controls sample-level concurrency. Every sample
runs in its own OWL subprocess with its own temporary directory, browser,
budgets, and trace files. The supplied five-sample validation config uses two
workers; the full 500-sample config uses four workers on the validated
14-core/36-GiB host. Start at two on an untested machine, then increase to four
after watching memory and provider rate-limit errors. Eight workers may improve
throughput on larger hosts, but it also doubles simultaneous Chromium and model
traffic; it should be treated as a separate capacity test rather than the
default benchmark setting.

For clusters with per-job wall-time limits, the repository includes a
conservative SLURM array launcher for the 423 IDs retained by the no-internet
filter. It permits two simultaneous shards, each running two workers, with a
ten-hour hard limit per primary shard. Retry-only shards use smaller batches
and an eight-hour limit. See
[`slurm/README.md`](slurm/README.md) for installation, submission, output, and
resume instructions.

By default the runner caps model API retries, retries sample errors 3 times,
records failed samples without failing the whole run, and continues after failed
samples. One bad sample will not stop the whole evaluation.

## Harness and tool behavior

### OWL browser and media

With `harness: owl`, the solver launches the isolated OWL/CAMEL Workforce and
does not construct the Inspect ReAct agent or its Exa tools. The OWL web worker
uses DuckDuckGo, Wikipedia, and `BrowserToolkit`; its browser feeds annotated
page screenshots to the configured model. With `owl_multimodal: true`, a
separate worker also receives image, video, rendered-PDF, and audio analysis
tools powered by the same primary model. The single-model design follows the
MM-BrowseComp OWL setup; PDF/audio adapters and video routing are local
extensions, not a claim of exact paper reproduction. Dataset records can
provide an `image_urls` list; those URLs are appended to the task prompt for
visual inspection. This mirrors the OWL baseline described in
[MM-BrowseComp Appendix C.1](https://arxiv.org/abs/2508.13186): temperature 0,
one primary model for all tool functions, and input image URLs included in the
prompt.

Video routing is controlled by `OWL_VIDEO_BACKEND`:

- `auto` (default): for YouTube with Gemini through OpenRouter, send native
  `video_url` input to the same model via Google AI Studio (not Vertex).
  If the API fails, try local downloading. Other sites/models use local
  downloading immediately.
- `download`: use the site's native `yt-dlp` extractor, decode frames locally,
  and send those images to the configured vision model. Bilibili uses this
  path. It does not use Google to extract frames.
- `native`: require native YouTube input for YouTube URLs; unsupported models
  and provider errors fail explicitly. Other sites still use downloading.

The download path accepts video-only MP4 streams because frame analysis does
not need a separate audio track. If downloading fails, the existing Chromium
HTML5 playback fallback attempts a timestamped contact sheet. This fallback is
best-effort, not proof of universal site access. Local frame analysis does not
listen to the video's audio.

Access is anonymous by default; the native route uses the model provider's
access, not your browser session. If a site rate-limits the host or requires
sign-in, these optional `.env` settings are supported but disabled unless
configured. Cookies do not guarantee access:

```dotenv
# Preferred: path to a user-exported Netscape cookies.txt file.
OWL_YTDLP_COOKIE_FILE=/absolute/path/to/youtube-cookies.txt

# Alternative, opt-in local browser cookie import (examples: chrome,
# chrome:Profile 1, firefox). This reads authentication cookies from that
# browser profile, so enable it only when intended.
OWL_YTDLP_COOKIES_FROM_BROWSER=chrome

# Optional proxy used by both yt-dlp and the Chromium frame fallback.
OWL_YTDLP_PROXY=http://127.0.0.1:8080
```

The configured OWL model must accept image inputs for screenshot, image, or
video-frame or rendered-PDF analysis. Open-source vision-language models exposed
through a compatible endpoint can use the download-and-frame path without a
Google perception model. Endpoint image support is required; text-only models
cannot inspect media. `z-ai/glm-4.6v` through OpenRouter has been live-tested on
the YouTube download-and-frame path. This does not imply that every GLM variant,
provider, or video is compatible. Ready-to-run OWL configs are also supplied
for OpenRouter's multimodal `z-ai/glm-5.3-flash`; validate the smoke run against
the provider currently selected by OpenRouter before starting a full benchmark.

`ask_question_about_pdf(pdf_path, question, pages="1")` renders up to eight
specified 1-based pages per call and reports the total page count. It does not
silently inspect the rest of a document. `ask_question_about_audio` downloads a
public audio file (or reads a local one) and sends base64 `input_audio` to the
same primary model; that endpoint must support audio input. No auxiliary
Whisper/Gemini key is used. PDF/audio downloads have a 32 MiB limit.

### ReAct tools

- `search_backend: internal` uses Inspect's standard provider-native web search
  for `openai`, `anthropic`, `gemini`, `grok`, `minimax`, `mistral`, and
  `perplexity`
- `search_backend: none` disables search
- `fetch_backend: none` disables page fetching
- `web` exposes the configured web tools
- `web_code` adds `bash()` and `python()`

The agent prompt adapts to the configured search and fetch backends, so it does
not ask the model to call tools that are disabled.
