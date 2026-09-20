# hyper-browsecomp

`hyper-browsecomp` is a compact Inspect AI harness for BrowseComp-style web research evals. It runs local JSONL datasets and the encrypted `afaji/HyperBrowseComp` Hugging Face dataset, gives the model web search and fetch tools, and scores answers with a BrowseComp-style judge.

## Install

```bash
uv sync --extra dev
```

This creates a local `.venv` and installs the project with its development
dependencies from `pyproject.toml` and `uv.lock`.

The OWL harness has an isolated runtime because OWL/CAMEL and Inspect require
incompatible Pydantic versions. Install its locked dependencies and Chromium
once before an OWL run:

```bash
uv sync --project owl_runtime
uv run --project owl_runtime playwright install chromium
```

Local video decoding also needs FFmpeg on `PATH`. YouTube downloads use the
locked `yt-dlp-ejs` dependency and enable Node.js when `node` is on `PATH`.

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

Only set the backend key for backends selected by the YAML config. For example,
`search_backend: exa` and `fetch_backend: exa` need `EXA_API_KEY`, while
`fetch_backend: none` does not need `FIRECRAWL_API_KEY`.

HyperBrowseComp on Hugging Face is also supported. Put the hex-encoded
AES-256-GCM master key in `.env`:

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

## Run

Use the single shell entrypoint with any YAML config:

```bash
uv run bash run_eval.sh configs/exa_dev/dev_openai.yaml
uv run bash run_eval.sh configs/internal_search_dev/dev_openai.yaml
uv run bash run_eval.sh configs/exa_full/openai.yaml
uv run bash run_eval.sh configs/web_code.yaml
uv run bash run_eval.sh configs/owl_dev/openrouter_gemini_flash.yaml
uv run bash run_eval.sh configs/owl_dev/openrouter_gemini_flash_video.yaml
uv run bash run_eval.sh configs/owl_dev/openrouter_gemini_flash_media.yaml
uv run bash run_eval.sh configs/owl_full/openrouter_gemini_flash.yaml
```

The OWL smoke config uses OpenRouter's multimodal
`google/gemini-3.7-flash` on a one-question image task. It runs an OWL
Workforce inside the Inspect evaluation shell, so the existing dataset,
scorer, `.eval` log, redaction, and resume behavior are retained while the
normal Inspect ReAct agent is bypassed.

The video smoke config searches for the canonical YouTube URL for `Me at the
zoo`, requires OWL to call `ask_question_about_video`, and asks for a visual
detail from the video. It deliberately supplies no URL so both live
search and actual video processing are exercised; a download/decoding failure
must be reported rather than guessed.

The media smoke config runs five end-to-end OWL tasks: YouTube, Bilibili,
image, rendered PDF, and audio. To test the same tools directly, without the
planner or judge, run `uv run python owl_runtime/smoke.py`. This additionally
tests both native and downloaded YouTube input. Use `--cases pdf,audio` for a
subset. Checks load `.env`, use no cookies/proxy unless
`--use-configured-cookies` is passed, and save reports in
`logs/owl/media-smoke/`. They require actual media input plus an expected
answer; these easy integration checks are not benchmark accuracy results.

The dev configs all run the single-question `data/dev.jsonl` dataset:

```text
configs/exa_dev/*.yaml
configs/internal_search_dev/*.yaml
```

The full Exa configs run the encrypted Hugging Face dataset and require
`HYPERBROWSECOMP_KEY`:

```text
configs/exa_full/*.yaml      afaji/HyperBrowseComp
```

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
- `data_path`, `sample_range`, `start_index`, `end_index`, `num_samples`
- `max_steps`
- `inspect_max_samples_parallel`, `inspect_model_max_retries`, `inspect_attempt_timeout`
- `inspect_retry_on_error`, `inspect_no_fail_on_error`, `inspect_continue_on_fail`,
  `inspect_ctl_server`
- `no_sandbox`

OWL-specific fields are `owl_model_name`, `owl_api_key_env`, `owl_base_url`,
`owl_headless`, `owl_multimodal`, `owl_browser_round_limit`,
`owl_task_timeout_seconds`, `owl_finalize_reserve_seconds`,
`owl_max_external_tool_calls`, `owl_max_model_calls`,
`owl_model_max_retries`, and `owl_max_tokens`.
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

### OWL parallelism

`inspect_max_samples_parallel` controls sample-level concurrency. Every sample
runs in its own OWL subprocess with its own temporary directory, browser,
budgets, and trace files. The supplied five-sample validation config uses two
workers; the full 500-sample config uses four workers on the validated
14-core/36-GiB host. Start at two on an untested machine, then increase to four
after watching memory and provider rate-limit errors. Eight workers may improve
throughput on larger hosts, but it also doubles simultaneous Chromium and model
traffic; it should be treated as a separate capacity test rather than the
default benchmark setting.

`sample_range` is the easiest way to run a subset. It uses 1-based inclusive sample
numbers, so `sample_range: "1-2"` runs dataset samples 1 and 2. The older
`start_index`, `end_index`, and `num_samples` fields are still available, but they
use Python slicing semantics.

By default the runner caps model API retries, retries sample errors 3 times,
records failed samples without failing the whole run, and continues after failed
samples. One bad sample will not stop the whole evaluation.

Removed from the old repo:

- OpenRouter-specific modes and adapters
- Browser/PDF/OCR/video tools from the original Inspect ReAct path (the
  isolated OWL harness now supplies its own browser and multimodal tools)
- dataset conversion scripts
- multiple wrapper scripts

## Tool behavior

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
provider, or video is compatible.

`ask_question_about_pdf(pdf_path, question, pages="1")` renders up to eight
specified 1-based pages per call and reports the total page count. It does not
silently inspect the rest of a document. `ask_question_about_audio` downloads a
public audio file (or reads a local one) and sends base64 `input_audio` to the
same primary model; that endpoint must support audio input. No auxiliary
Whisper/Gemini key is used. PDF/audio downloads have a 32 MiB limit.

- `search_backend: internal` uses Inspect's standard provider-native web search
  for `openai`, `anthropic`, `gemini`, `grok`, `mistral`, and `perplexity`
- `search_backend: none` disables search
- `fetch_backend: none` disables page fetching
- `web` exposes the configured web tools
- `web_code` adds `bash()` and `python()`

The agent prompt adapts to the configured search and fetch backends, so it does
not ask the model to call tools that are disabled.
