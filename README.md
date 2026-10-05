<div align="center">
<h1>HyperBrowseComp: A Multilingual and Multimodal Stress Test for Web-Browsing Agents</h1>

[![Paper](https://img.shields.io/badge/Paper-arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2610.03574)
[![Hugging Face paper](https://img.shields.io/badge/Paper-Hugging%20Face-yellow?logo=huggingface)](https://huggingface.co/papers/2610.03574)
[![Website](https://img.shields.io/badge/Website-Project%20Page-6f42c1.svg?logo=githubpages)](https://hyperbrowsecomp.github.io/)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-blue?logo=huggingface)](https://huggingface.co/datasets/afaji/HyperBrowseComp)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
</div>

Evaluation code for HyperBrowseComp, a multilingual, multimodal benchmark for
difficult open-web research. The benchmark contains 423 human-authored
questions across 13 languages.

This repository provides:

- Inspect AI evaluation and scoring
- provider-native search and Exa/Firecrawl retrieval configurations
- resumable runs and sanitized `.eval` logs
- an isolated OWL/CAMEL browser and multimodal harness

## Requirements

- Python 3.11 or 3.12
- [`uv`](https://docs.astral.sh/uv/)
- API credentials for the model, scorer, and retrieval backends you use

OWL runs also require Chromium. Local video processing requires FFmpeg on
`PATH`.

## Setup

```bash
git clone https://github.com/faizr206/hyper-browsecomp.git
cd hyper-browsecomp
uv sync --extra dev
cp .env.example .env
uv run pytest -q
```

Fill in only the credentials needed by your chosen config. `.env` is ignored
by Git; never commit credentials or decrypted benchmark data.

For the encrypted full dataset, set its AES-256-GCM master key:

```dotenv
HYPERBROWSECOMP_KEY=...
```

Development configs use the small fixtures in `data/` and do not require the
dataset key.

## Run an evaluation

All evaluations use the same entrypoint:

```bash
uv run hyper-browsecomp CONFIG.yaml
```

Examples:

```bash
# Provider-native search on a development fixture
uv run hyper-browsecomp configs/internal_search_dev/dev_openai.yaml

# Shared Exa search/fetch tools on a development fixture
uv run hyper-browsecomp configs/exa_dev/dev_openai.yaml

# Full encrypted dataset
uv run hyper-browsecomp configs/exa_full/openai.yaml
```

The main config groups are:

| Directory | Purpose |
| --- | --- |
| `configs/internal_search_dev/` | Small provider-native search runs |
| `configs/exa_dev/` | Small shared-retrieval runs |
| `configs/internal_full/` | Full provider-native search runs |
| `configs/exa_full/` | Full shared-retrieval runs |
| `configs/no_tools_full/` | Full runs without web tools |
| `configs/owl_dev/` | OWL smoke and media checks |
| `configs/owl_full/` | Full OWL runs |

Copy the closest config before changing models, tools, concurrency, or sample
selection. Configs are flat YAML files validated by
`src/hyper_browsecomp/config.py`.

### Common config fields

```yaml
provider: openai
model_name: gpt-5.4-mini
model_api_key_env: OPENAI_API_KEY
model_base_url: https://api.openai.com/v1

scorer_provider: openai
scorer_model_name: gpt-5.4-mini
scorer_api_key_env: OPENAI_API_KEY
scorer_base_url: https://api.openai.com/v1

data_path: data/dev.jsonl
sample_range: "1"
harness: react
tool_profile: web
search_backend: internal
fetch_backend: none
max_steps: 12
```

`sample_range` is 1-based and inclusive. You can instead use
`sample_ids_path` to pass IDs to Inspect or `sample_ids_file` to filter and
order the loaded dataset. These selectors cannot be combined.

`search_backend` accepts `internal`, `exa`, `firecrawl`, or `none`;
`fetch_backend` accepts `exa`, `firecrawl`, or `none`. The corresponding
`EXA_API_KEY` or `FIRECRAWL_API_KEY` is required only when that backend is
selected. `tool_profile: web_code` also enables shell and Python tools and
requires `no_sandbox: false`.

## OWL harness

OWL/CAMEL uses a separate environment because its dependency versions conflict
with the main Inspect environment:

```bash
uv sync --project owl_runtime
uv run --project owl_runtime playwright install chromium
```

Start with a development config before a full run:

```bash
uv run hyper-browsecomp configs/owl_dev/openrouter_gemini_flash.yaml
uv run hyper-browsecomp configs/owl_dev/openrouter_gemini_flash_media.yaml
```

To check media tools directly, without the planner or judge:

```bash
uv run python owl_runtime/smoke.py
uv run python owl_runtime/smoke.py --cases pdf,audio
```

OWL traces are written under `logs/owl/traces/`. The configured model endpoint
must support the media types exercised by a run. Video download mode also
requires FFmpeg; native YouTube processing depends on provider support.

## Outputs and resume

Runs write Inspect `.eval` files under `logs/`. OWL runs also write readable
console traces, structured JSON traces, and extracted media artifacts.
Generated outputs are ignored by Git.

Resume missing or errored samples from an existing log:

```bash
uv run hyper-browsecomp resume CONFIG.yaml logs/PARTIAL_RUN.eval
```

Render an OWL evaluation and its trace sidecars as Markdown:

```bash
uv run --extra dev python scripts/render_owl_trace_report.py \
  logs/owl/RUN.eval --output logs/owl/RUN.traces.md
```

## Repository layout

```text
configs/             Evaluation configurations
data/                Small public smoke-test fixtures
owl_runtime/         Isolated OWL/CAMEL worker and lockfile
scripts/             Trace and result utilities
src/hyper_browsecomp Core package
tests/               Offline unit tests
```

## Development

Run the offline test suite before submitting changes:

```bash
uv sync --extra dev
uv run pytest -q
```

Keep generated logs, downloaded datasets, credentials, and local environments
out of version control. See `.gitignore` for the excluded paths.
