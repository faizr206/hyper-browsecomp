# MiniMax M3: native search and direct Exa

Both configs call `MiniMax-M3` through MiniMax's own Anthropic-compatible API at
`https://api.minimax.io/anthropic`. They use the same 423 retained question IDs,
enable M3 adaptive thinking, and allow 25 research turns per question. Up to
five questions run concurrently. The same model grades the answers, without
search tools enabled for grading.

| Setup | Config | Retrieval |
| --- | --- | --- |
| Native search | `configs/internal_full/minimax_m3.yaml` | MiniMax's server-side `web_search_20250305` tool |
| Direct Exa | `configs/exa_full/minimax_m3.yaml` | The existing Exa search and page-fetch tools using `EXA_API_KEY` |

Neither setup uses OpenRouter. MiniMax manages native retrieval; its underlying
search provider is not identified by its documentation. The direct Exa config
requests five search results with text and highlights and retains the existing
20,000-character page-fetch limit.

## Credentials and a small connectivity test

Add your standard MiniMax pay-as-you-go API key to the repository's ignored
`.env` file. Keep the existing benchmark decryption key and, for the Exa setup,
your Exa key:

```dotenv
MINIMAX_API_KEY=your_minimax_api_key
EXA_API_KEY=your_exa_api_key
HYPERBROWSECOMP_KEY=your_existing_question_decryption_key
```

Use the public-question canary before starting a benchmark:

```bash
.venv/bin/python scripts/test_minimax_m3.py --setup both
```

Choose `--setup native` or `--setup exa` to test one setup. This is a small live
API test, so it uses credits. It does not run benchmark questions. Each setup is
limited to six model requests of at most 4,096 output tokens each; the Exa check
allows one search and one fetch. The checks enable adaptive thinking, verify
returned search results, and grade the public answer without tools.

Both setups passed a live check on 2026-09-22, using the Python 3.13.0 release
date as the public question. Native search returned ten results from one search
and used 3,778 total tokens, including grading. Direct Exa performed one search
and one page fetch and used 14,783 total tokens, including grading. These are
connectivity checks, not BrowseComp accuracy or workload estimates. Each canary prints its receipt location under `dump/minimax_m3_preflight/`;
receipts and `.eval` traces are local artifacts and are not included in Git.

## Start and resume

Run these commands from the repository directory. Each starts the corresponding
full 423-question experiment:

```bash
bash run_eval.sh configs/internal_full/minimax_m3.yaml
```

```bash
bash run_eval.sh configs/exa_full/minimax_m3.yaml
```

For a clean stop, press Ctrl+C once and wait for Inspect to save. Resume by
repeating the same command after restarting the computer. Completed questions
are reused, and unfinished questions resume from available turn checkpoints.
An in-flight request not yet checkpointed may repeat. Each experiment prevents
duplicate simultaneous launches. Keep its config and retained-ID file unchanged
after starting; a new experiment needs a new log directory.

Each setup publishes one canonical result, so their results stay separate:

```text
logs/minimax_m3_internal/HyperBrowseComp_MiniMax-M3.eval
logs/minimax_m3_direct_exa/HyperBrowseComp_MiniMax-M3.eval
```

Raw attempts, turn checkpoints, and earlier publications remain under
`dump/auto_resume/`. The canonical file is refreshed when a run stops or
finishes and when it resumes; ongoing progress is stored in the raw attempts.
To inspect the published results:

```bash
.venv/bin/inspect view --log-dir logs
```

## Comparing tokens

Use each setup's recorded usage independently, including research and grading.
Use recorded `total_tokens` for the total. Inspect's Anthropic-compatible
normalizer records uncached input separately from cache reads and writes, and
includes all three plus output in `total_tokens`. Reasoning, when reported, is a
subset of output. Do not add cache or reasoning tokens to `total_tokens` again;
other providers can use different input/cache column conventions. Native server-side retrieval can expose
different usage details than direct Exa's returned content. Its search-call
count and token total must be measured from its own run. Resume attempts can
also consume usage that a final retained question history does not include.

MiniMax references: [server tools](https://platform.minimax.io/docs/guides/server-tools)
and [text generation](https://platform.minimax.io/docs/guides/text-generation).
