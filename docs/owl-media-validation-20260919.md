# OWL media validation - 2026-09-19

## Outcome

Public Bilibili video processing works in the current environment. Both a direct
tool check and an end-to-end OWL task downloaded video bytes, decoded four frames,
submitted those frames to the primary model, and correctly identified elephants.
The saved frame was independently inspected. This is visual evidence, not just
a search result or a correct judge answer.

YouTube also passed using two independent paths: native video input through
OpenRouter/Google AI Studio, and anonymous local downloading plus frame analysis.
Image, rendered PDF, and audio checks passed. All five end-to-end smoke questions
were judged correct with no sample errors or OWL model-call errors. These are easy
integration checks, not HyperBrowseComp benchmark results or a site-wide success
rate. The video fixture is the well-known "Me at the zoo" clip and a subtitled
Bilibili mirror, so answer correctness alone would be weak evidence.

## What changed

- Installed the explicitly pinned `yt-dlp-ejs` dependency and enabled Node.js for
  extraction when present. The local download route accepts separate video-only
  streams, including Bilibili's format 30080.
- Added `OWL_VIDEO_BACKEND=auto|download|native`. Auto uses native YouTube input
  with OpenRouter Gemini through Google AI Studio; other sites/models use local
  download and frame extraction. Provider API failures in auto try downloading.
- Added same-primary-model rendered-PDF and native-audio adapters. Fixed a
  pypdfium2 4.x context-manager incompatibility found by the live PDF test.
- Preserved decoded video frames and PDF pages beside traces; recorded actual
  input media counts, media events, provider usage, and per-role statistics.
- Added direct integration checks, a five-task evaluation fixture, documentation,
  and regression coverage. `uv run --extra dev pytest -q`: **100 passed**.

The PDF skill's render-and-inspect workflow was used to verify the page image
actually submitted to the model, rather than checking extracted text alone.

## Direct tool checks

All used `google/gemini-3.7-flash` through OpenRouter, temperature 0, with
multimodal enabled. Cookie-file, browser-cookie, and proxy settings were explicitly
empty. No Chrome cookies were read. Times below measure the worker task, excluding
subprocess/import startup and any judge.

| Check | Actual media evidence | Input tokens | Output tokens | Seconds |
| --- | --- | ---: | ---: | ---: |
| YouTube native | 1 video input; 1,676 provider-reported video tokens | 1,724 | 531 | 4.897 |
| YouTube download | 223,779 bytes; 4 decoded image inputs | 4,663 | 517 | 9.948 |
| Bilibili download | 2,688,566 bytes; 4 decoded image inputs | 4,663 | 503 | 31.917 |
| Image | 1 image input | 1,132 | 77 | 6.221 |
| PDF | 1 rendered page image | 1,122 | 61 | 5.317 |
| Audio | 3,249,924 WAV bytes; 1 audio input | 484 | 73 | 6.270 |

Each direct check made one model call and one tool call. The initial PDF check
failed on the context-manager bug; the initial audio check failed because the
fixture URL returned HTTP 404. The corrected rerun passed both. Original failure
records remain intact.

Reports:

- `logs/owl/media-smoke/20260919T113708.775889Z.json` (first four passes and two initial errors)
- `logs/owl/media-smoke/20260919T114312.389116Z.json` (PDF/audio passes after correction)

## End-to-end OWL evaluation

Configuration: `configs/owl_dev/openrouter_gemini_flash_media.yaml`.
No Exa tools were used. Five tasks, two samples in parallel, 2 minutes 56 seconds
reported evaluation time, 5/5 correct, zero sample errors.

| Task | Input tokens | Output tokens | Turns/model calls | Tool calls | Worker seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| YouTube | 12,887 | 3,571 | 8 | 1 | 51.002 |
| Bilibili | 29,909 | 6,658 | 16 | 2 | 94.410 |
| Image | 11,730 | 2,942 | 8 | 1 | 48.549 |
| PDF | 11,942 | 2,987 | 8 | 1 | 34.424 |
| Audio | 11,237 | 2,864 | 8 | 1 | 34.404 |
| Total | 77,705 | 19,022 | 48 | 6 | - |

These statistics include OWL planning, coordination, perception, synthesis, and
internal quality checks, but exclude the external Inspect judge. Bilibili's
agent also called `browse_url`; its visual answer still had a successful actual
video-tool call. Summed model-call time can exceed wall time due to concurrency.
Reasoning tokens are a subset of output tokens and must not be added again.

Inspect's console token total (1,780) represents its judge calls, not the above
OWL work. The OWL totals live in each sample's `metadata.owl_harness.statistics`
and the sidecar JSON traces. Inspect emitted a nonfatal warning parsing a
signature-only Gemini `reasoning.text` field from OpenRouter. This warning remains;
it did not prevent any judge score, and the eval's final status is `success`.

Eval:
`logs/owl/owl_media_smoke_gemini-3.7-flash_20260919T114350Z_2draqBs3Dyj6VzsmB3XQNj.eval`

Trace stems (each has `.log` and `.json`):

- `logs/owl/traces/20260919T114350.370324Z_owl-media-youtube`
- `logs/owl/traces/20260919T114350.393753Z_owl-media-bilibili`
- `logs/owl/traces/20260919T114500.799475Z_owl-media-image`
- `logs/owl/traces/20260919T114529.000807Z_owl-media-pdf`
- `logs/owl/traces/20260919T114553.089742Z_owl-media-audio`

Bilibili frame evidence:
`logs/owl/traces/20260919T114350.393753Z_owl-media-bilibili.media/video-1/frame-002.jpg`.
The four saved images are indexed frames; these filenames are not timestamps.

To rerun just Bilibili directly:

```bash
uv run python owl_runtime/smoke.py --cases bilibili
```

To rerun all five full-agent tasks:

```bash
uv run bash run_eval.sh configs/owl_dev/openrouter_gemini_flash_media.yaml
```

The latter honors configured environment settings; the diagnostic script uses
anonymous settings unless `--use-configured-cookies` is supplied. For live model
activity, tail the `.log` path printed when a sample starts. The matching JSON is
finalized after completion; media artifacts are written during the tool call.

## Why other papers can run videos

Published implementations do not establish a universally reliable anonymous
YouTube downloader. Their code and released assets show several approaches:

- **Video-Browser** implements a pytubefix OAuth/cache path and an alternative
  yt-dlp path accepting `data/cookies.txt`, plus local video caching. This is
  evidence of the implementation strategy, not proof that its authentication
  path still works today or that every paper run used the same settings.
  [Author code](https://github.com/chrisx599/Video-Browser/blob/main/videobrowser/tools/fetch_video.py)
- **GUIDE** releases 453 MP4 files in 299 annotated directories, along with
  metadata, subtitles, audio, keyframes, and precomputed task results. Reusing
  these assets can avoid repeating live retrieval during reproduction.
  [Author repository and dataset description](https://github.com/sharryXR/GUIDE#dataset)
- **WildClawBench** uses yt-dlp and FFmpeg in setup and explicitly warns about
  YouTube's sign-in/bot errors, suggesting cookies or a JavaScript runtime.
  [Author setup instructions](https://github.com/InternLM/WildClawBench#setup)

The practical distinction is retrieval versus perception: downloaded frames
can go to a compatible open-source vision model without Google extraction.
Native YouTube input in this harness is specifically the OpenRouter Gemini
route. A subsequent live check using `z-ai/glm-4.6v` through OpenRouter's Novita
provider passed the local YouTube download-and-frame path: 223,779 video bytes,
four 512x384 image inputs, 1,418 input tokens, 473 output tokens, one model call,
and no model error. The report is
`logs/owl/media-smoke/20260919T122539.193438Z.json`; this validates that model and
route, not every open-source model.

## Remaining limits

- One short clip and its mirror do not prove access to every Bilibili/YouTube
  video, region, account-restricted item, livestream, or future site version.
- Native YouTube processing supplies the video to the provider; it does not
  produce local decoded-frame artifacts. Its evidence is the video input and
  provider-reported video usage.
- Local frame analysis does not process the video's audio. The separate audio
  tool was tested on a public WAV, not on extracted Bilibili/YouTube audio.
- PDF checks covered one rendered page; selected-page limits are explicit.
- The browser-playback fallback was not exercised successfully in this run;
  it remains best-effort, not an anti-bot guarantee.
- These five supplied-URL tasks do not test finding an unknown video by search.
  The existing separate video-search smoke config is available for that purpose.
