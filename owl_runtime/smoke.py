"""Live media checks using the exact OWL tool callables, without a judge.

Run from the repository: uv run python owl_runtime/smoke.py
Each check retains a normal OWL trace, usage, and decoded image artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hyper_browsecomp.owl_harness import run_owl_harness


CASES = {
    "youtube-native": {
        "backend": "native", "tool": "ask_question_about_video",
        "args": {"video_path": "https://www.youtube.com/watch?v=jNQXAC9IVRw", "question": "What animals are behind the speaker? Describe the visible scene at about 00:05."},
        "expected": ["elephant"], "media_field": "video_inputs",
    },
    "youtube-download": {
        "backend": "download", "tool": "ask_question_about_video",
        "args": {"video_path": "https://www.youtube.com/watch?v=jNQXAC9IVRw", "question": "What animals are behind the speaker? Describe visible details from the supplied frames."},
        "expected": ["elephant"], "media_field": "image_inputs",
    },
    "bilibili": {
        "backend": "download", "tool": "ask_question_about_video",
        "args": {"video_path": "https://www.bilibili.com/video/BV1PKEJzdEh3/", "question": "What animals are behind the speaker? Describe visible details from the supplied frames."},
        "expected": ["elephant"], "media_field": "image_inputs",
    },
    "image": {
        "tool": "ask_question_about_image",
        "args": {"image_path": "https://www.python.org/static/community_logos/python-logo.png", "question": "What two colors are the two snake shapes in this image?"},
        "expected": ["blue", "yellow"], "media_field": "image_inputs",
    },
    "pdf": {
        "tool": "ask_question_about_pdf",
        "args": {"pdf_path": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf", "question": "Read the large text visible on the first page.", "pages": "1"},
        "expected": ["dummy pdf file"], "media_field": "image_inputs",
    },
    "audio": {
        "tool": "ask_question_about_audio",
        "args": {"audio_path": "https://raw.githubusercontent.com/realpython/python-speech-recognition/master/audio_files/harvard.wav", "question": "Transcribe only the first sentence spoken in the audio."},
        "expected": ["stale smell", "old beer", "lingers"], "media_field": "audio_inputs",
    },
}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--model", default="google/gemini-3.7-flash")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--use-configured-cookies", action="store_true",
                        help="Opt in to existing OWL cookie/proxy settings. By default these checks are anonymous.")
    args = parser.parse_args()
    names = args.cases.split(",")
    if any(name not in CASES for name in names):
        parser.error("Unknown case. Choose from: " + ", ".join(CASES))
    load_dotenv(ROOT / ".env")
    if not args.use_configured_cookies:
        for key in ("OWL_YTDLP_COOKIE_FILE", "OWL_YTDLP_COOKIES_FROM_BROWSER", "OWL_YTDLP_PROXY"):
            os.environ[key] = ""
    output_dir = ROOT / "logs/owl/media-smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    report = {"model": args.model, "checks": [], "note": "Tool integration checks, not full agent benchmark accuracy."}
    for name in names:
        case = CASES[name]
        os.environ["OWL_VIDEO_BACKEND"] = case.get("backend", "auto")
        print(f"START {name}", flush=True)
        check = {"name": name, "tool": case["tool"], "source": next(iter(case["args"].values()))}
        try:
            result = await run_owl_harness(
                "Direct media-tool integration check.", model_name=args.model,
                api_key_env="OPENROUTER_API_KEY", base_url="https://openrouter.ai/api/v1",
                headless=True, multimodal=True, browser_round_limit=4,
                task_timeout_seconds=args.timeout, max_tokens=2048,
                sample_id="media-check-" + name, trace_dir=ROOT / "logs/owl/traces",
                tool_check=case["tool"], tool_args=case["args"],
            )
            trace = json.loads(Path(result["trace_files"]["events"]).read_text())
            media_sent = sum(call.get(case["media_field"], 0) for call in trace["model_calls"])
            answer_ok = all(value in result["completion"].lower() for value in case["expected"])
            check.update(status="passed" if media_sent and answer_ok else "failed",
                         media_inputs=media_sent, answer_matches=answer_ok,
                         answer=result["completion"], statistics=result["statistics"],
                         traces=result["trace_files"], media_events=result.get("media_events", []))
        except Exception as exc:
            check.update(status="error", error=str(exc))
        report["checks"].append(check)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"RESULT {name}: {check['status']}", flush=True)
    print(f"REPORT {report_path}", flush=True)
    return 0 if all(check["status"] == "passed" for check in report["checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
