from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw

from camel.agents import ChatAgent
from camel.logger import set_log_level
from camel.models import ModelFactory
from camel.societies import Workforce
from camel.societies.workforce.workforce_logger import WorkforceLogger
from camel.tasks import Task
from camel.tasks.task import TaskState
from camel.toolkits import (
    BrowserToolkit,
    FileToolkit,
    FunctionTool,
    ImageAnalysisToolkit,
    SearchToolkit,
    VideoAnalysisToolkit,
)
from camel.types import ModelPlatformType
from dotenv import load_dotenv
from media_tools import MediaTools, input_media_counts, is_youtube_url, native_youtube_supported


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "HYPER_BROWSECOMP_OWL_RESULT="
PROGRESS_PREFIX = "HYPER_BROWSECOMP_OWL_PROGRESS="


class BudgetExceededError(RuntimeError):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason

WEB_AGENT_PROMPT = """\
You are OWL's web research specialist. Use DuckDuckGo or Wikipedia to find candidate sources, then use the visual browser on promising URLs. The browser observes screenshots with a multimodal model, so use it whenever page layout, rendered content, or images matter. Inspect primary sources where possible, persist through failed approaches, keep track of evidence and URLs, and return concise findings to the workforce. Never claim that you inspected an image, video, or page unless a tool actually processed it. Do not spend the whole task issuing searches: make at most 8 consecutive search calls, browse the best candidate before searching again, and always return a best-effort evidence summary before reaching your iteration limit. For a question whose answer depends on a video or audio recording, your job is to identify a credible direct canonical media URL and any useful title/timestamp clues. Once you have one credible URL, return it immediately so the multimodal specialist can inspect the media; do not repeatedly browse search-result pages or infer the recording's contents from snippets.
"""

MULTIMODAL_AGENT_PROMPT = """\
You are OWL's multimodal evidence specialist. Analyze relevant images, videos, PDF pages, and audio with the provided media tools. For PDFs, select page numbers and request additional pages as needed. Pay attention to fine visual details, visible text, identity cues, colors, counts, and temporal context. Report evidence and its source precisely. The same primary model powers all perception tools. When a task depends on spoken or visible video content and a direct video URL is supplied, you must call ask_question_about_video before answering. Report tool failures explicitly; never substitute search snippets or remembered facts for inaccessible media.
"""

REASONING_AGENT_PROMPT = """\
You are OWL's synthesis specialist. Reconcile the evidence gathered by the browsing and multimodal workers, identify unsupported leaps, and produce the shortest defensible answer. Follow the exact response format requested by the task.
"""

TASK_MANAGER_PROMPT = """\
Decompose the research task into concrete browsing, multimodal inspection, and synthesis subtasks. Make sure the workforce actually inspects relevant visual content instead of relying on snippets or model memory. For a media-dependent question, plan a short ordered pipeline: locate one credible direct media URL, require the multimodal worker to inspect it, then synthesize. Do not duplicate completed subtasks or repeatedly re-plan the same search; allow at most one focused recovery attempt after a source failure before producing the best supported answer.
"""

COORDINATOR_PROMPT = """\
Coordinate the OWL workforce. Assign web navigation to the web specialist, image or video inspection to the multimodal specialist, and final evidence reconciliation to the synthesis specialist. For media-dependent questions, hand the first credible direct URL to the multimodal specialist promptly and do not accept an answer about media contents unless a media tool returned evidence or an explicit access failure. Do not re-open completed subtasks. If any tool reports RESEARCH_BUDGET_CLOSED, stop planning or delegating research and ask the synthesis specialist for the best available final answer. The final result must follow the user's exact requested format.
"""


def _read_payload() -> dict[str, Any]:
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("OWL worker input must be a JSON object.")
    return payload


def _response_usage(result: Any) -> dict[str, int]:
    usage = getattr(result, "usage", None)
    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(details, "reasoning_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _model_factory(
    payload: dict[str, Any],
    model_stats: dict[str, Any],
    on_progress: Callable[[], None],
) -> Callable[[str], Any]:
    api_key_env = str(payload["api_key_env"])
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {api_key_env}")

    model_name = str(payload["model_name"])
    base_url = str(payload["base_url"])
    max_tokens = int(payload.get("max_tokens", 8192))
    max_model_calls = int(payload.get("max_model_calls", 180))

    stats_lock = threading.Lock()

    def reserve_model_call(role: str) -> None:
        error: BudgetExceededError | None = None
        with stats_lock:
            if model_stats["model_call_attempts"] >= max_model_calls:
                model_stats["termination_reason"] = "model_call_limit"
                model_stats["budget_events"].append(
                    {
                        "kind": "model_call_limit",
                        "role": role,
                        "limit": max_model_calls,
                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                error = BudgetExceededError(
                    "model_call_limit",
                    f"Global model-call budget of {max_model_calls} is exhausted.",
                )
            else:
                model_stats["model_call_attempts"] += 1
        on_progress()
        if error is not None:
            raise error

    def create_model(role: str = "unspecified"):
        model_config = {"temperature": 0, "max_tokens": max_tokens}
        if role == "youtube_native":
            # Vertex does not accept YouTube URLs. Keep provider routing strict.
            model_config["extra_body"] = {
                "provider": {"only": ["google-ai-studio"], "allow_fallbacks": False}
            }
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=model_name,
            api_key=api_key,
            url=base_url,
            model_config_dict=model_config,
            timeout=float(payload.get("task_timeout_seconds", 900)),
            max_retries=int(payload.get("model_max_retries", 1)),
        )
        original_run = model.run
        original_arun = model.arun

        def record_error(started_at: str, started: float, exc: Exception) -> None:
            elapsed = time.perf_counter() - started
            with stats_lock:
                model_stats["model_errors"] += 1
                model_stats["model_time_seconds"] += elapsed
                model_stats["calls"].append(
                    {
                        "turn": len(model_stats["calls"]) + 1,
                        "role": role,
                        "started_at": started_at,
                        "duration_seconds": round(elapsed, 3),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        def record_success(started_at: str, started: float, response: Any, media: dict[str, int]) -> None:
            elapsed = time.perf_counter() - started
            usage = _response_usage(response)
            with stats_lock:
                model_stats["model_calls"] += 1
                model_stats["model_time_seconds"] += elapsed
                for key, value in usage.items():
                    model_stats[key] += value
                role_stats = model_stats["by_role"].setdefault(
                    role,
                    {
                        "model_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "reasoning_tokens": 0,
                        "total_tokens": 0,
                        "model_time_seconds": 0.0,
                    },
                )
                role_stats["model_calls"] += 1
                role_stats["model_time_seconds"] += elapsed
                for key, value in usage.items():
                    role_stats[key] += value
                model_stats["calls"].append(
                    {
                        "turn": len(model_stats["calls"]) + 1,
                        "role": role,
                        "started_at": started_at,
                        "duration_seconds": round(elapsed, 3),
                        "status": "success",
                        "provider": getattr(response, "provider", None),
                        "response_id": getattr(response, "id", None),
                        "usage_details": response.usage.model_dump() if getattr(response, "usage", None) else {},
                        **media,
                        **usage,
                    }
                )

        @wraps(original_run)
        def tracked_run(*args: Any, **kwargs: Any) -> Any:
            reserve_model_call(role)
            started_at = datetime.now(timezone.utc).isoformat()
            started = time.perf_counter()
            try:
                response = original_run(*args, **kwargs)
            except Exception as exc:
                record_error(started_at, started, exc)
                on_progress()
                raise
            record_success(started_at, started, response, input_media_counts(kwargs.get("messages", args[0] if args else [])))
            on_progress()
            return response

        @wraps(original_arun)
        async def tracked_arun(*args: Any, **kwargs: Any) -> Any:
            reserve_model_call(role)
            started_at = datetime.now(timezone.utc).isoformat()
            started = time.perf_counter()
            try:
                response = await original_arun(*args, **kwargs)
            except Exception as exc:
                record_error(started_at, started, exc)
                on_progress()
                raise
            record_success(started_at, started, response, input_media_counts(kwargs.get("messages", args[0] if args else [])))
            on_progress()
            return response

        model.run = tracked_run
        model.arun = tracked_arun
        return model

    return create_model


def _preserve_clone_step_timeout(agent: ChatAgent) -> ChatAgent:
    """Work around CAMEL 0.2.84 clone() dropping step_timeout.

    Workforce executes cloned agents asynchronously. CAMEL's clone constructor
    currently omits step_timeout, so every worker clone silently falls back to
    180 seconds even when its base agent has a larger, explicit budget.
    """
    original_clone = agent.clone

    @wraps(original_clone)
    def clone_with_timeout(*args: Any, **kwargs: Any) -> ChatAgent:
        cloned = original_clone(*args, **kwargs)
        cloned.step_timeout = agent.step_timeout
        return _preserve_clone_step_timeout(cloned)

    agent.clone = clone_with_timeout
    return agent


def build_workforce(
    payload: dict[str, Any],
    tool_registry: dict[str, Callable[..., Any]] | None = None,
) -> tuple[
    Workforce,
    WorkforceLogger,
    dict[str, Any],
    dict[str, int],
    dict[str, Any],
    Path,
]:
    model_stats: dict[str, Any] = {
        "model_calls": 0,
        "model_call_attempts": 0,
        "model_errors": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "model_time_seconds": 0.0,
        "by_role": {},
        "calls": [],
        "budget_events": [],
        "termination_reason": None,
        "task_timeout_seconds": int(payload.get("task_timeout_seconds", 900)),
        "finalize_reserve_seconds": int(payload.get("finalize_reserve_seconds", 120)),
        "max_external_tool_calls": int(payload.get("max_external_tool_calls", 50)),
        "max_model_calls": int(payload.get("max_model_calls", 180)),
        "model_max_retries": int(payload.get("model_max_retries", 1)),
    }
    tool_usage: dict[str, int] = defaultdict(int)
    wall_started = time.perf_counter()
    progress_lock = threading.Lock()
    tool_lock = threading.Lock()

    def emit_progress() -> None:
        with progress_lock:
            progress = {
                "tool_usage": dict(tool_usage),
                "statistics": _statistics(
                    model_stats, tool_usage, time.perf_counter() - wall_started
                ),
                "model_calls": list(model_stats["calls"]),
                "media_events": list(model_stats.get("media_events", [])),
            }
            print(
                f"{PROGRESS_PREFIX}{json.dumps(progress, ensure_ascii=False)}",
                flush=True,
            )

    create_model = _model_factory(payload, model_stats, emit_progress)
    max_external_tool_calls = int(payload.get("max_external_tool_calls", 50))
    task_timeout_seconds = float(payload.get("task_timeout_seconds", 900))
    finalize_reserve_seconds = float(payload.get("finalize_reserve_seconds", 120))
    research_deadline_seconds = max(0.0, task_timeout_seconds - finalize_reserve_seconds)

    def reserve_tool_call(name: str) -> str | None:
        closure_message: str | None = None
        elapsed = time.perf_counter() - wall_started
        with tool_lock:
            used = sum(tool_usage.values())
            if elapsed >= research_deadline_seconds:
                reason = "finalization_reserve"
                message = (
                    "External research is closed to reserve time for final synthesis. "
                    "Do not call another tool; return the best evidence-backed answer now."
                )
            elif used >= max_external_tool_calls:
                reason = "external_tool_call_limit"
                message = (
                    f"Global external-tool budget of {max_external_tool_calls} is "
                    "exhausted. Do not call another tool; return the best "
                    "evidence-backed answer now."
                )
            else:
                reason = ""
                message = ""
                tool_usage[name] += 1
            if reason:
                model_stats["budget_events"].append(
                    {
                        "kind": reason,
                        "tool": name,
                        "limit": (
                            research_deadline_seconds
                            if reason == "finalization_reserve"
                            else max_external_tool_calls
                        ),
                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                closure_message = (
                    f"RESEARCH_BUDGET_CLOSED ({reason}): {message}"
                )
        emit_progress()
        return closure_message

    def tracked(name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def call(*args: Any, **kwargs: Any) -> Any:
            closure_message = reserve_tool_call(name)
            if closure_message is not None:
                return closure_message
            try:
                return function(*args, **kwargs)
            finally:
                emit_progress()

        return call

    cache_root = Path(tempfile.mkdtemp(prefix="hyper-browsecomp-owl-"))
    media_events: list[dict[str, Any]] = []
    model_stats["media_events"] = media_events
    artifact_dir = Path(payload["media_artifact_dir"]) if payload.get("media_artifact_dir") else None
    media = MediaTools(create_model, cache_root, media_events, artifact_dir)
    search_toolkit = SearchToolkit(timeout=60)
    search_state: dict[str, Any] = {"consecutive": 0, "last_results": []}
    file_toolkit = FileToolkit(working_directory=str(cache_root))
    configured_round_limit = int(payload.get("browser_round_limit", 12))

    def search_duckduckgo(
        query: str, source: str = "text", number_of_result_pages: int = 5
    ) -> list[dict[str, Any]]:
        """Search DuckDuckGo, with a hard budget between browser visits."""
        closure_message = reserve_tool_call("search_duckduckgo")
        if closure_message is not None:
            return [
                {
                    "error": "Research budget closed.",
                    "instruction": closure_message,
                }
            ]
        search_state["consecutive"] += 1
        if search_state["consecutive"] > 8:
            return [
                {
                    "error": "Consecutive search budget exhausted.",
                    "instruction": "Stop searching. Browse the best URL already found or return a best-effort evidence summary now.",
                },
                *search_state["last_results"][:3],
            ]
        normalized_source = {
            "all": "text",
            "web": "text",
            "image": "images",
            "video": "videos",
        }.get((source or "text").lower(), (source or "text").lower())
        if normalized_source not in {"text", "images", "videos"}:
            normalized_source = "text"
        try:
            results = search_toolkit.search_duckduckgo(
                query=query,
                source=normalized_source,
                number_of_result_pages=max(1, min(number_of_result_pages, 5)),
            )
            search_state["last_results"] = results
            return results
        finally:
            emit_progress()

    def browse_url(task_prompt: str, start_url: str, round_limit: int = 12) -> str:
        """Visually browse a URL with screenshots and a bounded action loop."""
        closure_message = reserve_tool_call("browse_url")
        if closure_message is not None:
            return closure_message
        search_state["consecutive"] = 0
        # BrowserToolkit.browse_url() closes its Playwright browser at the end
        # of every call. A toolkit instance therefore cannot be reused for a
        # later URL: doing so raises "Event loop is closed". Give every page
        # visit its own toolkit and lifecycle instead.
        browser_toolkit = BrowserToolkit(
            headless=bool(payload.get("headless", True)),
            cache_dir=str(cache_root / "browser"),
            web_agent_model=create_model("visual_browser_web"),
            planning_agent_model=create_model("visual_browser_planner"),
        )
        # BaseToolkit's metaclass wraps browse_url itself with the toolkit's
        # default 180-second timeout. Video download, frame extraction, and a
        # multi-round browser loop can legitimately take longer, so align the
        # toolkit wrapper with the web agent's explicit tool budget.
        browser_toolkit.timeout = min(
            600.0, float(payload.get("task_timeout_seconds", 900))
        )
        # BrowserToolkit constructs its two internal ChatAgents with CAMEL's
        # 180-second default. That is shorter than this tool's 600-second
        # execution budget and caused real long-page visual browsing to fail
        # even though the surrounding OWL task was healthy. Keep the nested
        # calls within the outer boundary while honoring shorter task limits.
        browser_step_timeout = min(
            540.0, float(payload.get("task_timeout_seconds", 900))
        )
        browser_toolkit.web_agent.step_timeout = browser_step_timeout
        browser_toolkit.planning_agent.step_timeout = browser_step_timeout
        original_visit_page = browser_toolkit.browser.visit_page
        original_ask_question_about_video = (
            browser_toolkit.browser.ask_question_about_video
        )

        @wraps(original_visit_page)
        def visit_page(url: str) -> Any:
            return original_visit_page(url.strip().strip("\"'"))

        browser_toolkit.browser.visit_page = visit_page

        def find_text_on_page(search_text: str) -> str:
            # CAMEL 0.2.81 builds a multiline JavaScript string whose CSS
            # selector is split by a newline, producing a SyntaxError on every
            # call. Pass the text as an evaluate argument instead, which also
            # avoids interpolating quotes from model-generated search terms.
            page = browser_toolkit.browser.page
            if page is None:
                raise RuntimeError("Browser page is not initialized.")
            found = bool(
                page.evaluate("(text) => window.find(text)", search_text)
            )
            if found:
                return f"Found text '{search_text}' on the page."
            return f"Text '{search_text}' not found on the page."

        browser_toolkit.browser.find_text_on_page = find_text_on_page

        @wraps(original_ask_question_about_video)
        def ask_question_about_video(question: str) -> str:
            # CAMEL prompts with input() for y/n confirmation before video
            # analysis. Inspect workers receive their task through stdin and
            # then close it, so that redundant prompt always raises EOFError.
            # Enabling this harness's multimodal mode is the explicit consent
            # to invoke the configured video tool.
            with patch("builtins.input", return_value="y"):
                return original_ask_question_about_video(question)

        browser_toolkit.browser.ask_question_about_video = (
            ask_question_about_video
        )
        try:
            return browser_toolkit.browse_url(
                task_prompt=task_prompt,
                start_url=start_url.strip().strip("\"'"),
                round_limit=min(round_limit, configured_round_limit),
            )
        finally:
            emit_progress()

    web_tools = [
        FunctionTool(search_duckduckgo),
        FunctionTool(tracked("search_wiki", search_toolkit.search_wiki)),
        FunctionTool(browse_url),
        FunctionTool(tracked("read_file", file_toolkit.read_file)),
    ]
    web_agent = ChatAgent(
        WEB_AGENT_PROMPT,
        model=create_model("web_worker"),
        tools=web_tools,
        max_iteration=20,
        tool_execution_timeout=600,
        step_timeout=660,
    )

    multimodal = bool(payload.get("multimodal", True))
    multimodal_tools = []
    ytdlp_cookie_file = os.getenv("OWL_YTDLP_COOKIE_FILE", "").strip()
    ytdlp_browser = os.getenv("OWL_YTDLP_COOKIES_FROM_BROWSER", "").strip()
    ytdlp_proxy = os.getenv("OWL_YTDLP_PROXY", "").strip()
    video_backend = os.getenv("OWL_VIDEO_BACKEND", "auto").strip().lower()
    if video_backend not in {"auto", "download", "native"}:
        raise ValueError("OWL_VIDEO_BACKEND must be auto, download, or native.")
    native_video = native_youtube_supported(str(payload["model_name"]), str(payload["base_url"]))
    if multimodal:
        image_toolkit = ImageAnalysisToolkit(
            model=create_model("image_analysis"), timeout=180
        )
        video_toolkit = VideoAnalysisToolkit(
            working_directory=str(cache_root / "video"),
            model=create_model("video_analysis"),
            use_audio_transcription=False,
            use_ocr=False,
            frame_interval=4.0,
            timeout=300,
        )
        original_video_step = video_toolkit.vl_agent.step

        def video_step_with_evidence(message, *args, **kwargs):
            frames = message.image_list or []
            if not frames:
                raise RuntimeError("Video decoding returned no visual frames.")
            artifacts = []
            if artifact_dir:
                frame_dir = artifact_dir / f"video-{len(media_events)}"
                frame_dir.mkdir(parents=True, exist_ok=True)
                for index, frame in enumerate(frames):
                    path = frame_dir / f"frame-{index:03d}.jpg"
                    frame.convert("RGB").save(path, quality=85)
                    artifacts.append(str(path))
            media.record(kind="video", backend="decoded_frames", status="submitted",
                         image_inputs=len(frames), dimensions=[list(frame.size) for frame in frames],
                         artifacts=artifacts)
            return original_video_step(message, *args, **kwargs)

        video_toolkit.vl_agent.step = video_step_with_evidence

        def browser_video_frames(video_url: str, question: str) -> str:
            """Inspect timestamped frames by seeking the browser video element."""
            from playwright.sync_api import sync_playwright

            parsed = urlparse(video_url)
            video_id = ""
            if parsed.hostname in {"youtu.be", "www.youtu.be"}:
                video_id = parsed.path.strip("/").split("/")[0]
            elif parsed.hostname and "youtube.com" in parsed.hostname:
                video_id = parse_qs(parsed.query).get("v", [""])[0]
            browser_url = (
                f"https://www.youtube.com/embed/{video_id}?autoplay=1&mute=1"
                if video_id
                else video_url
            )
            frame_dir = cache_root / "browser-video-frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            frame_paths: list[tuple[float, Path]] = []

            with sync_playwright() as playwright:
                launch_options: dict[str, Any] = {
                    "headless": bool(payload.get("headless", True)),
                    "args": ["--autoplay-policy=no-user-gesture-required"],
                }
                if ytdlp_proxy:
                    launch_options["proxy"] = {"server": ytdlp_proxy}
                browser = playwright.chromium.launch(
                    **launch_options,
                )
                try:
                    page = browser.new_page(viewport={"width": 960, "height": 720})
                    page.goto(browser_url, wait_until="domcontentloaded", timeout=60_000)
                    video = page.locator("video").first
                    video.wait_for(state="attached", timeout=45_000)
                    ready_deadline = time.monotonic() + 45
                    while time.monotonic() < ready_deadline:
                        if int(video.evaluate("element => element.readyState")) >= 2:
                            break
                        time.sleep(1)
                    else:
                        raise RuntimeError("Browser video element did not become playable.")
                    duration = float(
                        video.evaluate(
                            "element => Number.isFinite(element.duration) "
                            "? element.duration : 0"
                        )
                    )
                    if duration <= 0:
                        raise RuntimeError("Browser video element has no playable duration.")
                    if duration <= 48:
                        timestamps = [
                            min(duration - 0.1, float(second))
                            for second in range(0, int(duration) + 1, 4)
                        ]
                    else:
                        timestamps = [
                            (duration - 0.2) * index / 11 for index in range(12)
                        ]
                    timestamps = sorted({max(0.0, value) for value in timestamps})
                    for index, timestamp in enumerate(timestamps):
                        page.evaluate(
                            """async ({timestamp}) => {
                                const video = document.querySelector('video');
                                video.muted = true;
                                video.currentTime = timestamp;
                                await new Promise((resolve) => {
                                  if (Math.abs(video.currentTime - timestamp) < 0.15 && video.readyState >= 2) {
                                    resolve();
                                    return;
                                  }
                                  const done = () => { video.removeEventListener('seeked', done); resolve(); };
                                  video.addEventListener('seeked', done, {once: true});
                                  setTimeout(done, 8000);
                                });
                            }""",
                            {"timestamp": timestamp},
                        )
                        path = frame_dir / f"frame-{index:02d}.jpg"
                        video.screenshot(path=str(path), type="jpeg", quality=80)
                        frame_paths.append((timestamp, path))
                finally:
                    browser.close()

            if not frame_paths:
                raise RuntimeError("Browser playback produced no video frames.")
            frames = [Image.open(path).convert("RGB") for _, path in frame_paths]
            cell_width = 480
            cell_height = int(frames[0].height * cell_width / frames[0].width)
            label_height = 28
            columns = 3
            rows = (len(frames) + columns - 1) // columns
            sheet = Image.new(
                "RGB",
                (columns * cell_width, rows * (cell_height + label_height)),
                "white",
            )
            draw = ImageDraw.Draw(sheet)
            for index, ((timestamp, _), frame) in enumerate(zip(frame_paths, frames)):
                left = (index % columns) * cell_width
                top = (index // columns) * (cell_height + label_height)
                resized = frame.resize((cell_width, cell_height))
                sheet.paste(resized, (left, top + label_height))
                draw.text((left + 8, top + 6), f"t={timestamp:.1f}s", fill="black")
            sheet_path = frame_dir / "contact-sheet.jpg"
            sheet.save(sheet_path, format="JPEG", quality=85)
            print(
                f"[video] yt-dlp unavailable; inspected {len(frame_paths)} "
                f"browser-decoded frames from {browser_url}",
                file=sys.stderr,
                flush=True,
            )
            if artifact_dir:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                kept_sheet = artifact_dir / f"browser-video-{len(media_events)}.jpg"
                shutil.copy2(sheet_path, kept_sheet)
            else:
                kept_sheet = sheet_path
            media.record(kind="video", backend="browser_frames", source=video_url,
                         status="submitted", frame_count=len(frame_paths), image_inputs=1,
                         timestamps=[timestamp for timestamp, _ in frame_paths],
                         artifacts=[str(kept_sheet)] if artifact_dir else [])
            return image_toolkit.ask_question_about_image(
                str(sheet_path),
                f"{question}\nThe image is a contact sheet of real decoded video "
                "frames; each frame has its timestamp printed immediately above it. "
                "Base the answer only on visible frame evidence and cite timestamps.",
            )

        def ask_question_about_video(video_path: str, question: str) -> str:
            """Download (when needed) and answer a question about a video."""
            source = video_path
            if is_youtube_url(video_path) and video_backend != "download":
                if native_video:
                    try:
                        return media.ask_youtube(video_path, question)
                    except Exception:
                        if video_backend == "native":
                            raise
                        print("[video] native YouTube request failed; trying local download", file=sys.stderr, flush=True)
                elif video_backend == "native":
                    raise ValueError("Native YouTube input requires Gemini through OpenRouter. Use OWL_VIDEO_BACKEND=download for other models.")
            parsed = urlparse(video_path)
            if parsed.scheme and parsed.netloc:
                # CAMEL currently forces yt-dlp's generic extractor and asks
                # for bestvideo+bestaudio, which fails on ordinary YouTube
                # links that do not expose that exact combination. Use the
                # native extractor and a progressive-stream fallback, then
                # hand the local file to CAMEL's frame analysis.
                import yt_dlp

                download_dir = cache_root / "video-downloads"
                download_dir.mkdir(parents=True, exist_ok=True)
                options = {
                    # Bilibili and many YouTube videos provide separate video
                    # and audio streams. Frame analysis only needs the video.
                    "format": "best[ext=mp4]/bestvideo[ext=mp4]/bestvideo/best",
                    "outtmpl": str(download_dir / "%(id)s.%(ext)s"),
                    "noplaylist": True,
                    "quiet": True,
                    "no_warnings": True,
                    "socket_timeout": 30,
                    "retries": 2,
                    "extractor_retries": 1,
                }
                if shutil.which("node"):
                    options["js_runtimes"] = {"node": {}}
                if ytdlp_cookie_file:
                    cookie_path = Path(ytdlp_cookie_file).expanduser()
                    if not cookie_path.is_file():
                        raise FileNotFoundError(
                            f"OWL_YTDLP_COOKIE_FILE does not exist: {cookie_path}"
                        )
                    options["cookiefile"] = str(cookie_path)
                if ytdlp_browser:
                    browser_parts = ytdlp_browser.split(":", 1)
                    options["cookiesfrombrowser"] = (
                        browser_parts[0],
                        browser_parts[1] if len(browser_parts) > 1 else None,
                        None,
                        None,
                    )
                if ytdlp_proxy:
                    options["proxy"] = ytdlp_proxy
                try:
                    with yt_dlp.YoutubeDL(options) as downloader:
                        info = downloader.extract_info(video_path, download=True)
                        video_path = downloader.prepare_filename(info)
                except Exception as download_error:
                    print(
                        f"[video] yt-dlp failed ({download_error}); trying "
                        "browser frame extraction",
                        file=sys.stderr,
                        flush=True,
                    )
                    return browser_video_frames(video_path, question)
                if not Path(video_path).exists():
                    candidates = sorted(download_dir.glob(f"{info.get('id', '*')}.*"))
                    if not candidates:
                        raise RuntimeError(
                            f"yt-dlp completed but no video file was found for {video_path}"
                        )
                    video_path = str(candidates[0])
                media.record(kind="video", backend="yt_dlp", source=source, status="downloaded",
                             extractor=info.get("extractor_key"), format_id=info.get("format_id"),
                             duration_seconds=info.get("duration"), bytes=Path(video_path).stat().st_size)
            answer = video_toolkit.ask_question_about_video(video_path, question)
            if answer.startswith(("Error:", "Failed to generate an answer")):
                media.record(kind="video", backend="decoded_frames", source=source, status="error", error=answer)
                raise RuntimeError(answer)
            media.record(kind="video", backend="decoded_frames", source=source, status="response_received")
            return answer

        ask_question_about_video_tracked = tracked(
            "ask_question_about_video", ask_question_about_video
        )
        multimodal_tools = [
            FunctionTool(tracked("image_to_text", image_toolkit.image_to_text)),
            FunctionTool(
                tracked("ask_question_about_image", image_toolkit.ask_question_about_image)
            ),
            FunctionTool(ask_question_about_video_tracked),
            FunctionTool(tracked("ask_question_about_pdf", media.ask_question_about_pdf)),
            FunctionTool(tracked("ask_question_about_audio", media.ask_question_about_audio)),
        ]
        if tool_registry is not None:
            tool_registry.update({
                "ask_question_about_video": ask_question_about_video_tracked,
                "ask_question_about_image": tracked("ask_question_about_image", image_toolkit.ask_question_about_image),
                "ask_question_about_pdf": tracked("ask_question_about_pdf", media.ask_question_about_pdf),
                "ask_question_about_audio": tracked("ask_question_about_audio", media.ask_question_about_audio),
            })

    multimodal_agent = ChatAgent(
        MULTIMODAL_AGENT_PROMPT,
        model=create_model("multimodal_worker"),
        tools=multimodal_tools,
        max_iteration=12,
        tool_execution_timeout=360,
        step_timeout=420,
    )
    reasoning_agent = ChatAgent(
        REASONING_AGENT_PROMPT,
        model=create_model("reasoning_worker"),
        max_iteration=8,
        step_timeout=660,
    )
    task_agent = ChatAgent(
        TASK_MANAGER_PROMPT,
        model=create_model("task_manager"),
        max_iteration=8,
        step_timeout=660,
    )
    coordinator_agent = ChatAgent(
        COORDINATOR_PROMPT,
        model=create_model("coordinator"),
        max_iteration=12,
        step_timeout=660,
    )
    new_worker_agent = ChatAgent(
        "You are an auxiliary OWL reasoning worker. Use only evidence supplied in the task; you have no retrieval tools.",
        model=create_model("auxiliary_worker"),
        max_iteration=8,
        step_timeout=660,
    )

    for agent in (
        web_agent,
        multimodal_agent,
        reasoning_agent,
        task_agent,
        coordinator_agent,
        new_worker_agent,
    ):
        _preserve_clone_step_timeout(agent)

    workforce_logger = WorkforceLogger(workforce_id="hyper-browsecomp-owl")
    workforce = Workforce(
        "HyperBrowseComp OWL Workforce",
        task_agent=task_agent,
        coordinator_agent=coordinator_agent,
        new_worker_agent=new_worker_agent,
        task_timeout_seconds=float(payload.get("task_timeout_seconds", 900)),
        callbacks=[workforce_logger],
    )
    workforce.add_single_agent_worker(
        "Searches the public web and visually browses rendered pages and screenshots.",
        worker=web_agent,
    )
    if multimodal:
        workforce.add_single_agent_worker(
            "Inspects images, videos, PDF pages and audio using the primary model.",
            worker=multimodal_agent,
        )
    workforce.add_single_agent_worker(
        "Synthesizes gathered evidence into the exact requested answer format.",
        worker=reasoning_agent,
    )

    capabilities = {
        "framework": "OWL Workforce (CAMEL)",
        "browser": True,
        "browser_uses_screenshots": True,
        "multimodal": multimodal,
        "video_access": {
            "backend": video_backend,
            "native_youtube_available": native_video,
            "cookie_file_configured": bool(ytdlp_cookie_file),
            "browser_cookies_configured": bool(ytdlp_browser),
            "proxy_configured": bool(ytdlp_proxy),
            "anonymous_browser_frame_fallback": True,
        },
        "image_analysis": multimodal,
        "video_analysis": multimodal,
        "pdf_analysis": multimodal,
        "audio_analysis": multimodal,
        "audio_requires_model_support": True,
        "search_backends": ["duckduckgo", "wikipedia"],
        "exa": False,
        "browser_round_limit": int(payload.get("browser_round_limit", 12)),
        "task_timeout_seconds": int(payload.get("task_timeout_seconds", 900)),
        "finalize_reserve_seconds": int(payload.get("finalize_reserve_seconds", 120)),
        "max_external_tool_calls": int(payload.get("max_external_tool_calls", 50)),
        "max_model_calls": int(payload.get("max_model_calls", 180)),
        "model_max_retries": int(payload.get("model_max_retries", 1)),
    }
    return (
        workforce,
        workforce_logger,
        capabilities,
        tool_usage,
        model_stats,
        cache_root,
    )


def _statistics(
    model_stats: dict[str, Any],
    tool_usage: dict[str, int],
    wall_time_seconds: float,
) -> dict[str, Any]:
    by_role = {
        role: {
            **values,
            "model_time_seconds": round(values["model_time_seconds"], 3),
        }
        for role, values in model_stats["by_role"].items()
    }
    return {
        "wall_time_seconds": round(wall_time_seconds, 3),
        "turns": len(model_stats["calls"]),
        "model_calls": model_stats["model_calls"],
        "model_call_attempts": model_stats["model_call_attempts"],
        "model_errors": model_stats["model_errors"],
        "input_tokens": model_stats["input_tokens"],
        "output_tokens": model_stats["output_tokens"],
        "reasoning_tokens": model_stats["reasoning_tokens"],
        "total_tokens": model_stats["total_tokens"],
        "model_time_seconds": round(model_stats["model_time_seconds"], 3),
        "tool_calls": sum(tool_usage.values()),
        "tool_calls_by_name": dict(tool_usage),
        "by_role": by_role,
        "limits": {
            "task_timeout_seconds": model_stats.get("task_timeout_seconds"),
            "finalize_reserve_seconds": model_stats.get("finalize_reserve_seconds"),
            "max_external_tool_calls": model_stats.get("max_external_tool_calls"),
            "max_model_calls": model_stats.get("max_model_calls"),
            "model_max_retries": model_stats.get("model_max_retries"),
        },
        "budget_events": list(model_stats.get("budget_events", [])),
    }


def run(payload: dict[str, Any]) -> dict[str, Any]:
    wall_started = time.perf_counter()
    tool_registry: dict[str, Callable[..., Any]] = {}
    (
        workforce,
        workforce_logger,
        capabilities,
        tool_usage,
        model_stats,
        cache_root,
    ) = build_workforce(payload, tool_registry)
    try:
        question = str(payload["question"])
        browser_round_limit = int(payload.get("browser_round_limit", 12))
        task_prompt = f"""\
{question}

Use the OWL workforce and its tools to research this question. The visual browser and visual analysis tools are enabled. When calling browse_url, set round_limit to at most {browser_round_limit}.
The complete task has a {int(payload.get("task_timeout_seconds", 900))}-second wall-clock limit and a global budget of {int(payload.get("max_external_tool_calls", 50))} external tool calls. Research tools close {int(payload.get("finalize_reserve_seconds", 120))} seconds before the hard deadline. If a tool reports that the research budget is closed, stop calling tools and synthesize the best supported answer immediately.

Your final response must use this exact format:

Explanation: {{brief evidence-based reasoning}}
Exact Answer: {{the shortest correct answer}}
Confidence: {{0-100%}}
"""
        if payload.get("tool_check"):
            # Diagnostic mode exercises the identical tool callable, with no
            # planning or judge; full evaluations still run the workforce.
            tool_name = str(payload["tool_check"])
            if tool_name not in tool_registry:
                raise ValueError(f"Unavailable multimodal tool: {tool_name}")
            completion = tool_registry[tool_name](**payload.get("tool_args", {}))
            workforce_state = "tool_check"
        else:
            processed_task = workforce.process_task(Task(content=task_prompt))
            completion = (processed_task.result or "").strip()
            workforce_state = processed_task.state.value
            has_formatted_answer = all(
                marker in completion
                for marker in ("Explanation:", "Exact Answer:", "Confidence:")
            )
            if processed_task.state != TaskState.DONE and not has_formatted_answer:
                raise RuntimeError(
                    f"OWL workforce ended in state {processed_task.state.value}: "
                    f"{completion or 'no result'}"
                )
        if not completion:
            raise RuntimeError("OWL workforce completed without a result.")
        return {
            "completion": completion,
            "termination_reason": model_stats.get("termination_reason") or "completed",
            "workforce_state": workforce_state,
            "capabilities": capabilities,
            "tool_usage": dict(tool_usage),
            "statistics": _statistics(
                model_stats, tool_usage, time.perf_counter() - wall_started
            ),
            "model_calls": model_stats["calls"],
            "media_events": model_stats["media_events"],
            "workforce_events": workforce_logger.log_entries,
        }
    except Exception as exc:
        reason = model_stats.get("termination_reason")
        if not reason:
            reason = getattr(exc, "reason", "worker_error")
        return {
            "completion": "",
            "termination_reason": reason,
            "error": f"{type(exc).__name__}: {exc}",
            "capabilities": capabilities,
            "tool_usage": dict(tool_usage),
            "statistics": _statistics(
                model_stats, tool_usage, time.perf_counter() - wall_started
            ),
            "model_calls": model_stats["calls"],
            "media_events": model_stats["media_events"],
            "workforce_events": workforce_logger.log_entries,
        }
    finally:
        shutil.rmtree(cache_root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Construct the configured toolkits and report capabilities without running a task.",
    )
    args = parser.parse_args(argv)

    load_dotenv(PROJECT_ROOT / ".env")
    set_log_level(level="INFO")
    payload = _read_payload()
    if args.self_check:
        (
            _,
            workforce_logger,
            capabilities,
            tool_usage,
            model_stats,
            cache_root,
        ) = build_workforce(payload)
        try:
            result = {
                "completion": "OWL self-check passed.",
                "capabilities": capabilities,
                "tool_usage": dict(tool_usage),
                "statistics": _statistics(model_stats, tool_usage, 0.0),
                "model_calls": model_stats["calls"],
                "workforce_events": workforce_logger.log_entries,
            }
        finally:
            shutil.rmtree(cache_root, ignore_errors=True)
    else:
        result = run(payload)
    print(f"{RESULT_PREFIX}{json.dumps(result, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
