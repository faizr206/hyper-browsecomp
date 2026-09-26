#!/usr/bin/env python3
"""Render an OWL Inspect eval and its JSON sidecars as readable Markdown."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log


def text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(text(item) for item in value)
    return str(value)


def one_line(value: Any, limit: int = 180) -> str:
    value = " ".join(text(value).split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def cell(value: Any) -> str:
    return one_line(value).replace("|", "\\|") or "-"


def local_link(label: str, path: str | Path | None) -> str:
    if not path:
        return "-"
    resolved = Path(path).expanduser().resolve()
    return f"[{label}]({resolved.as_posix()})"


def score_summary(scores: dict[str, Any] | None) -> tuple[str, str]:
    if not scores:
        return "-", ""
    values = []
    explanations = []
    for name, raw in scores.items():
        score = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
        value = score.get("value")
        if isinstance(value, dict):
            rendered = ", ".join(f"{key}={item}" for key, item in value.items())
        else:
            rendered = text(value)
        values.append(f"{name}: {rendered}")
        if score.get("explanation"):
            explanations.append(f"**{name}:** {score['explanation']}")
    return "; ".join(values), "\n\n".join(explanations)


def trace_cost(trace: dict[str, Any]) -> float:
    total = 0.0
    for call in trace.get("model_calls", []):
        usage = call.get("usage_details") or {}
        cost = usage.get("cost")
        if cost in (None, 0):
            cost = (usage.get("cost_details") or {}).get(
                "upstream_inference_cost"
            )
        if isinstance(cost, (int, float)):
            total += float(cost)
    return total


def error_message(error: Any) -> str:
    if error is None:
        return ""
    if hasattr(error, "message"):
        return text(error.message)
    if isinstance(error, dict) and error.get("message"):
        return text(error["message"])
    rendered = text(error)
    match = re.search(r'message=["\'](.+?)["\'] traceback=', rendered, re.DOTALL)
    return match.group(1) if match else rendered.splitlines()[0]


def load_trace(
    metadata: dict[str, Any], error: Any = None
) -> tuple[dict[str, Any], dict[str, str]]:
    owl = metadata.get("owl_harness", {})
    paths = dict(owl.get("trace_files", {}))
    events_path = paths.get("events")
    if not events_path:
        match = re.search(
            r"(?:OWL\s+)?trace:\s+([^\s'\"]+\.json)",
            error_message(error),
            re.IGNORECASE,
        )
        if match:
            events_path = match.group(1)
            paths["events"] = events_path
            paths["console"] = str(Path(events_path).with_suffix(".log"))
    if not events_path or not Path(events_path).is_file():
        return {}, paths
    return json.loads(Path(events_path).read_text(encoding="utf-8")), paths


def render_model_calls(lines: list[str], calls: list[dict[str, Any]]) -> None:
    lines.extend([
        "### Model calls",
        "",
        "| # | Role | Provider | Status | Seconds | Input | Output | Reasoning | Media | Cost |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ])
    for call in calls:
        media = ", ".join(
            f"{kind}={call.get(kind + '_inputs', 0)}"
            for kind in ("image", "video", "audio")
            if call.get(kind + "_inputs", 0)
        ) or "-"
        usage = call.get("usage_details", {})
        cost = usage.get("cost")
        if cost in (None, 0):
            cost = (usage.get("cost_details") or {}).get("upstream_inference_cost")
        lines.append(
            "| {turn} | {role} | {provider} | {status} | {seconds} | {input} | "
            "{output} | {reasoning} | {media} | {cost} |".format(
                turn=call.get("turn", "-"), role=cell(call.get("role")),
                provider=cell(call.get("provider")), status=cell(call.get("status")),
                seconds=call.get("duration_seconds", "-"),
                input=call.get("input_tokens", "-"), output=call.get("output_tokens", "-"),
                reasoning=call.get("reasoning_tokens", "-"), media=cell(media),
                cost=f"${cost:.6f}" if isinstance(cost, (int, float)) else "-",
            )
        )
    if not calls:
        lines.append("| - | - | - | - | - | - | - | - | - | - |")
    lines.append("")


def render_media(
    lines: list[str],
    events: list[dict[str, Any]],
    model_calls: list[dict[str, Any]],
) -> None:
    lines.extend([
        "### Media evidence",
        "",
        "| Kind | Backend | Status | Submitted input | Source/artifacts |",
        "| --- | --- | --- | --- | --- |",
    ])
    for event in events:
        submitted = []
        for key in ("image_inputs", "video_inputs", "audio_inputs", "frame_count", "bytes"):
            if event.get(key) is not None:
                submitted.append(f"{key}={event[key]}")
        locations = []
        if event.get("source"):
            locations.append(text(event["source"]))
        locations.extend(local_link(Path(path).name, path) for path in event.get("artifacts", []))
        lines.append(
            f"| {cell(event.get('kind'))} | {cell(event.get('backend'))} | "
            f"{cell(event.get('status'))} | {cell(', '.join(submitted))} | "
            f"{cell('; '.join(locations))} |"
        )
    if not events:
        media_calls = [
            call for call in model_calls
            if any(call.get(f"{kind}_inputs", 0) for kind in ("image", "video", "audio"))
        ]
        if media_calls:
            call = max(
                media_calls,
                key=lambda item: sum(
                    item.get(f"{kind}_inputs", 0) or 0
                    for kind in ("image", "video", "audio")
                ),
            )
            submitted = ", ".join(
                f"{kind}_inputs={call.get(kind + '_inputs', 0)}"
                for kind in ("image", "video", "audio")
                if call.get(kind + "_inputs", 0)
            )
            kind = "frame batch (inferred)" if call.get("image_inputs", 0) > 1 else "model media input"
            lines.append(
                f"| {kind} | {cell(call.get('provider'))} | {cell(call.get('status'))} | "
                f"{cell(submitted)} | model call #{call.get('turn', '-')}; explicit media hook not emitted |"
            )
        else:
            lines.append("| - | - | No recorded media input | - | - |")
    lines.append("")


def render_timeline(lines: list[str], events: list[dict[str, Any]]) -> None:
    workers = {
        event.get("worker_id"): event.get("role")
        for event in events if event.get("event_type") == "worker_created"
    }
    lines.extend([
        "### Workforce timeline",
        "",
        "| Time (UTC) | Event | Worker | Task/result |",
        "| --- | --- | --- | --- |",
    ])
    for event in events:
        event_type = event.get("event_type", "unknown")
        if event_type == "worker_created":
            continue
        timestamp = text(event.get("timestamp"))
        when = timestamp[11:19] if len(timestamp) >= 19 else timestamp
        worker = workers.get(event.get("worker_id"), event.get("worker_id", "-"))
        detail = event.get("description") or event.get("result_summary")
        if not detail and event.get("subtask_ids"):
            detail = "subtasks: " + ", ".join(event["subtask_ids"])
        if not detail:
            detail = event.get("task_id", "-")
        lines.append(f"| {cell(when)} | {cell(event_type)} | {cell(worker)} | {cell(detail)} |")
    if not events:
        lines.append("| - | No workforce events recorded | - | - |")
    lines.append("")


def render_report(eval_path: Path) -> str:
    log = read_eval_log(str(eval_path))
    samples = list(log.samples or [])
    lines = [
        "# OWL evaluation trace report",
        "",
        f"- Eval: {local_link(eval_path.name, eval_path)}",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Status: `{log.status}`",
        f"- Samples: {len(samples)}",
        "",
        "## Summary",
        "",
        "| Sample | Score | Termination | Seconds | Calls/attempts | Tools | Input tokens | Output tokens | Cost | Trace |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    loaded: list[tuple[Any, dict[str, Any], dict[str, str]]] = []
    aggregate = {
        "completed": 0, "wall": 0.0, "calls": 0, "tools": 0,
        "input": 0, "output": 0, "reasoning": 0, "total": 0, "cost": 0.0,
        "terminations": {},
    }
    for sample in samples:
        metadata = sample.metadata or {}
        trace, paths = load_trace(metadata, sample.error)
        loaded.append((sample, trace, paths))
        stats = ((metadata.get("owl_harness") or {}).get("statistics")
                 or trace.get("statistics", {}))
        cost = trace_cost(trace)
        termination = (
            (metadata.get("owl_harness") or {}).get("termination_reason")
            or trace.get("termination_reason")
            or "not_recorded"
        )
        aggregate["terminations"][termination] = (
            aggregate["terminations"].get(termination, 0) + 1
        )
        if stats:
            aggregate["completed"] += 1
            aggregate["wall"] += stats.get("wall_time_seconds", 0) or 0
            aggregate["calls"] += stats.get("model_calls", 0) or 0
            aggregate["tools"] += stats.get("tool_calls", 0) or 0
            aggregate["input"] += stats.get("input_tokens", 0) or 0
            aggregate["output"] += stats.get("output_tokens", 0) or 0
            aggregate["reasoning"] += stats.get("reasoning_tokens", 0) or 0
            aggregate["total"] += stats.get("total_tokens", 0) or 0
            aggregate["cost"] += cost
        score, _ = score_summary(sample.scores)
        trace_links = " · ".join(
            part for part in (
                local_link("log", paths.get("console")), local_link("json", paths.get("events"))
            ) if part != "-"
        ) or "-"
        lines.append(
            f"| {cell(sample.id)} | {cell(score)} | "
            f"{cell(termination)} | "
            f"{stats.get('wall_time_seconds', '-')} | "
            f"{stats.get('model_calls', '-')}/{stats.get('model_call_attempts', '-')} | "
            f"{stats.get('tool_calls', '-')} | "
            f"{stats.get('input_tokens', '-')} | {stats.get('output_tokens', '-')} | "
            f"{'$' + format(cost, '.4f') if cost else '-'} | {trace_links} |"
        )
    lines.extend([
        "",
        f"Traces with statistics: {aggregate['completed']}/{len(samples)}; "
        f"wall time: {aggregate['wall']:.3f}s; model calls: {aggregate['calls']}; "
        f"tool calls: {aggregate['tools']}.",
        "",
        "Termination counts: `"
        + json.dumps(aggregate["terminations"], sort_keys=True)
        + "`",
        "",
        f"Recorded usage: {aggregate['input']:,} input + {aggregate['output']:,} output "
        f"tokens ({aggregate['reasoning']:,} reasoning; {aggregate['total']:,} total). "
        f"Recorded upstream cost: ${aggregate['cost']:.6f}.",
        "",
    ])

    for index, (sample, trace, paths) in enumerate(loaded, 1):
        metadata = sample.metadata or {}
        owl = metadata.get("owl_harness", {})
        stats = owl.get("statistics") or trace.get("statistics", {})
        termination_reason = owl.get("termination_reason") or trace.get("termination_reason")
        score, score_explanation = score_summary(sample.scores)
        completion = getattr(sample.output, "completion", "") if sample.output else ""
        error = sample.error or trace.get("error")
        lines.extend([
            f"## {index}. {sample.id}",
            "",
            "### Question",
            "",
            text(sample.input),
            "",
            "### Reference answer",
            "",
            text(sample.target) or "Not available.",
            "",
            "### OWL final answer",
            "",
            completion or "No completion.",
            "",
            "### Judge",
            "",
            f"- Score: {score}",
            f"- Explanation: {score_explanation or 'Not available.'}",
            f"- Error: {one_line(error_message(error), 600) or 'None'}",
            f"- Termination reason: {termination_reason or 'Not recorded'}",
            "",
            "### Statistics",
            "",
            "| Wall seconds | Model calls | Attempts | Model errors | Tool calls | Input | Output | Reasoning | Total | Model seconds |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {stats.get('wall_time_seconds', '-')} | {stats.get('model_calls', '-')} | "
            f"{stats.get('model_call_attempts', '-')} | "
            f"{stats.get('model_errors', '-')} | {stats.get('tool_calls', '-')} | "
            f"{stats.get('input_tokens', '-')} | {stats.get('output_tokens', '-')} | "
            f"{stats.get('reasoning_tokens', '-')} | {stats.get('total_tokens', '-')} | "
            f"{stats.get('model_time_seconds', '-')} |",
            "",
            "Tool counts: `" + json.dumps(stats.get("tool_calls_by_name", {}), sort_keys=True) + "`",
            "",
            "Configured limits: `" + json.dumps(stats.get("limits", {}), sort_keys=True) + "`",
            "",
            "Budget events: `" + json.dumps(stats.get("budget_events", []), sort_keys=True) + "`",
            "",
        ])
        render_model_calls(lines, trace.get("model_calls", []))
        render_media(
            lines,
            trace.get("media_events", owl.get("media_events", [])),
            trace.get("model_calls", []),
        )
        render_timeline(lines, trace.get("workforce_events", []))
        lines.extend([
            "### Raw files",
            "",
            f"- {local_link('Console trace', paths.get('console'))}",
            f"- {local_link('Structured trace', paths.get('events'))}",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    eval_path = args.eval_log.expanduser().resolve()
    output = args.output or eval_path.with_suffix(".traces.md")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(eval_path), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
