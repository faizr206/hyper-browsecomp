from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


WHITESPACE_RE = re.compile(r"\s+")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {jsonl_path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected object in {jsonl_path} at line {line_number}, "
                    f"got {type(record).__name__}"
                )
            yield record


def ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().strip()
    kept: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category.startswith("P") and char not in {".", ",", "-", "%"}:
            kept.append(" ")
        else:
            kept.append(char)
    return WHITESPACE_RE.sub(" ", "".join(kept)).strip()


def clear_substring(needle: str, haystack: str) -> bool:
    if not needle:
        return False
    pattern = rf"(?<!\S){re.escape(needle)}(?!\S)"
    return re.search(pattern, haystack) is not None


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-._")
    return cleaned or "unnamed"
