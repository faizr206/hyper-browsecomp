"""Media adapters for OWL; all perception uses the configured primary model."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


def is_youtube_url(value: str) -> bool:
    host = (urlparse(value).hostname or "").lower()
    return host in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def native_youtube_supported(model_name: str, base_url: str) -> bool:
    return (
        urlparse(base_url).hostname == "openrouter.ai"
        and model_name.startswith("google/gemini-")
    )


def input_media_counts(messages: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"image_inputs": 0, "video_inputs": 0, "audio_inputs": 0}
    fields = {"image_url": "image_inputs", "video_url": "video_inputs", "input_audio": "audio_inputs"}
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in fields:
                    counts[fields[part["type"]]] += 1
    return counts


class MediaTools:
    def __init__(
        self,
        create_model: Callable[[str], Any],
        cache_root: Path,
        events: list[dict[str, Any]],
        artifact_dir: Path | None = None,
    ) -> None:
        self.create_model = create_model
        self.cache_root = cache_root
        self.events = events
        self.artifact_dir = artifact_dir

    def record(self, **event: Any) -> None:
        self.events.append(event)
        print("[media] " + json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True)

    def local_file(self, source: str, max_bytes: int = 32 * 1024 * 1024) -> Path:
        import httpx

        parsed = urlparse(source)
        if parsed.scheme not in {"http", "https"}:
            path = Path(source).expanduser()
            if not path.is_file():
                raise FileNotFoundError(source)
            if path.stat().st_size > max_bytes:
                raise ValueError("Media file exceeds the 32 MiB input limit.")
            return path
        suffix = Path(parsed.path).suffix[:12]
        path = self.cache_root / (hashlib.sha256(source.encode()).hexdigest()[:20] + suffix)
        if path.exists():
            return path
        try:
            with httpx.stream("GET", source, follow_redirects=True, timeout=45) as response:
                response.raise_for_status()
                total = 0
                with path.open("wb") as output:
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total > max_bytes:
                            raise ValueError("Media download exceeds the 32 MiB input limit.")
                        output.write(chunk)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path

    def ask(self, role: str, content: list[dict[str, Any]]) -> str:
        result = self.create_model(role).run([{"role": "user", "content": content}])
        if not result.choices or not result.choices[0].message.content:
            raise RuntimeError(f"{role} returned no answer.")
        return result.choices[0].message.content

    def ask_youtube(self, video_url: str, question: str) -> str:
        if not is_youtube_url(video_url):
            raise ValueError("Native YouTube input requires a YouTube URL.")
        self.record(kind="video", backend="openrouter_youtube", source=video_url,
                    status="submitted", provider="google-ai-studio", video_inputs=1,
                    local_frames=False)
        try:
            answer = self.ask("youtube_native", [
                {"type": "text", "text": question + "\nInspect the supplied video. Cite visible evidence and timestamps. If it is inaccessible, explicitly report that; do not answer from memory."},
                {"type": "video_url", "video_url": {"url": video_url}},
            ])
        except Exception as exc:
            self.record(kind="video", backend="openrouter_youtube", source=video_url,
                        status="error", error=str(exc))
            raise
        self.record(kind="video", backend="openrouter_youtube", source=video_url,
                    status="response_received", local_frames=False)
        return "[Video supplied to the primary model via OpenRouter/Google AI Studio; no local frame extraction.]\n" + answer

    def ask_question_about_pdf(self, pdf_path: str, question: str, pages: str = "1") -> str:
        """Render selected PDF pages and inspect them visually. pages is a comma-separated list of 1-based page numbers, at most 8 per call."""
        import pypdfium2 as pdfium

        requested = list(dict.fromkeys(int(value.strip()) for value in pages.split(",")))
        if not requested or len(requested) > 8 or min(requested) < 1:
            raise ValueError("Choose 1 to 8 page numbers, starting at 1.")
        path = self.local_file(pdf_path)
        content: list[dict[str, Any]] = [{"type": "text", "text": question + "\nUse only the rendered PDF pages; cite page numbers."}]
        artifacts = []
        document = pdfium.PdfDocument(path)
        try:
            page_count = len(document)
            if max(requested) > page_count:
                raise ValueError(f"PDF contains only {page_count} pages.")
            for number in requested:
                page = document[number - 1]
                try:
                    scale = min(1.5, 1600 / max(page.get_size()))
                    bitmap = page.render(scale=scale)
                    try:
                        image = bitmap.to_pil().convert("RGB")
                        encoded = io.BytesIO()
                        image.save(encoded, format="PNG")
                        if self.artifact_dir:
                            self.artifact_dir.mkdir(parents=True, exist_ok=True)
                            artifact = self.artifact_dir / f"pdf-{path.stem}-page-{number}.png"
                            artifact.write_bytes(encoded.getvalue())
                            artifacts.append(str(artifact))
                        content.extend([
                            {"type": "text", "text": f"PDF page {number} of {page_count}:"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode()}},
                        ])
                    finally:
                        bitmap.close()
                finally:
                    page.close()
        finally:
            document.close()
        self.record(kind="pdf", backend="rendered_pages", source=pdf_path,
                    status="submitted", pages=requested, total_pages=page_count,
                    image_inputs=len(requested), artifacts=artifacts)
        answer = self.ask("pdf_analysis", content)
        self.record(kind="pdf", backend="rendered_pages", source=pdf_path, status="response_received")
        return f"[Rendered pages {requested} of {page_count}; the PDF may contain other uninspected pages.]\n{answer}"

    def ask_question_about_audio(self, audio_path: str, question: str) -> str:
        """Listen to a local or public audio file with the primary model. Requires audio input support; WAV, MP3, FLAC, OGG, M4A, AAC, AIFF supported."""
        path = self.local_file(audio_path)
        audio_format = path.suffix.lower().lstrip(".")
        if audio_format not in {"wav", "mp3", "flac", "ogg", "m4a", "aac", "aiff"}:
            raise ValueError("Provide a supported audio file with a WAV/MP3/FLAC/OGG/M4A/AAC/AIFF extension.")
        data = path.read_bytes()
        self.record(kind="audio", backend="native_audio", source=audio_path,
                    status="submitted", format=audio_format, bytes=len(data), audio_inputs=1)
        answer = self.ask("audio_analysis", [
            {"type": "text", "text": question + "\nAnswer from the attached audio; report any inability to hear it."},
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(data).decode(), "format": audio_format}},
        ])
        self.record(kind="audio", backend="native_audio", source=audio_path, status="response_received")
        return answer
