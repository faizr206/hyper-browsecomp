from types import SimpleNamespace
import sys

import pytest

from owl_runtime.media_tools import MediaTools, input_media_counts, is_youtube_url, native_youtube_supported
from hyper_browsecomp.owl_harness import _sanitize_console_line


def test_native_youtube_is_limited_to_supported_provider_and_real_hosts():
    assert is_youtube_url("https://www.youtube.com/watch?v=test")
    assert is_youtube_url("https://youtu.be/test")
    assert not is_youtube_url("https://youtube.com.example.org/watch?v=test")
    assert not is_youtube_url("https://bilibili.com/video/test")
    assert native_youtube_supported("google/gemini-3.7-flash", "https://openrouter.ai/api/v1")
    assert not native_youtube_supported("qwen/qwen3-vl", "https://openrouter.ai/api/v1")
    assert not native_youtube_supported("z-ai/glm-4.6v", "https://openrouter.ai/api/v1")
    assert not native_youtube_supported("google/gemini-3.7-flash", "https://other.example/api")


def test_media_counts_ignore_text_urls_and_count_actual_content_parts():
    messages = [{"role": "user", "content": "Look at https://youtube.com/watch?v=test"}]
    assert input_media_counts(messages) == {"image_inputs": 0, "video_inputs": 0, "audio_inputs": 0}
    messages.append({"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,a"}},
        {"type": "video_url", "video_url": {"url": "https://youtu.be/test"}},
        {"type": "input_audio", "input_audio": {"data": "abc", "format": "wav"}},
    ]})
    assert input_media_counts(messages) == {"image_inputs": 1, "video_inputs": 1, "audio_inputs": 1}


def test_audio_bytes_are_sent_to_primary_model_and_not_written_to_events(tmp_path):
    calls = []
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF-example-audio")
    model = SimpleNamespace(run=lambda messages: calls.append(messages) or SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="spoken words"))]))
    roles = []
    def create_model(role):
        roles.append(role)
        return model
    events = []
    media = MediaTools(create_model, tmp_path, events)
    assert media.ask_question_about_audio(str(audio), "What is spoken?") == "spoken words"
    assert roles == ["audio_analysis"]
    part = calls[0][0]["content"][1]
    assert part["type"] == "input_audio"
    assert part["input_audio"]["format"] == "wav"
    assert all("data" not in event for event in events)


def test_native_youtube_failure_is_recorded_and_propagated(tmp_path):
    def fail(messages):
        raise RuntimeError("provider rejected video")
    events = []
    media = MediaTools(lambda role: SimpleNamespace(run=fail), tmp_path, events)
    with pytest.raises(RuntimeError, match="provider rejected"):
        media.ask_youtube("https://youtu.be/test", "What is visible?")
    assert [event["status"] for event in events] == ["submitted", "error"]


def test_console_redacts_audio_payloads_and_keeps_usage():
    raw = "{'input_audio': {'data': '" + "A" * 1000 + "', 'format': 'wav'}, 'audio_tokens': 123}"
    sanitized = _sanitize_console_line(raw)
    assert "A" * 100 not in sanitized
    assert "audio_tokens': 123" in sanitized
    assert "[INLINE_MEDIA_BASE64_REDACTED]" in sanitized
    assert "BASE64_REDACTED" in _sanitize_console_line("data:video/mp4;base64,AAAA")


def test_pdf_renders_and_closes_resources_without_context_manager(tmp_path, monkeypatch):
    closed = []
    calls = []
    source = tmp_path / "test.pdf"
    source.write_bytes(b"pdf fixture")
    image = SimpleNamespace(save=lambda stream, format: stream.write(b"png fixture"))
    bitmap = SimpleNamespace(
        to_pil=lambda: SimpleNamespace(convert=lambda mode: image),
        close=lambda: closed.append("bitmap"),
    )
    page = SimpleNamespace(
        get_size=lambda: (600, 800), render=lambda scale: bitmap,
        close=lambda: closed.append("page"),
    )

    class Document:
        # pypdfium2 4.x intentionally has no context-manager protocol.
        def __len__(self):
            return 2

        def __getitem__(self, index):
            assert index == 0
            return page

        def close(self):
            closed.append("document")

    monkeypatch.setitem(sys.modules, "pypdfium2", SimpleNamespace(PdfDocument=lambda path: Document()))
    model = SimpleNamespace(run=lambda messages: calls.append(messages) or SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="visible text"))]))
    events = []
    media = MediaTools(lambda role: model, tmp_path, events, tmp_path / "artifacts")
    answer = media.ask_question_about_pdf(str(source), "Read the page", "1,1")
    assert "pages [1] of 2" in answer
    assert closed == ["bitmap", "page", "document"]
    assert input_media_counts(calls[0])["image_inputs"] == 1
    assert events[0]["total_pages"] == 2
    assert len(events[0]["artifacts"]) == 1
    assert (tmp_path / "artifacts/pdf-test-page-1.png").read_bytes() == b"png fixture"
    with pytest.raises(ValueError, match="only 2 pages"):
        media.ask_question_about_pdf(str(source), "Read the page", "3")
    assert closed[-1] == "document"
    assert len(calls) == 1
