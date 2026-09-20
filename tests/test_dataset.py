import base64
import sys
from types import ModuleType

import pytest

from hyper_browsecomp.dataset import (
    HYPERBROWSECOMP_REDACTED_ANSWER,
    HYPERBROWSECOMP_REDACTED_QUESTION,
    confidential_answers,
    confidential_log_replacements,
    confidential_question,
    _hyperbrowsecomp_nonce,
    load_browsecomp_dataset,
    load_browsecomp_jsonl,
    record_to_sample,
)


def test_record_to_sample_keeps_metadata() -> None:
    sample = record_to_sample(
        {
            "id": "q1",
            "question": "Who?",
            "answers": ["Ada"],
            "answer_type": "entity",
            "language": "en",
        }
    )
    assert sample.id == "q1"
    assert sample.metadata["id"] == "q1"
    assert sample.target == ["Ada"]


def test_record_to_sample_keeps_image_urls_for_multimodal_harness() -> None:
    sample = record_to_sample(
        {
            "id": "visual-1",
            "question": "What is shown?",
            "answers": ["A logo"],
            "modalities": ["web", "image"],
            "image_urls": ["https://example.com/logo.png"],
        }
    )
    assert sample.metadata["image_urls"] == ["https://example.com/logo.png"]


def test_load_browsecomp_jsonl_rejects_empty(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="contains no samples"):
        load_browsecomp_jsonl(path)


def test_dev_dataset_loads() -> None:
    samples = load_browsecomp_jsonl("data/dev.jsonl")
    assert len(samples) == 1
    assert (
        samples[0].input
        == "As of July 22, 2026, what is the latest stable Python 3 release listed on the official Python website?"
    )
    assert samples[0].target == ["Python 3.14.6"]


def test_load_browsecomp_dataset_still_supports_jsonl() -> None:
    samples = load_browsecomp_dataset("data/dev.jsonl")
    assert len(samples) == 1
    assert samples[0].id == "dev-001"


def test_load_browsecomp_dataset_decrypts_hyperbrowsecomp(monkeypatch) -> None:
    cryptography = pytest.importorskip("cryptography.hazmat.primitives.ciphers.aead")

    key = bytes(range(32))
    canary = "test-canary"

    def encrypt(field: str, plaintext: str) -> str:
        aesgcm = cryptography.AESGCM(key)
        ciphertext = aesgcm.encrypt(_hyperbrowsecomp_nonce(canary, field), plaintext.encode(), None)
        return base64.b64encode(ciphertext).decode()

    fake_datasets = ModuleType("datasets")
    fake_datasets.load_dataset = lambda dataset_name, split: [
            {
                "id": "hbc-123",
                "language": "ko",
                "canary": canary,
                "question": encrypt("question", "Question?"),
            "answer": encrypt("answer", "Answer"),
        }
    ]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setenv("HYPERBROWSECOMP_KEY", key.hex())

    samples = load_browsecomp_dataset("afaji/HyperBrowseComp")

    assert len(samples) == 1
    assert samples[0].id == "hbc-123"
    assert samples[0].input == f"{HYPERBROWSECOMP_REDACTED_QUESTION}:hbc-123"
    assert samples[0].target == [f"{HYPERBROWSECOMP_REDACTED_ANSWER}:hbc-123"]
    assert samples[0].metadata["language"] == "ko"
    assert samples[0].metadata["source_metadata"]["hf_id"] == "hbc-123"
    assert confidential_question("hbc-123", "fallback") == "Question?"
    assert confidential_answers("hbc-123", ["fallback"]) == ["Answer"]
    assert confidential_log_replacements() == [
        ("Question?", f"{HYPERBROWSECOMP_REDACTED_QUESTION}:hbc-123")
    ]


def test_load_browsecomp_dataset_preserves_zero_hyperbrowsecomp_id(monkeypatch) -> None:
    cryptography = pytest.importorskip("cryptography.hazmat.primitives.ciphers.aead")

    key = bytes(range(32))
    canary = "test-canary-zero"

    def encrypt(field: str, plaintext: str) -> str:
        aesgcm = cryptography.AESGCM(key)
        ciphertext = aesgcm.encrypt(_hyperbrowsecomp_nonce(canary, field), plaintext.encode(), None)
        return base64.b64encode(ciphertext).decode()

    fake_datasets = ModuleType("datasets")
    fake_datasets.load_dataset = lambda dataset_name, split: [
        {
            "id": 0,
            "language": "en",
            "canary": canary,
            "question": encrypt("question", "Zero?"),
            "answer": encrypt("answer", "Zero"),
        }
    ]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setenv("HYPERBROWSECOMP_KEY", key.hex())

    samples = load_browsecomp_dataset("afaji/HyperBrowseComp")

    assert samples[0].id == "0"
    assert samples[0].metadata["id"] == "0"
    assert samples[0].metadata["source_metadata"]["hf_id"] == "0"
