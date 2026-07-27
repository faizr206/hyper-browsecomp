import base64
import sys
from types import ModuleType

import pytest

from hyper_browsecomp.dataset import (
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
    assert samples[0].id == "hyperbrowsecomp-001"
    assert samples[0].input == "Question?"
    assert samples[0].target == ["Answer"]
    assert samples[0].metadata["language"] == "ko"
