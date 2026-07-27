from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from inspect_ai.dataset import Sample
from pydantic import BaseModel, Field, ValidationError

from hyper_browsecomp.utils import read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HYPERBROWSECOMP_HF_DATASET = "afaji/HyperBrowseComp"
HYPERBROWSECOMP_KEY_ENV_NAMES = (
    "HYPERBROWSECOMP_KEY",
    "HYPER_BROWSECOMP_KEY",
    "HYPERBROWSECOMP_MASTER_KEY",
)


class BrowseCompRecord(BaseModel):
    id: str
    question: str
    answers: list[str] = Field(min_length=1)
    answer_type: str = "entity"
    language: str = "en"
    modalities: list[str] = Field(default_factory=lambda: ["web"])
    source_metadata: dict[str, str] = Field(default_factory=dict)


def record_to_sample(record: dict[str, Any], *, source: str = "<record>") -> Sample:
    try:
        item = BrowseCompRecord.model_validate(record)
    except ValidationError as exc:
        raise ValueError(f"Invalid BrowseComp record in {source}: {exc}") from exc

    metadata = {
        "id": item.id,
        "answer_type": item.answer_type,
        "language": item.language,
        "modalities": item.modalities,
    }
    if item.source_metadata:
        metadata["source_metadata"] = item.source_metadata

    return Sample(
        id=item.id,
        input=item.question,
        target=item.answers,
        metadata=metadata,
    )


def load_browsecomp_jsonl(path: str | Path) -> list[Sample]:
    dataset_path = Path(path)
    if not dataset_path.exists() and not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    samples = [
        record_to_sample(record, source=f"{dataset_path}:{index}")
        for index, record in enumerate(read_jsonl(dataset_path), start=1)
    ]
    if not samples:
        raise ValueError(f"Dataset contains no samples: {dataset_path}")
    return samples


def _load_optional_dependency(module_name: str, package_name: str):
    try:
        return __import__(module_name, fromlist=[""])
    except ImportError as exc:
        raise RuntimeError(
            f"Loading {HYPERBROWSECOMP_HF_DATASET} requires {package_name}. "
            f"Install project dependencies with `uv sync`."
        ) from exc


def _hyperbrowsecomp_key() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    for env_name in HYPERBROWSECOMP_KEY_ENV_NAMES:
        value = os.environ.get(env_name)
        if value:
            return value.strip()
    names = ", ".join(HYPERBROWSECOMP_KEY_ENV_NAMES)
    raise RuntimeError(
        f"{HYPERBROWSECOMP_HF_DATASET} requires a decryption key in .env "
        f"using one of: {names}."
    )


def _hyperbrowsecomp_nonce(canary: str, field: str) -> bytes:
    return hashlib.sha256(f"{canary}:{field}".encode("utf-8")).digest()[:12]


def _decrypt_hyperbrowsecomp_field(
    *,
    ciphertext_b64: str,
    key: bytes,
    canary: str,
    field: str,
) -> str:
    crypto_aead = _load_optional_dependency(
        "cryptography.hazmat.primitives.ciphers.aead",
        "cryptography",
    )
    aesgcm = crypto_aead.AESGCM(key)
    ciphertext = base64.b64decode(ciphertext_b64)
    return aesgcm.decrypt(_hyperbrowsecomp_nonce(canary, field), ciphertext, None).decode("utf-8")


def _hyperbrowsecomp_record_to_sample(row: dict[str, Any], *, index: int) -> Sample:
    key_text = _hyperbrowsecomp_key()
    try:
        key = bytes.fromhex(key_text)
    except ValueError as exc:
        raise ValueError("HyperBrowseComp key must be a hex-encoded AES-256 key.") from exc
    if len(key) != 32:
        raise ValueError("HyperBrowseComp key must decode to 32 bytes for AES-256-GCM.")

    canary = row.get("canary")
    if not isinstance(canary, str) or not canary:
        raise ValueError(f"HyperBrowseComp row {index} is missing a canary.")

    question = _decrypt_hyperbrowsecomp_field(
        ciphertext_b64=str(row["question"]),
        key=key,
        canary=canary,
        field="question",
    )
    answer = _decrypt_hyperbrowsecomp_field(
        ciphertext_b64=str(row["answer"]),
        key=key,
        canary=canary,
        field="answer",
    )

    record_id = str(row.get("id") or f"hyperbrowsecomp-{index:03d}")
    return record_to_sample(
        {
            "id": record_id,
            "question": question,
            "answers": [answer],
            "answer_type": "entity",
            "language": str(row.get("language") or "en"),
            "source_metadata": {
                "dataset": HYPERBROWSECOMP_HF_DATASET,
                "canary": canary,
            },
        },
        source=f"{HYPERBROWSECOMP_HF_DATASET}:test[{index - 1}]",
    )


def load_hyperbrowsecomp_hf(dataset_name: str = HYPERBROWSECOMP_HF_DATASET) -> list[Sample]:
    datasets = _load_optional_dependency("datasets", "datasets")
    hf_dataset = datasets.load_dataset(dataset_name, split="test")
    samples = [
        _hyperbrowsecomp_record_to_sample(dict(row), index=index)
        for index, row in enumerate(hf_dataset, start=1)
    ]
    if not samples:
        raise ValueError(f"Dataset contains no samples: {dataset_name}")
    return samples


def load_browsecomp_dataset(path: str | Path) -> list[Sample]:
    path_text = str(path)
    if path_text in {HYPERBROWSECOMP_HF_DATASET, f"hf://{HYPERBROWSECOMP_HF_DATASET}"}:
        return load_hyperbrowsecomp_hf()
    return load_browsecomp_jsonl(path)
