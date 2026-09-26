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
HYPERBROWSECOMP_REDACTED_QUESTION = "[REDACTED_HYPERBROWSECOMP_QUESTION]"
HYPERBROWSECOMP_REDACTED_ANSWER = "[REDACTED_HYPERBROWSECOMP_ANSWER]"
_HYPERBROWSECOMP_CONFIDENTIAL: dict[str, dict[str, Any]] = {}


def parse_sample_range(sample_range: str | int) -> tuple[int, int]:
    parts = str(sample_range).split("-", 1)
    try:
        start_sample = int(parts[0].strip())
        end_sample = int(parts[1].strip()) if len(parts) == 2 else start_sample
    except ValueError as exc:
        raise ValueError(
            "sample_range must be a 1-based sample number or range like '1-2'."
        ) from exc

    if start_sample < 1 or end_sample < 1:
        raise ValueError("sample_range must use 1-based sample numbers greater than 0.")
    if end_sample < start_sample:
        raise ValueError("sample_range end must be >= start.")

    return start_sample - 1, end_sample


def slice_dataset(
    dataset: list,
    *,
    sample_range: str | int | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    num_samples: int | None = None,
) -> list:
    if sample_range is not None:
        if start_index is not None or end_index is not None or num_samples is not None:
            raise ValueError(
                "sample_range cannot be combined with start_index, end_index, or num_samples."
            )
        start_index, end_index = parse_sample_range(sample_range)

    if start_index is not None and start_index < 0:
        raise ValueError("start_index must be >= 0.")
    if end_index is not None and end_index < 0:
        raise ValueError("end_index must be >= 0.")
    if num_samples is not None and num_samples < 0:
        raise ValueError("num_samples must be >= 0.")
    if start_index is not None and end_index is not None and end_index < start_index:
        raise ValueError("end_index must be >= start_index.")

    selected = dataset[start_index:end_index]
    if num_samples is not None:
        selected = selected[:num_samples]
    return selected


class BrowseCompRecord(BaseModel):
    id: str
    question: str
    answers: list[str] = Field(min_length=1)
    answer_type: str = "entity"
    language: str = "en"
    modalities: list[str] = Field(default_factory=lambda: ["web"])
    image_urls: list[str] = Field(default_factory=list)
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
    if item.image_urls:
        metadata["image_urls"] = item.image_urls

    return Sample(
        id=item.id,
        input=item.question,
        target=item.answers,
        metadata=metadata,
    )


def confidential_question(sample_id: str | int, fallback: str) -> str:
    item = _HYPERBROWSECOMP_CONFIDENTIAL.get(str(sample_id))
    if item and isinstance(item.get("question"), str):
        return item["question"]
    return fallback


def confidential_answers(sample_id: str | int, fallback: list[str]) -> list[str]:
    item = _HYPERBROWSECOMP_CONFIDENTIAL.get(str(sample_id))
    if item and isinstance(item.get("answers"), list):
        return [str(answer) for answer in item["answers"]]
    return fallback


def confidential_log_replacements() -> list[tuple[str, str]]:
    replacements: list[tuple[str, str]] = []
    for sample_id, item in _HYPERBROWSECOMP_CONFIDENTIAL.items():
        question = item.get("question")
        if isinstance(question, str) and question:
            replacements.append(
                (question, f"{HYPERBROWSECOMP_REDACTED_QUESTION}:{sample_id}")
            )
    return sorted(replacements, key=lambda item: len(item[0]), reverse=True)


def confidential_answer_replacements() -> list[tuple[str, str]]:
    replacements: list[tuple[str, str]] = []
    for sample_id, item in _HYPERBROWSECOMP_CONFIDENTIAL.items():
        for answer in item.get("answers", []):
            if isinstance(answer, str) and answer:
                replacements.append(
                    (answer, f"{HYPERBROWSECOMP_REDACTED_ANSWER}:{sample_id}")
                )
    return sorted(replacements, key=lambda item: len(item[0]), reverse=True)


def load_sample_ids(path: str | Path) -> list[str]:
    """Read one unique sample ID per line, resolving paths from the project root."""
    ids_path = Path(path)
    if not ids_path.is_absolute():
        ids_path = PROJECT_ROOT / ids_path
    sample_ids: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(ids_path.read_text(encoding="utf-8").splitlines(), start=1):
        sample_id = line.strip()
        if not sample_id:
            raise ValueError(f"Empty sample ID in {ids_path}:{line_number}.")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample ID in {ids_path}:{line_number}: {sample_id}")
        seen.add(sample_id)
        sample_ids.append(sample_id)
    if not sample_ids:
        raise ValueError(f"Sample ID file contains no IDs: {ids_path}")
    return sample_ids


def select_samples_by_ids(
    samples: list[Sample], sample_ids_file: str | Path | None
) -> list[Sample]:
    """Select the exact requested IDs in file order without exposing sample content."""
    if sample_ids_file is None:
        return samples
    sample_ids = load_sample_ids(sample_ids_file)
    samples_by_id: dict[str, Sample] = {}
    for sample in samples:
        if sample.id is None:
            raise ValueError("Sample ID selection requires every dataset sample to have an ID.")
        sample_id = str(sample.id)
        if sample_id in samples_by_id:
            raise ValueError(f"Dataset contains duplicate sample ID: {sample_id}")
        samples_by_id[sample_id] = sample
    missing = [sample_id for sample_id in sample_ids if sample_id not in samples_by_id]
    if missing:
        raise ValueError(f"Sample IDs not found in dataset: {', '.join(missing)}")
    return [samples_by_id[sample_id] for sample_id in sample_ids]


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


def _hyperbrowsecomp_record_id(row: dict[str, Any], *, index: int) -> str:
    if "id" not in row or row["id"] is None:
        return f"hyperbrowsecomp-{index:03d}"
    record_id = str(row["id"]).strip()
    return record_id or f"hyperbrowsecomp-{index:03d}"


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

    record_id = _hyperbrowsecomp_record_id(row, index=index)
    _HYPERBROWSECOMP_CONFIDENTIAL[record_id] = {
        "question": question,
        "answers": [answer],
    }

    return Sample(
        id=record_id,
        input=f"{HYPERBROWSECOMP_REDACTED_QUESTION}:{record_id}",
        target=[f"{HYPERBROWSECOMP_REDACTED_ANSWER}:{record_id}"],
        metadata={
            "id": record_id,
            "answer_type": "entity",
            "language": str(row.get("language") or "en"),
            "modalities": ["web"],
            "source_metadata": {
                "dataset": HYPERBROWSECOMP_HF_DATASET,
                "hf_id": record_id,
                "canary": canary,
            },
            "confidential": True,
        },
    )


def load_hyperbrowsecomp_hf(dataset_name: str = HYPERBROWSECOMP_HF_DATASET) -> list[Sample]:
    _HYPERBROWSECOMP_CONFIDENTIAL.clear()
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
