from __future__ import annotations

from pathlib import Path
from typing import Any

from inspect_ai.dataset import Sample
from pydantic import BaseModel, Field, ValidationError

from hyper_browsecomp.utils import read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
