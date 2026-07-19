import pytest

from hyper_browsecomp.dataset import load_browsecomp_jsonl, record_to_sample


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
    assert sample.metadata["id"] == "q1"
    assert sample.target == ["Ada"]


def test_load_browsecomp_jsonl_rejects_empty(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="contains no samples"):
        load_browsecomp_jsonl(path)
