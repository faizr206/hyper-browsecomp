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
