from pathlib import Path

import pytest
from inspect_ai.dataset import Sample

from hyper_browsecomp import dataset
from hyper_browsecomp.dataset import load_sample_ids, select_samples_by_ids


def test_sample_selection_uses_file_order_and_preserves_redacted_samples(tmp_path: Path) -> None:
    samples = [
        Sample(
            id=sample_id,
            input=f"[REDACTED_HYPERBROWSECOMP_QUESTION]:{sample_id}",
            target=[f"[REDACTED_HYPERBROWSECOMP_ANSWER]:{sample_id}"],
            metadata={"confidential": True},
        )
        for sample_id in ["a", "b", "c"]
    ]
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("c\na\n", encoding="utf-8")
    selected = select_samples_by_ids(samples, ids_file)
    assert [sample.id for sample in selected] == ["c", "a"]
    assert selected[0] is samples[2]
    assert selected[1] is samples[0]


def test_relative_sample_ids_path_is_independent_of_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "ids.txt").write_text("b\na\n", encoding="utf-8")
    (tmp_path / "ids.txt").write_text("wrong-id\n", encoding="utf-8")
    monkeypatch.setattr(dataset, "PROJECT_ROOT", project_root)
    monkeypatch.chdir(tmp_path)
    assert load_sample_ids("ids.txt") == ["b", "a"]


@pytest.mark.parametrize(
    ("contents", "error"),
    [
        ("", "contains no IDs"),
        ("\n", "Empty sample ID"),
        ("a\n\nb\n", "Empty sample ID"),
        ("a\na\n", "Duplicate sample ID"),
        ("a\n a \n", "Duplicate sample ID"),
    ],
)
def test_sample_ids_reject_invalid_files(tmp_path: Path, contents: str, error: str) -> None:
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_sample_ids(ids_file)


def test_selection_rejects_missing_ids(tmp_path: Path) -> None:
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("a\nmissing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Sample IDs not found in dataset: missing"):
        select_samples_by_ids([Sample(id="a", input="redacted")], ids_file)


def test_selection_rejects_duplicate_dataset_ids(tmp_path: Path) -> None:
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Dataset contains duplicate sample ID: a"):
        select_samples_by_ids(
            [Sample(id="a", input="redacted"), Sample(id="a", input="redacted")], ids_file
        )


def test_selection_rejects_dataset_samples_without_ids(tmp_path: Path) -> None:
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires every dataset sample to have an ID"):
        select_samples_by_ids([Sample(input="redacted")], ids_file)


def test_selection_supports_numeric_zero_id(tmp_path: Path) -> None:
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("0\n", encoding="utf-8")
    sample = Sample(id=0, input="redacted")
    assert select_samples_by_ids([sample], ids_file) == [sample]


def test_no_sample_ids_file_keeps_original_dataset() -> None:
    samples = [Sample(id="a", input="redacted")]
    assert select_samples_by_ids(samples, None) is samples


def test_retained_ids_config_contains_423_unique_ids() -> None:
    retained_ids = load_sample_ids("configs/internal_full/retained_ids.txt")
    assert len(retained_ids) == 423
    assert len(set(retained_ids)) == 423
