import pytest

from hyper_browsecomp.tools.web import BackendError, _truncate_text


def test_truncate_text_limits_output() -> None:
    assert _truncate_text("abcdef", 4).startswith("abcd")


def test_missing_backend_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    from hyper_browsecomp.tools.web import _require_env

    with pytest.raises(BackendError, match="EXA_API_KEY"):
        _require_env("EXA_API_KEY")
