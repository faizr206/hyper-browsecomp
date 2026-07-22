import pytest

from hyper_browsecomp.task import (
    DEFAULT_DOCKER_SANDBOX,
    parse_sample_range,
    resolve_sandbox,
    slice_dataset,
    web_tools,
)


def _tool_name(tool):
    return tool.__registry_info__.name


def test_slice_dataset_applies_range_and_num_samples() -> None:
    dataset = ["a", "b", "c", "d"]
    assert slice_dataset(dataset, start_index=1, end_index=4, num_samples=2) == ["b", "c"]


def test_sample_range_is_one_based_and_inclusive() -> None:
    dataset = ["a", "b", "c", "d"]
    assert parse_sample_range("1-2") == (0, 2)
    assert slice_dataset(dataset, sample_range="1-2") == ["a", "b"]
    assert slice_dataset(dataset, sample_range="3") == ["c"]


def test_sample_range_rejects_mixed_slice_options() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        slice_dataset(["a", "b"], sample_range="1-2", start_index=0)


def test_resolve_sandbox_can_disable_default() -> None:
    assert resolve_sandbox(None, no_sandbox=True) is None
    assert resolve_sandbox(None, no_sandbox=False) == DEFAULT_DOCKER_SANDBOX


def test_web_profile_exposes_only_search_and_fetch() -> None:
    tools = web_tools(
        tool_profile="web",
        search_backend="exa",
        fetch_backend="exa",
        model_provider="deepseek",
        search_max_results=5,
        search_timeout_seconds=60,
        fetch_timeout_seconds=60,
        fetch_max_chars=1000,
        bash_timeout=10,
        python_timeout=10,
    )
    assert [_tool_name(tool) for tool in tools] == ["web_search", "web_fetch"]


def test_web_code_profile_adds_bash_and_python() -> None:
    tools = web_tools(
        tool_profile="web_code",
        search_backend="exa",
        fetch_backend="firecrawl",
        model_provider="deepseek",
        search_max_results=5,
        search_timeout_seconds=60,
        fetch_timeout_seconds=60,
        fetch_max_chars=1000,
        bash_timeout=10,
        python_timeout=10,
    )
    names = [_tool_name(tool) for tool in tools]
    assert "web_search" in names
    assert "web_fetch" in names
    assert "inspect_ai/bash" in names
    assert "inspect_ai/python" in names


def test_internal_search_uses_inspect_web_search() -> None:
    tools = web_tools(
        tool_profile="web",
        search_backend="internal",
        fetch_backend="none",
        model_provider="openai",
        search_max_results=5,
        search_timeout_seconds=60,
        fetch_timeout_seconds=60,
        fetch_max_chars=1000,
        bash_timeout=10,
        python_timeout=10,
    )
    assert [_tool_name(tool) for tool in tools] == ["inspect_ai/web_search"]


def test_none_backends_disable_web_tools() -> None:
    tools = web_tools(
        tool_profile="web",
        search_backend="none",
        fetch_backend="none",
        model_provider="deepseek",
        search_max_results=5,
        search_timeout_seconds=60,
        fetch_timeout_seconds=60,
        fetch_max_chars=1000,
        bash_timeout=10,
        python_timeout=10,
    )
    assert tools == []


def test_internal_search_requires_supported_provider() -> None:
    with pytest.raises(ValueError, match="search_backend=internal"):
        web_tools(
            tool_profile="web",
            search_backend="internal",
            fetch_backend="none",
            model_provider="qwen",
            search_max_results=5,
            search_timeout_seconds=60,
            fetch_timeout_seconds=60,
            fetch_max_chars=1000,
            bash_timeout=10,
            python_timeout=10,
        )
