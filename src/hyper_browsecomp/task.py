from __future__ import annotations

from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai import model as inspect_model
from inspect_ai.agent import AgentState, react, run
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, bash, python, web_search as inspect_web_search
from inspect_ai.util import SandboxEnvironmentType

from hyper_browsecomp.dataset import load_browsecomp_jsonl
from hyper_browsecomp.prompts import QUERY_TEMPLATE, WEB_ONLY_AGENT_PROMPT
from hyper_browsecomp.scorer import browse_comp_scorer
from hyper_browsecomp.tools import build_web_fetch_tool, build_web_search_tool


ToolProfile = Literal["web", "web_code"]
SearchBackend = Literal["internal", "exa", "firecrawl", "none"]
FetchBackend = Literal["exa", "firecrawl", "none"]
DEFAULT_DOCKER_SANDBOX: SandboxEnvironmentType = "docker"
INTERNAL_SEARCH_PROVIDERS = {"openai", "anthropic", "gemini", "grok", "mistral", "perplexity"}
FINALIZE_AFTER_MAX_STEPS_PROMPT = """\
You have reached the maximum research step limit. Do not call any more tools.
Using only the information already gathered in this conversation, provide your best final answer now.
Use this format:

Explanation: <brief evidence-based reasoning>
Exact Answer: <the shortest correct answer; use your best guess if evidence is incomplete>
Confidence: <0-100%>
"""


def parse_sample_range(sample_range: str | int) -> tuple[int, int]:
    parts = str(sample_range).split("-", 1)
    try:
        start_sample = int(parts[0].strip())
        end_sample = int(parts[1].strip()) if len(parts) == 2 else start_sample
    except ValueError as exc:
        raise ValueError("sample_range must be a 1-based sample number or range like '1-2'.") from exc

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


def resolve_sandbox(
    sandbox: SandboxEnvironmentType | None,
    *,
    no_sandbox: bool = False,
) -> SandboxEnvironmentType | None:
    if no_sandbox:
        return None
    return sandbox or DEFAULT_DOCKER_SANDBOX


def sample_id(state: TaskState) -> str:
    metadata = getattr(state, "metadata", {}) or {}
    return str(metadata.get("id") or getattr(state, "sample_id", "unknown"))


def agent_prompt_for_backends(
    *,
    search_backend: SearchBackend,
    fetch_backend: FetchBackend,
    tool_profile: ToolProfile,
) -> str:
    retrieval_instructions: list[str] = []
    if search_backend == "internal":
        retrieval_instructions.append(
            "Use web_search for internet retrieval. This uses the model provider's built-in web search."
        )
    elif search_backend != "none":
        retrieval_instructions.append("Use web_search first to find candidate sources.")

    if fetch_backend != "none":
        retrieval_instructions.append("Use web_fetch to retrieve relevant page content.")
    elif search_backend != "none":
        retrieval_instructions.append("web_fetch is disabled; rely on web_search results and citations.")

    if not retrieval_instructions:
        retrieval_instructions.append(
            "Web search and fetch are disabled; answer from the model context and any non-web tools available."
        )

    if tool_profile == "web_code":
        retrieval_instructions.append(
            "Use bash or python only when you genuinely need computation, parsing, or transformation."
        )
    else:
        retrieval_instructions.append("Do not mention tools that are not available.")

    return WEB_ONLY_AGENT_PROMPT.replace(
        "If you need information from the internet, use web_search first to find candidate sources, then use web_fetch to retrieve the relevant page content. Prefer official and primary sources whenever possible.\n\n"
        "Do not use bash or python for internet retrieval. Use bash or python only when the web tools are insufficient and you genuinely need computation, parsing, or transformation that cannot be done from the tool outputs alone.",
        "\n".join(retrieval_instructions),
    )


async def final_answer_from_context(active_model, messages: list) -> str:
    response = await active_model.generate(messages)
    return response.completion.strip()


def web_tools(
    *,
    tool_profile: ToolProfile,
    search_backend: SearchBackend,
    fetch_backend: FetchBackend,
    model_provider: str | None,
    search_max_results: int,
    search_timeout_seconds: int,
    fetch_timeout_seconds: int,
    fetch_max_chars: int,
    bash_timeout: int,
    python_timeout: int,
) -> list[Tool]:
    tools: list[Tool] = []
    if search_backend == "internal":
        if model_provider not in INTERNAL_SEARCH_PROVIDERS:
            raise ValueError(
                "search_backend=internal requires provider to be one of: "
                f"{', '.join(sorted(INTERNAL_SEARCH_PROVIDERS))}."
            )
        tools.append(inspect_web_search(providers=model_provider))
    elif search_backend != "none":
        tools.append(
            build_web_search_tool(
                backend=search_backend,
                max_results=search_max_results,
                timeout_seconds=search_timeout_seconds,
            )
        )

    if fetch_backend != "none":
        tools.append(
            build_web_fetch_tool(
                backend=fetch_backend,
                timeout_seconds=fetch_timeout_seconds,
                max_chars=fetch_max_chars,
            )
        )

    if tool_profile == "web_code":
        tools.extend([bash(timeout=bash_timeout), python(timeout=python_timeout)])
    return tools


@solver
def web_research_solver(
    *,
    tool_profile: ToolProfile = "web",
    max_steps: int = 12,
    search_backend: SearchBackend = "exa",
    fetch_backend: FetchBackend = "exa",
    model_provider: str | None = None,
    search_max_results: int = 5,
    search_timeout_seconds: int = 60,
    fetch_timeout_seconds: int = 60,
    fetch_max_chars: int = 20000,
    bash_timeout: int = 120,
    python_timeout: int = 120,
) -> Solver:
    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        active_model = inspect_model.get_model()
        step_count = 0
        reached_limit = False

        async def limit_steps(_agent_state: AgentState) -> bool:
            nonlocal step_count, reached_limit
            step_count += 1
            if step_count >= max_steps:
                reached_limit = True
                return False
            return True

        agent = react(
            name="hyper_browsecomp_agent",
            description="Web research assistant for BrowseComp-style tasks",
            prompt=agent_prompt_for_backends(
                search_backend=search_backend,
                fetch_backend=fetch_backend,
                tool_profile=tool_profile,
            ),
            model=active_model,
            tools=web_tools(
                tool_profile=tool_profile,
                search_backend=search_backend,
                fetch_backend=fetch_backend,
                model_provider=model_provider,
                search_max_results=search_max_results,
                search_timeout_seconds=search_timeout_seconds,
                fetch_timeout_seconds=fetch_timeout_seconds,
                fetch_max_chars=fetch_max_chars,
                bash_timeout=bash_timeout,
                python_timeout=python_timeout,
            ),
            on_continue=limit_steps,
        )
        agent_state = await run(agent, QUERY_TEMPLATE.format(question=state.input_text))
        if reached_limit and not agent_state.output.completion.strip():
            messages = [
                *agent_state.messages,
                ChatMessageUser(content=FINALIZE_AFTER_MAX_STEPS_PROMPT),
            ]
            state.output.completion = await final_answer_from_context(active_model, messages)
        else:
            state.output.completion = agent_state.output.completion
        return state

    return solve


@task
def hyper_browsecomp(
    data_path: str = "data/dev.jsonl",
    tool_profile: ToolProfile = "web",
    search_backend: SearchBackend = "exa",
    fetch_backend: FetchBackend = "exa",
    model_provider: str | None = None,
    search_max_results: int = 5,
    search_timeout_seconds: int = 60,
    fetch_timeout_seconds: int = 60,
    fetch_max_chars: int = 20000,
    max_steps: int = 12,
    bash_timeout: int = 120,
    python_timeout: int = 120,
    sample_range: str | int | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    num_samples: int | None = None,
    scorer_model: str | inspect_model.Model | None = None,
    sandbox: SandboxEnvironmentType | None = None,
    no_sandbox: bool = True,
) -> Task:
    if tool_profile == "web_code" and no_sandbox:
        raise ValueError("tool_profile=web_code requires no_sandbox=false.")

    if scorer_model is None:
        judge_model = inspect_model.get_model()
    elif isinstance(scorer_model, str):
        judge_model = inspect_model.get_model(scorer_model)
    else:
        judge_model = scorer_model

    dataset = slice_dataset(
        load_browsecomp_jsonl(data_path),
        sample_range=sample_range,
        start_index=start_index,
        end_index=end_index,
        num_samples=num_samples,
    )
    if not dataset:
        raise ValueError(
            "Dataset slice contains no samples. Check sample_range, start_index, end_index, and num_samples."
        )

    return Task(
        dataset=dataset,
        solver=web_research_solver(
            tool_profile=tool_profile,
            max_steps=max_steps,
            search_backend=search_backend,
            fetch_backend=fetch_backend,
            model_provider=model_provider,
            search_max_results=search_max_results,
            search_timeout_seconds=search_timeout_seconds,
            fetch_timeout_seconds=fetch_timeout_seconds,
            fetch_max_chars=fetch_max_chars,
            bash_timeout=bash_timeout,
            python_timeout=python_timeout,
        ),
        scorer=browse_comp_scorer(judge_model),
        sandbox=resolve_sandbox(sandbox, no_sandbox=no_sandbox),
    )
