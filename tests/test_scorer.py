import asyncio
from types import SimpleNamespace

from inspect_ai.scorer import CORRECT, Score, Target

from hyper_browsecomp.scorer import browse_comp_rescorer


class FakeJudge:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, _prompt: str):
        self.calls += 1
        return SimpleNamespace(
            completion=(
                "extracted_final_answer: gold\n"
                "reasoning: It matches.\n"
                "correct: yes\n"
                "confidence: 80"
            )
        )


def state_with_scores(scores: dict) -> SimpleNamespace:
    return SimpleNamespace(
        scores=scores,
        output=SimpleNamespace(completion="Exact Answer: gold\nConfidence: 80%"),
        input_text="question",
        metadata={"id": "sample-1"},
        sample_id="sample-1",
    )


def test_rescorer_skips_samples_without_an_original_score() -> None:
    judge = FakeJudge()
    rescorer = browse_comp_rescorer(judge)

    result = asyncio.run(rescorer(state_with_scores({}), Target(["gold"])))

    assert result is None
    assert judge.calls == 0


def test_rescorer_scores_samples_that_have_an_original_score() -> None:
    judge = FakeJudge()
    rescorer = browse_comp_rescorer(judge)

    result = asyncio.run(
        rescorer(
            state_with_scores({"browse_comp_scorer": Score(value=CORRECT)}),
            Target(["gold"]),
        )
    )

    assert result is not None
    assert result.value == {"score": CORRECT, "confidence": 80}
    assert judge.calls == 1
