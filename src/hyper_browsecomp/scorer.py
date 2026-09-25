from __future__ import annotations

import math
import re

from inspect_ai import model as inspect_model
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    accuracy,
    metric,
    scorer,
    stderr,
    value_to_float,
)
from inspect_ai.solver import TaskState

from hyper_browsecomp.prompts import GRADER_TEMPLATE
from hyper_browsecomp.dataset import (
    confidential_answers,
    confidential_question,
    load_browsecomp_dataset,
)
from hyper_browsecomp.utils import clear_substring, ensure_list, normalize_text


FINAL_ANSWER_RE = re.compile(r"^\s*FINAL_ANSWER\s*:\s*(?P<answer>.+?)\s*$", re.I | re.M)
EXACT_ANSWER_RE = re.compile(r"^\s*Exact Answer\s*:\s*(?P<answer>.+?)\s*$", re.I | re.M)
CONFIDENCE_RE = re.compile(r"^\s*Confidence\s*:\s*(?P<confidence>\d{1,3})(?:\s*%)?\s*$", re.I | re.M)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:[,\s]\d{3})+|\d+)(?:[.,]\d+)?%?")


def extract_final_answer(completion: str) -> str:
    exact_match = EXACT_ANSWER_RE.search(completion)
    if exact_match:
        return exact_match.group("answer").strip()
    match = FINAL_ANSWER_RE.search(completion)
    if match:
        return match.group("answer").strip()
    return completion.strip()


def extract_confidence(completion: str) -> int:
    match = CONFIDENCE_RE.search(completion)
    if not match:
        return 100
    return max(0, min(int(match.group("confidence")), 100))


def _target_values(target: Target) -> list[str]:
    text = getattr(target, "text", target)
    return ensure_list(text)


def _state_metadata(state: TaskState) -> dict:
    metadata = getattr(state, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    sample = getattr(state, "sample", None)
    sample_metadata = getattr(sample, "metadata", None)
    if isinstance(sample_metadata, dict):
        return sample_metadata
    return {}


def _parse_number(value: str) -> float | None:
    token = value.strip().replace(" ", "")
    if token.endswith("%"):
        token = token[:-1]
    if "," in token and "." in token:
        token = token.replace(",", "")
    elif "," in token:
        parts = token.split(",")
        if len(parts[-1]) == 3 and all(len(part) == 3 for part in parts[1:]):
            token = "".join(parts)
        else:
            token = token.replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def extract_numbers(value: str) -> list[float]:
    numbers: list[float] = []
    for match in NUMBER_RE.finditer(value):
        parsed = _parse_number(match.group(0))
        if parsed is not None and math.isfinite(parsed):
            numbers.append(parsed)
    return numbers


def _numeric_match(prediction: str, golds: list[str], *, atol: float) -> bool:
    pred_numbers = extract_numbers(prediction)
    if not pred_numbers:
        return False

    for gold in golds:
        for gold_number in extract_numbers(gold):
            if any(math.isclose(pred, gold_number, rel_tol=0.0, abs_tol=atol) for pred in pred_numbers):
                return True
    return False


def _entity_match(prediction: str, golds: list[str]) -> bool:
    normalized_prediction = normalize_text(prediction)
    for gold in golds:
        normalized_gold = normalize_text(gold)
        if not normalized_gold:
            continue
        if normalized_prediction == normalized_gold:
            return True
        if clear_substring(normalized_gold, normalized_prediction):
            return True
    return False


@metric
def browse_comp_accuracy() -> Metric:
    to_float = value_to_float()

    def metric_func(scores: list[SampleScore]) -> float:
        values = []
        for item in scores:
            if isinstance(item.score.value, dict) and "score" in item.score.value:
                values.append(to_float(item.score.value["score"]))
        return sum(values) / len(values) if values else 0.0

    return metric_func


@metric
def calibration_error(num_bins: int = 10) -> Metric:
    to_float = value_to_float()

    def metric_func(scores: list[SampleScore]) -> float:
        predictions: list[float] = []
        confidences: list[float] = []
        for item in scores:
            value = item.score.value
            if not isinstance(value, dict):
                continue
            predictions.append(to_float(value.get("score", 0)))
            confidences.append(float(value.get("confidence", 100)) / 100.0)

        if not predictions:
            return 0.0

        error = 0.0
        for bin_index in range(num_bins):
            lower = bin_index / num_bins
            upper = (bin_index + 1) / num_bins
            in_bin = [
                index
                for index, confidence in enumerate(confidences)
                if (lower <= confidence <= upper if bin_index == num_bins - 1 else lower <= confidence < upper)
            ]
            if not in_bin:
                continue
            bin_accuracy = sum(predictions[index] for index in in_bin) / len(in_bin)
            bin_confidence = sum(confidences[index] for index in in_bin) / len(in_bin)
            error += (len(in_bin) / len(predictions)) * abs(bin_accuracy - bin_confidence)
        return error

    return metric_func


def _target_for_grader(target: Target) -> str:
    return " OR ".join(_target_values(target))


def _sample_id(state: TaskState) -> str:
    metadata = _state_metadata(state)
    return str(metadata.get("id") or getattr(state, "sample_id", "unknown"))


def _is_confidential_sample(state: TaskState) -> bool:
    return bool(_state_metadata(state).get("confidential"))


@scorer(metrics=[browse_comp_accuracy(), calibration_error()])
def browse_comp_scorer(scorer_model: str | inspect_model.Model) -> Scorer:
    judge = (
        inspect_model.get_model(scorer_model)
        if isinstance(scorer_model, str)
        else scorer_model
    )

    async def score(state: TaskState, target: Target) -> Score:
        completion = getattr(getattr(state, "output", None), "completion", "") or ""
        grading_response = await judge.generate(
            GRADER_TEMPLATE.format(
                question=confidential_question(_sample_id(state), state.input_text),
                response=completion,
                correct_answer=" OR ".join(
                    confidential_answers(_sample_id(state), _target_values(target))
                ),
            )
        )
        grader_text = grading_response.completion

        correct_match = re.search(r"correct:\s*(yes|no)", grader_text, re.I)
        is_correct = correct_match.group(1).casefold() == "yes" if correct_match else False
        confidence_match = re.search(r"confidence:\s*(\d{1,3})", grader_text, re.I)
        confidence = min(int(confidence_match.group(1)), 100) if confidence_match else extract_confidence(completion)
        reasoning_match = re.search(r"reasoning:\s*(.*?)(?=\n\s*correct:|$)", grader_text, re.I | re.S)
        answer_match = re.search(r"extracted_final_answer:\s*(.*)", grader_text, re.I)

        explanation = reasoning_match.group(1).strip() if reasoning_match else grader_text
        if _is_confidential_sample(state):
            explanation = "Grader reasoning hidden for confidential sample."

        return Score(
            value={"score": CORRECT if is_correct else INCORRECT, "confidence": confidence},
            answer=answer_match.group(1).strip() if answer_match else extract_final_answer(completion),
            explanation=explanation,
        )

    return score


@scorer(metrics=[browse_comp_accuracy(), calibration_error()])
def browse_comp_rescorer(
    scorer_model: str | inspect_model.Model,
    confidential_data_path: str | None = None,
) -> Scorer:
    """Re-run the BrowseComp judge only for samples that already have a score."""
    if confidential_data_path:
        # Published logs redact confidential questions and targets. Loading the
        # encrypted dataset repopulates the in-memory lookup used by the scorer.
        load_browsecomp_dataset(confidential_data_path)
    rescore = browse_comp_scorer(scorer_model)

    async def score(state: TaskState, target: Target) -> Score | None:
        # Failed/incomplete samples have no original score. Leaving them unscored
        # keeps the rescored metric denominator identical to the original one.
        if not state.scores:
            return None
        return await rescore(state, target)

    return score
