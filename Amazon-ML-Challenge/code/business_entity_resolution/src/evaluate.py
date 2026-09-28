"""Evaluation helpers matching the challenge's per-entity F0.5 objective."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MatchScore:
    candidate_id: str
    score: float


def fbeta_for_entity(
    truth: set[str],
    prediction: set[str],
    beta: float = 0.5,
) -> float:
    if not truth and not prediction:
        return 1.0
    if not truth or not prediction:
        return 0.0

    true_positive = len(truth & prediction)
    if true_positive == 0:
        return 0.0
    precision = true_positive / len(prediction)
    recall = true_positive / len(truth)
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    if denominator == 0:
        return 0.0
    return (1 + beta_squared) * precision * recall / denominator


def evaluate_predictions(
    expected_source1_ids: Iterable[str],
    truth_by_source1: Mapping[str, set[str]],
    predictions_by_source1: Mapping[str, set[str]],
    beta: float = 0.5,
) -> dict[str, float | int]:
    entity_ids = list(expected_source1_ids)
    if not entity_ids:
        return {
            "entities": 0,
            "macro_fbeta": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "predicted_matches": 0,
            "true_matches": 0,
            "correct_matches": 0,
            "candidate_to_match_rate": 0.0,
        }

    score_sum = 0.0
    predicted_total = 0
    truth_total = 0
    correct_total = 0
    predicted_nonempty = 0
    for source1_id in entity_ids:
        truth = truth_by_source1.get(source1_id, set())
        prediction = predictions_by_source1.get(source1_id, set())
        score_sum += fbeta_for_entity(truth, prediction, beta)
        predicted_total += len(prediction)
        truth_total += len(truth)
        correct_total += len(truth & prediction)
        predicted_nonempty += int(bool(prediction))

    return {
        "entities": len(entity_ids),
        "macro_fbeta": score_sum / len(entity_ids),
        "micro_precision": correct_total / predicted_total if predicted_total else 0.0,
        "micro_recall": correct_total / truth_total if truth_total else 0.0,
        "predicted_matches": predicted_total,
        "true_matches": truth_total,
        "correct_matches": correct_total,
        "candidate_to_match_rate": predicted_nonempty / len(entity_ids),
    }


def select_predictions(
    scores_by_source1: Mapping[str, Sequence[MatchScore]],
    threshold: float,
    strategy: str = "threshold",
    top_k: int | None = None,
    margin: float = 0.0,
) -> dict[str, set[str]]:
    if strategy not in {"threshold", "top_k", "margin"}:
        raise ValueError(f"Unknown decision strategy: {strategy}")
    predictions: dict[str, set[str]] = {}
    for source1_id, candidates in scores_by_source1.items():
        ranked = sorted(candidates, key=lambda candidate: (-candidate.score, candidate.candidate_id))
        selected = [candidate for candidate in ranked if candidate.score >= threshold]
        if strategy == "top_k" and top_k is not None:
            selected = selected[:top_k]
        elif strategy == "margin" and selected:
            best_score = ranked[0].score
            selected = [candidate for candidate in selected if best_score - candidate.score <= margin]
        predictions[source1_id] = {candidate.candidate_id for candidate in selected}
    return predictions


def threshold_search(
    expected_source1_ids: Iterable[str],
    truth_by_source1: Mapping[str, set[str]],
    scores_by_source1: Mapping[str, Sequence[MatchScore]],
    thresholds: Sequence[float],
    strategies: Sequence[tuple[str, int | None, float]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    source_ids = list(expected_source1_ids)
    for threshold in thresholds:
        for strategy, top_k, margin in strategies:
            predictions = select_predictions(
                scores_by_source1,
                threshold,
                strategy=strategy,
                top_k=top_k,
                margin=margin,
            )
            metrics = evaluate_predictions(source_ids, truth_by_source1, predictions)
            results.append(
                {
                    "threshold": threshold,
                    "strategy": strategy,
                    "top_k": top_k,
                    "margin": margin,
                    **metrics,
                }
            )
    return sorted(results, key=lambda result: (-result["macro_fbeta"], result["predicted_matches"]))