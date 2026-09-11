from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    cohen_kappa_score,
    mean_absolute_error,
)

from backend.app.services.text_utils import flesch_kincaid_grade

# cache the scorer per model - reloading roberta-large on every single call was segfaulting after a while
_BERTSCORER_CACHE: dict[str, "object"] = {}


def _bertscore_f1(predictions: list[str], references: list[str], model_type: str) -> list[float]:
    from bert_score import BERTScorer

    scorer = _BERTSCORER_CACHE.get(model_type)
    if scorer is None:
        scorer = BERTScorer(model_type=model_type, lang="en", rescale_with_baseline=False)
        _BERTSCORER_CACHE[model_type] = scorer
    _, _, f1 = scorer.score(predictions, references)
    return [float(value) for value in f1]


@dataclass
class RQ1Metrics:
    sari: float
    bertscore_f1: float
    mean_fk_grade: float
    pct_fk_le_8: float
    pct_fk_le_9: float
    n_examples: int


@dataclass
class BinaryMetrics:
    precision: float
    recall: float
    f1: float
    unsupported_precision: float
    unsupported_recall: float
    unsupported_f1: float
    accuracy: float
    predicted_hallucination_rate: float
    ground_truth_hallucination_rate: float
    missed_unsupported_rate: float
    confusion_matrix: list[list[int]]
    n_examples: int


def rq1_metrics(
    sources: list[str],
    predictions: list[str],
    references: list[str],
    *,
    bertscore_model: str,
) -> RQ1Metrics:
    if not sources or not (len(sources) == len(predictions) == len(references)):
        raise ValueError("RQ1 lists must be equally sized and non-empty")
    quality = rq1_example_quality(sources, predictions, references, bertscore_model=bertscore_model)
    sari = float(np.mean([item["sari"] for item in quality]))
    bert = float(np.mean([item["bertscore_f1"] for item in quality]))
    grades = [item["fk_grade"] for item in quality]
    return RQ1Metrics(
        round(sari, 6),
        round(bert, 6),
        round(float(np.mean(grades)), 6),
        round(float(np.mean([x <= 8 for x in grades])), 6),
        round(float(np.mean([x <= 9 for x in grades])), 6),
        len(sources),
    )


def rq1_example_quality(
    sources: list[str],
    predictions: list[str],
    references: list[str],
    *,
    bertscore_model: str,
) -> list[dict[str, float]]:
    """per-example SARI, BERTScore F1, and FK grade."""
    if not sources or not (len(sources) == len(predictions) == len(references)):
        raise ValueError("RQ1 lists must be equally sized and non-empty")
    sari_values = [sari_sentence(s, p, r) for s, p, r in zip(sources, predictions, references)]
    try:
        f1 = _bertscore_f1(predictions, references, bertscore_model)
        bert_values = [float(value) for value in f1]
    except Exception as exc:
        raise RuntimeError(
            "BERTScore could not be computed. Install/configure bert-score and its model before thesis evaluation; "
            "a lexical fallback is intentionally not mislabeled as BERTScore."
        ) from exc
    return [
        {"sari": float(sari), "bertscore_f1": float(bert), "fk_grade": float(flesch_kincaid_grade(prediction))}
        for sari, bert, prediction in zip(sari_values, bert_values, predictions)
    ]


def multiclass_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict:
    """six-class metrics - malformed labels go into __OTHER__, never silently coerced to Safe."""
    proposal_labels = list(labels)
    allowed = set(proposal_labels)
    other = "__OTHER__"
    normalised_true = [value if value in allowed else other for value in y_true]
    normalised_pred = [value if value in allowed else other for value in y_pred]
    matrix_labels = proposal_labels + [other]
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=proposal_labels, average="macro", zero_division=0
    )
    invalid = sum(value not in allowed for value in y_pred)
    return {
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(normalised_true, normalised_pred, labels=matrix_labels).tolist(),
        "labels": matrix_labels,
        "proposal_labels": proposal_labels,
        "per_class": classification_report(y_true, y_pred, labels=proposal_labels, output_dict=True, zero_division=0),
        "n_invalid_or_abstained": invalid,
        "invalid_or_abstention_rate": float(invalid / len(y_pred)) if y_pred else 0.0,
        "n_examples": len(y_true),
    }


def binary_faithfulness_metrics(labels_supported: Sequence[int], scores: Sequence[float], tau: float) -> BinaryMetrics:
    predictions = [1 if x >= tau else 0 for x in scores]
    p, r, f1, _ = precision_recall_fscore_support(
        labels_supported, predictions, average="binary", pos_label=1, zero_division=0
    )
    up, ur, uf1, _ = precision_recall_fscore_support(
        labels_supported, predictions, average="binary", pos_label=0, zero_division=0
    )
    matrix = confusion_matrix(labels_supported, predictions, labels=[0, 1]).tolist()
    false_supported = matrix[0][1]
    true_unsupported = matrix[0][0] + matrix[0][1]
    return BinaryMetrics(
        float(p),
        float(r),
        float(f1),
        float(up),
        float(ur),
        float(uf1),
        float(accuracy_score(labels_supported, predictions)),
        float(np.mean([x == 0 for x in predictions])),
        float(np.mean([x == 0 for x in labels_supported])),
        float(false_supported / true_unsupported) if true_unsupported else 0.0,
        matrix,
        len(scores),
    )


def attribution_accuracy(predicted: Sequence[str | None], expected: Sequence[str | None]) -> dict:
    eligible = [(p, g) for p, g in zip(predicted, expected) if g]
    if not eligible:
        return {"accuracy": None, "coverage": 0.0, "n_examples": 0}
    return {
        "accuracy": sum(p == g for p, g in eligible) / len(eligible),
        "coverage": sum(p is not None for p, _ in eligible) / len(eligible),
        "n_examples": len(eligible),
    }


def retrieval_metrics(rankings: Sequence[Sequence[str]], expected: Sequence[str | None], k: int) -> dict:
    rr, recall, precision, ndcg, first = [], [], [], [], []
    for ranking, gold in zip(rankings, expected):
        if not gold:
            continue
        top = list(ranking)[:k]
        if gold in top:
            rank = top.index(gold) + 1
            recall.append(1.0)
            precision.append(1.0 / max(len(top), 1))
            rr.append(1 / rank)
            ndcg.append(1 / math.log2(rank + 1))
            first.append(float(rank == 1))
        else:
            recall.append(0.0)
            precision.append(0.0)
            rr.append(0.0)
            ndcg.append(0.0)
            first.append(0.0)
    return {
        f"recall@{k}": float(np.mean(recall)) if recall else None,
        f"precision@{k}": float(np.mean(precision)) if precision else None,
        "mrr": float(np.mean(rr)) if rr else None,
        f"ndcg@{k}": float(np.mean(ndcg)) if ndcg else None,
        "hit_at_1": float(np.mean(first)) if first else None,
        "n_examples": len(recall),
    }


def retrieval_metrics_multi(rankings: Sequence[Sequence[str]], expected_sets: Sequence[set[str]], k: int) -> dict:
    """retrieval metrics for queries with multiple relevant chunks."""
    recalls, precisions, reciprocal_ranks, ndcgs, hits_at_1 = [], [], [], [], []
    for ranking, gold in zip(rankings, expected_sets):
        relevant = {item for item in gold if item}
        if not relevant:
            continue
        top = list(ranking)[:k]
        matches = [index + 1 for index, item in enumerate(top) if item in relevant]
        recalls.append(len(matches) / len(relevant))
        precisions.append(len(matches) / max(len(top), 1))
        reciprocal_ranks.append(1 / min(matches) if matches else 0.0)
        dcg = sum(1 / math.log2(rank + 1) for rank in matches)
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
        ndcgs.append(dcg / ideal if ideal else 0.0)
        hits_at_1.append(float(bool(top) and top[0] in relevant))
    return {
        f"recall@{k}": float(np.mean(recalls)) if recalls else None,
        f"precision@{k}": float(np.mean(precisions)) if precisions else None,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else None,
        "hit_at_1": float(np.mean(hits_at_1)) if hits_at_1 else None,
        "n_examples": len(recalls),
    }


def bootstrap_ci(values: Sequence[float], seed: int = 42, samples: int = 1000, confidence: float = 0.95):
    if not values:
        return {"mean": None, "lower": None, "upper": None}
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = np.asarray([rng.choice(arr, len(arr), replace=True).mean() for _ in range(samples)])
    alpha = (1 - confidence) / 2
    return {
        "mean": float(arr.mean()),
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1 - alpha)),
    }


def bootstrap_precision_recall_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    tau: float,
    *,
    seed: int = 42,
    samples: int = 2000,
    confidence: float = 0.95,
) -> dict:
    """bootstrap CI for precision/recall - resamples (label, score) pairs since these are ratios, not plain means."""
    n = len(labels)
    if n == 0:
        return {"n": 0, "precision": None, "recall": None}
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    alpha = (1 - confidence) / 2
    precisions: list[float] = []
    recalls: list[float] = []
    for _ in range(samples):
        idx = rng.integers(0, n, n)
        sample_labels = labels_arr[idx]
        sample_preds = (scores_arr[idx] >= tau).astype(int)
        p, r, _, _ = precision_recall_fscore_support(
            sample_labels, sample_preds, average="binary", pos_label=1, zero_division=0
        )
        precisions.append(float(p))
        recalls.append(float(r))
    point = binary_faithfulness_metrics(labels, scores, tau)
    return {
        "n": n,
        "precision": {
            "point_estimate": point.precision,
            "lower": float(np.quantile(precisions, alpha)),
            "upper": float(np.quantile(precisions, 1 - alpha)),
        },
        "recall": {
            "point_estimate": point.recall,
            "lower": float(np.quantile(recalls, alpha)),
            "upper": float(np.quantile(recalls, 1 - alpha)),
        },
        "method": f"Non-parametric bootstrap (n={samples} resamples, seed={seed}) over record-level (label, score) pairs.",
    }


def bootstrap_precision_recall_ci_clustered(
    labels: Sequence[int],
    scores: Sequence[float],
    tau: float,
    group_ids: Sequence[str],
    *,
    seed: int = 42,
    samples: int = 2000,
    confidence: float = 0.95,
) -> dict:
    """resamples whole clauses instead of individual records - FLB pairs a supported/unsupported record per clause, and they're correlated, so resampling them separately understates uncertainty."""
    n = len(labels)
    if n == 0:
        return {"n": 0, "precision": None, "recall": None}
    if len(group_ids) != n:
        raise ValueError("group_ids must have the same length as labels/scores")
    if any(not str(group_id).strip() for group_id in group_ids):
        raise ValueError("Clause-clustered bootstrap requires a non-empty group id for every record")

    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)
    groups: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(str(group_id), []).append(index)
    group_keys = sorted(groups)
    n_groups = len(group_keys)

    rng = np.random.default_rng(seed)
    alpha = (1 - confidence) / 2
    precisions: list[float] = []
    recalls: list[float] = []
    for _ in range(samples):
        chosen_groups = rng.integers(0, n_groups, n_groups)
        idx: list[int] = []
        for group_index in chosen_groups:
            idx.extend(groups[group_keys[group_index]])
        idx_arr = np.asarray(idx, dtype=int)
        sample_labels = labels_arr[idx_arr]
        sample_preds = (scores_arr[idx_arr] >= tau).astype(int)
        p, r, _, _ = precision_recall_fscore_support(
            sample_labels, sample_preds, average="binary", pos_label=1, zero_division=0
        )
        precisions.append(float(p))
        recalls.append(float(r))
    point = binary_faithfulness_metrics(labels, scores, tau)
    return {
        "n": n,
        "n_clauses": n_groups,
        "bootstrap_unit": "source_clause",
        "precision": {
            "point_estimate": point.precision,
            "lower": float(np.quantile(precisions, alpha)),
            "upper": float(np.quantile(precisions, 1 - alpha)),
        },
        "recall": {
            "point_estimate": point.recall,
            "lower": float(np.quantile(recalls, alpha)),
            "upper": float(np.quantile(recalls, 1 - alpha)),
        },
        "method": (
            f"Non-parametric clause-clustered bootstrap (n={samples} resamples, seed={seed}, "
            f"{n_groups} clauses resampled with replacement, all records per sampled clause "
            "included together)."
        ),
    }


def confidence_calibration(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: Sequence[float],
    *,
    bins: int = 10,
) -> dict:
    """Top-label confidence calibration for generative risk outputs."""
    if not (len(y_true) == len(y_pred) == len(confidences)) or not y_true:
        return {"ece": None, "brier_top_label": None, "bins": [], "n_examples": 0}
    conf = np.clip(np.asarray(confidences, dtype=float), 0.0, 1.0)
    correct = np.asarray([truth == pred for truth, pred in zip(y_true, y_pred)], dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (conf >= lower) & (conf < upper if index < bins - 1 else conf <= upper)
        if not mask.any():
            continue
        accuracy = float(correct[mask].mean())
        mean_confidence = float(conf[mask].mean())
        count = int(mask.sum())
        ece += count / len(conf) * abs(accuracy - mean_confidence)
        rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
            }
        )
    return {
        "ece": float(ece),
        "brier_top_label": float(np.mean((conf - correct) ** 2)),
        "bins": rows,
        "n_examples": len(y_true),
    }


def paired_bootstrap_difference(
    full_scores: Sequence[float],
    ablated_scores: Sequence[float],
    *,
    seed: int = 42,
    samples: int = 1000,
    confidence: float = 0.95,
) -> dict:
    """Paired bootstrap CI and two-sided sign probability for per-item scores."""
    if not full_scores or len(full_scores) != len(ablated_scores):
        return {"mean_difference": None, "lower": None, "upper": None, "p_two_sided": None, "n": 0}
    full = np.asarray(full_scores, dtype=float)
    ablated = np.asarray(ablated_scores, dtype=float)
    differences = full - ablated
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [differences[rng.integers(0, len(differences), len(differences))].mean() for _ in range(samples)]
    )
    alpha = (1 - confidence) / 2
    p = 2 * min(float(np.mean(means <= 0)), float(np.mean(means >= 0)))
    return {
        "mean_difference": float(differences.mean()),
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1 - alpha)),
        "p_two_sided": min(p, 1.0),
        "n": len(differences),
    }


def _token_f1(prediction, reference):
    pred, ref = Counter(prediction.lower().split()), Counter(reference.lower().split())
    overlap = sum((pred & ref).values())
    if not pred or not ref:
        return 0.0
    p, r = overlap / sum(pred.values()), overlap / sum(ref.values())
    return 2 * p * r / (p + r) if p + r else 0.0


def _ngrams(tokens, n):
    return {tuple(tokens[i : i + n]) for i in range(max(len(tokens) - n + 1, 0))}


def _prf(pred, gold):
    if not pred and not gold:
        return 1, 1, 1
    overlap = len(pred & gold)
    p = overlap / len(pred) if pred else 0
    r = overlap / len(gold) if gold else 0
    return p, r, 2 * p * r / (p + r) if p + r else 0


def sari_sentence(source, prediction, reference):
    src, pred, ref = source.lower().split(), prediction.lower().split(), reference.lower().split()
    scores = []
    for n in range(1, 5):
        s, p, r = _ngrams(src, n), _ngrams(pred, n), _ngrams(ref, n)
        keep = _prf(p & s, r & s)[2]
        add = _prf(p - s, r - s)[2]
        delete = _prf(s - p, s - r)[0]
        scores.append((keep + add + delete) / 3)
    return 100 * sum(scores) / len(scores)


_sari_sentence = sari_sentence


def ordinal_score_metrics(expected: Sequence[int], predicted: Sequence[int]) -> dict:
    if not expected or len(expected) != len(predicted):
        return {"mae": None, "quadratic_weighted_kappa": None, "n_examples": 0}
    return {
        "mae": float(mean_absolute_error(expected, predicted)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(expected, predicted, weights="quadratic")),
        "n_examples": len(expected),
    }


def mcnemar_exact(correct_a: list[bool], correct_b: list[bool]) -> dict:
    """exact binomial McNemar test on paired per-item correctness."""
    from math import comb

    if len(correct_a) != len(correct_b):
        raise ValueError("Paired correctness vectors must be the same length")
    b = sum(1 for a, y in zip(correct_a, correct_b) if a and not y)
    c = sum(1 for a, y in zip(correct_a, correct_b) if y and not a)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0, "significant_at_05": False}
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    p_value = min(1.0, 2 * tail)
    return {"b": b, "c": c, "n_discordant": n, "p_value": round(p_value, 6), "significant_at_05": p_value < 0.05}
