"""stops me from accidentally grading retrieval against gold chunks that came from the same pool being scored."""

from __future__ import annotations

from dataclasses import dataclass


class CircularEvaluationError(ValueError):
    """Raised when a gold/ground-truth id is drawn from its own candidate pool."""


def assert_not_circular_gold(gold_chunk_id: str | None, candidate_chunk_ids) -> None:
    """Raise if `gold_chunk_id` was selected from `candidate_chunk_ids`."""
    if gold_chunk_id is None:
        return
    pool = set(candidate_chunk_ids)
    if gold_chunk_id in pool:
        raise CircularEvaluationError(
            f"Gold chunk id {gold_chunk_id!r} was selected from the same candidate pool "
            "it is about to be evaluated against. This is circular: it cannot serve as "
            "independent evidence of retrieval or verifier quality. Use an independently "
            "adjudicated gold chunk id instead."
        )


def assert_pools_not_identical(evidence_pool, adjudication_pool) -> None:
    """same idea but catches pools that are literally the same object or slice."""
    if evidence_pool is adjudication_pool:
        raise CircularEvaluationError(
            "Verifier evidence pool and adjudication candidate pool are the same object."
        )
    if list(evidence_pool) == list(adjudication_pool) and len(evidence_pool) > 0:
        raise CircularEvaluationError(
            "Verifier evidence pool and adjudication candidate pool are an identical slice; "
            "ground truth cannot be constructed from the pool being scored."
        )


@dataclass
class IndependentRetrievalResult:
    status: str  # "ok" or "blocked"
    recall_at_k: float | None
    n_with_independent_gold: int
    n_total: int
    reason: str | None = None


def independent_retrieval_recall(records, retrieved_ids_by_record, k: int) -> IndependentRetrievalResult:
    """recall@k against independent_gold_chunk_id only - returns blocked instead of a number if nothing has independent gold yet."""
    n_total = len(records)
    eligible = [record for record in records if getattr(record, "independent_gold_chunk_id", None)]
    if not eligible:
        return IndependentRetrievalResult(
            status="blocked",
            recall_at_k=None,
            n_with_independent_gold=0,
            n_total=n_total,
            reason=(
                "No record has an independently adjudicated independent_gold_chunk_id. "
                "Independent retrieval evaluation requires a completed human/independent "
                "adjudication pass (see backend/evaluation/human_review.py); none exists yet."
            ),
        )
    # not circular here since the gold id was picked independently of this run's retrieval
    hits = 0
    for record in eligible:
        retrieved_ids = retrieved_ids_by_record.get(record.record_id, [])[:k]
        if record.independent_gold_chunk_id in retrieved_ids:
            hits += 1
    return IndependentRetrievalResult(
        status="ok",
        recall_at_k=hits / len(eligible),
        n_with_independent_gold=len(eligible),
        n_total=n_total,
    )
