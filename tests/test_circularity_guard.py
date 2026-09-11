"""guard against evaluating grounding on the same pool used to pick the gold chunk"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from backend.evaluation.circularity_guard import (
    CircularEvaluationError,
    assert_not_circular_gold,
    assert_pools_not_identical,
    independent_retrieval_recall,
)


def test_fails_when_gold_selected_from_current_retrieved_candidates():
    candidates = ["chunk_a", "chunk_b", "chunk_c"]
    with pytest.raises(CircularEvaluationError):
        assert_not_circular_gold("chunk_b", candidates)


def test_passes_when_gold_not_in_candidate_pool():
    candidates = ["chunk_a", "chunk_b", "chunk_c"]
    assert_not_circular_gold("chunk_z", candidates)
    assert_not_circular_gold(None, candidates)


def test_fails_when_evidence_and_adjudication_pools_are_same_object():
    pool = ["chunk_a", "chunk_b"]
    with pytest.raises(CircularEvaluationError):
        assert_pools_not_identical(pool, pool)


def test_fails_when_pools_are_identical_slices():
    with pytest.raises(CircularEvaluationError):
        assert_pools_not_identical(["chunk_a", "chunk_b"], ["chunk_a", "chunk_b"])


def test_passes_when_pools_differ():
    assert_pools_not_identical(["chunk_a", "chunk_b"], ["chunk_c", "chunk_d"])


@dataclass
class _FakeRecord:
    record_id: str
    independent_gold_chunk_id: str | None = None


def test_independent_recall_blocked_without_any_gold():
    records = [_FakeRecord("r1"), _FakeRecord("r2")]
    result = independent_retrieval_recall(records, {}, k=8)
    assert result.status == "blocked"
    assert result.recall_at_k is None
    assert result.n_with_independent_gold == 0
    assert "adjudicat" in result.reason.lower()


def test_independent_recall_computes_correctly_with_fixture_gold():
    records = [
        _FakeRecord("r1", independent_gold_chunk_id="gold1"),
        _FakeRecord("r2", independent_gold_chunk_id="gold2"),
        _FakeRecord("r3", independent_gold_chunk_id="gold3"),
    ]
    retrieved = {
        "r1": ["gold1", "x", "y"],  # hit
        "r2": ["x", "y", "z"],  # miss
        "r3": ["a", "gold3"],  # hit
    }
    result = independent_retrieval_recall(records, retrieved, k=8)
    assert result.status == "ok"
    assert result.n_with_independent_gold == 3
    assert result.recall_at_k == pytest.approx(2 / 3)


def test_independent_recall_respects_k_truncation():
    records = [_FakeRecord("r1", independent_gold_chunk_id="gold1")]
    retrieved = {"r1": ["x", "y", "gold1"]}
    assert independent_retrieval_recall(records, retrieved, k=2).recall_at_k == 0.0
    assert independent_retrieval_recall(records, retrieved, k=3).recall_at_k == 1.0
