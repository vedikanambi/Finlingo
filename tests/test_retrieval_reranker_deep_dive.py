"""Checks the saved retrieval/reranker deep-dive output for internal consistency."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULT_PATH = Path("reports/retrieval_reranker_deep_dive.json")


@pytest.mark.skipif(not RESULT_PATH.exists(), reason="deep-dive not yet generated in this environment")
def test_sanity_check_confirms_score_ordering():
    data = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    sanity = data["sanity_check_score_ordering"]
    assert sanity["ordering_correct"] is True
    assert sanity["relevant_pair_score"] > sanity["irrelevant_pair_score"]


@pytest.mark.skipif(not RESULT_PATH.exists(), reason="deep-dive not yet generated in this environment")
def test_rank_change_counts_are_consistent():
    data = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    summary = data["rank_change_summary"]
    total = summary["n_demoted"] + summary["n_promoted"] + summary["n_unchanged"]
    assert total == summary["n_with_gold_present_pre_rerank"]


@pytest.mark.skipif(not RESULT_PATH.exists(), reason="deep-dive not yet generated in this environment")
def test_recall_curves_are_monotonically_non_decreasing_in_k():
    data = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    for key in ("pre_rerank_recall_at_k", "post_rerank_recall_at_k_over_full_set"):
        values = [data[key][k] for k in sorted(data[key], key=int)]
        assert all(a <= b + 1e-9 for a, b in zip(values, values[1:])), f"{key} is not monotonic in k"


@pytest.mark.skipif(not RESULT_PATH.exists(), reason="deep-dive not yet generated in this environment")
def test_demoted_examples_have_positive_rank_delta():
    data = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    for example in data["demoted_examples"]:
        assert example["rank_delta"] > 0
        assert example["post_rerank_rank"] > example["pre_rerank_rank"]


@pytest.mark.skipif(not RESULT_PATH.exists(), reason="deep-dive not yet generated in this environment")
def test_promoted_examples_have_negative_rank_delta():
    data = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    for example in data["promoted_examples"]:
        assert example["rank_delta"] < 0
        assert example["post_rerank_rank"] < example["pre_rerank_rank"]
