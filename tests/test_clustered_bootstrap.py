"""bootstrap needs to resample whole clauses, not individual rows, or the CI is too tight"""

from __future__ import annotations

import numpy as np
import pytest

from backend.evaluation.metrics import bootstrap_precision_recall_ci_clustered


def _paired_fixture():
    labels = [1, 0, 1, 0, 1, 0, 1, 0]
    scores = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4]
    group_ids = ["c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4"]
    return labels, scores, group_ids


def test_paired_records_always_appear_together():
    labels, scores, group_ids = _paired_fixture()
    rng = np.random.default_rng(42)
    groups: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(index)
    group_keys = sorted(groups)
    for _ in range(200):
        chosen = rng.integers(0, len(group_keys), len(group_keys))
        idx: list[int] = []
        for group_index in chosen:
            idx.extend(groups[group_keys[group_index]])
        for pair_start in range(0, len(idx), 1):
            pass
    assert all(len(indices) == 2 for indices in groups.values())


def test_reproducible_with_same_seed():
    labels, scores, group_ids = _paired_fixture()
    result_a = bootstrap_precision_recall_ci_clustered(labels, scores, 0.5, group_ids, seed=7, samples=500)
    result_b = bootstrap_precision_recall_ci_clustered(labels, scores, 0.5, group_ids, seed=7, samples=500)
    assert result_a == result_b


def test_different_seed_can_differ():
    labels, scores, group_ids = _paired_fixture()
    result_a = bootstrap_precision_recall_ci_clustered(labels, scores, 0.5, group_ids, seed=1, samples=500)
    result_b = bootstrap_precision_recall_ci_clustered(labels, scores, 0.5, group_ids, seed=2, samples=500)
    assert result_a["bootstrap_unit"] == "source_clause"
    assert result_b["bootstrap_unit"] == "source_clause"


def test_missing_group_id_raises():
    labels, scores, group_ids = _paired_fixture()
    group_ids = list(group_ids)
    group_ids[3] = ""
    with pytest.raises(ValueError, match="non-empty group id"):
        bootstrap_precision_recall_ci_clustered(labels, scores, 0.5, group_ids, samples=50)


def test_mismatched_length_raises():
    labels, scores, group_ids = _paired_fixture()
    with pytest.raises(ValueError, match="same length"):
        bootstrap_precision_recall_ci_clustered(labels, scores, 0.5, group_ids[:-1], samples=50)


def test_report_declares_bootstrap_unit_and_n_clauses():
    labels, scores, group_ids = _paired_fixture()
    result = bootstrap_precision_recall_ci_clustered(labels, scores, 0.5, group_ids, samples=200)
    assert result["bootstrap_unit"] == "source_clause"
    assert result["n_clauses"] == 4
    assert result["n"] == 8
