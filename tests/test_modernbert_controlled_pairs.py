"""Checks the saved controlled-pair diagnostic output for internal consistency."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DIAGNOSIS_PATH = Path("reports/modernbert_controlled_pair_diagnosis.json")


@pytest.mark.skipif(not DIAGNOSIS_PATH.exists(), reason="diagnosis not yet generated in this environment")
def test_diagnosis_has_expected_pair_count():
    data = json.loads(DIAGNOSIS_PATH.read_text(encoding="utf-8"))
    assert data["n_pairs"] == 14
    assert len(data["results"]) == 14


@pytest.mark.skipif(not DIAGNOSIS_PATH.exists(), reason="diagnosis not yet generated in this environment")
def test_diagnosis_accuracy_matches_result_rows():
    data = json.loads(DIAGNOSIS_PATH.read_text(encoding="utf-8"))
    computed = sum(row["correct"] for row in data["results"]) / len(data["results"])
    assert computed == pytest.approx(data["accuracy"], abs=1e-4)


@pytest.mark.skipif(not DIAGNOSIS_PATH.exists(), reason="diagnosis not yet generated in this environment")
def test_diagnosis_distribution_sums_to_pair_count():
    data = json.loads(DIAGNOSIS_PATH.read_text(encoding="utf-8"))
    assert sum(data["predicted_label_distribution"].values()) == data["n_pairs"]
