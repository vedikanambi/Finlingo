"""3-reviewer human adjudication support"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from backend.evaluation.flb_builder import FLBBuilder, FLBRecord
from backend.evaluation.human_review import fleiss_kappa, score_review_file, validate_gold_export_ready


def _make_record(record_id: str, source_id: str) -> FLBRecord:
    return FLBRecord(
        record_id=record_id,
        source_id=source_id,
        document_id=source_id,
        original_clause="Original clause text.",
        reference_simplification="Simple version.",
        risk_label="Hidden Fee",
        hypothesis="A hypothesis sentence.",
        supported=1,
        source_dataset="test",
        provisional_evidence_text="Some regulatory text.",
        provisional_evidence_source="FCA",
    )


def test_fleiss_kappa_perfect_agreement():
    rows = [["Safe", "Safe", "Safe"], ["Hidden Fee", "Hidden Fee", "Hidden Fee"]] * 3
    assert fleiss_kappa(rows) == pytest.approx(1.0, abs=1e-6)


def test_fleiss_kappa_none_when_insufficient_categories():
    rows = [["Safe", "Safe", "Safe"]] * 3
    assert fleiss_kappa(rows) is None


def test_fleiss_kappa_skips_rows_with_fewer_than_two_ratings():
    rows = [["Safe"], ["Safe", "Hidden Fee", "Safe"], ["Hidden Fee", "Hidden Fee", "Hidden Fee"]]
    result = fleiss_kappa(rows)
    assert result is not None


def test_export_review_template_is_blinded_and_simple_by_default(tmp_path: Path):
    records = [_make_record(f"r{i}", f"s{i}") for i in range(3)]
    out = tmp_path / "review.csv"
    FLBBuilder.export_review_template(records, out, sample_size=3, seed=1)
    with out.open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert "risk_label" not in header
    assert "system_supported" not in header
    assert "nli_label" not in header
    assert "reviewer_1_risk_category" in header
    assert "reviewer_2_risk_category" in header
    assert "reviewer_3_risk_category" in header
    assert "reviewer_1_meaning_preserved" in header
    assert "reviewer_1_missing_or_changed" in header
    assert "reviewer_1_comments" in header
    assert "reviewer_1_risk_score" not in header
    assert "reviewer_1_chunk_id" not in header


def test_export_review_template_full_schema_when_not_simple(tmp_path: Path):
    records = [_make_record(f"r{i}", f"s{i}") for i in range(3)]
    out = tmp_path / "review_full.csv"
    FLBBuilder.export_review_template(records, out, sample_size=3, seed=1, simple=False)
    with out.open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert "reviewer_1_risk_label" in header
    assert "reviewer_1_expertise_category" in header
    assert "reviewer_1_review_timestamp" in header
    assert "reviewer_1_misleading_content" in header
    assert "reviewer_1_critical_fact_changed" in header


def test_export_review_template_unblinded_shows_system_fields(tmp_path: Path):
    records = [_make_record("r0", "s0")]
    out = tmp_path / "review_unblinded.csv"
    FLBBuilder.export_review_template(records, out, sample_size=1, seed=1, blinded=False)
    with out.open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert "risk_label" in header
    assert "system_supported" in header


def test_exported_rows_start_empty_and_are_not_counted_as_adjudicated(tmp_path: Path):
    records = [_make_record(f"r{i}", f"s{i}") for i in range(100)]
    out = tmp_path / "review100.csv"
    FLBBuilder.export_review_template(records, out, sample_size=100, seed=1)
    result = score_review_file(out)
    assert result["total_rows_in_sheet"] == 100
    assert result["adjudicated"] == 0
    assert result["adjudication_completion_fraction"] == 0.0


def test_three_reviewer_agreement_uses_fleiss(tmp_path: Path):
    path = tmp_path / "review.csv"
    fieldnames = [
        "record_id",
        "reviewer_1_risk_label",
        "reviewer_2_risk_label",
        "reviewer_3_risk_label",
        "reviewer_1_source_supported",
        "reviewer_2_source_supported",
        "reviewer_3_source_supported",
        "adjudicated_source_supported",
        "adjudicated_evidence_supported",
        "adjudicated_risk_label",
    ]
    rows = [
        {
            "record_id": "r1",
            "reviewer_1_risk_label": "Safe",
            "reviewer_2_risk_label": "Safe",
            "reviewer_3_risk_label": "Hidden Fee",
            "reviewer_1_source_supported": "1",
            "reviewer_2_source_supported": "1",
            "reviewer_3_source_supported": "0",
            "adjudicated_source_supported": "1",
            "adjudicated_evidence_supported": "1",
            "adjudicated_risk_label": "Safe",
        },
        {
            "record_id": "r2",
            "reviewer_1_risk_label": "Hidden Fee",
            "reviewer_2_risk_label": "Hidden Fee",
            "reviewer_3_risk_label": "Hidden Fee",
            "reviewer_1_source_supported": "1",
            "reviewer_2_source_supported": "1",
            "reviewer_3_source_supported": "1",
            "adjudicated_source_supported": "1",
            "adjudicated_evidence_supported": "1",
            "adjudicated_risk_label": "Hidden Fee",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    result = score_review_file(path)
    assert result["reviewer_count_detected"] == 3
    assert result["agreement_method"] == "fleiss_kappa"
    assert result["risk_kappa"] is not None
    assert result["adjudicated"] == 2
    assert result["adjudication_completion_fraction"] == 1.0
    assert result["conflict_count"]["risk_label"] == 1
    assert result["conflict_count"]["source_supported"] == 1


def test_gold_export_blocked_when_fields_missing(tmp_path: Path):
    path = tmp_path / "incomplete.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "adjudicated_source_supported"])
        writer.writeheader()
        writer.writerow({"record_id": "r1", "adjudicated_source_supported": "1"})
    with pytest.raises(RuntimeError, match="Cannot produce a frozen FLB-Gold export"):
        validate_gold_export_ready(path)


def test_gold_export_ready_when_fields_complete(tmp_path: Path):
    path = tmp_path / "complete.csv"
    fieldnames = ["record_id", "adjudicated_source_supported", "adjudicated_evidence_supported", "adjudicated_risk_label"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "record_id": "r1",
                "adjudicated_source_supported": "1",
                "adjudicated_evidence_supported": "1",
                "adjudicated_risk_label": "Safe",
            }
        )
    result = validate_gold_export_ready(path)
    assert result["status"] == "ready"
    assert result["rows"] == 1
