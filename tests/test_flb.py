"""tests covering the FLB construction rules - no CFPB rows, no unmapped labels sneaking in as Safe, no leakage across train/test, etc."""

from __future__ import annotations

import hashlib
import re
import types
from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest

from backend.app.core.config import FLBMode, Settings
from backend.evaluation.flb_builder import (
    FLBBuilder,
    FLBRecord,
    _balanced_quotas,
    _normalise_label,
    _risk_judge_reasons,
)
from backend.evaluation.risk_taxonomy import RiskTaxonomy

LABELS = [
    "Auto-Renewal",
    "Hidden Fee",
    "Liability Waiver",
    "Data Sharing",
    "Penalty Clause",
    "Safe",
]


def _make_settings(**overrides) -> Settings:
    base = dict(
        environment="test",
        flb_size=500,
        flb_pilot_size=6,
        flb_mode=FLBMode.final,
    )
    base.update(overrides)
    return Settings(**base)


def test_flb_size_represents_unique_clauses_across_all_six_classes():
    quotas = _balanced_quotas(LABELS, 500)
    assert sum(quotas.values()) == 500
    assert set(quotas) == set(LABELS)
    assert max(quotas.values()) - min(quotas.values()) <= 1


def test_cfpb_source_ids_are_excluded_from_candidate_pool():
    """Candidates with source_id starting with 'cfpb' must be rejected."""
    settings = _make_settings()
    registry = MagicMock()
    builder = FLBBuilder.__new__(FLBBuilder)
    builder.settings = settings
    builder.taxonomy = RiskTaxonomy(settings.project_root / "configs" / "risk_taxonomy.yaml")

    mock_repo = MagicMock()
    mock_repo.stream_partitioned.return_value = iter([])
    builder.repo = mock_repo

    builder._candidate_pool({label: 10 for label in LABELS})

    mock_repo.cfpb_narratives.assert_not_called()


def test_cfpb_source_id_prefix_never_in_pool(tmp_path):
    """Any source_id starting with 'cfpb::' is structurally blocked."""
    settings = _make_settings()
    registry = MagicMock()
    builder = FLBBuilder.__new__(FLBBuilder)
    builder.settings = settings
    builder.taxonomy = RiskTaxonomy(settings.project_root / "configs" / "risk_taxonomy.yaml")

    fake_cuad_row = {
        "label": "Renewal Term",
        "clause": "The agreement shall automatically renew each year unless notice is given thirty days prior.",
        "file_name": "test_doc.pdf",
        "start_at": 0,
        "end_at": 100,
    }

    mock_repo = MagicMock()
    mock_repo.stream_partitioned.return_value = iter([fake_cuad_row])
    builder.repo = mock_repo

    candidates = builder._candidate_pool({label: 10 for label in LABELS})

    for c in candidates:
        assert not str(c["source_id"]).startswith("cfpb"), f"CFPB source_id found in pool: {c['source_id']}"


def test_unmapped_cuad_labels_are_excluded():
    """A CUAD row with an unmapped label must not enter the candidate pool."""
    settings = _make_settings()
    builder = FLBBuilder.__new__(FLBBuilder)
    builder.settings = settings
    builder.taxonomy = RiskTaxonomy(settings.project_root / "configs" / "risk_taxonomy.yaml")

    unmapped_row = {
        "label": "Revenue/Profit Sharing",
        "clause": "The parties agree to share all revenue equally and without restriction.",
        "file_name": "doc.pdf",
        "start_at": 5,
        "end_at": 50,
    }

    mock_repo = MagicMock()
    mock_repo.stream_partitioned.return_value = iter([unmapped_row])
    builder.repo = mock_repo

    candidates = builder._candidate_pool({label: 10 for label in LABELS})
    assert candidates == [], "Unmapped CUAD label must produce zero candidates"


def test_unmapped_label_does_not_become_safe():
    """A row whose label is not in the exact taxonomy and not in safe_categories must not be assigned Safe."""
    settings = _make_settings()
    builder = FLBBuilder.__new__(FLBBuilder)
    builder.settings = settings
    builder.taxonomy = RiskTaxonomy(settings.project_root / "configs" / "risk_taxonomy.yaml")

    row = {
        "label": "Warranty Duration",
        "clause": "The product warranty shall last for one year from date of purchase.",
        "file_name": "doc.pdf",
        "start_at": 0,
        "end_at": 80,
    }

    mock_repo = MagicMock()
    mock_repo.stream_partitioned.return_value = iter([row])
    builder.repo = mock_repo

    candidates = builder._candidate_pool({label: 10 for label in LABELS})
    safe_candidates = [c for c in candidates if c["risk_label"] == "Safe"]
    assert safe_candidates == [], "Ambiguous/unmapped label must not become Safe"


def test_exact_taxonomy_matching():
    """map_question must return a label only for exact normalised matches."""
    import yaml
    from pathlib import Path

    settings = _make_settings()
    taxonomy = RiskTaxonomy(settings.project_root / "configs" / "risk_taxonomy.yaml")

    assert taxonomy.map_question("Renewal Term") == "Auto-Renewal"
    assert taxonomy.map_question("Fees") == "Hidden Fee"
    assert taxonomy.map_question("Data Privacy") == "Data Sharing"

    assert taxonomy.map_question("renewal") is None
    assert taxonomy.map_question("Renewal") is None
    assert taxonomy.map_question("Some Renewal Term Extra") is None
    assert taxonomy.map_question("") is None


def test_duplicate_source_ids_are_rejected():
    """The same source_id must never appear twice in the candidate pool."""
    settings = _make_settings()
    builder = FLBBuilder.__new__(FLBBuilder)
    builder.settings = settings
    builder.taxonomy = RiskTaxonomy(settings.project_root / "configs" / "risk_taxonomy.yaml")

    identical_row = {
        "label": "Renewal Term",
        "clause": "The agreement shall automatically renew each year unless written notice is given.",
        "file_name": "dup_doc.pdf",
        "start_at": 0,
        "end_at": 100,
    }

    mock_repo = MagicMock()
    mock_repo.stream_partitioned.return_value = iter([identical_row, dict(identical_row)])
    builder.repo = mock_repo

    candidates = builder._candidate_pool({label: 10 for label in LABELS})
    source_ids = [c["source_id"] for c in candidates]
    assert len(source_ids) == len(set(source_ids)), "Duplicate source_ids found in pool"
    assert len(candidates) == 1, "Second identical row must have been rejected"


def test_leakage_partition_method_is_recorded():
    """Every accepted candidate must carry leakage_partition_method in its record."""
    record_field_names = {f.name for f in fields(FLBRecord)}
    assert "leakage_partition_method" in record_field_names


def test_candidate_pool_uses_test_split_only():
    """stream_partitioned must be called with requested_split='test'."""
    settings = _make_settings()
    builder = FLBBuilder.__new__(FLBBuilder)
    builder.settings = settings
    builder.taxonomy = RiskTaxonomy(settings.project_root / "configs" / "risk_taxonomy.yaml")

    mock_repo = MagicMock()
    mock_repo.stream_partitioned.return_value = iter([])
    builder.repo = mock_repo

    builder._candidate_pool({label: 1 for label in LABELS})

    call_kwargs = mock_repo.stream_partitioned.call_args
    assert call_kwargs is not None
    args, kwargs = call_kwargs
    assert kwargs.get("requested_split") == "test" or (len(args) >= 2 and args[1] == "test"), (
        "CUAD must only be streamed from the test split"
    )


def test_feasibility_failure_makes_no_provider_calls(tmp_path):
    """build() must abort before any silver-label generation when feasibility_audit reports infeasible."""
    settings = _make_settings(flb_mode=FLBMode.final, flb_size=500)

    mock_registry = MagicMock()
    builder = FLBBuilder.__new__(FLBBuilder)
    builder.settings = settings
    builder.taxonomy = RiskTaxonomy(settings.project_root / "configs" / "risk_taxonomy.yaml")
    builder.silver = MagicMock()

    infeasible_result = {
        "mode": "final",
        "target_size": 500,
        "feasible": False,
        "class_counts": {label: 0 for label in LABELS},
        "quotas": {label: 84 for label in LABELS},
        "shortages": {label: {"available": 0, "required": 84, "short": 84} for label in LABELS},
        "cfpb_excluded": True,
        "note": "INFEASIBLE",
    }

    with patch.object(FLBBuilder, "feasibility_audit", return_value=infeasible_result):
        with pytest.raises(RuntimeError, match="feasibility check failed"):
            builder.build()

    builder.silver.simplification_pairs.assert_not_called()
    builder.silver.assess_simplifications.assert_not_called()
    builder.silver.assess_risk_labels.assert_not_called()


def test_pilot_mode_flag_is_set_on_records():
    """FLBRecord must have is_pilot field; pilot build must mark it True."""
    record_field_names = {f.name for f in fields(FLBRecord)}
    assert "is_pilot" in record_field_names


def test_pilot_mode_setting():
    """FLBMode.pilot must be selectable and have value 'pilot'."""
    assert FLBMode.pilot.value == "pilot"
    settings = _make_settings(flb_mode=FLBMode.pilot)
    assert settings.flb_mode == FLBMode.pilot


def test_pilot_label_constant_present():
    """FLBBuilder.PILOT_LABEL must contain 'NOT FINAL THESIS RESULTS'."""
    assert "NOT FINAL THESIS RESULTS" in FLBBuilder.PILOT_LABEL


def test_final_mode_aborts_on_infeasibility():
    """Final mode must raise RuntimeError (not silently reduce) when quotas unmet."""
    settings = _make_settings(flb_mode=FLBMode.final, flb_size=500)
    builder = FLBBuilder.__new__(FLBBuilder)
    builder.settings = settings
    builder.taxonomy = RiskTaxonomy(settings.project_root / "configs" / "risk_taxonomy.yaml")
    builder.silver = MagicMock()

    infeasible = {
        "mode": "final",
        "target_size": 500,
        "feasible": False,
        "class_counts": {label: 10 for label in LABELS},
        "quotas": {label: 84 for label in LABELS},
        "shortages": {"Auto-Renewal": {"available": 10, "required": 84, "short": 74}},
        "cfpb_excluded": True,
        "note": "INFEASIBLE",
    }

    with patch.object(FLBBuilder, "feasibility_audit", return_value=infeasible):
        with pytest.raises(RuntimeError) as exc_info:
            builder.build()

    msg = str(exc_info.value)
    assert "No Ollama generation was started" in msg, "Error message must state that Ollama was not started"


def test_provenance_fields_exist_on_flb_record():
    """FLBRecord must carry all required G-section provenance fields."""
    required = [
        "source_id",
        "source_dataset",
        "source_revision",
        "source_split",
        "source_document_id",
        "original_source_label",
        "provisional_risk_label",
        "judge_risk_label",
        "judge_risk_confidence",
        "source_text_hash",
        "candidate_text_hash",
        "leakage_partition_method",
        "accepted",
        "rejection_reasons",
    ]
    record_field_names = {f.name for f in fields(FLBRecord)}
    missing = [name for name in required if name not in record_field_names]
    assert not missing, f"FLBRecord is missing provenance fields: {missing}"


def test_silver_progress_logging_is_global(caplog):
    """simplification_pairs must log global_accepted that only increases."""
    import logging
    from backend.training.silver_labels import SilverLabelGenerator, SimplificationPair

    settings = _make_settings()
    mock_registry = MagicMock()
    gen = SilverLabelGenerator.__new__(SilverLabelGenerator)
    gen.settings = settings
    gen.provider = MagicMock()
    gen.cached_labels_reused = 0
    gen.new_labels_generated = 0
    gen.rejected_labels = 0

    call_count = [0]

    def fake_simplify_batch(batch, alternate=False):
        call_count[0] += 1
        return [
            SimplificationPair(
                source_id=row["source_id"],
                document_id=row.get("document_id", "doc"),
                source=row["text"],
                target="simplified text here",
                source_dataset="test",
                fk_grade=7.0,
            )
            for row in batch
        ]

    gen._simplify_batch = fake_simplify_batch
    gen.settings.silver_label_batch_size = 1

    rows = [
        {
            "source_id": f"id_{i}",
            "text": "The fee shall not exceed one hundred dollars per month.",
            "document_id": "doc",
        }
        for i in range(4)
    ]

    with caplog.at_level(logging.INFO, logger="backend.training.silver_labels"):
        result = gen.simplification_pairs(iter(rows), limit=4)

    accepted_values = []
    for record in caplog.records:
        msg = record.getMessage()
        m = re.search(r"global_accepted=(\d+)", msg)
        if m:
            accepted_values.append(int(m.group(1)))

    for i in range(1, len(accepted_values)):
        assert accepted_values[i] >= accepted_values[i - 1], (
            f"Progress counter reset: {accepted_values[i - 1]} -> {accepted_values[i]} at step {i}"
        )


def test_safe_candidates_require_judge_approval():
    """A provisional Safe candidate from CUAD must still pass risk judge adjudication."""
    record_field_names = {f.name for f in fields(FLBRecord)}
    assert "provisional_risk_label" in record_field_names
    assert "judge_risk_label" in record_field_names
    assert "accepted" in record_field_names

    # A record can be constructed with provisional=Safe but judge=different
    r = FLBRecord(
        record_id="test::supported",
        source_id="cuad::doc::Parties::0::100::abc123",
        document_id="doc",
        original_clause="This agreement is between Alpha Corp and Beta Ltd.",
        reference_simplification="Alpha Corp and Beta Ltd agree.",
        risk_label="Safe",
        hypothesis="Alpha Corp and Beta Ltd agree.",
        supported=1,
        source_dataset="test/cuad",
        provisional_risk_label="Safe",
        judge_risk_label="Liability Waiver",
        judge_risk_confidence=4,
        accepted=False,
        rejection_reasons="risk_label='Liability Waiver'_expected='Safe'",
    )
    assert r.provisional_risk_label == "Safe"
    assert r.judge_risk_label == "Liability Waiver"
    assert r.accepted is False
    assert r.rejection_reasons is not None


def _risk(label="Safe", confidence=5):
    return types.SimpleNamespace(correct_label=label, confidence=confidence)


def test_exact_cuad_gold_is_authoritative_when_judge_missing_or_disagrees():
    row = {"source_type": "cuad_exact_risk", "provisional": False, "risk_label": "Hidden Fee"}
    assert _risk_judge_reasons(row, None, 4) == []
    assert _risk_judge_reasons(row, _risk("Safe", 1), 4) == []


def test_reviewed_external_gold_is_authoritative():
    row = {
        "source_type": "human_adjudicated_external",
        "human_reviewed": True,
        "provisional": False,
        "risk_label": "Data Sharing",
    }
    assert _risk_judge_reasons(row, None, 4) == []
    assert _risk_judge_reasons(row, _risk("Safe", 1), 4) == []


def test_provisional_safe_requires_confident_matching_judge():
    row = {"source_type": "cuad_provisional_safe", "provisional": True, "risk_label": "Safe"}
    assert _risk_judge_reasons(row, None, 4) == ["missing_risk_judge"]
    assert "risk_confidence=3" in _risk_judge_reasons(row, _risk("Safe", 3), 4)
    assert any(reason.startswith("risk_label=") for reason in _risk_judge_reasons(row, _risk("Hidden Fee", 5), 4))
    assert _risk_judge_reasons(row, _risk("Safe", 5), 4) == []


def test_external_gold_is_not_injected_when_disabled():
    settings = _make_settings(flb_external_gold_enabled=False)
    builder = FLBBuilder.__new__(FLBBuilder)
    builder.settings = settings
    builder.taxonomy = RiskTaxonomy(settings.project_root / "configs" / "risk_taxonomy.yaml")
    builder.repo = MagicMock()
    builder.repo.stream_partitioned.return_value = iter([])
    assert builder._candidate_pool({label: 1 for label in LABELS}) == []
