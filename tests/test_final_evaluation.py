import pytest

from backend.app.core.config import Settings
from backend.evaluation.evaluator import validate_final_evaluation_records
from backend.evaluation.flb_builder import FLBRecord


def _record(**updates):
    values = dict(
        record_id="r1",
        source_id="s1",
        document_id="d1",
        original_clause="A fee applies.",
        reference_simplification="You pay a fee.",
        risk_label="Hidden Fee",
        hypothesis="You pay a fee.",
        supported=1,
        source_dataset="test",
        evidence_supported=1,
        ground_truth_chunk_id="chunk-1",
    )
    values.update(updates)
    return FLBRecord(**values)


def test_final_evaluation_rejects_provisional_labels():
    settings = Settings(_env_file=None)
    with pytest.raises(RuntimeError):
        validate_final_evaluation_records(settings, [_record()])


def test_final_evaluation_accepts_fully_human_adjudicated_labels():
    settings = Settings(_env_file=None)
    record = _record(
        human_reviewed=True,
        evidence_label_source="human_adjudicated",
        risk_label_source="human_adjudicated",
        human_risk_label="Hidden Fee",
        human_risk_score=4,
    )
    status = validate_final_evaluation_records(settings, [record])
    assert status["human_review_fraction"] == 1.0
    assert status["non_human_or_provisional"] == 0
