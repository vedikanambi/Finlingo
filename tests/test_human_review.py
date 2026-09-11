from pathlib import Path

from backend.evaluation.flb_builder import FLBRecord
from backend.evaluation.human_review import apply_adjudicated_reviews


def test_adjudicated_reviews_replace_provisional_labels(tmp_path: Path):
    record = FLBRecord(
        record_id="r1",
        source_id="s1",
        document_id="d1",
        original_clause="A fee applies.",
        reference_simplification="You must pay a fee.",
        risk_label="Hidden Fee",
        hypothesis="No fee applies.",
        supported=1,
        source_dataset="test",
        ground_truth_chunk_id="provisional",
    )
    review = tmp_path / "review.csv"
    review.write_text(
        "record_id,adjudicated_supported,adjudicated_chunk_id\nr1,0,\n",
        encoding="utf-8",
    )
    status = apply_adjudicated_reviews([record], review)
    assert status["applied"] == 1
    assert record.supported == 0
    assert record.ground_truth_chunk_id is None
    assert record.human_reviewed is True
    assert record.evidence_label_source == "human_adjudicated"
