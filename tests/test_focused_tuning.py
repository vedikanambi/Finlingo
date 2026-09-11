import json
from pathlib import Path

from backend.app.core.config import Settings
from scripts.evaluate_locked_tuned import load_locked_records
from scripts.run_focused_tuning import _parse_ints


def test_focused_profile_keeps_canonical_defaults_separate():
    canonical = Settings(_env_file=None)
    tuned = Settings(_env_file=Path("configs/focused_tuning_profile.env"))
    assert (canonical.top_k_retrieval, canonical.top_j_reranked) == (500, 15)
    assert (tuned.top_k_retrieval, tuned.top_j_reranked) == (40, 30)
    assert tuned.risk_classifier_backend == "frozen"
    assert tuned.risk_classifier_frozen_artifact.name == "classifier.joblib"


def test_parse_int_grid_deduplicates_and_sorts():
    assert _parse_ints("40,8,20,20") == (8, 20, 40)


def test_locked_record_loader_coerces_export_types(tmp_path):
    row = {
        "record_id": "r1",
        "source_id": "s1",
        "document_id": "d1",
        "original_clause": "A sufficiently long original clause for testing.",
        "reference_simplification": "A simpler clause.",
        "risk_label": "Safe",
        "hypothesis": "A simpler clause.",
        "supported": "1",
        "source_dataset": "fixture",
        "accepted": "True",
        "fk_grade": "8.5",
        "human_reviewed": "False",
    }
    source = tmp_path / "locked.jsonl"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    records = load_locked_records(source)
    assert records[0].supported == 1
    assert records[0].accepted is True
    assert records[0].human_reviewed is False
    assert records[0].fk_grade == 8.5
