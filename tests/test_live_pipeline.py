from __future__ import annotations

import os
from pathlib import Path

import pytest

from backend.app.core.config import Settings
from backend.app.core.preflight import run_preflight
from backend.app.pipeline import FinLingoPipeline


@pytest.mark.live
def test_live_seven_stage_pipeline() -> None:
    """Opt-in end to end run against a real document, skipped in CI."""
    if os.getenv("FINLINGO_RUN_LIVE_TESTS") != "1":
        pytest.skip("Set FINLINGO_RUN_LIVE_TESTS=1 to run the target-environment pipeline test")

    document_value = os.getenv("FINLINGO_LIVE_DOCUMENT", "").strip()
    if not document_value:
        pytest.fail("FINLINGO_LIVE_DOCUMENT is required for the live pipeline test")
    document = Path(document_value).expanduser().resolve()
    if not document.exists():
        pytest.fail(f"Live test document not found: {document}")

    settings = Settings()
    preflight = run_preflight(settings, final=True)
    assert preflight["passed"] is True

    report = FinLingoPipeline(settings).run(document)
    assert report.clauses, "S1-S7 returned no clauses"
    assert report.metrics.total_clauses == len(report.clauses)
    assert report.model_provenance.primary_faithfulness_premise == "retrieved_chunk"
    assert all(clause.simplification_status for clause in report.clauses)
    assert all(clause.extraction_method for clause in report.clauses)
