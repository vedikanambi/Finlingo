from __future__ import annotations

import json
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.services.model_registry import ModelRegistry
from backend.evaluation.evaluator import evaluate_records
from backend.evaluation.flb_builder import FLBBuilder, FLBRecord


def run_retrieval_query_ablation(
    settings: Settings,
    max_examples: int | None = None,
    output_path: Path | None = None,
    *,
    records: list[FLBRecord] | None = None,
) -> dict:
    """Compare original, simplified, and combined retrieval queries."""
    records = records or FLBBuilder(settings, ModelRegistry(settings)).build(max_examples or settings.flb_size)
    systems = {}
    # only adjudicate once, then reuse the same labels so query mode is the only thing that varies
    for index, mode in enumerate(("simplified", "original", "combined")):
        run_settings = settings.model_copy(update={"retrieval_query_mode": mode})
        systems[mode] = evaluate_records(
            run_settings,
            records,
            adjudicate_evidence=(index == 0),
        )
    payload = {
        "shared_benchmark_rows": len(records),
        "systems": systems,
        "purpose": "Measure whether simplification removes exact legal terms needed for regulatory retrieval.",
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
