from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from backend.app.core.config import Settings
from backend.app.services.model_registry import ModelRegistry
from backend.app.services.nli_service import NLIService
from backend.evaluation.flb_builder import FLBBuilder, FLBRecord
from backend.evaluation.metrics import binary_faithfulness_metrics


def run_tau_sweep(
    settings: Settings,
    max_examples: int | None = None,
    output_path: Path | None = None,
    *,
    records: list[FLBRecord] | None = None,
) -> dict:
    registry = ModelRegistry(settings)
    records = records or FLBBuilder(settings, registry).build(max_examples or settings.flb_size)
    scores = [
        x.entailment
        for x in NLIService(registry).score_pairs(
            [(r.original_clause, r.hypothesis) for r in records],
            settings.verifier_batch_size,
            settings.verifier_max_length,
        )
    ]
    labels = [r.supported for r in records]
    payload = {
        "model": settings.verifier_model,
        "adapter": settings.verifier_adapter,
        "shared_test_set_size": len(records),
        "results": {str(t): asdict(binary_faithfulness_metrics(labels, scores, t)) for t in settings.tau_values},
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
