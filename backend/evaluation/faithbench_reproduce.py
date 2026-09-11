"""reruns base vs v3 adapter on the fixed FaithBench sample and checks we still land near the frozen reference numbers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings
from backend.app.services.model_registry import ModelRegistry
from backend.app.services.nli_service import NLIService
from backend.evaluation.faithbench import load_faithbench
from backend.evaluation.metrics import bootstrap_ci, paired_bootstrap_difference


def _binary_metrics(y_true: list[int], probs: list[float], tau: float) -> dict[str, float]:
    y_pred = [1 if p >= tau else 0 for p in probs]
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sens / (precision + sens) if precision + sens else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "balanced_accuracy": round((sens + spec) / 2, 4),
        "hallucination_detection_recall": round(spec, 4),
        "supported_precision": round(precision, 4),
        "supported_recall": round(sens, 4),
        "supported_f1": round(f1, 4),
    }


def reproduce_faithbench(
    settings: Settings, max_examples: int | None = None, output_path: Path | None = None
) -> dict[str, Any]:
    records = load_faithbench(settings, max_examples)
    pairs = [(r.source, r.summary) for r in records]
    y_true = [r.supported for r in records]
    tau = settings.faithfulness_tau

    registry = ModelRegistry(settings)
    try:
        v3 = NLIService(registry, adapter_id=settings.verifier_adapter, require_adapter=True)
        v3_scores = [
            s.entailment for s in v3.score_pairs(pairs, settings.verifier_batch_size, settings.verifier_max_length)
        ]
        registry.release_nli()
        base = NLIService(
            registry,
            model_id=settings.verifier_model,
            adapter_id=None,
            require_adapter=False,
            use_verifier_calibration=False,
        )
        base_scores = [
            s.entailment for s in base.score_pairs(pairs, settings.verifier_batch_size, settings.verifier_max_length)
        ]
    finally:
        registry.close()

    v3_correct = [float((s >= tau) == bool(t)) for s, t in zip(v3_scores, y_true)]
    base_correct = [float((s >= tau) == bool(t)) for s, t in zip(base_scores, y_true)]

    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(records),
        "seed": settings.random_seed,
        "tau": tau,
        "verifier_v3": {
            **_binary_metrics(y_true, v3_scores, tau),
            "accuracy_ci95": bootstrap_ci(v3_correct, seed=settings.random_seed, samples=settings.bootstrap_samples),
        },
        "base_checkpoint": {
            **_binary_metrics(y_true, base_scores, tau),
            "accuracy_ci95": bootstrap_ci(base_correct, seed=settings.random_seed, samples=settings.bootstrap_samples),
        },
        "v3_minus_base_paired_bootstrap": paired_bootstrap_difference(
            v3_correct, base_correct, seed=settings.random_seed, samples=settings.bootstrap_samples
        ),
    }

    reference_path = settings.resolve(settings.faithbench_reference_path)
    gate: dict[str, Any] = {"reference_path": str(reference_path), "checked": False}
    if reference_path.exists():
        try:
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            # look for our own exact key shape first, so we don't accidentally match some other report's "accuracy" field
            ref_value = (
                reference.get("verifier_v3", {}).get("balanced_accuracy")
                if isinstance(reference.get("verifier_v3"), dict)
                else None
            )
            if ref_value is None:
                ref_value = _find_metric(reference, ("balanced_accuracy", "accuracy"))
            new_value = result["verifier_v3"]["balanced_accuracy"]
            if ref_value is not None:
                gate.update(
                    {
                        "checked": True,
                        "reference_balanced_accuracy": ref_value,
                        "reproduced_balanced_accuracy": new_value,
                        "tolerance": settings.faithbench_reproduction_tolerance,
                        "within_tolerance": abs(new_value - ref_value) <= settings.faithbench_reproduction_tolerance,
                    }
                )
        except (json.JSONDecodeError, TypeError) as exc:
            gate["error"] = f"Reference artefact unreadable: {exc}"
    result["reproduction_gate"] = gate

    out = settings.resolve(output_path or settings.faithbench_reproduction_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _find_metric(payload: Any, keys: tuple[str, ...]) -> float | None:
    """finds the first matching metric key anywhere in a nested report."""
    if isinstance(payload, dict):
        for key in keys:
            if key in payload and isinstance(payload[key], (int, float)):
                return float(payload[key])
        for value in payload.values():
            found = _find_metric(value, keys)
            if found is not None:
                return found
    if isinstance(payload, list):
        for value in payload:
            found = _find_metric(value, keys)
            if found is not None:
                return found
    return None
