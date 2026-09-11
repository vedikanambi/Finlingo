"""confusion matrix + bootstrap CIs, boundary FP/FN exemplars, and McNemar tests, all from a finished eval run."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings
from backend.evaluation.metrics import bootstrap_ci, mcnemar_exact


def _confusion(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    matrix = {t: {p: 0 for p in labels + ["<other>"]} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in matrix:
            matrix[t][p if p in matrix[t] else "<other>"] += 1
    per_class: dict[str, Any] = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[t][label] for t in labels if t != label)
        fn = sum(v for k, v in matrix[label].items() if k != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    return {"matrix": matrix, "per_class": per_class}


def run_error_analysis(
    settings: Settings,
    evaluation_path: Path,
    multiverifier_path: Path | None = None,
    output_path: Path | None = None,
    boundary_examples: int = 20,
) -> dict[str, Any]:
    payload = json.loads(settings.resolve(evaluation_path).read_text(encoding="utf-8"))
    result: dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(), "source": str(evaluation_path)}

    rq2 = payload.get("rq2") or payload.get("RQ2") or {}
    rows = rq2.get("per_record") or payload.get("risk_records") or []
    if rows:
        labels = list(settings.configured_risk_labels)
        y_true = [str(r.get("true_label")) for r in rows]
        y_pred = [str(r.get("predicted_label")) for r in rows]
        result["risk_confusion"] = _confusion(y_true, y_pred, labels)
        correct = [float(t == p) for t, p in zip(y_true, y_pred)]
        result["risk_accuracy_ci95"] = bootstrap_ci(
            correct, seed=settings.random_seed, samples=settings.bootstrap_samples
        )

    rq3 = payload.get("rq3") or payload.get("RQ3") or {}
    items = rq3.get("per_record") or payload.get("verifier_records") or []
    if items:
        tau = settings.faithfulness_tau
        fps = [i for i in items if i.get("gold_supported") is False and float(i.get("entailment", 0)) >= tau]
        fns = [i for i in items if i.get("gold_supported") is True and float(i.get("entailment", 1)) < tau]
        near = sorted(items, key=lambda i: abs(float(i.get("entailment", 0.5)) - tau))
        result["verifier_errors"] = {
            "tau": tau,
            "false_positive_count": len(fps),
            "false_negative_count": len(fns),
            "false_positives_nearest_boundary": sorted(fps, key=lambda i: float(i.get("entailment", 0)))[
                :boundary_examples
            ],
            "false_negatives_nearest_boundary": sorted(fns, key=lambda i: -float(i.get("entailment", 0)))[
                :boundary_examples
            ],
            "boundary_band_examples": near[:boundary_examples],
        }

    if multiverifier_path is not None:
        mv_file = settings.resolve(multiverifier_path)
        if mv_file.exists():
            mv = json.loads(mv_file.read_text(encoding="utf-8"))
            per_item = mv.get("per_item_correct") or {}
            names = sorted(per_item)
            tests = {}
            for i, a in enumerate(names):
                for b in names[i + 1 :]:
                    if len(per_item[a]) == len(per_item[b]):
                        tests[f"{a}__vs__{b}"] = mcnemar_exact(
                            [bool(x) for x in per_item[a]], [bool(x) for x in per_item[b]]
                        )
            if tests:
                result["mcnemar"] = tests
            else:
                result["mcnemar_note"] = (
                    "multiverifier.json has no per_item_correct arrays; rerun the "
                    "multiverifier command from this version to enable McNemar tests."
                )
    if output_path is not None:
        out = settings.resolve(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
