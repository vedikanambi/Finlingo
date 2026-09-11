"""Evaluate tuned settings against the previously exported locked FLB rows
without rebuilding the benchmark, to avoid LLM randomness in a before/after comparison."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import Settings
from backend.app.core.schemas import SimplificationResult
from backend.app.services.model_registry import ModelRegistry
from backend.app.services.text_utils import flesch_kincaid_grade
from backend.app.stages.stage5_risk_classifier import RISK_LABELS, Stage5RiskClassifier
from backend.evaluation.evaluator import evaluate_records
from backend.evaluation.flb_builder import FLBRecord


_INT_FIELDS = {"supported", "judge_risk_confidence", "evidence_supported", "human_risk_score",
               "judge_semantic_accuracy", "judge_readability"}
_FLOAT_FIELDS = {"sari", "bertscore_f1", "fk_grade"}
_BOOL_FIELDS = {"accepted", "human_reviewed", "quality_filter_passed", "is_pilot"}


def _optional(value):
    return None if value in {None, "", "None", "null"} else value


def load_locked_records(path: Path) -> list[FLBRecord]:
    allowed = {field.name for field in fields(FLBRecord)}
    records = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            values = {key: _optional(value) for key, value in raw.items() if key in allowed}
            for key in _INT_FIELDS:
                if values.get(key) is not None:
                    values[key] = int(values[key])
            for key in _FLOAT_FIELDS:
                if values.get(key) is not None:
                    values[key] = float(values[key])
            for key in _BOOL_FIELDS:
                if values.get(key) is not None:
                    values[key] = str(values[key]).strip().lower() in {"1", "true", "yes"}
            try:
                records.append(FLBRecord(**values))
            except TypeError as exc:
                raise ValueError(f"Invalid locked FLB row {line_number}: {exc}") from exc
    if not records:
        raise RuntimeError(f"No locked FLB rows found in {path}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("reports/flb_achievable_export.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/focused_locked_evaluation.json"))
    parser.add_argument(
        "--profile", type=Path, default=Path("configs/focused_tuning_profile.env"),
        help="Explicit tuning overrides; avoids dependence on shell environment inheritance.",
    )
    parser.add_argument(
        "--risk-only", action="store_true",
        help="Evaluate trained RQ2 without loading the external regulatory corpus.",
    )
    args = parser.parse_args()
    settings = Settings(_env_file=(Path(".env"), args.profile))
    records = load_locked_records(args.input)
    if args.risk_only:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        unique = {record.source_id: record for record in records}
        registry = ModelRegistry(settings)
        try:
            classifier = Stage5RiskClassifier(settings, registry)
            predictions = [classifier.classify(record.original_clause, []) for record in unique.values()]
        finally:
            registry.close()
        truth = [record.risk_label for record in unique.values()]
        predicted = [item.risk_label for item in predictions]
        report = classification_report(
            truth, predicted, labels=RISK_LABELS, output_dict=True, zero_division=0
        )
        result = {
            "mode": "locked_rq2_risk_only",
            "n_examples": len(truth),
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
            "accuracy": accuracy_score(truth, predicted),
            "labels": RISK_LABELS,
            "confusion_matrix": confusion_matrix(truth, predicted, labels=RISK_LABELS).tolist(),
            "per_class": {label: report[label] for label in RISK_LABELS},
            "configuration": {
                "model": settings.risk_classifier_model,
                "adapter": str(settings.risk_classifier_adapter),
            },
            "locked_benchmark": {"source": str(args.input), "selection_use": "none; evaluation only"},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(args.output), "macro_f1": result["macro_f1"]}, indent=2))
        return
    # Reuse the hypothesis already in the locked export instead of regenerating it -
    # the S2 adapter that produced it no longer exists.
    simplification_cache = {}
    for record in records:
        if record.source_id in simplification_cache or not record.supported:
            continue
        grade = flesch_kincaid_grade(record.hypothesis)
        simplification_cache[record.source_id] = SimplificationResult(
            text=record.hypothesis,
            fk_grade=grade,
            semantic_preservation_score=1.0,
            semantic_target_met=True,
            readability_target_met=grade <= settings.fk_generation_target,
            acceptance_target_met=grade <= settings.fk_acceptance_target,
            accepted=bool(record.accepted),
            attempts=1,
            status="accepted" if record.accepted else "needs_review",
        )
    result = evaluate_records(
        settings,
        records,
        adjudicate_evidence=False,
        simplification_cache=simplification_cache,
    )
    # hypothesis == reference in the locked export, so RQ1 here would just be comparing a string to itself
    result["rq1"] = {
        "status": "not_evaluated",
        "reason": "locked export hypothesis equals reference; self-comparison would be invalid",
    }
    result["locked_benchmark"] = {
        "source": str(args.input),
        "rows": len(records),
        "selection_use": "none; evaluation only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(records)}, indent=2))


if __name__ == "__main__":
    main()
