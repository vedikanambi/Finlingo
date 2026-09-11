from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

from backend.app.core.config import Settings
from backend.app.pipeline import FinLingoPipeline
from backend.app.services.model_registry import ModelRegistry
from backend.app.services.nli_service import NLIService
from backend.evaluation.flb_builder import FLBBuilder, FLBRecord
from backend.evaluation.metrics import (
    binary_faithfulness_metrics,
    multiclass_metrics,
    retrieval_metrics,
)

RISK_LABELS = ["Auto-Renewal", "Hidden Fee", "Liability Waiver", "Data Sharing", "Penalty Clause", "Safe"]

# explicit tuple so validate_rq4_conditions() can catch a condition being silently added/removed/renamed
EXPECTED_RQ4_CONDITIONS = (
    "baseline_clean",
    "scanned_ocr_noise_proxy",
    "reordered_clause_sentences",
    "technical_derivative_language",
    "very_short_or_truncated",
    "unseen_jurisdiction",
    "cross_clause_dependency",
)


def validate_rq4_conditions(payload: dict) -> None:
    """Raise if the RQ4 result doesn't have exactly the 7 expected conditions over the same clauses as baseline."""
    conditions = payload.get("rq4_failure_modes", payload)
    keys = list(conditions.keys())
    if len(keys) != len(EXPECTED_RQ4_CONDITIONS):
        raise ValueError(
            f"Expected exactly {len(EXPECTED_RQ4_CONDITIONS)} RQ4 conditions "
            f"(one baseline + six perturbations), found {len(keys)}: {sorted(keys)}"
        )
    missing = set(EXPECTED_RQ4_CONDITIONS) - set(keys)
    unexpected = set(keys) - set(EXPECTED_RQ4_CONDITIONS)
    if missing or unexpected:
        raise ValueError(f"RQ4 condition set mismatch. Missing: {sorted(missing)}. Unexpected: {sorted(unexpected)}.")
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate RQ4 condition name detected.")
    clause_id_sets = {
        name: set(data["pipeline"]["clause_ids"])
        for name, data in conditions.items()
        if isinstance(data, dict) and "pipeline" in data and "clause_ids" in data["pipeline"]
    }
    if clause_id_sets:
        baseline_ids = clause_id_sets.get("baseline_clean")
        if baseline_ids is not None:
            for name, ids in clause_id_sets.items():
                if ids != baseline_ids:
                    raise ValueError(
                        f"Condition {name!r} covers a different clause-id set than baseline_clean "
                        "-- every perturbation must be a transform of the same clean clauses, not a "
                        "different clause selection."
                    )


def _ocr_noise(text: str, rng: random.Random) -> str:
    substitutions = {"l": "1", "I": "l", "O": "0", "rn": "m", "cl": "d"}
    output = text
    positions = max(1, len(text) // 100)
    for _ in range(positions):
        source, replacement = rng.choice(list(substitutions.items()))
        candidates = [index for index in range(len(output)) if output.startswith(source, index)]
        if candidates:
            index = rng.choice(candidates)
            output = output[:index] + replacement + output[index + len(source) :]
    return output


def _reorder(text: str, rng: random.Random) -> str:
    parts = [part.strip() for part in text.replace(". ", ".|").split("|") if part.strip()]
    rng.shuffle(parts)
    return " ".join(parts) or text


def _technical_derivative(text: str, _: random.Random) -> str:
    return (
        "For the avoidance of doubt and subject to all applicable provisos, carve-outs, schedules, and "
        "cross-referenced definitions, " + text
    )


def _very_short(text: str, _: random.Random) -> str:
    return " ".join(text.split()[: min(30, len(text.split()))])


def _unseen_jurisdiction(text: str, _: random.Random) -> str:
    return "Under the law of an unspecified foreign jurisdiction, " + text


def _cross_clause_dependency(text: str, _: random.Random) -> str:
    return "Subject to Clause 14.3 and the exceptions in Schedule B, " + text


def run_failure_analysis(
    settings: Settings,
    max_examples: int = 100,
    output_path: Path | None = None,
    *,
    records: list[FLBRecord] | None = None,
    conditions_to_run: tuple[str, ...] | None = None,
) -> dict:
    """Run RQ4 stress tests through the actual S2-S6 pipeline. conditions_to_run lets you run a subset for a quick check."""
    records = records or FLBBuilder(settings, ModelRegistry(settings)).build(max_examples)
    records = _limit_unique_clauses(records, max_examples)
    unique: dict[str, FLBRecord] = {record.source_id: record for record in records}
    rng = random.Random(settings.random_seed)
    transforms: dict[str, Callable[[str, random.Random], str]] = {
        "baseline_clean": lambda text, _: text,
        "scanned_ocr_noise_proxy": _ocr_noise,
        "reordered_clause_sentences": _reorder,
        "technical_derivative_language": _technical_derivative,
        "very_short_or_truncated": _very_short,
        "unseen_jurisdiction": _unseen_jurisdiction,
        "cross_clause_dependency": _cross_clause_dependency,
    }

    if conditions_to_run is not None:
        unknown = set(conditions_to_run) - set(transforms)
        if unknown:
            raise ValueError(f"Unknown RQ4 condition(s) requested: {sorted(unknown)}")
        transforms = {name: transforms[name] for name in conditions_to_run}

    pipeline = FinLingoPipeline(settings)
    nli = NLIService(pipeline.registry)
    output: dict[str, dict] = {}
    for name, transform in transforms.items():
        logger.info("RQ4 condition start: %s (%d clauses)", name, len(unique))
        condition_started = time.perf_counter()
        transformed_by_source = {
            source_id: transform(record.original_clause, rng) for source_id, record in unique.items()
        }
        risk_true: list[str] = []
        risk_predicted: list[str] = []
        fk_grades: list[float] = []
        source_faithfulness: list[float] = []
        predicted_unsupported: list[int] = []
        retrieval_rankings: list[list[str]] = []
        retrieval_gold: list[str | None] = []
        per_clause_records: list[dict] = []

        for index, (source_id, record) in enumerate(unique.items(), 1):
            clause_started = time.perf_counter()
            clause = transformed_by_source[source_id]
            simplification = pipeline.stage2.simplify(clause)
            simplified, grade = simplification.text, simplification.fk_grade
            per_clause_records.append(
                {
                    "source_clause_id": source_id,
                    "condition": name,
                    "generation_mode": simplification.generation_mode,
                    "generation_seed": simplification.generation_seed,
                    "output_hash": simplification.output_hash,
                    "attempts": simplification.attempts,
                }
            )
            query = pipeline.stage3.build_query(clause, simplified)
            retrieved = pipeline.stage3.retrieve(simplified, "faiss", original_clause=clause)
            evidence = pipeline.stage4.rerank(query, retrieved)
            risk = pipeline.stage5.classify(clause, evidence)
            faith = pipeline.stage6.verify(clause, simplified, evidence)
            logger.info(
                "RQ4 %s: clause %d/%d done in %.1fs (simplifier attempts=%d, fk_grade=%.2f)",
                name, index, len(unique), time.perf_counter() - clause_started,
                simplification.attempts, grade,
            )

            risk_true.append(record.risk_label)
            risk_predicted.append(risk.risk_label)
            fk_grades.append(grade)
            source_faithfulness.extend(item.source_faithfulness for item in faith)
            predicted_unsupported.extend(int(item.unsupported) for item in faith)
            retrieval_rankings.append([chunk.chunk_id for chunk in evidence])
            retrieval_gold.append(record.ground_truth_chunk_id)

        fixed_pairs = [(transformed_by_source[record.source_id], record.hypothesis) for record in records]
        fixed_scores = [
            score.entailment
            for score in nli.score_pairs(
                fixed_pairs,
                settings.verifier_batch_size,
                settings.verifier_max_length,
            )
        ]
        fixed_labels = [record.supported for record in records]
        output[name] = {
            "pipeline": {
                "risk": multiclass_metrics(risk_true, risk_predicted, RISK_LABELS),
                "mean_fk_grade": float(np.mean(fk_grades)) if fk_grades else None,
                "pct_fk_le_8": float(np.mean([grade <= 8 for grade in fk_grades])) if fk_grades else None,
                "mean_generated_source_faithfulness": float(np.mean(source_faithfulness))
                if source_faithfulness
                else None,
                "generated_unsupported_flag_rate": float(np.mean(predicted_unsupported))
                if predicted_unsupported
                else None,
                "retrieval": retrieval_metrics(retrieval_rankings, retrieval_gold, settings.top_j_reranked),
                "n_unique_clauses": len(unique),
                "clause_ids": sorted(unique.keys()),
                "generation_mode": "deterministic" if settings.s2_deterministic_mode else "stochastic",
                "per_clause": per_clause_records,
            },
            "fixed_hypothesis_verifier": asdict(
                binary_faithfulness_metrics(
                    fixed_labels,
                    fixed_scores,
                    settings.faithfulness_tau,
                )
            ),
        }
        logger.info(
            "RQ4 condition done: %s in %.1fs total", name, time.perf_counter() - condition_started
        )

    if "baseline_clean" in output:
        baseline = output["baseline_clean"]["pipeline"]
        for name, values in output.items():
            current = values["pipeline"]
            current["delta_vs_clean"] = {
                "risk_macro_f1": _delta(current["risk"].get("macro_f1"), baseline["risk"].get("macro_f1")),
                "mean_fk_grade": _delta(current.get("mean_fk_grade"), baseline.get("mean_fk_grade")),
                "mean_source_faithfulness": _delta(
                    current.get("mean_generated_source_faithfulness"),
                    baseline.get("mean_generated_source_faithfulness"),
                ),
            }

    payload = {
        "rq4_failure_modes": output,
        "tau": settings.faithfulness_tau,
        "notes": {
            "ocr": "Text-level OCR corruption proxy; final thesis should also run representative scanned PDFs through S1.",
            "ground_truth": "Retrieval deltas are available only for records with adjudicated evidence chunk IDs.",
        },
    }
    if conditions_to_run is None:
        validate_rq4_conditions(payload)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _delta(value: float | None, baseline: float | None) -> float | None:
    return float(value - baseline) if value is not None and baseline is not None else None


def _limit_unique_clauses(records: list[FLBRecord], max_clauses: int) -> list[FLBRecord]:
    selected: set[str] = set()
    output: list[FLBRecord] = []
    for record in records:
        if record.source_id not in selected and len(selected) >= max_clauses:
            continue
        selected.add(record.source_id)
        output.append(record)
    return output


def run_document_failure_analysis(
    settings: Settings,
    manifest_path: Path,
    output_path: Path | None = None,
) -> dict:
    """Run RQ4 on real files described by a CSV/JSON manifest (needs a 'path' column)."""
    import csv

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if manifest_path.suffix.lower() == ".json":
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        with manifest_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    if not isinstance(rows, list) or not rows:
        raise ValueError("RQ4 document manifest must contain at least one row")

    pipeline = FinLingoPipeline(settings)
    results: list[dict] = []
    for row in rows:
        path = Path(str(row.get("path") or "")).expanduser()
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        metadata = {key: value for key, value in row.items() if key != "path"}
        try:
            report = pipeline.run(path)
            results.append(
                {
                    "path": str(path),
                    "metadata": metadata,
                    "status": "ok",
                    "metrics": report.metrics.model_dump(mode="json"),
                    "warnings": report.warnings,
                    "parser_methods": sorted(
                        {clause.extraction_method for clause in report.clauses if clause.extraction_method}
                    ),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "path": str(path),
                    "metadata": metadata,
                    "status": "error",
                    "error": str(exc),
                }
            )

    groups: dict[str, dict[str, list[float]]] = {}
    for item in results:
        if item["status"] != "ok":
            continue
        metadata = item["metadata"]
        metrics = item["metrics"]
        for dimension in ("document_type", "jurisdiction", "scan_quality", "length_band"):
            value = str(metadata.get(dimension) or "unspecified")
            key = f"{dimension}={value}"
            bucket = groups.setdefault(key, {"fk": [], "faith": [], "unsupported": []})
            if metrics.get("mean_fk_grade") is not None:
                bucket["fk"].append(float(metrics["mean_fk_grade"]))
            if metrics.get("mean_primary_faithfulness") is not None:
                bucket["faith"].append(float(metrics["mean_primary_faithfulness"]))
            if metrics.get("unsupported_flag_rate") is not None:
                bucket["unsupported"].append(float(metrics["unsupported_flag_rate"]))
    summary = {
        key: {
            "documents": max(len(values["fk"]), len(values["faith"]), len(values["unsupported"])),
            "mean_fk_grade": float(np.mean(values["fk"])) if values["fk"] else None,
            "mean_primary_faithfulness": float(np.mean(values["faith"])) if values["faith"] else None,
            "mean_unsupported_flag_rate": float(np.mean(values["unsupported"])) if values["unsupported"] else None,
        }
        for key, values in groups.items()
    }
    payload = {
        "mode": "real_document_manifest",
        "manifest": str(manifest_path),
        "documents": results,
        "group_summary": summary,
        "successful": sum(item["status"] == "ok" for item in results),
        "failed": sum(item["status"] == "error" for item in results),
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
