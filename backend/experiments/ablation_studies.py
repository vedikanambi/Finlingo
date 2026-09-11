from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from backend.app.core.config import Settings
from backend.app.pipeline import FinLingoPipeline
from backend.app.services.model_registry import ModelRegistry
from backend.evaluation.evaluator import evaluate_records
from backend.evaluation.flb_builder import FLBBuilder, FLBRecord
from backend.evaluation.metrics import paired_bootstrap_difference, rq1_metrics
from backend.experiments.baselines import (
    finbert_domain_baseline,
    gpt_zero_shot_baseline,
    ragas_style_paragraph_baseline,
    rule_based_baseline,
    selfcheckgpt_nli_baseline,
)


def _checkpoint(output_path: Path | None, records_len: int, results: dict, baselines: dict, status: str) -> None:
    """Save partial progress so an interrupted run isn't a total loss."""
    if not output_path:
        return
    payload = {
        "status": status,
        "shared_benchmark_size": records_len,
        "experiments": results,
        "baselines": baselines,
        "definitions": {
            "no_s6": "verifier omitted",
            "no_s4_reranking": "FAISS without cross-encoder",
            "zero_shot": "no FinLingo task adapters; general pretrained checkpoints are identified explicitly",
            "simplification_only": "S1-S2 only",
            "bm25": "BM25 replaces FAISS; downstream unchanged",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(output_path)


def _load_resume_state(output_path: Path | None) -> tuple[dict, dict]:
    """Resume from a checkpoint file if it looks like ours."""
    if not output_path or not output_path.exists():
        return {}, {}
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}, {}
    if "experiments" not in payload or "baselines" not in payload:
        return {}, {}
    return payload.get("experiments", {}), payload.get("baselines", {})


def run_ablations(
    settings: Settings,
    max_examples: int | None = None,
    output_path: Path | None = None,
    *,
    records: list[FLBRecord] | None = None,
    include_gpt: bool = True,
    resume: bool = False,
) -> dict:
    records = records or FLBBuilder(settings, ModelRegistry(settings)).build(max_examples or settings.flb_size)
    if resume:
        results, baselines = _load_resume_state(output_path)
        if results or baselines:
            print(
                f"Resuming: {len(results)} variant(s) and {len(baselines)} baseline(s) "
                f"already on disk will be skipped: {sorted(results)} / {sorted(baselines)}"
            )
    else:
        results, baselines = {}, {}

    # only reuse these caches for variants that don't touch the cached stage's inputs
    simplification_cache: dict[str, object] = {}
    risk_cache: dict[str, object] = {}

    if "full" not in results:
        results["full"] = evaluate_records(
            settings,
            records,
            simplification_cache=simplification_cache,
            risk_cache=risk_cache,
        )
        _checkpoint(output_path, len(records), results, baselines, "in_progress: full done")

    if "no_s6" not in results:
        results["no_s6"] = evaluate_records(
            settings,
            records,
            use_verifier=False,
            adjudicate_evidence=False,
            simplification_cache=simplification_cache,
            risk_cache=risk_cache,
        )
        _checkpoint(output_path, len(records), results, baselines, "in_progress: no_s6 done")

    if "no_s4_reranking" not in results:
        results["no_s4_reranking"] = evaluate_records(
            settings,
            records,
            use_reranker=False,
            adjudicate_evidence=False,
            simplification_cache=simplification_cache,
        )
        _checkpoint(output_path, len(records), results, baselines, "in_progress: no_s4_reranking done")

    if "bm25" not in results:
        results["bm25"] = evaluate_records(
            settings,
            records,
            retrieval_method="bm25",
            adjudicate_evidence=False,
            simplification_cache=simplification_cache,
        )
        _checkpoint(output_path, len(records), results, baselines, "in_progress: bm25 done")

    if "zero_shot" not in results:
        zero = settings.model_copy(
            update={
                "simplifier_adapter": None,
                "risk_adapter": None,
                "verifier_adapter": None,
                "allow_base_models": True,
                "verifier_require_adapter": False,
            }
        )
        results["zero_shot"] = evaluate_records(zero, records, adjudicate_evidence=False)
        _checkpoint(output_path, len(records), results, baselines, "in_progress: zero_shot done")

    if "simplification_only" not in results:
        pipeline = FinLingoPipeline(settings)
        unique = {r.source_id: r for r in records}
        sources = []
        pred = []
        refs = []
        for r in unique.values():
            if r.source_id in simplification_cache:
                simplification = simplification_cache[r.source_id]
            else:
                simplification = pipeline.stage2.simplify(r.original_clause)
                simplification_cache[r.source_id] = simplification
            sources.append(r.original_clause)
            pred.append(simplification.text)
            refs.append(r.reference_simplification)
        results["simplification_only"] = {
            "rq1": asdict(
                rq1_metrics(
                    sources,
                    pred,
                    refs,
                    bertscore_model=settings.bertscore_model,
                )
            )
        }
        _checkpoint(output_path, len(records), results, baselines, "in_progress: simplification_only done")

    baseline_registry = ModelRegistry(settings)
    if "rule_based" not in baselines or "plain_rag" not in baselines:
        baselines["rule_based"] = rule_based_baseline(records)
        baselines["plain_rag"] = results["no_s6"]
        _checkpoint(output_path, len(records), results, baselines, "in_progress: rule_based+plain_rag done")

    if "finbert" not in baselines:
        baselines["finbert"] = finbert_domain_baseline(settings, records)
        _checkpoint(output_path, len(records), results, baselines, "in_progress: finbert done")

    if "ragas_style" not in baselines:
        baselines["ragas_style"] = ragas_style_paragraph_baseline(settings, baseline_registry, records)
        _checkpoint(output_path, len(records), results, baselines, "in_progress: ragas_style done")

    if "selfcheckgpt_style" not in baselines:
        baselines["selfcheckgpt_style"] = selfcheckgpt_nli_baseline(settings, baseline_registry, records)
        _checkpoint(output_path, len(records), results, baselines, "in_progress: selfcheckgpt_style done")

    if include_gpt and "gpt4_zero_shot" not in baselines:
        baselines["gpt4_zero_shot"] = gpt_zero_shot_baseline(settings, baseline_registry, records)
        _checkpoint(output_path, len(records), results, baselines, "in_progress: gpt4_zero_shot done")

    significance = {}
    full_items = results["full"].get("paired_items", {})
    for name in ("no_s4_reranking", "bm25", "zero_shot"):
        other_items = results[name].get("paired_items", {})
        significance[name] = {
            "risk_accuracy_difference": paired_bootstrap_difference(
                full_items.get("risk_correct", []),
                other_items.get("risk_correct", []),
                seed=settings.random_seed,
                samples=settings.bootstrap_samples,
            ),
            "evidence_accuracy_difference": paired_bootstrap_difference(
                full_items.get("evidence_classification_correct", []),
                other_items.get("evidence_classification_correct", []),
                seed=settings.random_seed,
                samples=settings.bootstrap_samples,
            ),
        }
    payload = {
        "status": "complete",
        "shared_benchmark_size": len(records),
        "experiments": results,
        "baselines": baselines,
        "paired_significance": significance,
        "definitions": {
            "no_s6": "verifier omitted",
            "no_s4_reranking": "FAISS without cross-encoder",
            "zero_shot": "no FinLingo task adapters; general pretrained checkpoints are identified explicitly",
            "simplification_only": "S1-S2 only",
            "bm25": "BM25 replaces FAISS; downstream unchanged",
        },
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload
