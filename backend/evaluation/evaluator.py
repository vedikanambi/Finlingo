from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, create_model

from backend.app.core.config import Settings
from backend.app.core.provenance import runtime_manifest
from backend.app.pipeline import FinLingoPipeline
from backend.app.services.model_registry import ModelRegistry
from backend.evaluation.circularity_guard import independent_retrieval_recall
from backend.evaluation.flb_builder import FLBBuilder, FLBRecord
from backend.evaluation.metrics import (
    attribution_accuracy,
    binary_faithfulness_metrics,
    bootstrap_ci,
    bootstrap_precision_recall_ci,
    bootstrap_precision_recall_ci_clustered,
    confidence_calibration,
    multiclass_metrics,
    ordinal_score_metrics,
    retrieval_metrics,
    rq1_metrics,
)

logger = logging.getLogger(__name__)
RISK_LABELS = ["Auto-Renewal", "Hidden Fee", "Liability Waiver", "Data Sharing", "Penalty Clause", "Safe"]


class EvidenceDecision(BaseModel):
    chunk_id: str | None = None
    supported: bool
    confidence: float = Field(ge=0, le=1)
    rationale: str


def _provisional_evidence(
    settings: Settings,
    registry: ModelRegistry,
    record: FLBRecord,
    chunks,
) -> EvidenceDecision:
    """LLM-adjudicated provisional evidence, never final human ground truth."""
    if not chunks:
        return EvidenceDecision(
            chunk_id=None,
            supported=False,
            confidence=1.0,
            rationale="No regulatory candidates were retrieved.",
        )
    allowed = {chunk.chunk_id for chunk in chunks}
    evidence = "\n\n".join(f"[{chunk.chunk_id}] {chunk.source} {chunk.section or ''}\n{chunk.text}" for chunk in chunks)
    from backend.training.silver_label_providers import build_silver_label_provider

    provider_name = settings.silver_judge_provider or settings.silver_label_provider
    provider = build_silver_label_provider(settings, registry, task="judge", provider_name=provider_name)
    candidate_ids = tuple(sorted(allowed))
    call_schema = create_model(
        "EvidenceDecisionConstrained",
        chunk_id=(Literal[candidate_ids], ...),
        supported=(bool, ...),
        confidence=(float, Field(ge=0, le=1)),
        rationale=(str, ...),
    )
    result = provider.generate(
        schema=call_schema,
        system_prompt=(
            "Always select chunk_id as the single most topically relevant regulatory chunk to the hypothesis "
            "sentence, even if it does not fully entail it — this identifies what the retrieval system should "
            "have found. Separately decide supported=true or false using a strict, mechanical test, not an "
            "overall vibe of relevance:\n"
            "STEP 1: List the specific, checkable facts in the hypothesis sentence (exact numbers, durations, "
            "dollar/percentage amounts, named parties or obligations, concrete conditions).\n"
            "STEP 2: For supported=true, the chunk text must explicitly state or directly entail EVERY one of "
            "those specific facts, not merely discuss the same general subject area. A chunk about the right "
            "topic that only states a general principle, a disclosure procedure, or a different specific number "
            "is NOT supported — set supported=false in that case, even if it is still the best/most relevant "
            "candidate available.\n"
            "WORKED COUNTER-EXAMPLE: hypothesis 'the deal auto-renews yearly unless you cancel with 60 days "
            "notice' against a chunk that only says businesses must disclose auto-renewal terms clearly — this "
            "is topically about auto-renewal but never states the specific 60-day figure or a yearly term, so "
            "supported=false, even though chunk_id should still point at that chunk as the most relevant one "
            "found. Most retrieved chunks in this task will genuinely be unsupported by this strict test; do "
            "not default to true out of an instinct to be helpful. Return JSON only."
        ),
        user_prompt=(
            'Return {"chunk_id":"id of the most relevant candidate","supported":true,"confidence":0.0,'
            '"rationale":"..."}.\n'
            f"RISK LABEL (context only): {record.risk_label}\n"
            f"SOURCE CLAUSE:\n{record.original_clause}\n\n"
            f"HYPOTHESIS SENTENCE:\n{record.hypothesis}\n\nCANDIDATES:\n{evidence}"
        ),
    )
    if result.parsed is None:
        raise RuntimeError(f"Evidence adjudication returned no valid structured response: {result.metadata}")
    decision = result.parsed
    if decision.chunk_id not in allowed:
        decision.chunk_id = None
        decision.supported = False
    return decision


def validate_final_evaluation_records(settings: Settings, records: list[FLBRecord]) -> dict:
    reviewed = sum(record.human_reviewed for record in records)
    fraction = reviewed / len(records) if records else 0.0
    missing_evidence_labels = sum(record.evidence_supported is None for record in records)
    provisional = sum(
        record.evidence_label_source != "human_adjudicated" or record.risk_label_source != "human_adjudicated"
        for record in records
    )
    status = {
        "records": len(records),
        "human_reviewed": reviewed,
        "human_review_fraction": fraction,
        "missing_evidence_labels": missing_evidence_labels,
        "non_human_or_provisional": provisional,
        "human_risk_labels": sum(record.risk_label_source == "human_adjudicated" for record in records),
        "human_risk_scores": sum(record.human_risk_score is not None for record in records),
    }
    if settings.final_evaluation_require_all_human_review:
        if fraction < settings.final_evaluation_min_review_fraction:
            raise RuntimeError(
                "Final evaluation requires human adjudication for at least "
                f"{settings.final_evaluation_min_review_fraction:.0%} of FLB rows; found {fraction:.1%}."
            )
        if missing_evidence_labels:
            raise RuntimeError(f"Final evaluation has {missing_evidence_labels} rows without evidence-support labels")
        if not settings.final_evaluation_allow_provisional_evidence and provisional:
            raise RuntimeError(f"Final evaluation has {provisional} non-human/provisional evidence labels")
        if settings.final_evaluation_require_risk_scores:
            missing_scores = sum(record.human_risk_score is None for record in records)
            if missing_scores:
                raise RuntimeError(f"Final evaluation has {missing_scores} rows without adjudicated risk scores")
    return status


def evaluate_records(
    settings: Settings,
    records: list[FLBRecord],
    *,
    retrieval_method: str = "bm25",
    # reranker off by default - it actually hurts recall on this corpus, see pipeline.py
    use_reranker: bool = False,
    use_verifier: bool = True,
    adjudicate_evidence: bool = True,
    final_evaluation: bool = False,
    simplification_cache: dict[str, object] | None = None,
    risk_cache: dict[str, object] | None = None,
    adjudication_candidate_limit: int | None = None,
) -> dict:
    """simplification_cache/risk_cache let you reuse stage outputs across calls - only safe if the upstream settings match the run that populated them."""
    if final_evaluation:
        final_status = validate_final_evaluation_records(settings, records)
    else:
        final_status = None

    pipeline = FinLingoPipeline(settings)
    sources: list[str] = []
    predictions: list[str] = []
    references: list[str] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    risk_confidences: list[float] = []
    human_risk_scores: list[int] = []
    predicted_risk_scores: list[int] = []
    evidence_labels: list[int] = []
    evidence_scores: list[float] = []
    evidence_chunk_retrieved: list[bool] = []
    source_labels: list[int] = []
    source_scores: list[float] = []
    source_group_ids: list[str] = []
    attribution_pred: list[str | None] = []
    attribution_gold: list[str | None] = []
    latencies: list[float] = []
    generated_output_scores: list[float] = []
    generated_output_unsupported: list[int] = []
    risk_item_scores: list[float] = []
    evidence_item_scores: list[float] = []
    unique_cache: dict[str, dict] = {}
    pending_adjudication: list[tuple] = []

    for index, record in enumerate(records, 1):
        key = record.source_id
        if key not in unique_cache:
            if simplification_cache is not None and key in simplification_cache:
                simplification = simplification_cache[key]
            else:
                simplification = pipeline.stage2.simplify(record.original_clause)
                if simplification_cache is not None:
                    simplification_cache[key] = simplification
            simplified = simplification.text
            query = pipeline.stage3.build_query(record.original_clause, simplified)
            retrieved = pipeline.stage3.retrieve(
                simplified,
                retrieval_method,
                original_clause=record.original_clause,
            )
            evidence = (
                pipeline.stage4.rerank(query, retrieved) if use_reranker else retrieved[: settings.top_j_reranked]
            )
            if risk_cache is not None and key in risk_cache:
                risk = risk_cache[key]
            else:
                risk = pipeline.stage5.classify(record.original_clause, evidence)
                if risk_cache is not None:
                    risk_cache[key] = risk
            if use_verifier:
                generated = pipeline.stage6.verify(record.original_clause, simplified, evidence)
                generated_output_scores.extend(
                    item.faithfulness_score for item in generated if item.faithfulness_score is not None
                )
                generated_output_unsupported.extend(int(item.unsupported) for item in generated)
            unique_cache[key] = {
                "simplification": simplification,
                "retrieved": retrieved,
                "evidence": evidence,
                "risk": risk,
                "query": query,
            }
            sources.append(record.original_clause)
            predictions.append(simplified)
            references.append(record.reference_simplification)
            y_true.append(record.risk_label)
            y_pred.append(risk.risk_label)
            risk_confidences.append(risk.confidence)
            risk_item_scores.append(float(record.risk_label == risk.risk_label))
            if record.human_risk_score is not None:
                human_risk_scores.append(record.human_risk_score)
                predicted_risk_scores.append(risk.risk_score)

        cached = unique_cache[key]
        evidence = cached["evidence"]
        retrieved = cached["retrieved"]

        if adjudicate_evidence and record.evidence_supported is None:
            candidate_chunks = retrieved[: settings.top_j_reranked]
            record.evidence_candidates_json = json.dumps(
                [
                    {
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "section": chunk.section,
                        "source_url": chunk.source_url,
                        "text": chunk.text,
                    }
                    for chunk in candidate_chunks
                ],
                ensure_ascii=False,
            )
            pending_adjudication.append((record, candidate_chunks))

        if index % 25 == 0:
            logger.info("Evaluated %d/%d FLB rows (stage2/3/5 pass)", index, len(records))

    if pending_adjudication:
        from concurrent.futures import ThreadPoolExecutor

        def _adjudicate(item: tuple) -> None:
            record, candidate_chunks = item
            try:
                decision = _provisional_evidence(settings, pipeline.registry, record, candidate_chunks)
                record.evidence_supported = int(decision.supported)
                record.ground_truth_chunk_id = decision.chunk_id
                if decision.chunk_id:
                    selected = next((chunk for chunk in candidate_chunks if chunk.chunk_id == decision.chunk_id), None)
                    if selected is not None:
                        record.provisional_evidence_text = selected.text
                        record.provisional_evidence_source = selected.source
                        record.provisional_evidence_url = selected.source_url
                record.evidence_label_source = "gpt_provisional"
            except Exception as exc:
                logger.warning("Evidence adjudication failed for %s: %s", record.record_id, exc)

        # capped at 2 workers - cranking this up to 4 just stalled for hours, single-GPU ollama seems to serialize requests anyway
        workers = min(2, settings.pipeline_ollama_concurrency, len(pending_adjudication))
        logger.info("Running %d evidence adjudications with %d concurrent workers", len(pending_adjudication), workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_adjudicate, pending_adjudication))

    for index, record in enumerate(records, 1):
        if not use_verifier:
            break
        cached = unique_cache[record.source_id]
        evidence = cached["evidence"]
        started = time.perf_counter()
        source_nli = pipeline.stage6.nli.score_pairs(
            [(record.original_clause, record.hypothesis)],
            settings.verifier_batch_size,
            settings.verifier_max_length,
        )[0]
        best_id: str | None = None
        best_score = 0.0
        if evidence:
            reg_scores = pipeline.stage6.nli.score_pairs(
                [(chunk.text, record.hypothesis) for chunk in evidence],
                settings.verifier_batch_size,
                settings.verifier_max_length,
            )
            best_index = max(range(len(reg_scores)), key=lambda item: reg_scores[item].entailment)
            best_score = reg_scores[best_index].entailment
            if best_score >= settings.attribution_tau:
                best_id = evidence[best_index].chunk_id
        latencies.append((time.perf_counter() - started) * 1000)
        source_labels.append(record.supported)
        source_scores.append(source_nli.entailment)
        source_group_ids.append(record.source_id)
        if record.evidence_supported is not None:
            evidence_labels.append(record.evidence_supported)
            evidence_scores.append(best_score)
            evidence_item_scores.append(
                float((best_score >= settings.faithfulness_tau) == bool(record.evidence_supported))
            )
            attribution_pred.append(best_id if record.evidence_supported else None)
            attribution_gold.append(record.ground_truth_chunk_id if record.evidence_supported else None)
            evidence_chunk_retrieved.append(
                bool(record.ground_truth_chunk_id)
                and record.ground_truth_chunk_id in {chunk.chunk_id for chunk in evidence}
            )
        if index % 25 == 0:
            logger.info("Evaluated %d/%d FLB rows (verifier pass)", index, len(records))

    pre_rankings: list[list[str]] = []
    post_rankings: list[list[str]] = []
    retrieval_gold: list[str | None] = []
    for record in records:
        if record.ground_truth_chunk_id:
            cached = unique_cache[record.source_id]
            pre_rankings.append([chunk.chunk_id for chunk in cached["retrieved"]])
            post_rankings.append([chunk.chunk_id for chunk in cached["evidence"]])
            retrieval_gold.append(record.ground_truth_chunk_id)

    evidence_sources = {record.evidence_label_source or "unlabelled" for record in records}
    results = {
        "rq1": asdict(
            rq1_metrics(
                sources,
                predictions,
                references,
                bertscore_model=settings.bertscore_model,
            )
        )
        if sources
        else {},
        "rq2": multiclass_metrics(y_true, y_pred, RISK_LABELS) if y_true else {},
        "risk_confidence_calibration": confidence_calibration(y_true, y_pred, risk_confidences),
        "risk_score_validation": ordinal_score_metrics(human_risk_scores, predicted_risk_scores),
        "retrieval": {
            "before_reranking": retrieval_metrics(pre_rankings, retrieval_gold, settings.top_k_retrieval),
            "after_reranking": retrieval_metrics(post_rankings, retrieval_gold, settings.top_j_reranked),
        },
        "paired_items": {
            "risk_correct": risk_item_scores,
            "evidence_classification_correct": evidence_item_scores,
        },
        "configuration": {
            "retrieval_method": retrieval_method,
            "retrieval_query_mode": settings.retrieval_query_mode,
            "use_reranker": use_reranker,
            "use_verifier": use_verifier,
            "tau": settings.faithfulness_tau,
            "attribution_tau": settings.attribution_tau,
            "primary_premise": settings.faithfulness_primary_premise,
            "n_records": len(records),
            "n_unique_clauses": len({record.source_id for record in records}),
            "human_reviewed_records": sum(record.human_reviewed for record in records),
            "dataset_revisions": settings.dataset_revisions,
            "model_revisions": settings.model_revisions,
            "evidence_label_sources": sorted(evidence_sources),
            "final_evaluation": final_evaluation,
            "final_validation": final_status,
        },
    }

    if use_verifier and evidence_scores:
        chunk_metrics = binary_faithfulness_metrics(evidence_labels, evidence_scores, settings.faithfulness_tau)
        source_metrics = binary_faithfulness_metrics(source_labels, source_scores, settings.faithfulness_tau)
        retrieved_labels = [
            label for label, retrieved_flag in zip(evidence_labels, evidence_chunk_retrieved) if retrieved_flag
        ]
        retrieved_scores = [
            score for score, retrieved_flag in zip(evidence_scores, evidence_chunk_retrieved) if retrieved_flag
        ]
        chunk_metrics_retrieval_conditioned = (
            binary_faithfulness_metrics(retrieved_labels, retrieved_scores, settings.faithfulness_tau)
            if retrieved_labels
            else None
        )
        chunk_metrics_retrieval_conditioned_ci = (
            bootstrap_precision_recall_ci(
                retrieved_labels,
                retrieved_scores,
                settings.faithfulness_tau,
                seed=settings.random_seed,
                samples=settings.bootstrap_samples,
            )
            if retrieved_labels
            else None
        )
        retrieval_coverage = (
            sum(evidence_chunk_retrieved) / len(evidence_chunk_retrieved) if evidence_chunk_retrieved else 0.0
        )
        premise_mode = settings.faithfulness_primary_premise
        if premise_mode == "source_clause":
            primary_labels, primary_scores, primary = source_labels, source_scores, source_metrics
            primary_definition = "P(entailment | generated sentence, source clause)"
        elif premise_mode == "dual":
            primary_labels, primary_scores, primary = evidence_labels, evidence_scores, chunk_metrics
            primary_definition = "P(entailment | generated sentence, min(source clause, retrieved regulatory chunk))"
        else:
            primary_labels, primary_scores, primary = evidence_labels, evidence_scores, chunk_metrics
            primary_definition = "P(entailment | generated sentence, retrieved regulatory chunk)"
        source_metrics_ci_clustered = bootstrap_precision_recall_ci_clustered(
            source_labels,
            source_scores,
            settings.faithfulness_tau,
            source_group_ids,
            seed=settings.bootstrap_seed,
            samples=settings.bootstrap_resamples,
        )
        source_metrics_ci_record_level = bootstrap_precision_recall_ci(
            source_labels,
            source_scores,
            settings.faithfulness_tau,
            seed=settings.bootstrap_seed,
            samples=settings.bootstrap_resamples,
        )
        results["rq3"] = {
            "primary_definition": primary_definition,
            "primary_tau": asdict(primary),
            "source_clause_precision_recall_ci": {
                "bootstrap_unit": settings.bootstrap_unit,
                "clause_clustered": source_metrics_ci_clustered,
                "record_level_legacy_sensitivity": source_metrics_ci_record_level,
                "note": (
                    "clause_clustered resamples whole source clauses (both the supported and "
                    "unsupported companion record travel together) and is the primary reported "
                    "interval. record_level_legacy_sensitivity resamples individual records "
                    "independently and is kept only as a historical sensitivity comparison -- "
                    "it can understate uncertainty when paired records share a clause."
                ),
            },
            "tau_sweep": {
                str(tau): asdict(binary_faithfulness_metrics(primary_labels, primary_scores, tau))
                for tau in settings.tau_values
            },
            "source_clause_preservation_metrics": asdict(source_metrics),
            "retrieved_chunk_grounding_metrics": asdict(chunk_metrics),
            "retrieval_conditioned_chunk_grounding_metrics": (
                asdict(chunk_metrics_retrieval_conditioned) if chunk_metrics_retrieval_conditioned else None
            ),
            "retrieval_conditioned_chunk_grounding_ci": chunk_metrics_retrieval_conditioned_ci,
            "retrieval_conditioned_note": (
                "DIAGNOSTIC DECOMPOSITION, NOT A REPLACEMENT HEADLINE. The proposal-exact "
                "unconditional primary_chunk_grounded metric above (evaluated on all n=182 rows) "
                "remains the reported RQ3 headline. This retrieval-conditioned view isolates "
                "verifier calibration from retrieval-stage misses by scoring only the subset of "
                "records (see chunk_retrieval_coverage.n_retrieved) where the grounding chunk was "
                "actually present in the evidence set the verifier scored -- it explains the "
                "mechanism behind the near-zero headline, it does not substitute for it."
            ),
            "chunk_retrieval_coverage": {
                "value": retrieval_coverage,
                "n_retrieved": sum(evidence_chunk_retrieved),
                "n_total": len(evidence_chunk_retrieved),
                "circular_diagnostic": True,
                "definition": (
                    "Fraction of records where the regulatory chunk judged (by the silver "
                    "adjudicator) to actually ground the hypothesis was present in the "
                    "post-rerank evidence set the verifier scored against. Low coverage means "
                    "the primary chunk-grounded metric is bottlenecked by retrieval recall, not "
                    "verifier calibration -- retrieval_conditioned_chunk_grounding_metrics isolates "
                    "verifier performance on the subset where the correct premise was available."
                ),
                "circularity_note": (
                    "This figure is a legacy, within-benchmark diagnostic, not independent evidence "
                    "of a retrieval bottleneck: the adjudicator's ground-truth chunk_id is drawn "
                    "from the same retrieved[:top_j_reranked] candidate pool retrieval coverage is "
                    "measured against (see backend/evaluation/circularity_guard.py). It is retained "
                    "for continuity with prior reported runs, marked circular_diagnostic=true so it "
                    "cannot be mistaken for the independent evaluation below."
                ),
            },
            "independent_retrieval_evaluation": asdict(
                independent_retrieval_recall(
                    records,
                    {
                        record.record_id: [chunk.chunk_id for chunk in unique_cache[record.source_id]["retrieved"]]
                        for record in records
                        if record.source_id in unique_cache
                    },
                    settings.top_k_retrieval,
                )
            ),
            "attribution": attribution_accuracy(attribution_pred, attribution_gold),
            "generated_output": {
                "mean_primary_faithfulness": (
                    sum(generated_output_scores) / len(generated_output_scores) if generated_output_scores else None
                ),
                "unsupported_flag_rate": (
                    sum(generated_output_unsupported) / len(generated_output_unsupported)
                    if generated_output_unsupported
                    else None
                ),
                "sentence_count": len(generated_output_scores),
                "definition": (
                    "Operational fraction of generated sentences below tau. This is not labelled as a true "
                    "hallucination rate without independent human ground truth."
                ),
            },
            "validated_missed_unsupported_rate": primary.missed_unsupported_rate,
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "entailment_score_ci": bootstrap_ci(evidence_scores, settings.random_seed, settings.bootstrap_samples),
        }

    rq1 = results.get("rq1", {})
    rq2 = results.get("rq2", {})
    rq3 = results.get("rq3", {})
    primary = rq3.get("primary_tau", {}) if isinstance(rq3, dict) else {}
    attribution = rq3.get("attribution", {}) if isinstance(rq3, dict) else {}
    results["target_checks"] = {
        "sari_gt_40": _target(rq1.get("sari"), 40.0, "gt"),
        "bertscore_f1_gt_0_88": _target(rq1.get("bertscore_f1"), 0.88, "gt"),
        "mean_fk_grade_le_9": _target(rq1.get("mean_fk_grade"), settings.fk_acceptance_target, "le"),
        "risk_macro_f1_gt_target": _target(rq2.get("macro_f1"), settings.target_risk_macro_f1, "gt"),
        "faithfulness_precision_gt_target": _target(
            primary.get("precision"), settings.target_faithfulness_precision, "gt"
        ),
        "faithfulness_recall_gt_target": _target(primary.get("recall"), settings.target_faithfulness_recall, "gt"),
        "attribution_accuracy_gt_target": _target(
            attribution.get("accuracy"), settings.target_attribution_accuracy, "gt"
        ),
        "validated_missed_unsupported_rate_lt_target": _target(
            rq3.get("validated_missed_unsupported_rate") if isinstance(rq3, dict) else None,
            settings.target_hallucination_rate,
            "lt",
        ),
    }
    results["runtime_manifest"] = runtime_manifest(settings)
    return results


def _target(value: float | None, threshold: float, operator: str) -> dict:
    if value is None:
        return {"value": None, "threshold": threshold, "operator": operator, "passed": None}
    comparisons = {
        "gt": value > threshold,
        "lt": value < threshold,
        "le": value <= threshold,
        "ge": value >= threshold,
    }
    return {"value": value, "threshold": threshold, "operator": operator, "passed": comparisons[operator]}


def evaluate_finlingo(
    settings: Settings,
    *,
    max_examples: int | None = None,
    output_path: Path | None = None,
    export_flb_path: Path | None = None,
    review_template_path: Path | None = None,
    review_file_path: Path | None = None,
    review_all: bool = False,
    final_evaluation: bool = False,
) -> dict:
    registry = ModelRegistry(settings)
    builder = FLBBuilder(settings, registry)
    records = builder.build(max_examples or settings.flb_size)
    review_status = None
    if review_file_path is not None:
        from backend.evaluation.human_review import apply_adjudicated_reviews

        review_status = apply_adjudicated_reviews(records, review_file_path)
    if final_evaluation:
        validate_final_evaluation_records(settings, records)
    result = evaluate_records(
        settings,
        records,
        adjudicate_evidence=not final_evaluation,
        final_evaluation=final_evaluation,
    )
    result["human_review"] = {
        "review_file": str(review_file_path) if review_file_path else None,
        "status": review_status,
        "human_reviewed_records": sum(record.human_reviewed for record in records),
        "provisional_records": sum(record.evidence_label_source != "human_adjudicated" for record in records),
    }
    if export_flb_path:
        builder.export(records, export_flb_path)
    if review_template_path:
        builder.export_review_template(
            records,
            review_template_path,
            len(records) if review_all else settings.flb_human_review_size,
            settings.random_seed,
        )
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
