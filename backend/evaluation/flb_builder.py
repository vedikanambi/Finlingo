from __future__ import annotations

import hashlib
import re
import yaml

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from backend.app.core.config import FLBMode, Settings
from backend.app.services.dataset_streams import StreamingDatasetRepository
from backend.app.services.model_registry import ModelRegistry
from backend.app.services.text_utils import normalise_text
from backend.evaluation.metrics import rq1_example_quality
from backend.evaluation.risk_taxonomy import RiskTaxonomy
from backend.training.silver_labels import SilverLabelGenerator

logger = logging.getLogger(__name__)


def _risk_judge_reasons(row: dict, risk_assessment, min_score: int) -> list[str]:
    """Apply mandatory risk confirmation only to provisional benchmark labels."""
    source_type = str(row.get("source_type") or "")
    provisional = bool(row.get("provisional", False)) or source_type == "cuad_provisional_safe"
    if not provisional:
        return []
    if risk_assessment is None:
        return ["missing_risk_judge"]

    reasons: list[str] = []
    expected = _normalise_label(row["risk_label"]).replace(" ", "")
    observed = _normalise_label(risk_assessment.correct_label).replace(" ", "")
    if observed != expected:
        reasons.append(f"risk_label={risk_assessment.correct_label!r}_expected={row['risk_label']!r}")
    if risk_assessment.confidence < min_score:
        reasons.append(f"risk_confidence={risk_assessment.confidence}")
    return reasons


def _normalise_label(value: str) -> str:
    """Normalise a source label for exact dictionary lookup."""
    value = str(value or "").strip().lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _canonical_label(value: str, configured_labels: list[str] | tuple[str, ...]) -> str | None:
    """Map formatting variants to the exact configured label."""
    normalized = _normalise_label(value)
    compact = normalized.replace(" ", "")
    by_normalized = {_normalise_label(label): label for label in configured_labels}
    by_compact = {_normalise_label(label).replace(" ", ""): label for label in configured_labels}
    return by_normalized.get(normalized) or by_compact.get(compact)


@dataclass
class FLBRecord:
    record_id: str
    source_id: str
    document_id: str
    original_clause: str
    reference_simplification: str
    risk_label: str
    hypothesis: str
    supported: int
    source_dataset: str
    source_revision: str | None = None
    source_split: str = "test"
    source_document_id: str | None = None
    original_source_label: str | None = None
    provisional_risk_label: str | None = None
    judge_risk_label: str | None = None
    judge_risk_confidence: int | None = None
    source_text_hash: str | None = None
    candidate_text_hash: str | None = None
    leakage_partition_method: str | None = None
    accepted: bool = True
    rejection_reasons: str | None = None
    nli_label: str | None = None
    dataset_split: str = "test"
    ground_truth_chunk_id: str | None = None
    evidence_candidates_json: str | None = None
    provisional_evidence_text: str | None = None
    provisional_evidence_source: str | None = None
    provisional_evidence_url: str | None = None
    evidence_supported: int | None = None
    evidence_label_source: str | None = None
    risk_label_source: str = "gpt_provisional"
    human_risk_label: str | None = None
    human_risk_score: int | None = None
    human_reviewed: bool = False
    alternate_reference: str | None = None
    sari: float | None = None
    bertscore_f1: float | None = None
    fk_grade: float | None = None
    judge_semantic_accuracy: int | None = None
    judge_readability: int | None = None
    judge_rationale: str | None = None
    judge_risk_rationale: str | None = None
    generation_model: str | None = None
    quality_filter_passed: bool = False
    is_pilot: bool = False

    # ground_truth_chunk_id is judge-selected from the same pool being scored (circular); these
    # independent_gold_* fields stay unset until real human adjudication exists, so retrieval eval is blocked, not faked.
    independent_gold_chunk_id: str | None = None
    independent_gold_document_id: str | None = None
    independent_gold_source: str | None = None


class FLBBuilder:
    """Builds FLB in RAM from a leakage-safe CUAD test partition: independently rewrites each
    clause twice, filters by SARI/BERTScore/FK, and scores via the GPT judge; CFPB is excluded."""

    PILOT_LABEL = "PILOT / NOT FINAL THESIS RESULTS"

    def __init__(self, settings: Settings, registry: ModelRegistry) -> None:
        self.settings = settings
        self.repo = StreamingDatasetRepository(settings)
        self.taxonomy = RiskTaxonomy(settings.resolve(settings.risk_taxonomy_path))
        self.silver = SilverLabelGenerator(settings, registry)

    @classmethod
    def feasibility_audit(
        cls,
        settings: Settings,
        target_size: int | None = None,
    ) -> dict:
        """Return per-class supply counts and shortages; no provider is called, safe without API keys."""
        from backend.app.services.model_registry import ModelRegistry  # local to avoid circular

        if target_size is None:
            target_size = settings.flb_pilot_size if settings.flb_mode == FLBMode.pilot else settings.flb_size

        taxonomy = RiskTaxonomy(settings.resolve(settings.risk_taxonomy_path))
        labels = list(settings.configured_risk_labels)
        safe_label = settings.safe_risk_label
        quotas = _balanced_quotas(labels, target_size)

        safe_config_path = settings.resolve(settings.flb_safe_candidates_path)
        safe_config = yaml.safe_load(safe_config_path.read_text(encoding="utf-8"))
        safe_categories: set[str] = set()
        for value in safe_config.get("provisional_safe_categories", []):
            safe_categories.add(_normalise_label(str(value)))

        repo = StreamingDatasetRepository(settings)
        counts: dict[str, int] = {label: 0 for label in labels}
        seen_ids: set[str] = set()
        doc_ids: set[str] = set()

        external_path = settings.resolve(settings.flb_external_gold_path)

        if settings.flb_external_gold_enabled and external_path.exists():
            external_payload = yaml.safe_load(external_path.read_text(encoding="utf-8")) or {}

            for external_row in external_payload.get("records") or []:
                if not isinstance(external_row, dict):
                    continue

                source_id = str(external_row.get("source_id") or "").strip()
                risk_label = _canonical_label(
                    str(external_row.get("risk_label") or ""),
                    labels,
                )
                text_value = normalise_text(str(external_row.get("text") or ""))

                if risk_label is None or risk_label not in counts:
                    continue

                if not source_id or len(text_value.split()) < 8:
                    continue

                if not bool(external_row.get("human_reviewed")):
                    continue

                if not bool(external_row.get("excluded_from_training")):
                    continue

                if source_id in seen_ids:
                    continue

                seen_ids.add(source_id)
                doc_ids.add(str(external_row.get("document_id") or source_id))
                counts[risk_label] += 1

        silver_path = settings.resolve(settings.flb_external_silver_path)
        if settings.flb_external_silver_enabled and silver_path.exists():
            silver_payload = yaml.safe_load(silver_path.read_text(encoding="utf-8")) or {}
            for silver_row in silver_payload.get("records") or []:
                if not isinstance(silver_row, dict):
                    continue

                source_id = str(silver_row.get("source_id") or "").strip()
                risk_label = _canonical_label(str(silver_row.get("risk_label") or ""), tuple(labels))
                text_value = normalise_text(str(silver_row.get("text") or ""))

                if risk_label is None or risk_label not in counts:
                    continue
                if not source_id or len(text_value.split()) < 8:
                    continue
                if not bool(silver_row.get("excluded_from_training")):
                    continue
                if source_id in seen_ids:
                    continue

                seen_ids.add(source_id)
                doc_ids.add(str(silver_row.get("document_id") or source_id))
                counts[risk_label] += 1

        try:
            cuad_rows = repo.stream_partitioned(
                settings.cuad_dataset,
                requested_split="test",
                config=settings.cuad_config,
                key_fields=("file_name", "title", "document_id", "id"),
                shuffle=False,
            )
            for row in cuad_rows:
                raw_label = str(row.get("label") or row.get("question") or "").strip()
                mapped = taxonomy.map_question(raw_label)

                if mapped is not None:
                    risk_label = mapped
                elif _normalise_label(raw_label) in safe_categories:
                    risk_label = safe_label
                else:
                    continue

                text_value = str(row.get("clause") or row.get("text") or "").strip()
                if not text_value or len(text_value.split()) < 8:
                    continue

                text_hash = hashlib.sha256(text_value.encode("utf-8", errors="ignore")).hexdigest()[:12]
                source_id = "::".join(
                    [
                        "cuad",
                        str(row.get("file_name") or "unknown-file"),
                        str(row.get("class_id") or raw_label or "unknown-class"),
                        str(row.get("start_at") if row.get("start_at") is not None else "unknown-start"),
                        str(row.get("end_at") if row.get("end_at") is not None else "unknown-end"),
                        text_hash,
                    ]
                )
                if source_id in seen_ids:
                    continue
                seen_ids.add(source_id)
                doc_id = str(row.get("file_name") or row.get("title") or source_id)
                doc_ids.add(doc_id)
                counts[risk_label] = counts.get(risk_label, 0) + 1
        finally:
            repo.close()

        shortages = {
            label: {
                "available": counts.get(label, 0),
                "required": quotas[label],
                "short": quotas[label] - counts.get(label, 0),
            }
            for label in labels
            if counts.get(label, 0) < quotas[label]
        }

        feasible = not shortages
        result = {
            "mode": settings.flb_mode.value,
            "target_size": target_size,
            "unique_documents": len(doc_ids),
            "unique_clauses": len(seen_ids),
            "class_counts": {label: counts.get(label, 0) for label in labels},
            "quotas": quotas,
            "feasible": feasible,
            "shortages": shortages,
            "cfpb_excluded": True,
            "note": (
                cls.PILOT_LABEL
                if settings.flb_mode == FLBMode.pilot
                else ("FEASIBLE — safe to proceed" if feasible else "INFEASIBLE — do not start Ollama generation")
            ),
        }
        logger.info("FLB feasibility_audit result=%s", result)
        return result

    def build(self, size: int | None = None) -> list[FLBRecord]:
        mode = self.settings.flb_mode
        is_pilot = mode == FLBMode.pilot

        if is_pilot:
            source_target = size if size is not None else self.settings.flb_pilot_size
            logger.warning(
                "FLB pilot mode: building %d-record benchmark. %s",
                source_target,
                self.PILOT_LABEL,
            )
        else:
            source_target = size if size is not None else self.settings.flb_size

        labels = list(self.settings.configured_risk_labels)
        accepted_plan = None
        if mode == FLBMode.achievable:
            from backend.evaluation.flb_plan import load_accepted_plan

            accepted_plan = load_accepted_plan(self.settings)
            quotas = {label: int(accepted_plan["achievable_quotas"][label]) for label in labels}
            source_target = sum(quotas.values())
            logger.info(
                "FLB achievable mode: building to accepted plan quotas=%s (total=%d)",
                quotas,
                source_target,
            )
        else:
            quotas = _balanced_quotas(labels, source_target)
        pool_quotas = {
            label: max(quota * self.settings.flb_candidate_multiplier, quota) for label, quota in quotas.items()
        }
        if accepted_plan is not None:
            supply_limited = accepted_plan.get("supply_limited_classes") or {}
            for label, info in supply_limited.items():
                if label in pool_quotas and isinstance(info, dict) and info.get("quality_yield_capped"):
                    pool_quotas[label] = max(pool_quotas[label], int(info.get("available", 0)))

        audit = self.feasibility_audit(self.settings, source_target)
        if accepted_plan is not None:
            counts = audit["class_counts"]
            plan_shortage = {
                label: {"available": counts.get(label, 0), "required_pool": pool_quotas[label]}
                for label in labels
                if counts.get(label, 0) < pool_quotas[label]
            }
            if plan_shortage:
                raise RuntimeError(
                    "FLB achievable plan no longer feasible against live supply: "
                    f"{plan_shortage}. Regenerate and re-accept the plan."
                )
        elif not audit["feasible"]:
            missing = audit["shortages"]
            if is_pilot:
                zero_supply = {k: v for k, v in missing.items() if v["available"] == 0}
                if zero_supply:
                    raise RuntimeError(
                        f"FLB pilot feasibility check failed: classes with zero source candidates: "
                        f"{zero_supply}. No Ollama generation was started."
                    )
                raise RuntimeError(f"FLB pilot feasibility check failed: {missing}. No Ollama generation was started.")
            else:
                raise RuntimeError(
                    "FLB final feasibility check failed before generation. "
                    f"Available class counts={audit['class_counts']}; "
                    f"required quotas={quotas}; "
                    f"insufficient classes={missing}. "
                    "No Ollama generation was started. "
                    "Run the offline feasibility audit to see exact shortages."
                )

        try:
            candidates = self._candidate_pool(
                pool_quotas,
                include_external_gold=True,
                include_external_silver=True,
            )
        finally:
            self.repo.close()

        if len(candidates) < source_target:
            raise RuntimeError(
                f"Only {len(candidates)} balanced CUAD candidates were available; need at least {source_target}"
            )

        available_counts = {label: sum(1 for row in candidates if row["risk_label"] == label) for label in labels}
        missing_supply = {
            label: {"available": available_counts[label], "required": quotas[label]}
            for label in labels
            if available_counts[label] < quotas[label]
        }
        if missing_supply:
            raise RuntimeError(
                "FLB source feasibility check failed before generation. "
                f"Available class counts={available_counts}; "
                f"required quotas={quotas}; "
                f"insufficient classes={missing_supply}. "
                "No Ollama generation was started."
            )

        global_accepted = 0
        global_attempted = 0
        per_class_accepted: dict[str, int] = {label: 0 for label in labels}
        per_class_attempted: dict[str, int] = {label: 0 for label in labels}

        primary_pairs = self.silver.simplification_pairs(
            iter(candidates),
            len(candidates),
            alternate=False,
        )
        alternate_pairs = self.silver.simplification_pairs(
            iter(candidates),
            len(candidates),
            alternate=True,
        )

        primary = {pair.source_id: pair for pair in primary_pairs}
        alternate = {pair.source_id: pair for pair in alternate_pairs}

        for row in candidates:
            source_id = row["source_id"]
            if source_id not in primary:
                recovered = self.silver.regenerate_simplification(
                    row,
                    failure_feedback=["missing_primary_variant"],
                    attempt=0,
                )
                if recovered is not None:
                    primary[source_id] = recovered
            if source_id not in alternate:
                recovered = self.silver.regenerate_simplification(
                    row,
                    failure_feedback=["missing_alternate_variant"],
                    attempt=0,
                )
                if recovered is not None:
                    alternate[source_id] = recovered

        aligned = [row for row in candidates if row["source_id"] in primary and row["source_id"] in alternate]

        dropped_counts: dict[str, int] = {label: 0 for label in labels}
        for row in candidates:
            if row["source_id"] not in primary or row["source_id"] not in alternate:
                label = _canonical_label(row["risk_label"], labels)
                if label is not None:
                    dropped_counts[label] += 1
        if any(dropped_counts.values()):
            logger.warning(
                "FLB candidates still missing an independent variant after recovery: %s",
                dropped_counts,
            )

        if not aligned:
            raise RuntimeError("No independently generated FLB simplification pairs passed critical-token validation")

        variants: list[dict] = []
        for row in aligned:
            source_id = row["source_id"]
            primary_text = primary[source_id].target
            alternate_text = alternate[source_id].target
            variants.extend(
                [
                    {
                        "source_id": source_id,
                        "judge_id": f"{source_id}::primary",
                        "variant": "primary",
                        "source": row["text"],
                        "candidate": primary_text,
                        "reference": alternate_text,
                    },
                    {
                        "source_id": source_id,
                        "judge_id": f"{source_id}::alternate",
                        "variant": "alternate",
                        "source": row["text"],
                        "candidate": alternate_text,
                        "reference": primary_text,
                    },
                ]
            )

        automatic = rq1_example_quality(
            [item["source"] for item in variants],
            [item["candidate"] for item in variants],
            [item["reference"] for item in variants],
            bertscore_model=self.settings.bertscore_model,
        )

        judge_rows = [
            {
                "source_id": item["judge_id"],
                "source": item["source"],
                "candidate": item["candidate"],
            }
            for item in variants
        ]
        judged = self.silver.assess_simplifications(
            judge_rows,
            len(judge_rows),
        )
        risk_judged = self.silver.assess_risk_labels(
            aligned,
            len(aligned),
        )

        variants_by_source: dict[str, list[dict]] = {}
        for variant, metrics in zip(variants, automatic):
            evaluated = dict(variant)
            evaluated["metrics"] = metrics
            evaluated["assessment"] = judged.get(variant["judge_id"])
            variants_by_source.setdefault(
                variant["source_id"],
                [],
            ).append(evaluated)

        accepted: list[dict] = []
        accepted_counts = {label: 0 for label in labels}
        rejection_summary: dict[str, dict[str, int]] = {label: {} for label in labels}

        for row in aligned:
            source_id = row["source_id"]
            risk_label = _canonical_label(row["risk_label"], labels)
            if risk_label is None:
                raise RuntimeError(f"Candidate has an unconfigured risk label: {row['risk_label']!r}")

            global_attempted += 1
            per_class_attempted[risk_label] = per_class_attempted.get(risk_label, 0) + 1

            if accepted_counts[risk_label] >= quotas[risk_label]:
                continue

            risk_assessment = risk_judged.get(source_id)
            row_for_risk = dict(row)
            row_for_risk["risk_label"] = risk_label
            shared_reasons = _risk_judge_reasons(
                row_for_risk,
                risk_assessment,
                self.settings.flb_judge_min_score,
            )

            source_type = str(row.get("source_type") or "")
            provisional = bool(row.get("provisional", False))
            requires_risk_confirmation = provisional or source_type == "cuad_provisional_safe"

            if not requires_risk_confirmation:
                if risk_assessment is None:
                    logger.warning(
                        "FLB gold-labelled source has no advisory risk judge source_id=%s risk=%s",
                        source_id,
                        risk_label,
                    )
                else:
                    observed = _canonical_label(
                        risk_assessment.correct_label,
                        labels,
                    )
                    if observed != risk_label:
                        logger.warning(
                            "FLB advisory risk judge disagreed with "
                            "authoritative gold label source_id=%s "
                            "gold=%s judge=%s confidence=%s",
                            source_id,
                            risk_label,
                            risk_assessment.correct_label,
                            risk_assessment.confidence,
                        )

            passing_variants: list[dict] = []
            combined_reasons: list[str] = []

            for variant in variants_by_source.get(source_id, []):
                metrics = variant["metrics"]
                assessment = variant["assessment"]
                reasons = list(shared_reasons)

                if assessment is None:
                    reasons.append(f"{variant['variant']}:missing_simplification_judge")
                else:
                    if not assessment.faithful:
                        reasons.append(f"{variant['variant']}:not_faithful")
                    if assessment.semantic_accuracy < self.settings.flb_judge_min_score:
                        reasons.append(f"{variant['variant']}:semantic_accuracy={assessment.semantic_accuracy}")
                    if assessment.readability < self.settings.flb_judge_min_score:
                        reasons.append(f"{variant['variant']}:readability={assessment.readability}")

                if metrics["sari"] < self.settings.flb_min_sari:
                    reasons.append(f"{variant['variant']}:sari={metrics['sari']:.4f}")
                if metrics["bertscore_f1"] < self.settings.flb_min_bertscore:
                    reasons.append(f"{variant['variant']}:bertscore_f1={metrics['bertscore_f1']:.4f}")
                if metrics["fk_grade"] > self.settings.fk_acceptance_target:
                    reasons.append(f"{variant['variant']}:fk_grade={metrics['fk_grade']:.4f}")

                if reasons:
                    combined_reasons.extend(reasons)
                    continue

                selected = dict(variant)
                selected["selection_key"] = (
                    assessment.semantic_accuracy,
                    assessment.readability,
                    metrics["bertscore_f1"],
                    metrics["sari"],
                    -metrics["fk_grade"],
                )
                passing_variants.append(selected)

            if not passing_variants and not shared_reasons:
                for retry_attempt in range(
                    1,
                    self.settings.flb_quality_regeneration_retries + 1,
                ):
                    regenerated = self.silver.regenerate_simplification(
                        row,
                        failure_feedback=sorted(set(combined_reasons)),
                        attempt=retry_attempt,
                    )
                    if regenerated is None:
                        continue

                    retry_variant_name = f"regenerated_{retry_attempt}"
                    retry_judge_id = f"{source_id}::{retry_variant_name}"
                    retry_reference = alternate[source_id].target
                    retry_metrics = rq1_example_quality(
                        [row["text"]],
                        [regenerated.target],
                        [retry_reference],
                        bertscore_model=self.settings.bertscore_model,
                    )[0]
                    retry_assessment = self.silver.assess_simplifications(
                        [
                            {
                                "source_id": retry_judge_id,
                                "source": row["text"],
                                "candidate": regenerated.target,
                            }
                        ],
                        1,
                    ).get(retry_judge_id)

                    retry_reasons: list[str] = []
                    if retry_assessment is None:
                        retry_reasons.append(f"{retry_variant_name}:missing_simplification_judge")
                    else:
                        if not retry_assessment.faithful:
                            retry_reasons.append(f"{retry_variant_name}:not_faithful")
                        if retry_assessment.semantic_accuracy < self.settings.flb_judge_min_score:
                            retry_reasons.append(
                                f"{retry_variant_name}:semantic_accuracy={retry_assessment.semantic_accuracy}"
                            )
                        if retry_assessment.readability < self.settings.flb_judge_min_score:
                            retry_reasons.append(f"{retry_variant_name}:readability={retry_assessment.readability}")

                    if retry_metrics["sari"] < self.settings.flb_min_sari:
                        retry_reasons.append(f"{retry_variant_name}:sari={retry_metrics['sari']:.4f}")
                    if retry_metrics["bertscore_f1"] < self.settings.flb_min_bertscore:
                        retry_reasons.append(f"{retry_variant_name}:bertscore_f1={retry_metrics['bertscore_f1']:.4f}")
                    if retry_metrics["fk_grade"] > self.settings.fk_acceptance_target:
                        retry_reasons.append(f"{retry_variant_name}:fk_grade={retry_metrics['fk_grade']:.4f}")

                    if retry_reasons:
                        combined_reasons.extend(retry_reasons)
                        logger.warning(
                            "FLB quality regeneration failed source_id=%s risk=%s attempt=%d reasons=%s",
                            source_id,
                            risk_label,
                            retry_attempt,
                            retry_reasons,
                        )
                        continue

                    passing_variants.append(
                        {
                            "source_id": source_id,
                            "judge_id": retry_judge_id,
                            "variant": retry_variant_name,
                            "source": row["text"],
                            "candidate": regenerated.target,
                            "reference": retry_reference,
                            "metrics": retry_metrics,
                            "assessment": retry_assessment,
                            "selection_key": (
                                retry_assessment.semantic_accuracy,
                                retry_assessment.readability,
                                retry_metrics["bertscore_f1"],
                                retry_metrics["sari"],
                                -retry_metrics["fk_grade"],
                            ),
                        }
                    )
                    logger.info(
                        "FLB quality regeneration passed source_id=%s risk=%s attempt=%d",
                        source_id,
                        risk_label,
                        retry_attempt,
                    )
                    break

            if not passing_variants:
                for reason in sorted(set(combined_reasons)):
                    rejection_summary[risk_label][reason] = rejection_summary[risk_label].get(reason, 0) + 1

                logger.warning(
                    "FLB rejected source_id=%s risk=%s class_accepted=%d/%d global_accepted=%d/%d reasons=%s",
                    source_id,
                    risk_label,
                    per_class_accepted.get(risk_label, 0),
                    quotas[risk_label],
                    global_accepted,
                    source_target,
                    sorted(set(combined_reasons)),
                )
                continue

            selected = max(
                passing_variants,
                key=lambda item: item["selection_key"],
            )
            metrics = selected["metrics"]
            assessment = selected["assessment"]

            accepted_row = dict(row)
            accepted_row.update(
                {
                    "risk_label": risk_label,
                    "reference": selected["candidate"],
                    "alternate_reference": selected["reference"],
                    "selected_variant": selected["variant"],
                    "sari": metrics["sari"],
                    "bertscore_f1": metrics["bertscore_f1"],
                    "fk_grade": metrics["fk_grade"],
                    "judge": assessment,
                    "risk_judge": risk_assessment,
                }
            )
            accepted.append(accepted_row)
            accepted_counts[risk_label] += 1
            per_class_accepted[risk_label] = per_class_accepted.get(risk_label, 0) + 1
            global_accepted += 1

            logger.info(
                "FLB progress class=%s class_accepted=%d/%d "
                "global_accepted=%d/%d global_attempted=%d "
                "selected_variant=%s",
                risk_label,
                per_class_accepted[risk_label],
                quotas[risk_label],
                global_accepted,
                source_target,
                global_attempted,
                selected["variant"],
            )

            if len(accepted) >= source_target and all(accepted_counts[label] >= quotas[label] for label in labels):
                break

        incomplete = {
            label: {
                "accepted": accepted_counts[label],
                "required": quotas[label],
                "rejections": rejection_summary[label],
            }
            for label in labels
            if accepted_counts[label] < quotas[label]
        }
        if incomplete:
            raise RuntimeError(
                "FLB could not satisfy balanced quality quotas after "
                "evaluating both independently generated variants. "
                f"Accepted={accepted_counts}; required={quotas}; "
                f"incomplete={incomplete}. "
                "Configured thresholds were not lowered or bypassed."
            )

        synthetic = self.silver.synthetic_nli(iter(accepted), len(accepted) * 3)
        negatives: dict[str, dict[str, dict]] = {}
        for item in synthetic:
            if item["label"] in {"neutral", "contradiction"}:
                negatives.setdefault(item["source_id"], {})[item["label"]] = item

        # not every clause gets a neutral/contradiction label back on the first pass, but the supported/
        # unsupported pair design needs one for every accepted row, so retry just the missing ones directly
        missing_rows = [row for row in accepted if row["source_id"] not in negatives]
        retry_round = 0
        while missing_rows and retry_round < self.settings.flb_synthetic_nli_repair_retries:
            retry_round += 1
            logger.warning(
                "Retrying synthetic-NLI generation for %d clause(s) missing an "
                "unsupported companion (round %d): %s",
                len(missing_rows),
                retry_round,
                [row["source_id"] for row in missing_rows],
            )
            for row in missing_rows:
                retried = self.silver.synthetic_nli(iter([row]), 3, repair_round=retry_round)
                for item in retried:
                    if item["label"] in {"neutral", "contradiction"}:
                        negatives.setdefault(item["source_id"], {})[item["label"]] = item
            missing_rows = [row for row in accepted if row["source_id"] not in negatives]

        records: list[FLBRecord] = []
        for row_index, row in enumerate(accepted):
            judge = row["judge"]
            risk_judge = row["risk_judge"]

            src_hash = hashlib.sha256(row["text"].encode("utf-8", errors="ignore")).hexdigest()[:16]
            cand_text = row["reference"]
            cand_hash = hashlib.sha256(cand_text.encode("utf-8", errors="ignore")).hexdigest()[:16]

            revisions = {}
            try:
                revisions = self.settings.dataset_revisions
            except Exception:
                pass
            source_revision = revisions.get(row.get("dataset", ""), None)

            common = dict(
                source_id=row["source_id"],
                document_id=row["document_id"],
                original_clause=row["text"],
                reference_simplification=row["reference"],
                alternate_reference=row["alternate_reference"],
                risk_label=row["risk_label"],
                source_dataset=row["dataset"],
                dataset_split="test",
                source_revision=source_revision,
                source_split=row.get("split", "test"),
                source_document_id=row["document_id"],
                original_source_label=row.get("source_label", row.get("question", "")),
                provisional_risk_label=row["risk_label"],
                judge_risk_label=(risk_judge.correct_label if risk_judge is not None else None),
                judge_risk_confidence=(risk_judge.confidence if risk_judge is not None else None),
                source_text_hash=src_hash,
                candidate_text_hash=cand_hash,
                leakage_partition_method="cuad_test_split",
                accepted=True,
                rejection_reasons=None,
                sari=row["sari"],
                bertscore_f1=row["bertscore_f1"],
                fk_grade=row["fk_grade"],
                judge_semantic_accuracy=judge.semantic_accuracy,
                judge_readability=judge.readability,
                judge_rationale=judge.rationale,
                judge_risk_rationale=(risk_judge.rationale if risk_judge is not None else None),
                generation_model=self.silver.model,
                quality_filter_passed=True,
                is_pilot=is_pilot,
            )
            records.append(
                FLBRecord(
                    record_id=f"{row['source_id']}::supported",
                    hypothesis=row["reference"],
                    supported=1,
                    nli_label="entailment",
                    **common,
                )
            )
            negative_options = negatives.get(row["source_id"], {})
            preferred = "neutral" if row_index % 2 == 0 else "contradiction"
            negative = negative_options.get(preferred) or next(iter(negative_options.values()), None)
            if negative is None:
                raise RuntimeError(f"No controlled unsupported NLI item was generated for {row['source_id']}")
            records.append(
                FLBRecord(
                    record_id=f"{row['source_id']}::unsupported",
                    hypothesis=negative["hypothesis"],
                    supported=0,
                    nli_label=negative["label"],
                    **common,
                )
            )

        expected_records = source_target * 2
        if len(records) != expected_records:
            raise RuntimeError(
                f"FLB has {len(records)} verifier rows after NLI generation; "
                f"expected exactly {expected_records} from {source_target} unique clauses"
            )

        if is_pilot:
            logger.warning(
                "FLB build complete: %s — %d records. DO NOT USE AS FINAL THESIS RESULTS.",
                self.PILOT_LABEL,
                len(records),
            )

        return records

    def _candidate_pool(
        self,
        target_counts: dict[str, int],
        *,
        include_external_gold: bool = False,
        include_external_silver: bool = False,
    ) -> list[dict]:
        """Build a leakage-safe provisional pool from CUAD only - CFPB is excluded from FLB entirely."""

        labels = list(target_counts)
        counts = {label: 0 for label in labels}
        candidates: list[dict] = []
        seen_source_ids: set[str] = set()

        def add_candidate(candidate: dict) -> bool:
            risk_label = str(candidate["risk_label"])
            source_id = str(candidate["source_id"])

            if risk_label not in target_counts:
                return False

            if counts[risk_label] >= target_counts[risk_label]:
                return False

            if source_id in seen_source_ids:
                logger.warning(
                    "Skipping duplicate FLB source clause source_id=%s",
                    source_id,
                )
                return False

            text_value = normalise_text(str(candidate.get("text") or ""))
            if not text_value or len(text_value.split()) < 8:
                return False

            max_words = int(getattr(self.settings, "simplification_source_max_words", 500))
            words = text_value.split()
            if len(words) > max_words:
                text_value = " ".join(words[:max_words])

            candidate["text"] = text_value
            seen_source_ids.add(source_id)
            candidates.append(candidate)
            counts[risk_label] += 1
            return True

        silver_path = self.settings.resolve(self.settings.flb_external_silver_path)

        if include_external_silver and self.settings.flb_external_silver_enabled and silver_path.exists():
            silver_payload = yaml.safe_load(silver_path.read_text(encoding="utf-8")) or {}

            for silver_row in silver_payload.get("records") or []:
                if not isinstance(silver_row, dict):
                    continue

                source_id = str(silver_row.get("source_id") or "").strip()

                risk_label = _canonical_label(
                    str(silver_row.get("risk_label") or ""),
                    tuple(target_counts),
                )

                text_value = normalise_text(str(silver_row.get("text") or ""))

                if risk_label is None:
                    continue

                if not source_id or len(text_value.split()) < 8:
                    continue

                if not bool(silver_row.get("excluded_from_training")):
                    continue

                add_candidate(
                    {
                        "source_id": source_id,
                        "document_id": str(silver_row.get("document_id") or source_id),
                        "text": text_value,
                        "question": str(silver_row.get("question") or risk_label),
                        "risk_label": risk_label,
                        "dataset": str(silver_row.get("dataset") or "external_silver"),
                        "split": str(silver_row.get("split") or "external_test"),
                        "source_type": str(silver_row.get("source_type") or "external_silver"),
                        "source_label": str(silver_row.get("source_label") or risk_label),
                        "source_url": str(silver_row.get("source_url") or ""),
                        "source_section": str(silver_row.get("source_section") or ""),
                        "source_revision": str(silver_row.get("source_revision") or ""),
                        "human_reviewed": bool(
                            silver_row.get(
                                "human_reviewed",
                                False,
                            )
                        ),
                        "excluded_from_training": True,
                        "provisional": True,
                    }
                )

        external_path = self.settings.resolve(self.settings.flb_external_gold_path)

        if include_external_gold and self.settings.flb_external_gold_enabled and external_path.exists():
            external_payload = yaml.safe_load(external_path.read_text(encoding="utf-8")) or {}

            for external_row in external_payload.get("records") or []:
                if not isinstance(external_row, dict):
                    continue

                source_id = str(external_row.get("source_id") or "").strip()
                risk_label = _canonical_label(
                    str(external_row.get("risk_label") or ""),
                    tuple(target_counts),
                )
                text_value = normalise_text(str(external_row.get("text") or ""))

                if risk_label is None or risk_label not in target_counts:
                    continue

                if not source_id or len(text_value.split()) < 8:
                    continue

                if not bool(external_row.get("human_reviewed")):
                    continue

                if not bool(external_row.get("excluded_from_training")):
                    continue

                add_candidate(
                    {
                        "source_id": source_id,
                        "document_id": str(external_row.get("document_id") or source_id),
                        "text": text_value,
                        "question": str(external_row.get("question") or risk_label),
                        "risk_label": risk_label,
                        "dataset": str(external_row.get("dataset") or "external_gold"),
                        "split": str(external_row.get("split") or "external_test"),
                        "source_type": str(external_row.get("source_type") or "human_adjudicated_external"),
                        "source_label": str(external_row.get("source_label") or risk_label),
                        "source_url": str(external_row.get("source_url") or ""),
                        "source_section": str(external_row.get("source_section") or ""),
                        "source_revision": str(external_row.get("source_revision") or ""),
                        "human_reviewed": True,
                        "excluded_from_training": True,
                        "provisional": False,
                    }
                )

        safe_config_path = self.settings.resolve(self.settings.flb_safe_candidates_path)
        safe_config = yaml.safe_load(safe_config_path.read_text(encoding="utf-8"))
        safe_categories = {_normalise_label(value) for value in safe_config.get("provisional_safe_categories", [])}
        safe_label = self.settings.safe_risk_label

        cuad_rows = self.repo.stream_partitioned(
            self.settings.cuad_dataset,
            requested_split="test",
            config=self.settings.cuad_config,
            key_fields=("file_name", "title", "document_id", "id"),
            shuffle=False,
        )

        for row in cuad_rows:
            raw_label = str(row.get("label") or row.get("question") or "").strip()
            mapped = self.taxonomy.map_question(raw_label)
            risk_label: str | None = None
            source_type: str | None = None

            if mapped is not None and counts.get(mapped, 0) < target_counts.get(mapped, 0):
                risk_label = mapped
                source_type = "cuad_exact_risk"
            elif _normalise_label(raw_label) in safe_categories and counts.get(safe_label, 0) < target_counts.get(
                safe_label, 0
            ):
                risk_label = safe_label
                source_type = "cuad_provisional_safe"

            if risk_label is None:
                continue

            text_value = normalise_text(str(row.get("clause") or row.get("text") or ""))
            if not text_value or len(text_value.split()) < 8:
                continue

            clause_hash = hashlib.sha256(text_value.encode("utf-8", errors="ignore")).hexdigest()[:12]

            source_id = "::".join(
                [
                    "cuad",
                    str(row.get("file_name") or "unknown-file"),
                    str(row.get("class_id") or raw_label or "unknown-class"),
                    str(row.get("start_at") if row.get("start_at") is not None else "unknown-start"),
                    str(row.get("end_at") if row.get("end_at") is not None else "unknown-end"),
                    clause_hash,
                ]
            )

            add_candidate(
                {
                    "source_id": source_id,
                    "document_id": str(row.get("file_name") or row.get("title") or source_id),
                    "text": text_value,
                    "question": raw_label,
                    "risk_label": risk_label,
                    "dataset": self.settings.cuad_dataset,
                    "split": "test",
                    "source_type": source_type,
                    "source_label": raw_label,
                    "provisional": risk_label == safe_label,
                }
            )

        logger.info(
            "FLB candidate pool (CUAD only, CFPB excluded): total=%d class_counts=%s",
            len(candidates),
            counts,
        )

        return candidates

    @staticmethod
    def export(records: list[FLBRecord], path: Path) -> None:
        """Explicit release export only; normal evaluation does not read from it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(asdict(record), ensure_ascii=False) for record in records), encoding="utf-8"
        )

    @staticmethod
    def export_review_template(
        records: list[FLBRecord],
        path: Path,
        sample_size: int,
        seed: int = 42,
        *,
        reviewers: int = 3,
        blinded: bool = True,
        as_xlsx: bool = False,
        simple: bool = True,
    ) -> None:
        """Export a human-review sheet - blinded hides the system's own predictions so reviewers can't anchor on them, and simple=True uses plain-English headers for non-technical reviewers."""
        import random

        rng = random.Random(seed)
        sample = rng.sample(records, min(sample_size, len(records)))
        path.parent.mkdir(parents=True, exist_ok=True)

        if simple:
            shown_fields = ["record_id", "original_clause", "hypothesis", "provisional_evidence_text"]
            shown_headers = {
                "record_id": "ID",
                "original_clause": "Original Contract Clause",
                "hypothesis": "Plain-English Version",
                "provisional_evidence_text": "Supporting Regulation Text (if any)",
            }
            per_reviewer_fields = [
                "risk_category",
                "meaning_preserved",
                "missing_or_changed",
                "missing_or_changed_details",
                "comments",
            ]
            reviewer_headers = {
                "risk_category": "Risk Category (pick one)",
                "meaning_preserved": "Does the Plain-English Version Mean the Same Thing? (Yes / No)",
                "missing_or_changed": "Is Anything Important Missing, Wrong, or Changed? (Yes / No)",
                "missing_or_changed_details": "If Yes, what's missing or wrong? (leave blank if No)",
                "comments": "Comments (optional)",
            }
            adjudicated_fields = ["adjudicated_risk_label", "adjudicated_notes"]
            adjudicated_headers = {
                "adjudicated_risk_label": "Final Agreed Risk Category",
                "adjudicated_notes": "Final Notes",
            }
        else:
            shown_fields = [
                "record_id",
                "original_clause",
                "hypothesis",
                "provisional_evidence_source",
                "provisional_evidence_url",
                "provisional_evidence_text",
                "evidence_candidates_json",
            ]
            shown_headers = {}
            per_reviewer_fields = [
                "risk_label",
                "risk_score",
                "source_supported",
                "evidence_supported",
                "chunk_id",
                "meaning_preserved",
                "readability_score",
                "misleading_content",
                "critical_fact_changed",
                "expertise_category",
                "review_timestamp",
                "comments",
            ]
            reviewer_headers = {}
            adjudicated_fields = [
                "adjudicated_risk_label",
                "adjudicated_risk_score",
                "adjudicated_source_supported",
                "adjudicated_evidence_supported",
                "adjudicated_chunk_id",
                "adjudication_notes",
            ]
            adjudicated_headers = {}

        if not blinded:
            shown_fields[1:1] = [
                "risk_label",
                "system_supported",
                "system_source_supported",
                "system_evidence_supported",
                "nli_label",
            ]

        reviewer_columns: list[str] = []
        column_headers: dict[str, str] = dict(shown_headers)
        for field, header in adjudicated_headers.items():
            column_headers[field] = header
        for reviewer_index in range(1, reviewers + 1):
            for field in per_reviewer_fields:
                column = f"reviewer_{reviewer_index}_{field}"
                reviewer_columns.append(column)
                if field in reviewer_headers:
                    column_headers[column] = f"Reviewer {reviewer_index}: {reviewer_headers[field]}"

        fields = shown_fields + reviewer_columns + adjudicated_fields
        if not simple:
            fields = fields + ["ground_truth_chunk_id"]

        rows = []
        for record in sample:
            row = {
                "record_id": record.record_id,
                "original_clause": record.original_clause,
                "hypothesis": record.hypothesis,
                "provisional_evidence_text": record.provisional_evidence_text or "",
            }
            if not simple:
                row.update(
                    {
                        "provisional_evidence_source": record.provisional_evidence_source or "",
                        "provisional_evidence_url": record.provisional_evidence_url or "",
                        "evidence_candidates_json": record.evidence_candidates_json or "",
                        "ground_truth_chunk_id": record.ground_truth_chunk_id or "",
                    }
                )
            if not blinded:
                row.update(
                    {
                        "risk_label": record.risk_label,
                        "system_supported": record.supported,
                        "system_source_supported": record.supported,
                        "system_evidence_supported": (
                            record.evidence_supported if record.evidence_supported is not None else ""
                        ),
                        "nli_label": record.nli_label or "",
                    }
                )
            for column in reviewer_columns + adjudicated_fields:
                row[column] = ""
            rows.append(row)

        if as_xlsx:
            import pandas as pd

            xlsx_fields = [f for f in fields if f != "evidence_candidates_json"]
            frame = pd.DataFrame(rows, columns=xlsx_fields)
            frame = frame.rename(columns=column_headers)
            if simple:
                instructions = pd.DataFrame(
                    {
                        "How to review each clause": [
                            "Each row is one contract clause and its plain-English rewrite.",
                            "Read the 'Original Contract Clause' and the 'Plain-English Version' next to it.",
                            "Fill in your own columns only (they say 'Reviewer 1', 'Reviewer 2', etc. -- "
                            "use the one assigned to you).",
                            "Risk Category, Meaning Preserved, and Missing/Wrong/Changed each have a "
                            "dropdown -- click the cell and choose from the list, don't type your own text.",
                            "Meaning Preserved: choose Yes if the plain-English version means the same "
                            "as the original, No if it changes the meaning.",
                            "Missing/Wrong/Changed: choose No if everything important carried over "
                            "correctly. If something important (a number, date, fee, deadline, or "
                            "condition) is missing or wrong, choose Yes and briefly say what in the "
                            "'If Yes, what's missing or wrong?' cell right next to it.",
                            "Comments: optional, anything else worth noting.",
                            "Please do not look at another reviewer's answers before finishing your own.",
                            "Leave the 'Final Agreed' columns blank -- those are filled in afterwards, "
                            "once all reviewers are done.",
                        ]
                    }
                )
                with pd.ExcelWriter(path, engine="openpyxl") as writer:
                    instructions.to_excel(writer, sheet_name="Instructions", index=False)
                    frame.to_excel(writer, sheet_name="Review", index=False)
                    _format_review_sheet(writer, "Review", frame)
                    _format_instructions_sheet(writer, "Instructions")
            else:
                frame.to_excel(path, index=False, engine="openpyxl")
        else:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)


RISK_CATEGORY_OPTIONS = (
    "Auto-Renewal",
    "Hidden Fee",
    "Liability Waiver",
    "Data Sharing",
    "Penalty Clause",
    "Safe",
)
YES_NO_OPTIONS = ("Yes", "No")


def _format_review_sheet(writer, sheet_name: str, frame) -> None:
    """Make the reviewer-facing sheet readable and hard to fill in wrong - sized columns, dropdown validation, and a highlight on required cells still blank."""
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    sheet = writer.sheets[sheet_name]
    wide_wrap_columns = {
        "ID": 12,
        "Original Contract Clause": 55,
        "Plain-English Version": 55,
        "Supporting Regulation Text (if any)": 45,
    }
    narrow_columns_by_keyword = [
        ("Risk Category", 22),
        ("Mean the Same Thing", 14),
        ("Missing, Wrong, or Changed", 14),
        ("what's missing or wrong", 35),
        ("Comments", 30),
        ("Final Agreed", 22),
        ("Final Notes", 30),
    ]
    for index, column_name in enumerate(frame.columns, start=1):
        letter = sheet.cell(row=1, column=index).column_letter
        width = wide_wrap_columns.get(column_name)
        if width is None:
            width = next((w for keyword, w in narrow_columns_by_keyword if keyword in column_name), 22)
        sheet.column_dimensions[letter].width = width
        for row in range(1, len(frame) + 2):
            sheet.cell(row=row, column=index).alignment = Alignment(wrap_text=True, vertical="top")

    for index in range(1, len(frame.columns) + 1):
        cell = sheet.cell(row=1, column=index)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    sheet.row_dimensions[1].height = 45
    sheet.freeze_panes = "A2"

    last_row = len(frame) + 1
    required_yellow = PatternFill(start_color="FFF9C4", end_color="FFF9C4", fill_type="solid")
    risk_validation = DataValidation(
        type="list", formula1='"' + ",".join(RISK_CATEGORY_OPTIONS) + '"', allow_blank=True, showErrorMessage=True
    )
    risk_validation.error = "Please choose one option from the dropdown list."
    risk_validation.errorTitle = "Invalid risk category"
    sheet.add_data_validation(risk_validation)

    yes_no_validation = DataValidation(
        type="list", formula1='"' + ",".join(YES_NO_OPTIONS) + '"', allow_blank=True, showErrorMessage=True
    )
    yes_no_validation.error = "Please choose Yes or No from the dropdown list."
    yes_no_validation.errorTitle = "Invalid answer"
    sheet.add_data_validation(yes_no_validation)

    for index, column_name in enumerate(frame.columns, start=1):
        letter = sheet.cell(row=1, column=index).column_letter
        cell_range = f"{letter}2:{letter}{last_row}"
        is_required = False
        if "Risk Category" in column_name and "Final" not in column_name:
            risk_validation.add(cell_range)
            is_required = True
        elif ("Mean the Same Thing" in column_name) or (
            "Missing, Wrong, or Changed" in column_name and "what's missing" not in column_name
        ):
            yes_no_validation.add(cell_range)
            is_required = True
        if is_required:
            sheet.conditional_formatting.add(
                cell_range,
                FormulaRule(formula=[f"{letter}2=\"\""], fill=required_yellow),
            )


def _format_instructions_sheet(writer, sheet_name: str) -> None:
    from openpyxl.styles import Alignment, Font

    sheet = writer.sheets[sheet_name]
    sheet.column_dimensions["A"].width = 110
    for row in sheet.iter_rows():
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    header_cell = sheet.cell(row=1, column=1)
    header_cell.font = Font(bold=True)


def _balanced_quotas(labels: list[str], total: int) -> dict[str, int]:
    if total < len(labels):
        raise ValueError(f"FLB size must be at least {len(labels)} to represent every risk class")
    base, remainder = divmod(total, len(labels))
    return {label: base + (1 if index < remainder else 0) for index, label in enumerate(labels)}
