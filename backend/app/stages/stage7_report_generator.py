"""Stage 7 - puts everything together into the final report, plus a
provenance manifest (model/adapter versions) so a run is traceable."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.core.provenance import adapter_sha256, runtime_manifest, sha256_path
from backend.app.core.schemas import ClauseResult, DocumentMetrics, DocumentResult, ModelProvenance
from backend.app.services.review_routing import route_clauses, routing_summary


class Stage7ReportGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build(
        self,
        filename: str,
        clauses: list[ClauseResult],
        *,
        total_runtime_ms: float,
        regulatory_sources: list[str],
        warnings: list[str],
        document_path: Path | None = None,
        regulatory_corpus_sha256: str | None = None,
        regulatory_source_manifests: list[dict] | None = None,
    ) -> DocumentResult:
        decisions = route_clauses(clauses, self.settings)
        for clause, decision in zip(clauses, decisions):
            clause.routed_for_review = decision.routed_for_review
            clause.routing_reasons = decision.reasons
            clause.routing_policy = decision.policy
        routing = routing_summary(decisions, self.settings)

        source = [item.source_faithfulness for clause in clauses for item in clause.faithfulness]
        regulatory = [
            item.regulatory_support
            for clause in clauses
            for item in clause.faithfulness
            if item.regulatory_support is not None
        ]
        primary = [
            item.faithfulness_score
            for clause in clauses
            for item in clause.faithfulness
            if item.faithfulness_score is not None
        ]
        risk_regulatory = [item.regulatory_support for clause in clauses for item in clause.risk_explanation_support]
        unsupported = [item for clause in clauses for item in clause.faithfulness if item.unsupported]
        attributed = [item for clause in clauses for item in clause.faithfulness if item.attribution_chunk_id]
        all_sentences = [item for clause in clauses for item in clause.faithfulness]
        retrieved_clause_count = sum(bool(clause.evidence) for clause in clauses)
        grades = [clause.readability_grade for clause in clauses if clause.readability_grade is not None]
        semantic_flags = [clause.semantic_target_met for clause in clauses if clause.semantic_target_met is not None]
        unsupported_flag_rate = round(len(unsupported) / len(all_sentences), 6) if all_sentences else None
        supported_attribution_coverage = round(len(attributed) / len(all_sentences), 6) if all_sentences else None
        metrics = DocumentMetrics(
            clause_count=len(clauses),
            sentence_count=len(all_sentences),
            risky_clause_count=sum(clause.risk_label not in {"Safe", "Needs Review"} for clause in clauses),
            needs_review_count=sum(clause.risk_label == "Needs Review" for clause in clauses),
            simplification_review_count=sum(clause.simplification_status == "needs_review" for clause in clauses),
            unsupported_sentence_count=len(unsupported),
            mean_source_faithfulness=_mean(source),
            mean_regulatory_support=_mean(regulatory),
            mean_primary_faithfulness=_mean(primary),
            mean_risk_regulatory_support=_mean(risk_regulatory),
            unsupported_flag_rate=unsupported_flag_rate,
            retrieval_coverage=(round(retrieved_clause_count / len(clauses), 6) if clauses else None),
            supported_attribution_coverage=supported_attribution_coverage,
            mean_fk_grade=_mean(grades),
            pct_fk_le_8=(round(sum(value <= 8 for value in grades) / len(grades), 6) if grades else None),
            pct_fk_le_9=(round(sum(value <= 9 for value in grades) / len(grades), 6) if grades else None),
            pct_semantic_target_met=(
                round(sum(bool(value) for value in semantic_flags) / len(semantic_flags), 6) if semantic_flags else None
            ),
            routing_policy=routing["routing_policy"],
            routed_clause_count=routing["routed_clause_count"],
            routing_reason_counts=routing["reason_counts"],
        )

        source_manifests = regulatory_source_manifests or []
        document_hash = sha256_path(document_path)
        simplifier_hash = adapter_sha256(self.settings, self.settings.simplifier_adapter)
        risk_hash = adapter_sha256(self.settings, self.settings.risk_adapter)
        verifier_hash = adapter_sha256(self.settings, self.settings.verifier_adapter)
        fingerprint_payload = {
            "simplifier": self.settings.simplifier_model,
            "domain_warmup_adapter": self.settings.domain_warmup_adapter,
            "simplifier_adapter": self.settings.simplifier_adapter,
            "simplifier_adapter_sha256": simplifier_hash,
            "embedding": self.settings.embedding_model,
            "reranker": self.settings.reranker_model,
            "risk_model": self.settings.risk_model or self.settings.simplifier_model,
            "risk_adapter": self.settings.risk_adapter,
            "risk_adapter_sha256": risk_hash,
            "verifier": self.settings.verifier_model,
            "verifier_adapter": self.settings.verifier_adapter,
            "verifier_adapter_sha256": verifier_hash,
            "tau": self.settings.faithfulness_tau,
            "attribution_tau": self.settings.attribution_tau,
            "primary_premise": self.settings.faithfulness_primary_premise,
            "chunk": self.settings.chunk_size_tokens,
            "overlap": self.settings.chunk_overlap_tokens,
            "top_k": self.settings.top_k_retrieval,
            "top_j": self.settings.top_j_reranked,
            "query_mode": self.settings.retrieval_query_mode,
            "document_sha256": document_hash,
            "regulatory_corpus_sha256": regulatory_corpus_sha256,
            "dataset_revisions": self.settings.dataset_revisions,
            "model_revisions": self.settings.model_revisions,
        }
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        models = ModelProvenance(
            simplifier_model=self.settings.simplifier_model,
            simplifier_adapter=self.settings.simplifier_adapter,
            simplifier_adapter_sha256=simplifier_hash,
            embedding_model=self.settings.embedding_model,
            embedding_dimensions=self.settings.embedding_dimensions,
            reranker_model=self.settings.reranker_model,
            risk_model=self.settings.risk_model or self.settings.simplifier_model,
            risk_adapter=self.settings.risk_adapter,
            risk_adapter_sha256=risk_hash,
            verifier_model=self.settings.verifier_model,
            verifier_adapter=self.settings.verifier_adapter,
            verifier_adapter_sha256=verifier_hash,
            faithfulness_tau=self.settings.faithfulness_tau,
            attribution_tau=self.settings.attribution_tau,
            faithfulness_primary_premise=self.settings.faithfulness_primary_premise,
            document_sha256=document_hash,
            regulatory_corpus_sha256=regulatory_corpus_sha256,
            regulatory_source_manifests=source_manifests,
            runtime_manifest=runtime_manifest(self.settings),
            config_fingerprint=fingerprint,
        )
        warnings = list(warnings)
        if any(clause.simplification_status == "needs_review" for clause in clauses):
            warnings.append(
                "One or more simplifications did not meet both FK and semantic-preservation gates and require review."
            )
        warnings.append(
            "The displayed unsupported flag rate is an operational verifier output, not a true hallucination rate "
            "unless compared with independently human-adjudicated labels."
        )
        result = DocumentResult(
            filename=filename,
            clauses=clauses,
            metrics=metrics,
            models=models,
            regulatory_sources=regulatory_sources,
            total_runtime_ms=round(total_runtime_ms, 3),
            warnings=_dedupe(warnings),
        )
        if self.settings.persist_reports:
            output = self.settings.resolve(self.settings.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            (output / f"{Path(filename).stem[:80]}_finlingo_report.json").write_text(
                result.model_dump_json(indent=2), encoding="utf-8"
            )
        return result


def _mean(values):
    return round(sum(values) / len(values), 6) if values else None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output
