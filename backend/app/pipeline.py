"""Runs a document through the 7 stages in order; _for_each lets Ollama stages batch clauses concurrently
while local model stages stay single-threaded (not safe to hit a loaded HF model from multiple threads)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings, get_settings
from backend.app.core.schemas import ClauseResult, DocumentResult
from backend.app.services.in_memory_retriever import InMemoryRetriever
from backend.app.services.model_registry import ModelRegistry
from backend.app.stages.stage1_document_parser import Stage1DocumentParser
from backend.app.stages.stage2_simplification import Stage2Simplification
from backend.app.stages.stage3_retrieval import Stage3Retrieval
from backend.app.stages.stage4_reranking import Stage4Reranking
from backend.app.stages.stage5_risk_classifier import Stage5RiskClassifier
from backend.app.stages.stage6_faithfulness import Stage6Faithfulness
from backend.app.stages.stage7_report_generator import Stage7ReportGenerator

ProgressCallback = Callable[[dict[str, Any]], None]


class FinLingoPipeline:
    """Seven-stage executable pipeline with live sources and RAM-only indexes."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.registry = ModelRegistry(self.settings)
        self.retriever = InMemoryRetriever(self.settings, self.registry)
        self.stage1 = Stage1DocumentParser(self.settings)
        self.stage2 = Stage2Simplification(self.settings, self.registry)
        self.stage3 = Stage3Retrieval(self.settings, self.retriever)
        self.stage4 = Stage4Reranking(self.settings, self.registry)
        self.stage5 = Stage5RiskClassifier(self.settings, self.registry)
        self.stage6 = Stage6Faithfulness(self.settings, self.registry)
        self.stage7 = Stage7ReportGenerator(self.settings)

    def _for_each(
        self,
        stage: str,
        work: list[dict[str, Any]],
        fn: Callable[[dict[str, Any]], None],
        *,
        concurrent: bool,
        progress_callback: ProgressCallback | None,
    ) -> None:
        """Run fn(item) for every item in work, optionally in parallel - only used for stateless Ollama calls, never for a loaded local model."""
        total = len(work)
        if not concurrent or self.settings.pipeline_ollama_concurrency <= 1 or total <= 1:
            for index, item in enumerate(work, 1):
                fn(item)
                self._progress(progress_callback, stage, index, total, item["clause"].clause_id)
            return

        completed = 0
        lock = threading.Lock()

        def _run(item: dict[str, Any]) -> None:
            nonlocal completed
            fn(item)
            with lock:
                completed += 1
                current = completed
            self._progress(progress_callback, stage, current, total, item["clause"].clause_id)

        with ThreadPoolExecutor(max_workers=min(self.settings.pipeline_ollama_concurrency, total)) as pool:
            list(pool.map(_run, work))

    def run(
        self,
        file_path: Path,
        *,
        retrieval_method: str = "bm25",
        # probe showed the reranker collapsed BM25 recall from 0.77 to as low as 0.10 on this corpus; off by default.
        use_reranker: bool = False,
        use_verifier: bool = True,
        simplification_only: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> DocumentResult:
        overall = time.perf_counter()

        self._notify(progress_callback, "stage_start", stage="S1", label="Document parsing")
        started = time.perf_counter()
        parsed = self.stage1.parse(file_path)
        s1_ms = _ms(started)
        self._notify(progress_callback, "stage_complete", stage="S1", elapsed_ms=s1_ms, clauses=len(parsed))

        work: list[dict[str, Any]] = [
            {"clause": clause, "timings": {"S1": s1_ms / max(len(parsed), 1)}} for clause in parsed
        ]

        self._notify(progress_callback, "stage_start", stage="S2", label="QLoRA simplification")

        def _run_s2(item: dict[str, Any]) -> None:
            started = time.perf_counter()
            item["simplification"] = self.stage2.simplify(item["clause"].text)
            item["generated_simplified"] = item["simplification"].text
            item["simplification_suppressed"] = bool(
                self.settings.suppress_unaccepted_simplification and not item["simplification"].accepted
            )
            item["simplified"] = (
                item["clause"].text if item["simplification_suppressed"] else item["generated_simplified"]
            )
            item["grade"] = item["simplification"].fk_grade
            item["timings"]["S2"] = _ms(started)

        self._for_each(
            "S2",
            work,
            _run_s2,
            concurrent=self.settings.backend_for("S2 simplifier") == "ollama",
            progress_callback=progress_callback,
        )
        self._notify(progress_callback, "stage_complete", stage="S2")

        if self.settings.release_causal_models_between_stages:
            self.registry.release_causal("S2 simplifier")

        if simplification_only:
            results = [self._simplification_only_result(item) for item in work]
            self._notify(progress_callback, "stage_start", stage="S7", label="Report generation")
            report = self.stage7.build(
                file_path.name,
                results,
                total_runtime_ms=(time.perf_counter() - overall) * 1000,
                regulatory_sources=[],
                warnings=["Simplification-only ablation: S3-S6 were intentionally disabled."],
                document_path=file_path,
                regulatory_corpus_sha256=None,
                regulatory_source_manifests=[],
            )
            self._notify(progress_callback, "stage_complete", stage="S7")
            return report

        self._notify(progress_callback, "stage_start", stage="S3", label=f"{retrieval_method.upper()} retrieval")
        for index, item in enumerate(work, 1):
            started = time.perf_counter()
            item["retrieval_query"] = self.stage3.build_query(item["clause"].text, item["simplified"])
            item["retrieved"] = self.stage3.retrieve(
                item["simplified"], retrieval_method, original_clause=item["clause"].text
            )
            item["timings"]["S3"] = _ms(started)
            self._progress(progress_callback, "S3", index, len(work), item["clause"].clause_id)
        self._notify(progress_callback, "stage_complete", stage="S3")

        self._notify(progress_callback, "stage_start", stage="S4", label="Cross-encoder reranking")
        for index, item in enumerate(work, 1):
            started = time.perf_counter()
            if use_reranker:
                item["evidence"] = self.stage4.rerank(item["retrieval_query"], item["retrieved"])
            else:
                item["evidence"] = item["retrieved"][: self.settings.top_j_reranked]
            item["timings"]["S4"] = _ms(started) if use_reranker else 0.0
            self._progress(progress_callback, "S4", index, len(work), item["clause"].clause_id)
        self._notify(progress_callback, "stage_complete", stage="S4", disabled=not use_reranker)

        self._notify(progress_callback, "stage_start", stage="S5", label="Six-class risk classification")

        def _run_s5(item: dict[str, Any]) -> None:
            started = time.perf_counter()
            item["risk"] = self.stage5.classify(item["clause"].text, item["evidence"])
            item["timings"]["S5"] = _ms(started)

        s5_concurrent = (
            self.settings.risk_classifier_backend == "prompted"
            and self.settings.backend_for("S5 risk classifier") == "ollama"
        )
        self._for_each("S5", work, _run_s5, concurrent=s5_concurrent, progress_callback=progress_callback)
        self._notify(progress_callback, "stage_complete", stage="S5")

        if self.settings.release_causal_models_between_stages:
            self.registry.release_causal("S5 risk classifier")

        self._notify(progress_callback, "stage_start", stage="S6", label="Sentence-level NLI verification")
        for index, item in enumerate(work, 1):
            started = time.perf_counter()
            item["faith"] = (
                self.stage6.verify(item["clause"].text, item["simplified"], item["evidence"]) if use_verifier else []
            )
            item["risk_support"] = (
                self.stage6.verify_regulatory_text(item["risk"].explanation, item["evidence"]) if use_verifier else []
            )
            item["timings"]["S6"] = _ms(started) if use_verifier else 0.0
            self._progress(progress_callback, "S6", index, len(work), item["clause"].clause_id)
        self._notify(progress_callback, "stage_complete", stage="S6", disabled=not use_verifier)

        results = [self._complete_result(item) for item in work]
        self._notify(progress_callback, "stage_start", stage="S7", label="Report generation")
        report = self.stage7.build(
            file_path.name,
            results,
            total_runtime_ms=(time.perf_counter() - overall) * 1000,
            regulatory_sources=self.retriever.sources,
            warnings=self.retriever.warnings,
            document_path=file_path,
            regulatory_corpus_sha256=self.retriever.corpus_sha256,
            regulatory_source_manifests=self.retriever.source_manifests,
        )
        self._notify(progress_callback, "stage_complete", stage="S7")
        return report

    def _complete_result(self, item: dict[str, Any]) -> ClauseResult:
        clause = item["clause"]
        simplification = item["simplification"]
        risk = item["risk"]
        faith = item["faith"]
        source_scores = [row.source_faithfulness for row in faith]
        regulatory_scores = [row.regulatory_support for row in faith if row.regulatory_support is not None]
        primary_scores = [row.faithfulness_score for row in faith if row.faithfulness_score is not None]
        risk_support = item.get("risk_support", [])
        retrieval_coverage = (1.0 if item.get("evidence") else 0.0) if faith else None
        supported_attribution_coverage = (
            round(sum(bool(row.attribution_chunk_id) for row in faith) / len(faith), 6) if faith else None
        )
        unsupported_flag_rate = round(sum(row.unsupported for row in faith) / len(faith), 6) if faith else None
        return ClauseResult(
            clause_id=clause.clause_id,
            original_text=clause.text,
            page_number=clause.page_number,
            section_heading=clause.section_heading,
            extraction_method=clause.extraction_method,
            simplified_text=item["simplified"],
            generated_simplified_text=item.get("generated_simplified"),
            simplification_suppressed=bool(item.get("simplification_suppressed")),
            readability_grade=simplification.fk_grade,
            readability_target_met=simplification.readability_target_met,
            semantic_preservation_score=simplification.semantic_preservation_score,
            semantic_target_met=simplification.semantic_target_met,
            simplification_accepted=simplification.accepted,
            simplification_attempts=simplification.attempts,
            simplification_status=simplification.status,
            risk_label=risk.risk_label,
            risk_score=risk.risk_score,
            risk_confidence=risk.confidence,
            risk_explanation=risk.explanation,
            secondary_risk_labels=risk.secondary_risk_labels,
            evidence=item["evidence"],
            faithfulness=faith,
            risk_explanation_support=risk_support,
            mean_source_faithfulness=_mean(source_scores),
            mean_regulatory_support=_mean(regulatory_scores),
            mean_primary_faithfulness=_mean(primary_scores),
            mean_risk_regulatory_support=_mean([row.regulatory_support for row in risk_support]),
            unsupported_flag_rate=unsupported_flag_rate,
            retrieval_coverage=retrieval_coverage,
            supported_attribution_coverage=supported_attribution_coverage,
            stage_timings_ms=item["timings"],
        )

    def _simplification_only_result(self, item: dict[str, Any]) -> ClauseResult:
        clause = item["clause"]
        simplification = item["simplification"]
        return ClauseResult(
            clause_id=clause.clause_id,
            original_text=clause.text,
            page_number=clause.page_number,
            section_heading=clause.section_heading,
            extraction_method=clause.extraction_method,
            simplified_text=item["simplified"],
            generated_simplified_text=item.get("generated_simplified"),
            simplification_suppressed=bool(item.get("simplification_suppressed")),
            readability_grade=simplification.fk_grade,
            readability_target_met=simplification.readability_target_met,
            semantic_preservation_score=simplification.semantic_preservation_score,
            semantic_target_met=simplification.semantic_target_met,
            simplification_accepted=simplification.accepted,
            simplification_attempts=simplification.attempts,
            simplification_status=simplification.status,
            risk_label="Needs Review",
            risk_score=1,
            risk_explanation="Risk and faithfulness stages disabled for simplification-only ablation.",
            stage_timings_ms=item["timings"],
        )

    @staticmethod
    def _notify(callback: ProgressCallback | None, event: str, **payload: Any) -> None:
        if callback is not None:
            callback({"event": event, **payload})

    @classmethod
    def _progress(
        cls,
        callback: ProgressCallback | None,
        stage: str,
        current: int,
        total: int,
        clause_id: str,
    ) -> None:
        cls._notify(
            callback,
            "stage_progress",
            stage=stage,
            current=current,
            total=total,
            percent=round(current * 100 / max(total, 1), 2),
            clause_id=clause_id,
        )


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None
