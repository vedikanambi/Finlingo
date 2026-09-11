from __future__ import annotations

import json
import time
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.pipeline import FinLingoPipeline
from backend.app.services.model_registry import ModelRegistry
from backend.evaluation.evaluator import _provisional_evidence
from backend.evaluation.flb_builder import FLBBuilder, FLBRecord
from backend.evaluation.metrics import retrieval_metrics


def run_reranker_comparison(
    settings: Settings,
    max_examples: int | None = None,
    output_path: Path | None = None,
    *,
    records: list[FLBRecord] | None = None,
) -> dict:
    """Compare MiniLM L-6 and L-12 on the same FLB queries and FAISS top-k."""
    records = records or FLBBuilder(settings, ModelRegistry(settings)).build(max_examples or settings.flb_size)
    unique = {record.source_id: record for record in records}
    pipeline = FinLingoPipeline(settings)

    queries: list[str] = []
    retrieved_sets = []
    gold_ids: list[str | None] = []
    for record in unique.values():
        simplification = pipeline.stage2.simplify(record.original_clause)
        simplified = simplification.text
        query = pipeline.stage3.build_query(record.original_clause, simplified)
        retrieved = pipeline.stage3.retrieve(simplified, "faiss", original_clause=record.original_clause)
        gold = record.ground_truth_chunk_id
        if gold is None:
            broad = {chunk.chunk_id: chunk for chunk in retrieved}
            for chunk in pipeline.stage3.retrieve(simplified, "bm25", original_clause=record.original_clause):
                broad.setdefault(chunk.chunk_id, chunk)
            decision = _provisional_evidence(settings, pipeline.registry, record, list(broad.values()))
            gold = decision.chunk_id if decision.supported else None
        queries.append(query)
        retrieved_sets.append(retrieved)
        gold_ids.append(gold)

    before = retrieval_metrics(
        [[chunk.chunk_id for chunk in chunks] for chunks in retrieved_sets],
        gold_ids,
        settings.top_k_retrieval,
    )
    systems = {}
    for model_id in settings.reranker_models:
        comparison_settings = settings.model_copy(update={"reranker_model": model_id})
        cross_encoder = ModelRegistry(comparison_settings).cross_encoder()
        rankings: list[list[str]] = []
        latencies = []
        for query, chunks in zip(queries, retrieved_sets):
            started = time.perf_counter()
            scores = cross_encoder.predict(
                [(query, chunk.text) for chunk in chunks],
                batch_size=settings.reranker_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            ordered = [
                chunk.chunk_id
                for chunk, _ in sorted(zip(chunks, scores), key=lambda pair: float(pair[1]), reverse=True)
            ][: settings.top_j_reranked]
            latencies.append((time.perf_counter() - started) * 1000)
            rankings.append(ordered)
        systems[model_id] = {
            "retrieval": retrieval_metrics(rankings, gold_ids, settings.top_j_reranked),
            "mean_rerank_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        }

    payload = {
        "shared_queries": len(queries),
        "faiss_before_reranking": before,
        "systems": systems,
        "evidence_ground_truth": "GPT-provisional; confirm with the generated FLB human-review workflow",
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
