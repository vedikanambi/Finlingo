"""Tunes retrieval top-k/top-j on FinanceBench (kept separate from the locked
FLB set); computes embeddings/cross-encoder scores once at the largest k and
reuses them for every grid point instead of rerunning retrieval per config."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import Settings
from backend.app.services.model_registry import ModelRegistry
from backend.evaluation.financebench_eval import (
    _deduplicate_evidence,
    _embed,
    load_financebench_stream,
)
from backend.evaluation.metrics import retrieval_metrics_multi


def _parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def tune_retrieval(
    settings: Settings,
    *,
    max_examples: int,
    top_ks: tuple[int, ...],
    top_js: tuple[int, ...],
) -> dict:
    """selects top-k/top-j on FinanceBench without touching locked FLB rows"""
    import faiss
    import numpy as np

    started = time.perf_counter()
    examples = load_financebench_stream(settings, max_examples)
    registry = ModelRegistry(settings)
    evidence = list(_deduplicate_evidence(examples).values())
    evidence_matrix = _embed(settings, registry, [item.text for item in evidence])
    query_matrix = _embed(settings, registry, [item.question for item in examples])

    index = faiss.IndexFlatL2(settings.embedding_dimensions)
    index.add(evidence_matrix)
    max_k = min(max(top_ks), len(evidence))
    _, indices = index.search(query_matrix, max_k)
    expected = [{item.chunk_id for item in example.evidence} for example in examples]
    cross_encoder = registry.cross_encoder()

    # Score the max candidate set once; smaller-k trials just take a prefix.
    score_rows: list[np.ndarray] = []
    candidate_rows: list[list] = []
    for example, row_indices in zip(examples, indices):
        candidates = [evidence[int(i)] for i in row_indices if i >= 0]
        scores = cross_encoder.predict(
            [(example.question, item.text) for item in candidates],
            batch_size=settings.reranker_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        candidate_rows.append(candidates)
        score_rows.append(np.asarray(scores))

    trials = []
    for top_k in top_ks:
        effective_k = min(top_k, max_k)
        before = [[item.chunk_id for item in row[:effective_k]] for row in candidate_rows]
        before_metrics = retrieval_metrics_multi(before, expected, effective_k)
        for top_j in top_js:
            if top_j > effective_k:
                continue
            after = []
            for candidates, scores in zip(candidate_rows, score_rows):
                pairs = sorted(
                    zip(candidates[:effective_k], scores[:effective_k]),
                    key=lambda pair: float(pair[1]),
                    reverse=True,
                )
                after.append([item.chunk_id for item, _ in pairs[:top_j]])
            after_metrics = retrieval_metrics_multi(after, expected, top_j)
            trials.append(
                {
                    "top_k": effective_k,
                    "top_j": top_j,
                    "faiss": before_metrics,
                    "reranked": after_metrics,
                }
            )

    def objective(trial: dict) -> tuple[float, float, float, int, int]:
        metrics = trial["reranked"]
        return (
            float(metrics.get(f"recall@{trial['top_j']}", 0.0)),
            float(metrics.get(f"ndcg@{trial['top_j']}", 0.0)),
            float(metrics.get("mrr", 0.0)),
            -trial["top_j"],
            -trial["top_k"],
        )

    selected = max(trials, key=objective)
    return {
        "status": "complete",
        "selection_benchmark": "PatronusAI/financebench (development only; FLB remains locked)",
        "objective": "maximize reranked recall, then nDCG/MRR, then minimize cost",
        "n_examples": len(examples),
        "n_unique_evidence_chunks": len(evidence),
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.ollama_embedding_model
        if settings.embedding_provider == "ollama"
        else settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "selected": selected,
        "trials": trials,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-examples", type=int, default=150)
    parser.add_argument("--top-k", type=_parse_ints, default=_parse_ints("8,20,40,80"))
    parser.add_argument("--top-j", type=_parse_ints, default=_parse_ints("5,10,20,30"))
    parser.add_argument("--output", type=Path, default=Path("reports/focused_retrieval_tuning.json"))
    args = parser.parse_args()
    result = tune_retrieval(
        Settings(), max_examples=args.max_examples, top_ks=args.top_k, top_js=args.top_j
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected": result["selected"]}, indent=2))


if __name__ == "__main__":
    main()
