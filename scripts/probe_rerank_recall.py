"""Cheap probe for whether the reranker keeps the gold chunk inside top_j -
BM25 + cross-encoder only, no LLM calls, for fast top_k/top_j iteration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import Settings
from backend.app.services.in_memory_retriever import InMemoryRetriever
from backend.app.services.model_registry import ModelRegistry
from backend.app.stages.stage3_retrieval import Stage3Retrieval
from backend.app.stages.stage4_reranking import Stage4Reranking


def main() -> None:
    export_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("reports/flb_final_check3_export.json")
    top_ks = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["500"])]
    top_js = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["10", "30", "50", "80", "120"])]
    query_mode = sys.argv[4] if len(sys.argv) > 4 else "combined"
    rerank_query_mode = sys.argv[5] if len(sys.argv) > 5 else query_mode

    settings = Settings()
    registry = ModelRegistry(settings)
    retriever = InMemoryRetriever(settings, registry)
    stage3 = Stage3Retrieval(settings, retriever)
    stage4 = Stage4Reranking(settings, registry)

    lines = export_path.read_text(encoding="utf-8").splitlines()
    seen = {}
    for line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        sid = rec["source_id"]
        if sid in seen or not rec.get("ground_truth_chunk_id"):
            continue
        seen[sid] = rec

    print(f"{len(seen)} unique clauses with ground truth")
    for top_k in top_ks:
        pre_hits = 0
        post_hits_by_j = {j: 0 for j in top_js}
        total = 0
        for rec in seen.values():
            gold = rec["ground_truth_chunk_id"]
            original = rec["original_clause"]
            simplified = rec.get("reference_simplification") or original
            combined = f"LEGAL CLAUSE:\n{original}\n\nPLAIN-LANGUAGE INTERPRETATION:\n{simplified}"
            queries = {"original": original, "simplified": simplified, "combined": combined}
            query = queries[query_mode]
            rerank_query = queries[rerank_query_mode]
            retrieved = retriever.retrieve(query, top_k, "bm25")
            total += 1
            pre_ids = {c.chunk_id for c in retrieved}
            if gold in pre_ids:
                pre_hits += 1
            reranked = stage4.rerank(rerank_query, retrieved)
            for j in top_js:
                if gold in {c.chunk_id for c in reranked[:j]}:
                    post_hits_by_j[j] += 1
        print(f"top_k={top_k}: pre-rerank recall={pre_hits/total:.4f} ({pre_hits}/{total})")
        for j in top_js:
            print(f"  post-rerank recall@{j} = {post_hits_by_j[j]/total:.4f} ({post_hits_by_j[j]}/{total})")


if __name__ == "__main__":
    main()
