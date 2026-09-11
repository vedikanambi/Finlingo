from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

import numpy as np

from backend.app.core.config import Settings
from backend.app.core.schemas import EvidenceChunk
from backend.app.services.model_registry import ModelRegistry
from backend.app.services.regulatory_corpus import RegulatoryCorpusLoader

logger = logging.getLogger(__name__)


@dataclass
class RetrievalState:
    chunks: list[EvidenceChunk]
    warnings: list[str]
    source_manifests: list[dict]
    corpus_sha256: str
    faiss_index: object
    bm25: object
    chunk_id_to_index: dict[str, int]


class InMemoryRetriever:
    """Live-source regulatory retrieval with an ephemeral FAISS/BM25 index."""

    def __init__(self, settings: Settings, registry: ModelRegistry) -> None:
        self.settings = settings
        self.registry = registry
        self._state: RetrievalState | None = None
        self._lock = threading.RLock()

    @property
    def warnings(self) -> list[str]:
        return list(self._ensure_state().warnings)

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self._ensure_state().chunks})

    @property
    def source_manifests(self) -> list[dict]:
        return list(self._ensure_state().source_manifests)

    @property
    def corpus_sha256(self) -> str:
        return self._ensure_state().corpus_sha256

    def rebuild(self) -> None:
        with self._lock:
            self._state = self._build_state()

    def retrieve(self, query: str, k: int | None = None, method: str = "bm25") -> list[EvidenceChunk]:
        # bm25 wins on recall for this legal text so it's the default; faiss/hybrid stay around for comparison
        state = self._ensure_state()
        k = min(k or self.settings.top_k_retrieval, len(state.chunks))
        if method == "bm25":
            return self._bm25(state, query, k)
        if method == "hybrid":
            return self._hybrid(state, query, k)
        if method != "faiss":
            raise ValueError("retrieval method must be 'faiss', 'bm25', or 'hybrid'")
        return self._faiss(state, query, k)

    def _faiss(self, state: RetrievalState, query: str, k: int) -> list[EvidenceChunk]:
        vector = self._embed([query])
        distances, indices = state.faiss_index.search(vector, k)
        output = []
        for distance, index in zip(distances[0], indices[0]):
            if index < 0:
                continue
            chunk = state.chunks[int(index)].model_copy(deep=True)
            chunk.retrieval_score = float(1 / (1 + max(float(distance), 0)))
            output.append(chunk)
        return output

    def _hybrid(self, state: RetrievalState, query: str, k: int) -> list[EvidenceChunk]:
        """Reciprocal-rank fusion of dense (FAISS) and lexical (BM25) rankings."""
        pool_k = min(max(k * 10, 200), len(state.chunks))
        dense = self._faiss(state, query, pool_k)
        lexical = self._bm25(state, query, pool_k)

        # plain unweighted RRF actually did worse than bm25 alone here - dense noise was
        # bumping good lexical hits out of the top-k, so bm25 gets most of the weight
        rrf_const = 60.0
        bm25_weight = 5.0
        dense_weight = 1.0
        fused_scores: dict[int, float] = {}
        for rank, chunk in enumerate(dense):
            idx = state.chunk_id_to_index.get(chunk.chunk_id)
            if idx is None:
                continue
            fused_scores[idx] = fused_scores.get(idx, 0.0) + dense_weight / (rrf_const + rank + 1)
        for rank, chunk in enumerate(lexical):
            idx = state.chunk_id_to_index.get(chunk.chunk_id)
            if idx is None:
                continue
            fused_scores[idx] = fused_scores.get(idx, 0.0) + bm25_weight / (rrf_const + rank + 1)

        ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:k]
        output = []
        for idx, score in ranked:
            chunk = state.chunks[idx].model_copy(deep=True)
            chunk.retrieval_score = float(score)
            output.append(chunk)
        return output

    def _bm25(self, state: RetrievalState, query: str, k: int) -> list[EvidenceChunk]:
        scores = state.bm25.get_scores(_tokens(query))
        best = np.argsort(scores)[::-1][:k]
        max_score = float(max(scores)) if len(scores) else 0.0
        output = []
        for index in best:
            chunk = state.chunks[int(index)].model_copy(deep=True)
            chunk.retrieval_score = float(scores[index]) / max_score if max_score > 0 else 0.0
            output.append(chunk)
        return output

    def _ensure_state(self) -> RetrievalState:
        with self._lock:
            if self._state is None:
                self._state = self._build_state()
            return self._state

    def _build_state(self) -> RetrievalState:
        import faiss
        from rank_bm25 import BM25Okapi

        corpus = RegulatoryCorpusLoader(self.settings).load()
        matrix = self._embed([c.text for c in corpus.chunks])
        index = faiss.IndexFlatL2(self.settings.embedding_dimensions)
        index.add(matrix)
        bm25 = BM25Okapi([_tokens(c.text) for c in corpus.chunks])
        chunk_id_to_index = {chunk.chunk_id: i for i, chunk in enumerate(corpus.chunks)}
        logger.info("Built ephemeral FAISS Flat-L2 index with %d chunks", len(corpus.chunks))
        return RetrievalState(
            corpus.chunks,
            corpus.warnings,
            corpus.source_manifests,
            corpus.corpus_sha256,
            index,
            bm25,
            chunk_id_to_index,
        )

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.settings.embedding_dimensions), dtype="float32")
        encoder = self.registry.sentence_encoder()
        matrix = encoder.encode(
            texts,
            batch_size=self.settings.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        return np.ascontiguousarray(matrix, dtype="float32")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
