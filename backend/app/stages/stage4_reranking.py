"""Stage 4 - optional cross-encoder reranking of retrieved chunks; off by default since it hurt recall
on this corpus, kept for the reranker-comparison ablation."""

from backend.app.core.config import Settings
from backend.app.core.schemas import EvidenceChunk
from backend.app.services.model_registry import ModelRegistry


class Stage4Reranking:
    def __init__(self, settings: Settings, registry: ModelRegistry) -> None:
        self.settings, self.registry = settings, registry

    def rerank(self, query: str, chunks: list[EvidenceChunk]) -> list[EvidenceChunk]:
        if not chunks:
            return []
        scores = self.registry.cross_encoder().predict(
            [(query, chunk.text) for chunk in chunks],
            batch_size=self.settings.reranker_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        ranked: list[EvidenceChunk] = []
        for chunk, score in zip(chunks, scores):
            copy = chunk.model_copy(deep=True)
            copy.rerank_score = float(score)
            if self.settings.reranker_min_score is None or copy.rerank_score >= self.settings.reranker_min_score:
                ranked.append(copy)
        ranked.sort(
            key=lambda item: item.rerank_score if item.rerank_score is not None else float("-inf"),
            reverse=True,
        )
        return ranked[: self.settings.top_j_reranked]
