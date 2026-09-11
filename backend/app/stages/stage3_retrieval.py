"""Stage 3 - finds the regulatory text chunks relevant to a clause. BM25 by
default (tried dense/FAISS, BM25 worked better on this corpus)."""

from backend.app.core.config import Settings
from backend.app.core.schemas import EvidenceChunk
from backend.app.services.in_memory_retriever import InMemoryRetriever


class Stage3Retrieval:
    def __init__(self, settings: Settings, retriever: InMemoryRetriever) -> None:
        self.settings = settings
        self.retriever = retriever

    def build_query(self, original_clause: str, simplified_clause: str) -> str:
        mode = self.settings.retrieval_query_mode
        if mode == "original":
            return original_clause
        if mode == "combined":
            return f"LEGAL CLAUSE:\n{original_clause}\n\nPLAIN-LANGUAGE INTERPRETATION:\n{simplified_clause}"
        return simplified_clause

    def retrieve(
        self,
        simplified_clause: str,
        method: str = "bm25",
        k: int | None = None,
        *,
        original_clause: str | None = None,
    ) -> list[EvidenceChunk]:
        query = self.build_query(original_clause or simplified_clause, simplified_clause)
        return self.retriever.retrieve(query, k or self.settings.top_k_retrieval, method)
