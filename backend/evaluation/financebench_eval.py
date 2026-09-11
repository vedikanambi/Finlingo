from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from backend.app.core.config import Settings
from backend.app.services.dataset_streams import StreamingDatasetRepository
from backend.app.services.model_registry import ModelRegistry
from backend.app.services.nli_service import NLIService
from backend.app.services.text_generation import TextGenerator
from backend.evaluation.metrics import bootstrap_ci, retrieval_metrics_multi

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinanceBenchEvidence:
    chunk_id: str
    text: str
    document_name: str | None
    page_number: int | None


@dataclass(frozen=True)
class FinanceBenchExample:
    example_id: str
    question: str
    answer: str
    evidence: tuple[FinanceBenchEvidence, ...]


def normalise_financebench_row(row: dict[str, Any], index: int = 0) -> FinanceBenchExample | None:
    """normalises FinanceBench rows - the hub export format has varied, so this accepts a few different evidence shapes."""
    example_id = str(row.get("financebench_id") or row.get("id") or f"financebench-{index}")
    question = str(row.get("question") or row.get("query") or "").strip()
    answer = str(row.get("answer") or row.get("gold_answer") or row.get("reference_answer") or "").strip()
    raw_evidence = row.get("evidence") or row.get("evidences") or row.get("contexts") or row.get("context")
    evidence: list[FinanceBenchEvidence] = []

    if isinstance(raw_evidence, str):
        raw_items: list[Any] = [raw_evidence]
    elif isinstance(raw_evidence, dict):
        list_lengths = [len(value) for value in raw_evidence.values() if isinstance(value, list)]
        if list_lengths:
            raw_items = []
            for item_index in range(max(list_lengths)):
                raw_items.append(
                    {
                        key: (value[item_index] if isinstance(value, list) and item_index < len(value) else value)
                        for key, value in raw_evidence.items()
                    }
                )
        else:
            raw_items = [raw_evidence]
    elif isinstance(raw_evidence, (list, tuple)):
        raw_items = list(raw_evidence)
    else:
        raw_items = []

    for item_index, item in enumerate(raw_items):
        if isinstance(item, dict):
            text = str(
                item.get("evidence_text") or item.get("text") or item.get("context") or item.get("content") or ""
            ).strip()
            document = item.get("doc_name") or item.get("document_name") or item.get("filename") or row.get("doc_name")
            page = item.get("page_number") or item.get("page")
        else:
            text, document, page = str(item).strip(), row.get("doc_name"), None
        if not text:
            continue
        digest = hashlib.sha256(f"{example_id}\0{item_index}\0{text}".encode("utf-8", errors="ignore")).hexdigest()[:16]
        try:
            page_number = int(page) if page not in (None, "") else None
        except (TypeError, ValueError):
            page_number = None
        evidence.append(FinanceBenchEvidence(f"FB-{digest}", text, str(document) if document else None, page_number))

    if not question or not answer or not evidence:
        return None
    return FinanceBenchExample(example_id, question, answer, tuple(evidence))


def load_financebench_stream(settings: Settings, max_examples: int | None = None) -> list[FinanceBenchExample]:
    repository = StreamingDatasetRepository(settings)
    try:
        output: list[FinanceBenchExample] = []
        for index, row in enumerate(repository.financebench(limit=max_examples or settings.financebench_max_examples)):
            example = normalise_financebench_row(row, index)
            if example is not None:
                output.append(example)
        if not output:
            raise RuntimeError("No usable FinanceBench rows were returned by the live Hugging Face stream")
        return output
    finally:
        repository.close()


def evaluate_financebench(
    settings: Settings,
    *,
    max_examples: int | None = None,
    output_path: Path | None = None,
    include_generation: bool = True,
    include_gpt_judge: bool = False,
) -> dict[str, Any]:
    """runs retrieval/reranking and optionally grounded generation - everything stays in memory, only the summary gets written."""
    examples = load_financebench_stream(settings, max_examples)
    registry = ModelRegistry(settings)
    evidence_by_id = _deduplicate_evidence(examples)
    evidence = list(evidence_by_id.values())
    matrix = _embed(settings, registry, [item.text for item in evidence])

    import faiss

    index = faiss.IndexFlatL2(settings.embedding_dimensions)
    index.add(matrix)
    query_matrix = _embed(settings, registry, [item.question for item in examples])
    k = min(settings.top_k_retrieval, len(evidence))
    distances, indices = index.search(query_matrix, k)

    cross_encoder = registry.cross_encoder()
    before_rankings: list[list[str]] = []
    after_rankings: list[list[str]] = []
    expected: list[set[str]] = []
    retrieval_latencies_ms: list[float] = []
    rerank_latencies_ms: list[float] = []
    selected_evidence: list[list[FinanceBenchEvidence]] = []

    for row_index, example in enumerate(examples):
        started = time.perf_counter()
        retrieved: list[FinanceBenchEvidence] = []
        for item_index in indices[row_index]:
            if item_index >= 0:
                retrieved.append(evidence[int(item_index)])
        retrieval_latencies_ms.append((time.perf_counter() - started) * 1000)
        before_rankings.append([item.chunk_id for item in retrieved])
        expected.append({item.chunk_id for item in example.evidence})

        rerank_started = time.perf_counter()
        scores = (
            cross_encoder.predict(
                [(example.question, item.text) for item in retrieved],
                batch_size=settings.reranker_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            if retrieved
            else []
        )
        ranked = [item for item, _ in sorted(zip(retrieved, scores), key=lambda pair: float(pair[1]), reverse=True)]
        ranked = ranked[: settings.top_j_reranked]
        rerank_latencies_ms.append((time.perf_counter() - rerank_started) * 1000)
        after_rankings.append([item.chunk_id for item in ranked])
        selected_evidence.append(ranked)

    result: dict[str, Any] = {
        "benchmark": settings.financebench_dataset,
        "storage_mode": "Hugging Face streaming; evidence, embeddings, and FAISS index held in RAM only",
        "n_examples": len(examples),
        "n_unique_evidence_chunks": len(evidence),
        "retrieval": {
            "faiss_top_k": retrieval_metrics_multi(before_rankings, expected, settings.top_k_retrieval),
            "cross_encoder_top_j": retrieval_metrics_multi(after_rankings, expected, settings.top_j_reranked),
            "retrieval_latency_ms": _summary(retrieval_latencies_ms, settings),
            "rerank_latency_ms": _summary(rerank_latencies_ms, settings),
        },
        "configuration": {
            "embedding_provider": settings.embedding_provider,
            "embedding_model": (
                settings.ollama_embedding_model if settings.embedding_provider == "ollama" else settings.embedding_model
            ),
            "embedding_dimensions": settings.embedding_dimensions,
            "ollama_base_url": (settings.ollama_base_url if settings.embedding_provider == "ollama" else None),
            "reranker_provider": "huggingface_cross_encoder",
            "reranker_model": settings.reranker_model,
            "top_k": settings.top_k_retrieval,
            "top_j": settings.top_j_reranked,
        },
    }

    if include_generation:
        result["generation"] = _evaluate_generation(
            settings, registry, examples, selected_evidence, include_gpt_judge=include_gpt_judge
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _evaluate_generation(
    settings: Settings,
    registry: ModelRegistry,
    examples: list[FinanceBenchExample],
    evidence_sets: list[list[FinanceBenchEvidence]],
    *,
    include_gpt_judge: bool,
) -> dict[str, Any]:
    generator = TextGenerator(settings, registry)
    nli = NLIService(registry)
    model_id = settings.financebench_generation_model or settings.simplifier_model
    adapter_id = settings.financebench_generation_adapter or settings.simplifier_adapter
    predictions: list[str] = []
    answer_latencies: list[float] = []
    support_scores: list[float] = []
    judge_scores: list[float] = []

    for example, chunks in zip(examples, evidence_sets):
        context = "\n\n".join(f"[{item.chunk_id}] {item.text}" for item in chunks)
        started = time.perf_counter()
        answer = generator.generate(
            instructions=(
                "Answer the financial question using only the supplied evidence. "
                "If the evidence is insufficient, say so. Preserve all figures and units."
            ),
            user_input=f"QUESTION:\n{example.question}\n\nEVIDENCE:\n{context}\n\nReturn a concise answer.",
            model_id=model_id,
            adapter_id=adapter_id,
            purpose="FinanceBench grounded generation",
            max_new_tokens=settings.risk_max_new_tokens,
            temperature=settings.financebench_temperature,
            max_input_tokens=settings.financebench_max_input_tokens,
        )
        answer_latencies.append((time.perf_counter() - started) * 1000)
        predictions.append(answer)
        if chunks:
            premise = "\n\n".join(item.text for item in chunks)
            support_scores.append(
                nli.score_pairs([(premise, answer)], settings.verifier_batch_size, settings.verifier_max_length)[
                    0
                ].entailment
            )
        else:
            support_scores.append(0.0)
        if include_gpt_judge:
            judge_scores.append(_judge_answer(settings, registry, example, answer, chunks))

    lexical_f1 = [_token_f1(prediction, example.answer) for prediction, example in zip(predictions, examples)]
    bert_values = _bertscore(
        predictions,
        [example.answer for example in examples],
        settings.bertscore_model,
    )
    payload: dict[str, Any] = {
        "model": model_id,
        "adapter": adapter_id,
        "answer_token_f1": _summary(lexical_f1, settings),
        "answer_bertscore_f1": _summary(bert_values, settings),
        "evidence_entailment": _summary(support_scores, settings),
        "latency_ms": _summary(answer_latencies, settings),
    }
    if judge_scores:
        payload["gpt_correctness_1_to_5"] = _summary(judge_scores, settings)
    return payload


def _judge_answer(
    settings: Settings,
    registry: ModelRegistry,
    example: FinanceBenchExample,
    prediction: str,
    chunks: list[FinanceBenchEvidence],
) -> float:
    context = "\n\n".join(item.text for item in chunks)
    response = registry.openai().responses.create(
        model=settings.judge_model,
        instructions=(
            "Score the candidate answer from 1 to 5 for correctness against the reference and supplied evidence. "
            'Return JSON only: {"score":1,"rationale":"..."}.'
        ),
        input=(
            f"QUESTION:\n{example.question}\n\nREFERENCE:\n{example.answer}\n\n"
            f"CANDIDATE:\n{prediction}\n\nEVIDENCE:\n{context}"
        ),
    )
    try:
        value = json.loads(response.output_text)
        return float(min(5, max(1, int(value["score"]))))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"GPT FinanceBench judge returned invalid JSON: {response.output_text!r}") from exc


def _deduplicate_evidence(examples: list[FinanceBenchExample]) -> dict[str, FinanceBenchEvidence]:
    return {item.chunk_id: item for example in examples for item in example.evidence}


def _embed(settings: Settings, registry: ModelRegistry, texts: list[str]) -> np.ndarray:
    """embeds texts via ollama, normalized."""
    del registry

    if not texts:
        return np.empty(
            (0, int(settings.embedding_dimensions)),
            dtype="float32",
        )

    if settings.embedding_provider != "ollama":
        raise RuntimeError("FinanceBench requires EMBEDDING_PROVIDER=ollama.")

    base_url = settings.ollama_base_url.rstrip("/")
    model = settings.ollama_embedding_model

    vectors: list[list[float]] = []

    with httpx.Client(timeout=120.0) as client:
        for start in range(
            0,
            len(texts),
            settings.embedding_batch_size,
        ):
            batch = [value.replace(chr(0), " ") for value in texts[start : start + settings.embedding_batch_size]]

            response = client.post(
                f"{base_url}/api/embed",
                json={
                    "model": model,
                    "input": batch,
                },
            )
            response.raise_for_status()

            payload = response.json()
            batch_vectors = payload.get("embeddings")

            if not isinstance(batch_vectors, list):
                raise RuntimeError("Ollama response is missing the embeddings list.")

            if len(batch_vectors) != len(batch):
                raise RuntimeError(f"Ollama returned {len(batch_vectors)} embeddings for {len(batch)} input texts.")

            vectors.extend(batch_vectors)

    matrix = np.asarray(vectors, dtype="float32")

    if matrix.ndim != 2:
        raise RuntimeError(f"Ollama embeddings must be a two-dimensional matrix. Received shape={matrix.shape!r}.")

    expected_dimensions = int(settings.embedding_dimensions)
    actual_dimensions = int(matrix.shape[1])

    if actual_dimensions != expected_dimensions:
        raise RuntimeError(
            "Ollama embedding dimension mismatch: "
            f"model={model!r}, "
            f"expected={expected_dimensions}, "
            f"actual={actual_dimensions}."
        )

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms == 0, 1.0, norms)

    logger.info(
        "Generated FinanceBench embeddings through Ollama model=%s rows=%d dimensions=%d",
        model,
        matrix.shape[0],
        matrix.shape[1],
    )

    return np.ascontiguousarray(matrix, dtype="float32")


def _bertscore(predictions: list[str], references: list[str], model_type: str) -> list[float]:
    try:
        from bert_score import score

        _, _, f1 = score(
            predictions,
            references,
            model_type=model_type,
            lang="en",
            verbose=False,
            rescale_with_baseline=False,
        )
        return [float(value) for value in f1]
    except Exception as exc:
        raise RuntimeError(
            "FinanceBench BERTScore could not be computed. Token F1 is already reported separately, so it is "
            "not substituted under the BERTScore label."
        ) from exc


def _token_f1(prediction: str, reference: str) -> float:
    from collections import Counter

    pred = Counter(prediction.lower().split())
    ref = Counter(reference.lower().split())
    if not pred or not ref:
        return 0.0
    overlap = sum((pred & ref).values())
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _summary(values: list[float], settings: Settings) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "lower": None, "upper": None}
    return bootstrap_ci(values, settings.random_seed, settings.bootstrap_samples)
