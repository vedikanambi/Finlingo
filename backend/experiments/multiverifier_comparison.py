from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

from backend.app.core.config import Settings
from backend.app.services.model_registry import ModelRegistry
from backend.app.services.nli_service import NLIService
from backend.evaluation.faithbench import load_faithbench
from backend.evaluation.flb_builder import FLBBuilder, FLBRecord
from backend.evaluation.metrics import binary_faithfulness_metrics, bootstrap_ci


class JudgeResult(BaseModel):
    supported: bool
    confidence: float = Field(ge=0, le=1)
    rationale: str | None = None


def _nli_pairs(
    settings: Settings,
    pairs: list[tuple[str, str]],
    model_id: str,
    adapter_id: str | None,
) -> dict:
    """Benchmark one verifier in its own registry, after a warm-up pair."""
    import torch

    registry = ModelRegistry(settings)
    service = NLIService(
        registry,
        model_id,
        adapter_id,
        require_adapter=True,
    )
    try:
        if pairs:
            service.score_pairs(
                [pairs[0]],
                min(settings.verifier_batch_size, 1),
                settings.verifier_max_length,
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        scores = [
            item.entailment
            for item in service.score_pairs(
                pairs,
                settings.verifier_batch_size,
                settings.verifier_max_length,
            )
        ]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak_mb = float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else None
        return {
            "scores": scores,
            "mean_latency_ms": elapsed * 1000 / max(len(pairs), 1),
            "throughput_pairs_per_second": len(pairs) / elapsed if elapsed else None,
            "peak_gpu_memory_mb": peak_mb,
            "model": model_id,
            "adapter": adapter_id,
            "measurement": "one warm-up pair; model-load time excluded; isolated registry",
        }
    finally:
        registry.close()


def _gpt_pairs(settings: Settings, registry: ModelRegistry, pairs: list[tuple[str, str]]) -> dict:
    client = registry.openai()
    started = time.perf_counter()
    scores: list[float] = []
    input_tokens = 0
    output_tokens = 0
    for premise, hypothesis in pairs:
        response = client.responses.create(
            model=settings.judge_model,
            instructions=(
                "Judge whether the hypothesis is fully entailed by the premise. "
                "Any material addition, deletion, changed amount/date/party/condition/exception/negation is unsupported. "
                "Return JSON only."
            ),
            input=(
                'Return {"supported":true,"confidence":0.0,"rationale":"brief"}.\n'
                f"PREMISE:\n{premise}\n\nHYPOTHESIS:\n{hypothesis}"
            ),
        )
        result = JudgeResult.model_validate_json(response.output_text)
        scores.append(result.confidence if result.supported else 1.0 - result.confidence)
        usage = getattr(response, "usage", None)
        input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
    elapsed = time.perf_counter() - started
    estimated_cost = None
    if settings.gpt_input_cost_per_million is not None and settings.gpt_output_cost_per_million is not None:
        estimated_cost = (
            input_tokens / 1_000_000 * settings.gpt_input_cost_per_million
            + output_tokens / 1_000_000 * settings.gpt_output_cost_per_million
        )
    return {
        "scores": scores,
        "mean_latency_ms": elapsed * 1000 / max(len(pairs), 1),
        "throughput_pairs_per_second": len(pairs) / elapsed if elapsed else None,
        "peak_gpu_memory_mb": None,
        "model": settings.judge_model,
        "adapter": None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": estimated_cost,
        "cost_currency": settings.gpt_cost_currency if estimated_cost is not None else None,
    }


def _evaluate_system(labels: list[int], run: dict, settings: Settings) -> dict:
    scores = run.pop("scores")
    run["score_bootstrap_ci"] = bootstrap_ci(scores, settings.random_seed, settings.bootstrap_samples)
    run["threshold_metrics"] = {
        str(tau): asdict(binary_faithfulness_metrics(labels, scores, tau)) for tau in settings.tau_values
    }
    tau = settings.faithfulness_tau
    run["per_item_correct"] = [int((score >= tau) == bool(label)) for score, label in zip(scores, labels)]
    return run


def _run_dataset(
    settings: Settings, registry: ModelRegistry, pairs: list[tuple[str, str]], labels: list[int], include_gpt: bool
) -> dict:
    systems = {
        "deberta": _nli_pairs(settings, pairs, settings.verifier_model, settings.verifier_adapter),
        # has to use the base encoder here, not the NLI-finetuned checkpoint - the adapter was trained on the base
        "modernbert": _nli_pairs(settings, pairs, settings.modernbert_training_base_model, settings.modernbert_adapter),
    }
    if include_gpt:
        systems["gpt_judge"] = _gpt_pairs(settings, registry, pairs)
    evaluated = {name: _evaluate_system(labels, run, settings) for name, run in systems.items()}
    return {
        "shared_test_set_size": len(pairs),
        "class_counts": {"supported": sum(labels), "unsupported": len(labels) - sum(labels)},
        "systems": evaluated,
        "per_item_correct": {name: run.get("per_item_correct", []) for name, run in evaluated.items()},
    }


def run_multiverifier(
    settings: Settings,
    max_examples: int | None = None,
    output_path: Path | None = None,
    *,
    include_gpt: bool = True,
    flb_records: list[FLBRecord] | None = None,
) -> dict:
    """Compare all verifiers on the same FLB and FaithBench subsets."""
    registry = ModelRegistry(settings)
    flb = flb_records or FLBBuilder(settings, registry).build(max_examples or settings.flb_size)
    faith_limit = min(max_examples or settings.faithbench_max_examples, settings.faithbench_max_examples)
    faithbench = load_faithbench(settings, faith_limit)

    datasets = {
        "flb": (
            [(record.original_clause, record.hypothesis) for record in flb],
            [record.supported for record in flb],
        ),
        "faithbench": (
            [(record.source, record.summary) for record in faithbench],
            [record.supported for record in faithbench],
        ),
    }
    payload = {
        "thresholds": list(settings.tau_values),
        "primary_threshold": settings.faithfulness_tau,
        "label_definition": {
            "positive": "supported/entailed",
            "negative": "unsupported/hallucinated",
            "faithbench": "official worst aggregation; Questionable and Unwanted are negative, Consistent and Benign are positive",
        },
        "datasets": {
            name: _run_dataset(settings, registry, pairs, labels, include_gpt)
            for name, (pairs, labels) in datasets.items()
        },
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
