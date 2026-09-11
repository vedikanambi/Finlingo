from __future__ import annotations

import json
from dataclasses import asdict
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression

from backend.app.core.config import Settings
from backend.app.services.dataset_streams import StreamingDatasetRepository
from backend.app.services.model_registry import ModelRegistry
from backend.app.services.nli_service import NLIService
from backend.app.services.text_generation import TextGenerator
from backend.evaluation.flb_builder import FLBRecord
from backend.evaluation.metrics import binary_faithfulness_metrics, multiclass_metrics, rq1_metrics
from backend.evaluation.risk_taxonomy import RiskTaxonomy

RISK_LABELS = ["Auto-Renewal", "Hidden Fee", "Liability Waiver", "Data Sharing", "Penalty Clause", "Safe"]
_RULES = {
    "Auto-Renewal": ["automatic renewal", "auto-renew", "renewal", "recurring"],
    "Hidden Fee": ["fee", "charge", "surcharge", "additional cost"],
    "Liability Waiver": ["not liable", "waive", "indemnify", "hold harmless", "limitation of liability"],
    "Data Sharing": ["personal data", "personal information", "third party", "affiliate", "share data"],
    "Penalty Clause": ["penalty", "late fee", "default charge", "liquidated damages", "termination fee"],
}


class GPTResult(BaseModel):
    simplified: str
    risk_label: Literal[
        "Auto-Renewal",
        "Hidden Fee",
        "Liability Waiver",
        "Data Sharing",
        "Penalty Clause",
        "Safe",
    ]
    risk_score: int = Field(ge=1, le=5)


class ParagraphJudgeResult(BaseModel):
    support_score: float = Field(ge=0, le=1)
    supported_claims: int = Field(ge=0)
    total_claims: int = Field(ge=1)
    rationale: str


def _unique(records: list[FLBRecord]) -> list[FLBRecord]:
    seen: set[str] = set()
    output: list[FLBRecord] = []
    for record in records:
        if record.source_id not in seen:
            seen.add(record.source_id)
            output.append(record)
    return output


def rule_based_baseline(records: list[FLBRecord]) -> dict:
    unique = _unique(records)
    predicted = []
    for record in unique:
        text = record.original_clause.lower()
        scores = {label: sum(term in text for term in terms) for label, terms in _RULES.items()}
        label, score = max(scores.items(), key=lambda item: item[1])
        predicted.append(label if score else "Safe")
    return {"rq2": multiclass_metrics([record.risk_label for record in unique], predicted, RISK_LABELS)}


def gpt_zero_shot_baseline(settings: Settings, registry: ModelRegistry, records: list[FLBRecord]) -> dict:
    unique = _unique(records)
    simplified, labels = [], []
    for record in unique:
        response = registry.openai().responses.create(
            model=settings.judge_model,
            instructions=(
                "Zero-shot consumer-finance assistant. Faithfully simplify and classify into Auto-Renewal, "
                "Hidden Fee, Liability Waiver, Data Sharing, Penalty Clause, or Safe. Return JSON only."
            ),
            input=(
                f'Return {{"simplified":"...","risk_label":"...","risk_score":1}}.\nCLAUSE:\n{record.original_clause}'
            ),
        )
        value = GPTResult.model_validate_json(response.output_text)
        simplified.append(value.simplified)
        labels.append(value.risk_label)
    return {
        "rq1": asdict(
            rq1_metrics(
                [record.original_clause for record in unique],
                simplified,
                [record.reference_simplification for record in unique],
                bertscore_model=settings.bertscore_model,
            )
        ),
        "rq2": multiclass_metrics([record.risk_label for record in unique], labels, RISK_LABELS),
        "model": settings.judge_model,
    }


def ragas_style_paragraph_baseline(
    settings: Settings,
    registry: ModelRegistry,
    records: list[FLBRecord],
) -> dict:
    """Paragraph-level LLM-as-judge baseline, named RAGAS-style since it mirrors that premise."""
    from backend.training.silver_label_providers import build_silver_label_provider

    provider_name = settings.silver_judge_provider or settings.silver_label_provider
    provider = build_silver_label_provider(settings, registry, task="judge", provider_name=provider_name)

    scores: list[float] = []
    labels = [record.supported for record in records]
    for record in records:
        result = provider.generate(
            schema=ParagraphJudgeResult,
            system_prompt=(
                "Evaluate paragraph-level faithfulness. Break the hypothesis into factual claims, count how many are "
                "fully supported by the premise, and return their fraction. Material omissions or changed legal "
                "conditions are unsupported. Return JSON only."
            ),
            user_prompt=(
                'Return {"support_score":0.0,"supported_claims":0,"total_claims":1,"rationale":"..."}.\n'
                f"PREMISE:\n{record.original_clause}\n\nHYPOTHESIS:\n{record.hypothesis}"
            ),
        )
        if result.parsed is None:
            raise RuntimeError(f"RAGAS-style judge returned no valid structured response: {result.metadata}")
        scores.append(result.parsed.support_score)
    return {
        "name": "RAGAS-style paragraph LLM-as-judge",
        "granularity": "paragraph",
        "model": provider_name,
        "threshold_metrics": {
            str(tau): asdict(binary_faithfulness_metrics(labels, scores, tau)) for tau in settings.tau_values
        },
    }


def selfcheckgpt_nli_baseline(
    settings: Settings,
    registry: ModelRegistry,
    records: list[FLBRecord],
) -> dict:
    """Sentence-level cross-sample consistency baseline inspired by SelfCheckGPT - never sees the source, only compares against its own resampled generations."""
    generator = TextGenerator(settings, registry)
    nli = NLIService(registry)
    samples_by_source: dict[str, list[str]] = {}
    for record in _unique(records):
        samples = []
        for _ in range(settings.selfcheck_samples):
            samples.append(
                generator.generate(
                    instructions=(
                        "Explain the clause in plain English. Preserve legal meaning, but produce a natural independent sample."
                    ),
                    user_input=record.original_clause,
                    model_id=settings.simplifier_model,
                    adapter_id=settings.simplifier_adapter,
                    # needs the "S2" prefix to resolve to the right backend, else it falls back to one with no adapter configured and crashes
                    purpose="S2 SelfCheckGPT baseline sampling",
                    max_new_tokens=settings.simplifier_max_new_tokens,
                    temperature=settings.selfcheck_temperature,
                    max_input_tokens=settings.simplifier_max_input_tokens,
                )
            )
        samples_by_source[record.source_id] = samples

    scores: list[float] = []
    for record in records:
        samples = samples_by_source[record.source_id]
        entailments = nli.score_pairs(
            [(sample, record.hypothesis) for sample in samples],
            settings.verifier_batch_size,
            settings.verifier_max_length,
        )
        scores.append(float(np.mean([score.entailment for score in entailments])))
    labels = [record.supported for record in records]
    return {
        "name": "SelfCheckGPT-style NLI cross-sample consistency",
        "granularity": "sentence",
        "sample_count": settings.selfcheck_samples,
        "threshold_metrics": {
            str(tau): asdict(binary_faithfulness_metrics(labels, scores, tau)) for tau in settings.tau_values
        },
    }


# these CUAD categories are purely administrative so they're safe to use as the Safe class here
_FINBERT_SAFE_SOURCE_CATEGORIES = {"Parties", "Document Name", "Effective Date", "Agreement Date", "Governing Law"}


def finbert_domain_baseline(settings: Settings, records: list[FLBRecord]) -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer

    repository = StreamingDatasetRepository(settings)
    taxonomy = RiskTaxonomy(settings.project_root / "configs" / "risk_taxonomy.yaml")
    texts: list[str] = []
    labels: list[str] = []
    safe_count = 0
    try:
        for row in repository.stream_partitioned(
            settings.cuad_dataset,
            requested_split="train",
            config=settings.cuad_config,
            key_fields=("file_name",),
            shuffle=True,
        ):
            source_label = str(row.get("label") or "")
            clause_text = str(row.get("clause") or "").strip()
            if not clause_text or len(clause_text.split()) < 6:
                continue
            mapped = taxonomy.map_question(source_label)
            if mapped:
                texts.append(clause_text)
                labels.append(mapped)
            elif source_label in _FINBERT_SAFE_SOURCE_CATEGORIES and safe_count < settings.finbert_train_samples // 6:
                texts.append(clause_text)
                labels.append("Safe")
                safe_count += 1
            if len(texts) >= settings.finbert_train_samples:
                break
    finally:
        repository.close()
    if len(set(labels)) < 2:
        raise RuntimeError(
            f"Insufficient CUAD labels for FinBERT baseline: only found {sorted(set(labels))} "
            f"from {len(texts)} scanned rows"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(settings.finbert_model, token=settings.hf_token)
    model = AutoModel.from_pretrained(settings.finbert_model, token=settings.hf_token).to(device).eval()

    def encode(items: list[str], batch_size: int = 16) -> np.ndarray:
        output = []
        for start in range(0, len(items), batch_size):
            encoded = tokenizer(
                items[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                hidden = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            output.append(pooled.cpu().numpy())
        return np.concatenate(output)

    classifier = LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=settings.random_seed)
    classifier.fit(encode(texts), labels)
    unique = _unique(records)
    predictions = classifier.predict(encode([record.original_clause for record in unique])).tolist()
    return {
        "rq2": multiclass_metrics([record.risk_label for record in unique], predictions, RISK_LABELS),
        "model": settings.finbert_model,
        "train_examples": len(texts),
        "method": "frozen FinBERT mean-pooled embeddings + logistic regression",
    }
