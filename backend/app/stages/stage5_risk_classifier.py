"""Stage 5 - labels clause risk category via prompted LLM or fine-tuned classifier;
the fine-tuned one is deployed, achieving 0.79 macro-F1."""

from __future__ import annotations

import json
from functools import lru_cache

import yaml
from pydantic import ValidationError

from backend.app.core.config import Settings
from backend.app.core.schemas import EvidenceChunk, RiskPrediction
from backend.app.core.risk_taxonomy import load_risk_taxonomy
from backend.app.services.model_registry import ModelRegistry
from backend.app.services.text_generation import TextGenerator, parse_json_object
from backend.training.train_risk_classifier import LABEL2ID, RISK_LABELS

# the trained classifier only outputs a label, not a score, so map each label to a fixed score
_TRAINED_RISK_SCORE = {
    "Auto-Renewal": 4,
    "Hidden Fee": 3,
    "Liability Waiver": 4,
    "Data Sharing": 3,
    "Penalty Clause": 3,
    "Safe": 1,
}


class Stage5RiskClassifier:
    def __init__(self, settings: Settings, registry: ModelRegistry) -> None:
        self.settings = settings
        self.generator = TextGenerator(settings, registry)
        self.taxonomy = load_risk_taxonomy(settings.resolve(settings.risk_taxonomy_path))

    def classify(self, original_clause: str, chunks: list[EvidenceChunk]) -> RiskPrediction:
        if self.settings.risk_classifier_backend == "frozen":
            artifact = self.settings.resolve(self.settings.risk_classifier_frozen_artifact)
            if artifact.exists():
                return self._classify_frozen(original_clause, artifact)
            if self.settings.risk_classifier_require_adapter:
                raise RuntimeError(f"risk_classifier_backend=frozen but no artifact found at {artifact}")
        if self.settings.risk_classifier_backend == "trained":
            adapter_path = self.settings.resolve(
                self.settings.risk_classifier_adapter or self.settings.risk_classifier_adapter_output
            )
            if (adapter_path / "adapter_config.json").exists():
                return self._classify_trained(original_clause, adapter_path)
            if self.settings.risk_classifier_require_adapter:
                raise RuntimeError(f"risk_classifier_backend=trained but no adapter found at {adapter_path}")
        return self._classify_prompted(original_clause, chunks)

    def _classify_frozen(self, original_clause: str, artifact_path) -> RiskPrediction:
        model, tokenizer, classifier = _load_frozen_classifier(
            self.settings.risk_classifier_model,
            str(artifact_path),
            self.settings.risk_classifier_max_length,
        )
        import torch

        encoded = tokenizer(
            original_clause,
            padding=True,
            truncation=True,
            max_length=self.settings.risk_classifier_max_length,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        probabilities = classifier.predict_proba(pooled.float().cpu().numpy())[0]
        top_index = int(probabilities.argmax())
        label = str(classifier.classes_[top_index])
        confidence = float(probabilities[top_index])
        return RiskPrediction(
            risk_label=label,
            risk_score=_TRAINED_RISK_SCORE[label],
            confidence=confidence,
            explanation=f"Frozen FinBERT embedding classifier predicted {label} with confidence {confidence:.3f}.",
            evidence_chunk_ids=[],
            secondary_risk_labels=[],
            parse_attempts=1,
        )

    def _classify_trained(self, original_clause: str, adapter_path) -> RiskPrediction:
        model, tokenizer = _load_trained_classifier(
            self.settings.risk_classifier_model,
            str(adapter_path),
            self.settings.risk_classifier_max_length,
        )
        import torch

        encoded = tokenizer(
            original_clause,
            truncation=True,
            max_length=self.settings.risk_classifier_max_length,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[0]
        top_index = int(torch.argmax(probs).item())
        label = RISK_LABELS[top_index]
        confidence = float(probs[top_index].item())
        return RiskPrediction(
            risk_label=label,
            risk_score=_TRAINED_RISK_SCORE[label],
            confidence=confidence,
            explanation=(
                f"Trained classifier ({self.settings.risk_classifier_model} + LoRA fine-tune) "
                f"predicted {label} with confidence {confidence:.3f}."
            ),
            evidence_chunk_ids=[],
            secondary_risk_labels=[],
            parse_attempts=1,
        )

    def _classify_prompted(self, original_clause: str, chunks: list[EvidenceChunk]) -> RiskPrediction:
        evidence = "\n\n".join(
            f"[{chunk.chunk_id}] {chunk.source} {chunk.section or ''}\n{chunk.text}" for chunk in chunks
        )
        few_shot = self._few_shot_prompt()
        labels_text = ", ".join(self.taxonomy.labels)
        prompt = f"""Classify the clause into exactly one PRIMARY label: {labels_text}. "Needs Review" is an abstention, not a seventh research class, and is
mandatory only when the CLAUSE TEXT ITSELF is genuinely ambiguous (you cannot tell which of the taxonomy categories
applies even after reading it carefully) or the requested JSON truly cannot be produced. A thin, missing, or
off-topic EVIDENCE section is NOT by itself a reason to answer "Needs Review" -- classify primarily from the
CLAUSE TEXT, which is always available and usually sufficient on its own; EVIDENCE is a corroborating aid you cite
when it genuinely helps, never a precondition for making a decision. Retrieval in this system frequently returns no
closely-matching chunk, and defaulting to "Needs Review" whenever that happens would make the classifier useless.
You may list additional applicable proposal labels in secondary_risk_labels. Apply the risk-score rubric consistently.
Cite a chunk ID only if it actually appears in EVIDENCE and genuinely supports your reasoning; do not cite anything
if no EVIDENCE chunk is relevant, and do not let a missing citation change your PRIMARY label.

EVIDENCE CAN ONLY ESCALATE, NEVER JUSTIFY SAFE: retrieved chunks may corroborate or escalate a risk finding
(e.g. evidence showing a clause violates a specific disclosure rule supports a HIGHER risk_score), but evidence
can never be the reason to classify a clause as Safe. A Safe classification must be fully supportable from the
CLAUSE text alone — do not cite evidence_chunk_ids for a Safe decision, and do not reason "this is compliant
because the evidence discusses similar regulated topics." Retrieved regulatory text is frequently topically
adjacent but not specific to this clause (e.g. generic mortgage-disclosure text is not evidence of compliance
for an auto-renewal, hidden-fee, or data-sharing clause); treating such text as compliance justification is the
single most common error to avoid.

DO NOT DEFAULT TO SAFE: Safe is a specific finding -- the clause's topic falls fully outside all five risk
categories, or it demonstrably protects the consumer -- not a fallback for neutral, procedural, or formally-worded
text. Most real contract clauses in every one of the five risk categories are written in calm, standard legal
register (that is what boilerplate looks like); a clause is not Safe merely because it avoids alarming language.
Before answering Safe, check concretely: does the clause set any renewal/continuation term, mention any fee/cost/
reimbursement, cap or limit any party's liability or require indemnification, name any category of third party the
information may reach, or impose any consequence for late payment/default/early termination? If yes to any of
those, it belongs in that risk category (at whatever risk_score the specific wording earns, even a low one) and is
not Safe, regardless of how neutral or standard the wording sounds.

RISK-SCORE RUBRIC AND TAXONOMY EXAMPLES:
{few_shot}

Return only this JSON shape:
{{"risk_label":"...","risk_score":1,"confidence":0.0,"explanation":"...",\
"evidence_chunk_ids":["..."],"secondary_risk_labels":[]}}

CLAUSE:\n{original_clause}\n\nEVIDENCE:\n{evidence or "No evidence retrieved."}"""
        model_id = self.settings.risk_model or self.settings.simplifier_model
        adapter = self.settings.risk_adapter
        last_error: Exception | None = None
        for attempt in range(1, self.settings.risk_parse_retries + 2):
            raw = self.generator.generate(
                instructions=(
                    "Be a conservative consumer-finance risk classifier. Use only supplied clause and evidence. "
                    "Do not infer unstated facts. Return valid JSON."
                ),
                user_input=prompt,
                model_id=model_id,
                adapter_id=adapter,
                purpose="S5 risk classifier",
                max_new_tokens=self.settings.risk_max_new_tokens,
                temperature=self.settings.risk_temperature,
                require_adapter=self.settings.risk_require_adapter,
                max_input_tokens=self.settings.risk_max_input_tokens,
            )
            try:
                value = parse_json_object(raw)
                value["parse_attempts"] = attempt
                prediction = RiskPrediction.model_validate(value)
                self._validate_prediction(prediction, chunks)
                return prediction
            except (ValueError, ValidationError) as exc:
                last_error = exc
                prompt += f"\nPrevious output invalid: {exc}. Correct it and return only JSON."
        return RiskPrediction(
            risk_label="Needs Review",
            risk_score=3,
            confidence=0.0,
            explanation=f"Output could not be validated: {last_error}",
            evidence_chunk_ids=[],
            secondary_risk_labels=[],
            parse_attempts=self.settings.risk_parse_retries + 1,
        )

    def _validate_prediction(self, prediction: RiskPrediction, chunks: list[EvidenceChunk]) -> None:
        allowed = {chunk.chunk_id for chunk in chunks}
        if any(chunk_id not in allowed for chunk_id in prediction.evidence_chunk_ids):
            raise ValueError("Cited chunk was not retrieved")
        if prediction.risk_label in prediction.secondary_risk_labels:
            raise ValueError("Primary risk label cannot also appear as a secondary label")
        if len(prediction.secondary_risk_labels) > self.settings.risk_max_secondary_labels:
            raise ValueError("Too many secondary risk labels")
        if prediction.risk_label == "Safe" and prediction.risk_score > 2:
            raise ValueError("Safe predictions must have risk_score 1 or 2")
        if prediction.risk_label == "Safe" and prediction.evidence_chunk_ids:
            raise ValueError(
                "Safe classification cannot cite evidence as justification; Safe must be "
                "supportable from the clause text alone"
            )
        if prediction.risk_label == "Needs Review" and prediction.confidence > 0.5:
            raise ValueError("Needs Review must not claim high confidence")
        if (
            self.settings.risk_require_evidence_for_risk
            and chunks
            and prediction.risk_label not in {"Safe", "Needs Review"}
            and not prediction.evidence_chunk_ids
        ):
            raise ValueError("A non-Safe risk prediction must cite at least one retrieved evidence chunk")

    @lru_cache(maxsize=1)
    def _few_shot_prompt(self) -> str:
        path = self.settings.resolve(self.settings.risk_few_shot_path)
        if not path.exists():
            raise RuntimeError(f"Risk few-shot configuration not found: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        rubric = payload.get("risk_score_rubric") or {}
        examples = payload.get("examples") or []
        if not rubric or not examples:
            raise RuntimeError("Risk few-shot configuration must contain a rubric and examples")
        rubric_text = "\n".join(f"{score}: {description}" for score, description in sorted(rubric.items()))
        example_text = []
        for example in examples:
            label = str(example.get("risk_label") or "")
            if label not in self.taxonomy.labels:
                raise RuntimeError(f"Invalid few-shot risk label: {label}")
            output = {
                "risk_label": label,
                "risk_score": int(example["risk_score"]),
                "confidence": 0.90,
                "explanation": str(example["explanation"]),
                "evidence_chunk_ids": [],
                "secondary_risk_labels": list(example.get("secondary_risk_labels") or []),
            }
            example_text.append(
                f"EXAMPLE CLAUSE: {example['clause']}\nEXAMPLE OUTPUT: {json.dumps(output, ensure_ascii=False)}"
            )
        return f"RUBRIC:\n{rubric_text}\n\n" + "\n\n".join(example_text)


@lru_cache(maxsize=1)
def _load_trained_classifier(base_model_id: str, adapter_path: str, max_length: int):
    """Load the fine-tuned S5 sequence classifier once per process."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        num_labels=len(RISK_LABELS),
        # base FinBERT head is 3-class sentiment, ours is 6-class, sizes won't match
        ignore_mismatched_sizes=True,
        label2id=LABEL2ID,
        id2label={v: k for k, v in LABEL2ID.items()},
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, tokenizer


@lru_cache(maxsize=1)
def _load_frozen_classifier(base_model_id: str, artifact_path: str, max_length: int):
    import joblib
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModel.from_pretrained(base_model_id)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    classifier = joblib.load(artifact_path)
    return model, tokenizer, classifier
