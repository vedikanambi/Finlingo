from __future__ import annotations

import json
import hashlib
import logging
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Literal, TypeVar

from pydantic import BaseModel, Field, ValidationError

from backend.app.core.config import Settings
from backend.app.services.model_registry import ModelRegistry
from backend.training.silver_label_providers import (
    GroqProvider,
    build_silver_label_provider,
)
from backend.app.services.text_utils import flesch_kincaid_grade, normalise_text

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class ProviderQuotaExhausted(RuntimeError):
    def __init__(self, message: str, retry_after_seconds: float | None = None) -> None:
        self.retry_after_seconds = retry_after_seconds
        suffix = f" Retry after approximately {retry_after_seconds:.1f}s." if retry_after_seconds else ""
        super().__init__(message + suffix)


def _normalise_risk_label(value: str) -> str:
    value = str(value or "").strip().lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


class SimplificationItem(BaseModel):
    id: str
    simplified: str


class SimplificationBatch(BaseModel):
    items: list[SimplificationItem]


class SyntheticNLIItem(BaseModel):
    source_id: str
    hypothesis: str
    label: str = Field(pattern="^(entailment|neutral|contradiction)$")


class SyntheticNLIBatch(BaseModel):
    items: list[SyntheticNLIItem]


class SimplificationQualityItem(BaseModel):
    id: str
    semantic_accuracy: int = Field(ge=1, le=5)
    readability: int = Field(ge=1, le=5)
    faithful: bool
    rationale: str


class SimplificationQualityBatch(BaseModel):
    items: list[SimplificationQualityItem]


class RiskLabelReviewItem(BaseModel):
    id: str
    correct_label: str
    confidence: int = Field(ge=1, le=5)
    rationale: str


class RiskLabelReviewBatch(BaseModel):
    items: list[RiskLabelReviewItem]


@dataclass
class SimplificationPair:
    source_id: str
    document_id: str
    source: str
    target: str
    source_dataset: str
    fk_grade: float
    quality_weight: float = 1.0
    sari_score: float | None = None
    semantic_entailment: float | None = None


class SilverLabelGenerator:
    """Generate silver references via the configured chat provider; retain them only in RAM."""

    def __init__(self, settings: Settings, registry: ModelRegistry) -> None:
        self.settings = settings

        if (
            settings.silver_label_provider == "local_qwen"
            and not hasattr(registry, "causal_lm")
            and hasattr(registry, "groq_client")
        ):
            self.provider = GroqProvider(settings, registry)
        else:
            self.provider = build_silver_label_provider(
                settings,
                registry,
                task="simplification",
                provider_name=settings.silver_label_provider,
            )
        self.judge_provider_name = settings.silver_judge_provider or settings.silver_label_provider
        self.regeneration_provider_name = settings.silver_regeneration_provider or settings.silver_label_provider
        self.judge_provider = build_silver_label_provider(
            settings,
            registry,
            task="judge",
            provider_name=self.judge_provider_name,
        )
        self.regeneration_provider = build_silver_label_provider(
            settings,
            registry,
            task="simplification",
            provider_name=self.regeneration_provider_name,
        )

        self.cached_labels_reused = 0
        self.new_labels_generated = 0
        self.rejected_labels = 0
        self._cache_lock = threading.Lock()

    @property
    def model(self) -> str:
        if self.settings.silver_label_provider == "local_qwen":
            return self.settings.silver_local_model
        if self.settings.silver_label_provider == "gemini":
            return self.settings.gemini_silver_label_model
        return self.settings.groq_silver_label_model

    def simplification_pairs(
        self, rows: Iterable[dict[str, str]], limit: int, *, alternate: bool = False
    ) -> list[SimplificationPair]:
        output, batch = [], []
        global_attempted = 0
        skipped_batches = 0
        for row in rows:
            batch.append(row)
            global_attempted += 1
            if len(batch) >= self.settings.silver_label_batch_size:
                new_pairs = self._simplify_batch_or_skip(batch, alternate=alternate)
                if new_pairs is None:
                    skipped_batches += 1
                else:
                    output.extend(new_pairs)
                batch = []
                logger.info(
                    "Silver simplification_pairs: global_accepted=%d/%d global_attempted=%d "
                    "skipped_batches=%d alternate=%s",
                    len(output),
                    limit,
                    global_attempted,
                    skipped_batches,
                    alternate,
                )
                if len(output) >= limit:
                    return output[:limit]
        if batch and len(output) < limit:
            new_pairs = self._simplify_batch_or_skip(batch, alternate=alternate)
            if new_pairs is not None:
                output.extend(new_pairs)
            logger.info(
                "Silver simplification_pairs (final batch): global_accepted=%d/%d global_attempted=%d "
                "skipped_batches=%d alternate=%s",
                len(output),
                limit,
                global_attempted,
                skipped_batches,
                alternate,
            )
        return output[:limit]

    def _simplify_batch_or_skip(self, batch, *, alternate: bool) -> list["SimplificationPair"] | None:
        """Skip a batch that keeps failing rather than losing hours of otherwise-cached progress."""
        try:
            return self._simplify_batch(batch, alternate=alternate)
        except ProviderQuotaExhausted:
            raise
        except RuntimeError as exc:
            ids = [str(row.get("source_id")) for row in batch]
            logger.warning(
                "Skipping unrecoverable silver-label batch (%d items, ids=%s): %s",
                len(batch),
                ids,
                exc,
            )
            return None

    def regenerate_simplification(
        self,
        row: dict[str, str],
        *,
        failure_feedback: list[str],
        attempt: int,
    ) -> SimplificationPair | None:
        """Regenerate one failed candidate using measured quality feedback."""
        payload = {
            "id": row["source_id"],
            "clause": row["text"],
            "failed_checks": failure_feedback,
            "retry_attempt": attempt,
        }
        try:
            parsed = self._validated_response(
                SimplificationBatch,
                model=(self.settings.ollama_model if self.regeneration_provider_name == "ollama" else self.model),
                provider_override=self.regeneration_provider,
                provider_name=self.regeneration_provider_name,
                instructions=(
                    "Rewrite the legal or financial clause again in faithful plain English. "
                    "Correct every failed quality check supplied by the evaluator. "
                    "Use shorter sentences and simpler wording without removing or changing "
                    "any number, date, condition, exception, party, obligation, or negation. "
                    "Return JSON only."
                ),
                input_text=(
                    'Return exactly {"items":[{"id":"...","simplified":"..."}]}. '
                    f"The required maximum Flesch-Kincaid grade is "
                    f"{self.settings.fk_acceptance_target}. "
                    "The failed_checks values are diagnostic feedback, not text to copy.\n\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
            )
        except RuntimeError as exc:
            logger.warning(
                "Regeneration attempt failed structurally source_id=%s attempt=%d error=%s",
                row["source_id"],
                attempt,
                exc,
            )
            return None

        by_id = {item.id: normalise_text(item.simplified) for item in parsed.items}
        target = by_id.get(row["source_id"], "")
        if not target or not _critical_tokens_preserved(row["text"], target):
            self.rejected_labels += 1
            logger.warning(
                "Rejected regenerated silver label %s attempt=%d because critical numbers/negations changed",
                row["source_id"],
                attempt,
            )
            return None
        return SimplificationPair(
            row["source_id"],
            row.get("document_id", row["source_id"]),
            row["text"],
            target,
            row.get("dataset", "unknown"),
            flesch_kincaid_grade(target),
        )

    def synthetic_nli(
        self, rows: Iterable[dict[str, str]], limit: int, *, repair_round: int = 0
    ) -> list[dict[str, str]]:
        """Same batching as before, but dispatched concurrently in waves so we stop early once `limit` is hit."""
        batches: list[list[dict[str, str]]] = []
        current: list[dict[str, str]] = []
        for row in rows:
            current.append(row)
            if len(current) >= self.settings.synthetic_nli_batch_size:
                batches.append(current)
                current = []
        if current:
            batches.append(current)

        concurrency = max(1, self.settings.synthetic_nli_concurrency)
        output: list[dict[str, str]] = []
        if concurrency == 1:
            for batch in batches:
                output.extend(self._nli_batch_or_skip(batch, repair_round=repair_round))
                if len(output) >= limit:
                    return output[:limit]
            return output[:limit]

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for wave_start in range(0, len(batches), concurrency):
                wave = batches[wave_start : wave_start + concurrency]
                for result in executor.map(
                    lambda batch: self._nli_batch_or_skip(batch, repair_round=repair_round), wave
                ):
                    output.extend(result)
                if len(output) >= limit:
                    return output[:limit]
        return output[:limit]

    def _nli_batch_or_skip(self, batch, *, repair_round: int = 0) -> list[dict[str, str]]:
        """Same idea as _simplify_batch_or_skip but for NLI batches."""
        try:
            return self._nli_batch(batch, repair_round=repair_round)
        except ProviderQuotaExhausted:
            raise
        except RuntimeError as exc:
            ids = [str(row.get("source_id")) for row in batch]
            logger.warning(
                "Skipping unrecoverable synthetic-NLI batch (%d items, ids=%s): %s",
                len(batch),
                ids,
                exc,
            )
            return []

    def assess_simplifications(
        self, rows: Iterable[dict[str, str]], limit: int
    ) -> dict[str, SimplificationQualityItem]:
        output: dict[str, SimplificationQualityItem] = {}
        batch: list[dict[str, str]] = []

        def store(items, requested_batch):
            items = list(items)
            if len(requested_batch) == 1 and len(items) == 1:
                requested_id = requested_batch[0]["source_id"]
                item = items[0]
                if item.id != requested_id:
                    logger.warning(
                        "Rebound singleton simplification judge id returned=%s requested=%s",
                        item.id,
                        requested_id,
                    )
                    item = item.model_copy(update={"id": requested_id})
                output[requested_id] = item
                return
            for item in items:
                output[item.id] = item

        for row in rows:
            batch.append(row)
            if len(batch) >= self._judge_batch_size:
                current = batch
                store(self._quality_batch(current), current)
                batch = []
                if len(output) >= limit:
                    break
        if batch and len(output) < limit:
            store(self._quality_batch(batch), batch)
        return output

    def assess_risk_labels(self, rows: Iterable[dict[str, str]], limit: int) -> dict[str, RiskLabelReviewItem]:
        """Independently validate FLB's provisional risk-class mapping."""
        output: dict[str, RiskLabelReviewItem] = {}
        batch: list[dict[str, str]] = []

        def store(items, requested_batch):
            items = list(items)
            if len(requested_batch) == 1 and len(items) == 1:
                requested_id = requested_batch[0]["source_id"]
                item = items[0]
                if item.id != requested_id:
                    logger.warning(
                        "Rebound singleton risk judge id returned=%s requested=%s",
                        item.id,
                        requested_id,
                    )
                    item = item.model_copy(update={"id": requested_id})
                output[requested_id] = item
                return
            for item in items:
                output[item.id] = item

        for row in rows:
            batch.append(row)
            if len(batch) >= self._judge_batch_size:
                current = batch
                store(self._risk_batch(current), current)
                batch = []
                if len(output) >= limit:
                    break
        if batch and len(output) < limit:
            store(self._risk_batch(batch), batch)
        return output

    @property
    def _judge_batch_size(self) -> int:
        return 1 if self.judge_provider_name in {"local_qwen", "ollama"} else self.settings.silver_label_batch_size

    def _validated_response(
        self,
        schema: type[T],
        *,
        model: str,
        instructions: str,
        input_text: str,
        provider_override=None,
        provider_name: str | None = None,
    ) -> T:
        cache_path = self._cache_path(schema, model, instructions, input_text)
        if cache_path is not None:
            with self._cache_lock:
                cached_now = cache_path.exists()
                if cached_now:
                    try:
                        parsed = schema.model_validate_json(cache_path.read_text(encoding="utf-8"))
                        self.cached_labels_reused += len(getattr(parsed, "items", []))
                        return parsed
                    except (OSError, ValidationError, json.JSONDecodeError) as exc:
                        logger.warning("Ignoring invalid silver-label cache entry %s: %s", cache_path, exc)
        last_error: Exception | None = None
        selected_provider_name = provider_name or (
            self.settings.silver_label_provider if schema is SimplificationBatch else self.judge_provider_name
        )
        attempts = 1 if selected_provider_name == "local_qwen" else max(1, self.settings.groq_max_retries)
        seed = int(hashlib.sha256(input_text.encode("utf-8")).hexdigest()[:8], 16)

        for attempt in range(1, attempts + 1):
            try:
                provider = provider_override or (
                    self.provider if schema is SimplificationBatch else self.judge_provider
                )
                logger.info(
                    "Silver-label execution schema=%s task=%s",
                    schema.__name__,
                    "simplification" if schema is SimplificationBatch else "judge",
                )
                provider_result = provider.generate(
                    schema=schema,
                    system_prompt=instructions,
                    user_prompt=input_text,
                )
            except Exception as exc:
                status = getattr(
                    getattr(exc, "response", None),
                    "status_code",
                    None,
                )

                if status == 429:
                    raise ProviderQuotaExhausted(
                        f"{selected_provider_name} quota exhausted; cached batches remain intact.",
                        _retry_after_seconds(exc),
                    ) from exc

                last_error = exc

                if attempt >= attempts:
                    break

                _bounded_backoff(attempt, seed)
                continue

            parsed = provider_result.parsed

            if parsed is not None:
                if cache_path is not None:
                    temporary = cache_path.with_suffix(f".{threading.get_ident()}.tmp")
                    with self._cache_lock:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        temporary.write_text(
                            parsed.model_dump_json(),
                            encoding="utf-8",
                        )
                        temporary.replace(cache_path)

                self.new_labels_generated += len(getattr(parsed, "items", []))

                logger.info(
                    "Silver-label provider=%s model=%s malformed_attempts=%d",
                    provider_result.metadata.get("provider"),
                    provider_result.metadata.get("model"),
                    provider_result.malformed_attempts,
                )

                return parsed

            last_error = RuntimeError(
                f"Provider returned no valid structured response. metadata={provider_result.metadata}"
            )

            logger.warning(
                "Invalid structured response attempt %d/%d using provider=%s",
                attempt,
                attempts,
                selected_provider_name,
            )

            if attempt < attempts:
                _bounded_backoff(attempt, seed)

        raise RuntimeError(f"Invalid structured response after retries: {last_error}")

    def _cache_path(self, schema: type[T], model: str, instructions: str, input_text: str):
        directory = self.settings.silver_label_cache_dir
        if directory is None:
            return None
        payload = "\0".join((schema.__name__, model, instructions, input_text))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.settings.resolve(directory) / schema.__name__ / f"{digest}.json"

    def _simplify_batch(self, batch, *, alternate: bool):
        payload = [{"id": row["source_id"], "clause": row["text"]} for row in batch]
        variation = (
            "Use a different sentence structure from a literal rewrite while preserving every legal fact. "
            if alternate
            else "Prefer direct consumer wording and short sentences. "
        )
        parsed = self._validated_response(
            SimplificationBatch,
            model=self.model,
            instructions=(
                "Create faithful plain-English legal/financial references. "
                + variation
                + "Preserve all numbers, dates, conditions, exceptions, parties, obligations and negations. Return JSON only."
            ),
            input_text=(
                'Return exactly {"items":[{"id":"...","simplified":"..."}]}. '
                "Target Flesch-Kincaid grade 8 or lower.\n\n" + json.dumps(payload, ensure_ascii=False)
            ),
        )
        by_id = {item.id: normalise_text(item.simplified) for item in parsed.items}
        output = []
        for row in batch:
            target = by_id.get(row["source_id"], "")
            if not target or not _critical_tokens_preserved(row["text"], target):
                self.rejected_labels += 1
                logger.warning("Rejected silver label %s because critical numbers/negations changed", row["source_id"])
                continue
            grade = flesch_kincaid_grade(target)
            output.append(
                SimplificationPair(
                    row["source_id"],
                    row.get("document_id", row["source_id"]),
                    row["text"],
                    target,
                    row.get("dataset", "unknown"),
                    grade,
                )
            )
        return output

    def _quality_batch(self, batch: list[dict[str, str]]) -> list[SimplificationQualityItem]:
        payload = [{"id": row["source_id"], "source": row["source"], "candidate": row["candidate"]} for row in batch]
        parsed = self._validated_response(
            SimplificationQualityBatch,
            model=self.model,
            instructions=(
                "Act as a strict legal-financial simplification reviewer. Check every amount, date, party, obligation, "
                "condition, exception and negation. Use this exact 1-to-5 scale, where higher is always better: "
                "1=very poor, 2=poor, 3=acceptable, 4=good, 5=excellent. "
                "Semantic accuracy 5 means no material change or omission. "
                "Readability 5 means clear consumer language. "
                "Do not copy example score values; assign scores based on the clause and candidate. "
                "Never echo or return source or candidate text. Return only id, scores, faithful, and a short rationale."
            ),
            input_text=(
                'Return exactly {"items":[{"id":"...","semantic_accuracy":5,"readability":5,'
                '"faithful":true,"rationale":"brief"}]}. Scores must reflect your assessment; '
                "5 is best and 1 is worst.\n\n" + json.dumps(payload, ensure_ascii=False)
            ),
        )
        return parsed.items

    def _risk_batch(self, batch: list[dict[str, str]]) -> list[RiskLabelReviewItem]:
        labels = self.settings.configured_risk_labels
        safe_label = self.settings.safe_risk_label
        allowed_labels = ", ".join(labels)

        payload = [
            {
                "id": row["source_id"],
                "clause": row["text"],
                "proposed_label": row["risk_label"],
            }
            for row in batch
        ]

        example = {
            "items": [
                {
                    "id": "...",
                    "correct_label": safe_label,
                    "confidence": 5,
                    "rationale": "brief",
                }
            ]
        }

        parsed = self._validated_response(
            RiskLabelReviewBatch,
            model=self.model,
            instructions=(
                "Classify each contract clause into exactly one configured "
                f"FinLingo++ consumer-risk class: {allowed_labels}. "
                f"{safe_label} means none of the other configured risks is "
                "materially present. Judge the clause itself, not the proposed "
                "label. Use this confidence scale, where higher is always more "
                "confident: 1=very uncertain, 2=uncertain, 3=moderate, "
                "4=confident, 5=highly confident. Do not copy the example "
                "confidence value. Return JSON only."
            ),
            input_text=(
                "Return exactly "
                + json.dumps(example, ensure_ascii=False)
                + ". Confidence must reflect your assessment; 5 is highest "
                "and 1 is lowest. Never echo or return the clause text. "
                "Return only id, correct_label, confidence, and a short "
                "rationale.\n\n" + json.dumps(payload, ensure_ascii=False)
            ),
        )

        allowed = {_normalise_risk_label(label).replace(" ", ""): label for label in labels}

        for item in parsed.items:
            normalized = _normalise_risk_label(item.correct_label).replace(" ", "")
            canonical = allowed.get(normalized)
            if canonical is None:
                raise ValueError(f"Risk judge returned an unconfigured label: {item.correct_label!r}")
            item.correct_label = canonical

        return parsed.items

    def _nli_batch(self, batch, *, repair_round: int = 0):
        payload = [{"source_id": row["source_id"], "premise": row["text"]} for row in batch]
        instructions = (
            "Generate controlled financial/legal NLI hypotheses. Per premise: one entailment, one neutral, one "
            "contradiction. Contradictions must materially change amount, date, party, condition, exception, "
            "obligation, or negation. Return JSON only."
        )
        if repair_round > 0:
            # need to change the prompt each round, otherwise temp-0 just replays the same incomplete cached output
            instructions += (
                f" (Repair attempt {repair_round}: a previous attempt omitted one of the three required "
                "labels for this premise. You MUST return exactly three items per premise: exactly one "
                "entailment, one neutral, and one contradiction — no fewer, no duplicates.)"
            )
        parsed = self._validated_response(
            SyntheticNLIBatch,
            model=self.model,
            instructions=instructions,
            input_text=(
                'Return exactly {"items":[{"source_id":"...","hypothesis":"...",'
                '"label":"entailment|neutral|contradiction"}]}.\n\n' + json.dumps(payload, ensure_ascii=False)
            ),
        )
        premise = {row["source_id"]: row["text"] for row in batch}
        return [
            {
                "source_id": item.source_id,
                "premise": premise[item.source_id],
                "hypothesis": normalise_text(item.hypothesis),
                "label": item.label,
                "evidence_chunk_id": None,
                "dataset": "finlingo-synthetic",
            }
            for item in parsed.items
            if item.source_id in premise
        ]


def _normalise_number_token(token: str) -> str:
    """Strip currency/percent symbols and punctuation so "1.5%" and "1.5" compare equal."""
    return re.sub(r"[€£$%,\s]", "", token).rstrip(".")


def _critical_tokens_preserved(source: str, target: str) -> bool:
    source_numbers = {_normalise_number_token(t) for t in re.findall(r"(?:€|£|\$)?\d[\d,.%]*", source)}
    target_numbers = {_normalise_number_token(t) for t in re.findall(r"(?:€|£|\$)?\d[\d,.%]*", target)}
    # ok to drop an incidental number (cross-refs etc) but never invent one that isn't in the source
    if not target_numbers.issubset(source_numbers):
        return False
    source_neg = bool(re.search(r"\b(no|not|never|without|unless)\b", source, re.I))
    target_neg = bool(re.search(r"\b(no|not|never|without|unless)\b", target, re.I))
    return source_neg == target_neg


def _bounded_backoff(attempt: int, seed: int) -> None:
    jitter = random.Random(seed + attempt).uniform(0.0, 0.25)
    time.sleep(min(2 ** (attempt - 1), 8.0) + jitter)


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw:
        try:
            return max(float(raw), 0.0)
        except (TypeError, ValueError):
            pass
    message = str(exc)
    match = re.search(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)s", message, re.I)
    if match:
        return 60 * float(match.group(1) or 0) + float(match.group(2))
    return None
