from __future__ import annotations
import logging
import re

import json
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from backend.app.core.config import Settings
from backend.app.services.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class ProviderResult:
    parsed: BaseModel | None
    malformed_attempts: int
    metadata: dict[str, Any]


class SilverLabelProvider(Protocol):
    def generate(self, schema: type[T], system_prompt: str, user_prompt: str) -> ProviderResult: ...


class OllamaProvider:
    """Local Ollama provider using structured JSON responses."""

    def __init__(self, settings, registry=None):
        self.settings = settings
        self.registry = registry
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model
        self.timeout_seconds = settings.ollama_timeout_seconds

    def generate(
        self,
        *,
        schema,
        system_prompt: str,
        user_prompt: str,
    ):
        payload = {
            "model": self.model,
            "stream": False,
            "format": schema.model_json_schema(),
            "keep_alive": self.settings.ollama_keep_alive,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "options": {
                "temperature": self.settings.ollama_temperature,
                "num_ctx": self.settings.ollama_num_ctx,
                "num_predict": self.settings.ollama_num_predict,
            },
        }

        response = self._post_with_retries(payload)
        response.raise_for_status()

        response_data = response.json()
        content = response_data["message"]["content"]

        malformed_attempts = 0
        parsed = None

        try:
            if isinstance(content, str):
                parsed = _parse_structured_response(content, schema)
            else:
                parsed = schema.model_validate(content)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
            malformed_attempts = 1

        return SimpleNamespace(
            parsed=parsed,
            malformed_attempts=malformed_attempts,
            metadata={
                "provider": "ollama",
                "model": self.model,
                "prompt_eval_count": response_data.get("prompt_eval_count", 0),
                "eval_count": response_data.get("eval_count", 0),
            },
        )

    def _post_with_retries(self, payload: dict) -> httpx.Response:
        # retry on connection blips (ollama restarting mid-request) with backoff, then give up and let the caller skip this batch
        last_error: Exception | None = None
        attempts = max(self.settings.ollama_connection_retries, 0) + 1
        for attempt in range(attempts):
            try:
                return httpx.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                last_error = exc
                if attempt < attempts - 1:
                    delay = self.settings.ollama_connection_retry_backoff_seconds * (2**attempt)
                    logger.warning(
                        "Ollama connection error (attempt %d/%d): %s. Retrying in %.1fs.",
                        attempt + 1,
                        attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
        raise RuntimeError(f"Ollama connection failed after {attempts} attempts: {last_error}")


def build_silver_label_provider(
    settings: Settings,
    registry: ModelRegistry | None = None,
    *,
    task: str = "simplification",
    provider_name: str | None = None,
) -> SilverLabelProvider:
    selected = provider_name or settings.silver_label_provider

    if selected == "ollama":
        return OllamaProvider(settings, registry)

    registry = registry or ModelRegistry(settings)
    if selected == "local_qwen":
        return LocalQwenProvider(settings, registry, task=task)
    if selected == "gemini":
        return GeminiProvider(settings)
    if selected == "groq":
        return GroqProvider(settings, registry)
    raise ValueError(f"Unsupported silver-label provider: {selected}")


class LocalQwenProvider:
    def __init__(self, settings: Settings, registry: ModelRegistry, *, task: str = "simplification") -> None:
        self.settings = settings
        self.registry = registry
        self.task = task
        self._simplifier_handle = None
        self._judge_handle = None

    def _model(self, schema: type[T] | None = None):
        """Simplification uses the S2 adapter; judging uses the plain judge model."""
        schema_name = schema.__name__ if schema is not None else None

        is_simplification = schema_name == "SimplificationBatch" or (schema is None and self.task == "simplification")

        if is_simplification:
            if self._simplifier_handle is None:
                adapter = self.settings.simplifier_adapter or None
                model_name = self.settings.silver_local_model

                if adapter:
                    adapter_config_path = Path(adapter) / "adapter_config.json"
                    if not adapter_config_path.exists():
                        raise RuntimeError(f"Adapter config not found: {adapter_config_path}")

                    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
                    model_name = adapter_config.get("base_model_name_or_path")
                    if not model_name:
                        raise RuntimeError("adapter_config.json does not contain base_model_name_or_path")

                logger.info(
                    "Loading local simplifier handle schema=%s task=%s model=%s adapter=%s",
                    schema_name or "SimplificationBatch",
                    self.task,
                    model_name,
                    adapter,
                )
                self._simplifier_handle = self.registry.causal_lm(
                    model_name,
                    adapter,
                    f"local simplification inference ({self.task})",
                    require_adapter=adapter is not None,
                )

            return self._simplifier_handle

        judge_model = self.settings.silver_judge_model or self.settings.silver_local_model
        if self._judge_handle is None:
            logger.info(
                "Loading local judge handle schema=%s task=%s model=%s adapter=None",
                schema_name or "JudgeBatch",
                self.task,
                judge_model,
            )
            self._judge_handle = self.registry.causal_lm(
                judge_model,
                None,
                f"local quality and risk judge ({self.task})",
                require_adapter=False,
            )

        return self._judge_handle

    def generate(self, schema: type[T], system_prompt: str, user_prompt: str) -> ProviderResult:
        import torch

        handle = self._model(schema)

        schema_name = getattr(schema, "__name__", "")

        if schema_name == "SimplificationQualityBatch":
            system_prompt = (
                system_prompt.rstrip()
                + """
            
STRICT OUTPUT CONTRACT:
Return only one valid JSON object in exactly this form:
{
  "items": [
    {
      "id": "<copy the supplied id exactly>",
      "semantic_accuracy": <integer from 1 to 5>,
      "readability": <integer from 1 to 5>,
      "faithful": <true or false>,
      "rationale": "<short explanation>"
    }
  ]
}

Every object inside items must contain exactly:
id, semantic_accuracy, readability, faithful, rationale.

Do not return source.
Do not return candidate.
Do not return scores.
Do not return reasoning.
Do not place scores or faithful at the top level.
Do not reproduce the input clause or simplification.
Do not add markdown or JSON fences.
"""
            )

        elif schema_name in {"RiskLabelReviewBatch", "RiskAssessmentBatch"}:
            system_prompt = (
                system_prompt.rstrip()
                + """
            
STRICT OUTPUT CONTRACT:
Return only one valid JSON object in exactly this form:
{
  "items": [
    {
      "id": "<copy the supplied id exactly>",
      "correct_label": "<one allowed risk label>",
      "confidence": <integer from 1 to 5>,
      "rationale": "<short explanation>"
    }
  ]
}

Every object inside items must contain exactly:
id, correct_label, confidence, rationale.

Do not return source.
Do not return candidate.
Do not return scores.
Do not return reasoning.
Do not place fields outside items.
Do not reproduce the input clause.
Do not add markdown or JSON fences.
"""
            )
        malformed = 0
        started = time.perf_counter()
        prompt = user_prompt
        parsed: BaseModel | None = None
        raw = ""
        last_parse_error: str | None = None
        for attempt in range(self.settings.silver_local_malformed_retries + 1):
            torch.manual_seed(self.settings.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.settings.random_seed)
                torch.cuda.reset_peak_memory_stats()
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            encoded = handle.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
            )
            encoded = {key: value.to(handle.model.device) for key, value in encoded.items()}
            kwargs: dict[str, Any] = {
                "max_new_tokens": self.settings.silver_local_max_new_tokens,
                "do_sample": self.settings.silver_local_do_sample,
                "pad_token_id": handle.tokenizer.pad_token_id,
            }
            if self.settings.silver_local_do_sample:
                kwargs["temperature"] = self.settings.silver_local_temperature
            with torch.inference_mode():
                generated = handle.model.generate(**encoded, **kwargs)
            new_tokens = generated[0, encoded["input_ids"].shape[1] :]
            raw = handle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            try:
                parsed = _parse_structured_response(raw, schema)
                break
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                recovered = _recover_single_plain_simplification(
                    raw=raw,
                    schema=schema,
                    user_prompt=user_prompt,
                )
                if recovered is not None:
                    parsed = recovered
                    logger.warning(
                        "Recovered single plain-text simplification as structured output schema=%s",
                        schema_name,
                    )
                    break

                malformed += 1
                last_parse_error = f"{type(exc).__name__}: {exc}"
                prompt = (
                    "Repair the malformed answer below. Return exactly one valid JSON object only, "
                    "with the requested schema and no markdown or explanation.\n\n"
                    "Malformed answer:\n" + raw[:4000] + "\n\nOriginal request:\n" + user_prompt
                )
        precision = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16"
        return ProviderResult(
            parsed,
            malformed,
            {
                "provider": "local_qwen",
                "model": handle.model_id,
                "adapter": handle.adapter_id,
                "task": self.task,
                "revision": self.settings.model_revision(handle.model_id),
                "device": handle.device,
                "precision": f"4bit_nf4_{precision}" if self.settings.silver_local_use_4bit else precision,
                "generation_parameters": {
                    "max_new_tokens": self.settings.silver_local_max_new_tokens,
                    "temperature": self.settings.silver_local_temperature,
                    "do_sample": self.settings.silver_local_do_sample,
                    "seed": self.settings.random_seed,
                },
                "generation_seconds": time.perf_counter() - started,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                "raw_output_length": len(raw),
                "raw_output_preview": raw[:2000],
                "last_parse_error": last_parse_error,
            },
        )


class GroqProvider:
    def __init__(self, settings: Settings, registry: ModelRegistry) -> None:
        self.settings = settings
        self.client = registry.groq_client()

    def generate(self, schema: type[T], system_prompt: str, user_prompt: str) -> ProviderResult:
        started = time.perf_counter()
        completion = self.client.chat.completions.create(
            model=self.settings.groq_silver_label_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or ""
        try:
            parsed = schema.model_validate_json(raw)
            malformed = 0
        except (ValidationError, ValueError):
            parsed, malformed = None, 1
        return ProviderResult(
            parsed,
            malformed,
            {
                "provider": "groq",
                "model": self.settings.groq_silver_label_model,
                "revision": None,
                "device": "remote",
                "precision": "provider_managed",
                "generation_parameters": {"temperature": 0.0},
                "generation_seconds": time.perf_counter() - started,
            },
        )


class GeminiProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(self, schema: type[T], system_prompt: str, user_prompt: str) -> ProviderResult:

        if not self.settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required when SILVER_LABEL_PROVIDER=gemini")
        started = time.perf_counter()
        url = f"{self.settings.gemini_api_base_url}/models/{self.settings.gemini_silver_label_model}:generateContent"
        response = httpx.post(
            url,
            params={"key": self.settings.gemini_api_key},
            json={
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"parts": [{"text": user_prompt}]}],
                "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
            },
            timeout=self.settings.regulatory_request_timeout,
        )
        response.raise_for_status()
        raw = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        try:
            parsed = schema.model_validate_json(raw)
            malformed = 0
        except (ValidationError, ValueError):
            parsed, malformed = None, 1
        return ProviderResult(
            parsed,
            malformed,
            {
                "provider": "gemini",
                "model": self.settings.gemini_silver_label_model,
                "revision": None,
                "device": "remote",
                "precision": "provider_managed",
                "generation_parameters": {"temperature": 0.0},
                "generation_seconds": time.perf_counter() - started,
            },
        )


def _recover_single_plain_simplification(
    *,
    raw: str,
    schema: type[T],
    user_prompt: str,
):
    """Recover a plain-text simplification the model returned without its JSON envelope."""
    if getattr(schema, "__name__", "") != "SimplificationBatch":
        return None

    value = str(raw or "").strip()
    if not value or value.startswith(("{", "[", "```")):
        return None

    encoded_ids = re.findall(
        r'"id"\s*:\s*"((?:\\.|[^"\\])*)"',
        user_prompt,
    )
    requested_ids: list[str] = []
    for encoded_id in encoded_ids:
        try:
            decoded_id = json.loads(f'"{encoded_id}"')
        except json.JSONDecodeError:
            continue
        if decoded_id and decoded_id != "...":
            requested_ids.append(str(decoded_id))

    unique_ids = list(dict.fromkeys(requested_ids))
    if len(unique_ids) != 1:
        return None

    try:
        return schema.model_validate(
            {
                "items": [
                    {
                        "id": unique_ids[0],
                        "simplified": value,
                    }
                ]
            }
        )
    except ValidationError:
        return None


def _strip_json_fence(raw: str) -> str:
    """Strip a Markdown code fence if present."""
    value = raw.strip()

    if value.startswith("```"):
        first_newline = value.find("\n")
        if first_newline >= 0:
            value = value[first_newline + 1 :]

        if value.rstrip().endswith("```"):
            value = value.rstrip()[:-3]

    return value.strip()


def _extract_json_value(raw: str):
    """Parse the first complete JSON object or array from model output, tolerating fences and stray text."""
    cleaned = _strip_json_fence(raw)

    decoder = json.JSONDecoder()

    starts = [index for index, character in enumerate(cleaned) if character in ("{", "[")]

    last_error: Exception | None = None

    for start in starts:
        try:
            value, _ = decoder.raw_decode(cleaned[start:])
            return value
        except json.JSONDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise ValueError(f"No complete JSON value in local model output: {last_error}") from last_error

    raise ValueError("No JSON object or array in local model output")


def _parse_structured_response(raw: str, schema: type[T]):
    """Parse local-model output, normalizing the various batch-envelope shapes models return before validating."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Structured response is empty.")

    candidate = raw.strip()

    if candidate.startswith("```"):
        lines = candidate.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        candidate = "\n".join(lines).strip()

    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as whole_error:
        object_index = candidate.find("{")
        array_index = candidate.find("[")

        indexes = [index for index in (object_index, array_index) if index >= 0]

        if not indexes:
            raise ValueError(f"No JSON object or array found: {whole_error}") from whole_error

        json_start = min(indexes)
        decoder = json.JSONDecoder()

        try:
            value, _ = decoder.raw_decode(candidate[json_start:])
        except json.JSONDecodeError as embedded_error:
            raise ValueError(f"Invalid JSON response: {embedded_error}") from embedded_error

    schema_name = getattr(schema, "__name__", "")

    batch_schema_names = {
        "SimplificationBatch",
        "SimplificationQualityBatch",
        "RiskAssessmentBatch",
        "RiskLabelReviewBatch",
    }

    if schema_name in batch_schema_names:
        if isinstance(value, list):
            value = {"items": value}

        elif isinstance(value, dict):
            if "items" not in value:
                value = {"items": [value]}

        else:
            raise ValueError(f"{schema_name} requires a JSON object or array.")

        while (
            isinstance(value, dict)
            and isinstance(value.get("items"), list)
            and len(value["items"]) == 1
            and isinstance(value["items"][0], dict)
            and isinstance(value["items"][0].get("items"), list)
        ):
            value = {"items": value["items"][0]["items"]}

        items = value.get("items")

        if not isinstance(items, list):
            raise ValueError(f"{schema_name}.items must be a JSON array.")

        normalized_items = []

        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Every {schema_name} item must be a JSON object.")

            normalized_item = dict(item)

            # some models call it "clause" instead of "simplified" / "proposed_label" instead of "correct_label"
            if schema_name == "SimplificationBatch":
                if "simplified" not in normalized_item and "clause" in normalized_item:
                    normalized_item["simplified"] = normalized_item.pop("clause")

            if schema_name == "RiskLabelReviewBatch":
                if "correct_label" not in normalized_item and "proposed_label" in normalized_item:
                    normalized_item["correct_label"] = normalized_item.pop("proposed_label")

            normalized_items.append(normalized_item)

        value = {"items": normalized_items}

    return schema.model_validate(value)


def _json_object(raw: str) -> str:
    """Kept for backward compatibility with older callers/tests."""
    value = _extract_json_value(raw)

    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object in local model output")

    return json.dumps(value, ensure_ascii=False)
