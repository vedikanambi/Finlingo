"""Local Ollama runtime for S5 generation - keeps a second big HF model off the GPU."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from backend.app.core.config import Settings

logger = logging.getLogger(__name__)


class OllamaRuntimeError(RuntimeError):
    pass


class OllamaRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.ollama_base_url.rstrip("/")
        self._resolved_tags: dict[str, str] = {}

    def health(self) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=self.settings.ollama_health_timeout_seconds) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                tags = [m.get("name", "") for m in response.json().get("models", [])]
        except Exception as exc:
            return {"ok": False, "error": str(exc), "base_url": self.base_url}
        wanted = self.settings.ollama_generation_model_resolved
        resolved = self._match_tag(wanted, tags)
        return {
            "ok": resolved is not None,
            "base_url": self.base_url,
            "requested_model": wanted,
            "resolved_model": resolved,
            "available_models": tags,
            "error": None if resolved else f"Model '{wanted}' not pulled. Run: ollama pull {wanted}",
        }

    @staticmethod
    def _match_tag(wanted: str, tags: list[str]) -> str | None:
        if wanted in tags:
            return wanted
        family = wanted.split(":")[0]
        prefixed = [t for t in tags if t.startswith(wanted)]
        if prefixed:
            return sorted(prefixed)[0]
        same_family = [t for t in tags if t.split(":")[0] == family]
        return sorted(same_family)[0] if same_family else None

    def resolve_model(self, wanted: str | None = None) -> str:
        wanted = wanted or self.settings.ollama_generation_model_resolved
        if wanted in self._resolved_tags:
            return self._resolved_tags[wanted]
        report = self.health()
        if not report["ok"]:
            raise OllamaRuntimeError(
                f"Ollama unavailable for generation: {report['error']} "
                f"(base_url={self.base_url}). Start `ollama serve` and pull the model."
            )
        self._resolved_tags[wanted] = report["resolved_model"]
        return report["resolved_model"]

    def chat(
        self,
        *,
        instructions: str,
        user_input: str,
        max_new_tokens: int,
        temperature: float,
        json_mode: bool | None = None,
        retries: int = 2,
    ) -> str:
        model = self.resolve_model()
        json_mode = self.settings.ollama_generation_json_mode if json_mode is None else json_mode
        payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "keep_alive": self.settings.ollama_keep_alive,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_input},
            ],
            "options": {
                "temperature": max(temperature, 0.0),
                "num_predict": max_new_tokens,
                "num_ctx": self.settings.ollama_num_ctx,
                "seed": self.settings.random_seed,
            },
        }
        if json_mode:
            payload["format"] = "json"
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                with httpx.Client(timeout=self.settings.ollama_timeout_seconds) as client:
                    response = client.post(f"{self.base_url}/api/chat", json=payload)
                    response.raise_for_status()
                    content = response.json().get("message", {}).get("content", "")
                if content.strip():
                    return content.strip()
                last_error = OllamaRuntimeError("Ollama returned an empty completion")
            except Exception as exc:
                last_error = exc
                logger.warning("Ollama chat attempt %d failed: %s", attempt + 1, exc)
        raise OllamaRuntimeError(f"Ollama chat failed after {retries + 1} attempts: {last_error}")

    def provenance_id(self) -> str:
        try:
            return f"ollama::{self.resolve_model()}"
        except OllamaRuntimeError:
            return f"ollama::{self.settings.ollama_generation_model_resolved}(unresolved)"
