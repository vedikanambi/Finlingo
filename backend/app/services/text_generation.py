from __future__ import annotations

import json
import re
from typing import Any

import httpx

from backend.app.core.config import Settings
from backend.app.services.model_registry import ModelRegistry

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class TextGenerator:
    def __init__(self, settings: Settings, registry: ModelRegistry) -> None:
        self.settings = settings
        self.registry = registry

    def generate(
        self,
        *,
        instructions: str,
        user_input: str,
        model_id: str,
        adapter_id: str | None,
        purpose: str,
        max_new_tokens: int,
        temperature: float,
        require_adapter: bool = True,
        max_input_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        backend = self.settings.backend_for(purpose)
        if backend == "ollama":
            return self._ollama(instructions, user_input, purpose, max_new_tokens, temperature)
        if backend == "hf_endpoint":
            return self._endpoint(
                instructions,
                user_input,
                model_id,
                adapter_id,
                purpose,
                max_new_tokens,
                temperature,
            )
        return self._local(
            instructions,
            user_input,
            model_id,
            adapter_id,
            purpose,
            max_new_tokens,
            temperature,
            require_adapter,
            max_input_tokens,
            seed,
        )

    def _ollama(
        self,
        instructions: str,
        user_input: str,
        purpose: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        from backend.app.services.ollama_runtime import OllamaRuntime

        runtime = OllamaRuntime(self.settings)
        # only force json mode for the structured-output stage, free text stages shouldn't be constrained
        json_mode = purpose.startswith("S5")
        return runtime.chat(
            instructions=instructions,
            user_input=user_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )

    def _local(
        self,
        instructions: str,
        user_input: str,
        model_id: str,
        adapter_id: str | None,
        purpose: str,
        max_new_tokens: int,
        temperature: float,
        require_adapter: bool,
        max_input_tokens: int | None,
        seed: int | None = None,
    ) -> str:
        import torch

        if seed is not None:
            torch.manual_seed(seed)

        handle = self.registry.causal_lm(model_id, adapter_id, purpose, require_adapter=require_adapter)
        messages = [{"role": "system", "content": instructions}, {"role": "user", "content": user_input}]
        prompt = (
            handle.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if hasattr(handle.tokenizer, "apply_chat_template")
            else f"System: {instructions}\nUser: {user_input}\nAssistant:"
        )
        inputs = handle.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens or self.settings.simplifier_max_input_tokens,
        )
        device = next(handle.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "num_beams": 1,
            "use_cache": True,
            "pad_token_id": handle.tokenizer.pad_token_id,
            "eos_token_id": handle.tokenizer.eos_token_id,
        }
        if temperature > 0:
            kwargs.update(temperature=max(temperature, 1e-5), top_p=0.9)
        with torch.inference_mode():
            output = handle.model.generate(**inputs, **kwargs)
        generated = output[0, inputs["input_ids"].shape[1] :]
        return handle.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _endpoint(
        self,
        instructions: str,
        user_input: str,
        model_id: str,
        adapter_id: str | None,
        purpose: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        endpoint = self.settings.endpoint_for(purpose)
        if not endpoint or not self.settings.hf_token:
            raise RuntimeError(
                f"A task-specific Hugging Face endpoint and HF_TOKEN are required for {purpose}. "
                "Set HF_SIMPLIFIER_ENDPOINT_URL for S2 and HF_RISK_ENDPOINT_URL for S5."
            )
        payload: dict[str, Any] = {
            "inputs": f"<s>[INST] {instructions}\n\n{user_input} [/INST]",
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": max(temperature, 0.01),
                "do_sample": temperature > 0,
                "return_full_text": False,
            },
        }
        headers = {"Authorization": f"Bearer {self.settings.hf_token}"}
        if self.settings.hf_endpoint_supports_routing:
            payload["finlingo_route"] = {
                "purpose": purpose,
                "model_id": model_id,
                "adapter_id": adapter_id,
            }
            headers.update(
                {
                    "X-FinLingo-Purpose": purpose,
                    "X-FinLingo-Model": model_id,
                    "X-FinLingo-Adapter": adapter_id or "",
                }
            )
        response = httpx.post(endpoint, headers=headers, json=payload, timeout=180.0)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return str(data[0]["generated_text"]).strip()
        if isinstance(data, dict) and "generated_text" in data:
            return str(data["generated_text"]).strip()
        raise RuntimeError(f"Unexpected Hugging Face endpoint response: {type(data).__name__}")


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        match = _JSON_RE.search(candidate)
        if not match:
            raise ValueError("Model output did not contain a JSON object")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Model output JSON must be an object")
    return value
