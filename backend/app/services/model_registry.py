from __future__ import annotations

import logging
import gc
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class CausalLMHandle:
    tokenizer: Any
    model: Any
    device: str
    model_id: str
    adapter_id: str | None


@dataclass
class NLIHandle:
    tokenizer: Any
    model: Any
    device: str
    entailment_index: int
    contradiction_index: int | None
    neutral_index: int | None
    model_id: str
    adapter_id: str | None


class ModelRegistry:
    """Thread-safe lazy cache for executable model artefacts only."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._objects: dict[str, Any] = {}

    def _device(self) -> str:
        import torch

        requested = self.settings.device
        if requested != "auto":
            if requested == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("DEVICE=cuda requested but CUDA is unavailable")
            if requested == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("DEVICE=mps requested but MPS is unavailable")
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def sentence_encoder(self):
        key = f"senc::{self.settings.embedding_model}"
        with self._lock:
            if key not in self._objects:
                from sentence_transformers import SentenceTransformer

                device = self._device()
                revision = self.settings.model_revision(self.settings.embedding_model)
                kwargs: dict[str, Any] = {"device": device}
                if revision:
                    kwargs["revision"] = revision
                model = SentenceTransformer(self.settings.embedding_model, **kwargs)
                self._objects[key] = model
                logger.info("Loaded local embedding model %s on %s", self.settings.embedding_model, device)
            return self._objects[key]

    def groq_client(self):
        with self._lock:
            if "groq" not in self._objects:
                if not self.settings.groq_api_key:
                    raise RuntimeError(
                        "GROQ_API_KEY is required for silver-label generation and synthetic NLI. "
                        "Add it to .env or set the GROQ_API_KEY environment variable."
                    )
                from groq import Groq

                self._objects["groq"] = Groq(api_key=self.settings.groq_api_key, max_retries=0)
            return self._objects["groq"]

    def openai(self):
        with self._lock:
            if "openai" not in self._objects:
                if not self.settings.openai_api_key:
                    raise RuntimeError(
                        "OPENAI_API_KEY is required for the optional GPT-judge comparison. "
                        "If you do not need GPT judge, omit --include-gpt-judge."
                    )
                if self.settings.azure_endpoint:
                    from openai import AzureOpenAI

                    self._objects["openai"] = AzureOpenAI(
                        api_key=self.settings.openai_api_key,
                        azure_endpoint=self.settings.azure_endpoint,
                        api_version=self.settings.azure_api_version or "2024-12-01-preview",
                        max_retries=self.settings.openai_max_retries,
                    )
                else:
                    from openai import OpenAI

                    self._objects["openai"] = OpenAI(
                        api_key=self.settings.openai_api_key,
                        max_retries=self.settings.openai_max_retries,
                    )
            return self._objects["openai"]

    def causal_lm(
        self, model_id: str, adapter_id: str | None, purpose: str, *, require_adapter: bool = True
    ) -> CausalLMHandle:
        key = f"causal::{model_id}::{adapter_id or '-'}::{purpose}"
        with self._lock:
            if key in self._objects:
                return self._objects[key]
            if not adapter_id and require_adapter and not self.settings.allow_base_models:
                raise RuntimeError(
                    f"No trained adapter configured for {purpose}. Train it and set the corresponding *_ADAPTER, "
                    "or set ALLOW_BASE_MODELS=true only for a documented zero-shot baseline."
                )
            device = self._device()
            if device == "cpu":
                raise RuntimeError(
                    f"Local {model_id} execution on CPU is disabled. Use a CUDA/MPS machine or MODEL_BACKEND=hf_endpoint."
                )
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            token = self.settings.hf_token
            cache_dir = (
                str(self.settings.resolve(self.settings.model_cache_dir)) if self.settings.model_cache_dir else None
            )
            revision = self.settings.model_revision(model_id)
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, token=token, cache_dir=cache_dir, use_fast=True, revision=revision
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            kwargs: dict[str, Any] = {
                "token": token,
                "cache_dir": cache_dir,
                "device_map": "auto",
                "trust_remote_code": self.settings.model_trust_remote_code,
                "revision": revision,
            }
            if device == "cuda":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=self.settings.qlora_quant_type,
                    bnb_4bit_use_double_quant=self.settings.qlora_use_double_quant,
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                )
            else:
                kwargs["torch_dtype"] = torch.float16
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            if adapter_id:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, self._adapter_reference(adapter_id), token=token)
            if device != "cuda":
                model = model.to(device)
            model.eval()
            handle = CausalLMHandle(tokenizer, model, device, model_id, adapter_id)
            self._objects[key] = handle
            logger.info("Loaded %s model=%s adapter=%s", purpose, model_id, adapter_id)
            return handle

    def cross_encoder(self):
        key = f"cross::{self.settings.reranker_model}"
        with self._lock:
            if key not in self._objects:
                from sentence_transformers import CrossEncoder

                kwargs = {"device": self._device()}
                revision = self.settings.model_revision(self.settings.reranker_model)
                if revision:
                    kwargs["revision"] = revision
                self._objects[key] = CrossEncoder(
                    self.settings.reranker_model,
                    **kwargs,
                )
            return self._objects[key]

    def nli(
        self, model_id: str | None = None, adapter_id: str | None = None, *, require_adapter: bool | None = None
    ) -> NLIHandle:
        explicit_model = model_id is not None
        model_id = model_id or self.settings.verifier_model
        require_adapter = self.settings.verifier_require_adapter if require_adapter is None else require_adapter
        if adapter_id is None and not (explicit_model and require_adapter is False):
            adapter_id = self.settings.verifier_adapter
        key = f"nli::{model_id}::{adapter_id or '-'}::{int(require_adapter)}"
        with self._lock:
            if key in self._objects:
                return self._objects[key]
            if not adapter_id and require_adapter and not self.settings.allow_base_models:
                raise RuntimeError(
                    f"No trained verifier adapter configured for {model_id}. Set VERIFIER_ADAPTER (or the comparator "
                    "adapter), or explicitly disable VERIFIER_REQUIRE_ADAPTER only for a documented zero-shot baseline."
                )
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = self._device()
            token = self.settings.hf_token
            cache_dir = (
                str(self.settings.resolve(self.settings.model_cache_dir)) if self.settings.model_cache_dir else None
            )
            revision = self.settings.model_revision(model_id)
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=token,
                cache_dir=cache_dir,
                trust_remote_code=self.settings.model_trust_remote_code,
                revision=revision,
            )
            kwargs: dict[str, Any] = {
                "token": token,
                "cache_dir": cache_dir,
                "trust_remote_code": self.settings.model_trust_remote_code,
                "revision": revision,
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            }
            if device == "cuda":
                kwargs["device_map"] = "auto"
            if adapter_id:
                # label order has to match what the adapter was trained with or the saved head won't load right
                label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
                kwargs.update(
                    num_labels=3,
                    label2id=label2id,
                    id2label={value: key for key, value in label2id.items()},
                )
            model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
            if adapter_id:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, self._adapter_reference(adapter_id), token=token)
            elif (
                model_id in {"microsoft/deberta-v3-large", "answerdotai/ModernBERT-large"}
                and not self.settings.allow_base_models
            ):
                raise RuntimeError(f"{model_id} is a base checkpoint. Configure a trained NLI adapter.")
            if device != "cuda":
                model = model.to(device)
            model.eval()
            id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
            entail = next((i for i, label in id2label.items() if "entail" in label), None)
            if entail is None:
                raise RuntimeError(f"Model {model_id} has no entailment label in id2label={id2label}")
            contradiction = next((i for i, label in id2label.items() if "contrad" in label), None)
            neutral = next((i for i, label in id2label.items() if "neutral" in label), None)
            handle = NLIHandle(tokenizer, model, device, entail, contradiction, neutral, model_id, adapter_id)
            self._objects[key] = handle
            return handle

    def _adapter_reference(self, adapter_id: str) -> str:
        direct = Path(adapter_id).expanduser()
        if direct.exists():
            return str(direct.resolve())
        project_relative = self.settings.resolve(direct)
        if project_relative.exists():
            return str(project_relative)
        return adapter_id

    def release_causal(self, purpose: str) -> None:
        # frees the stage's model so two 7B copies don't sit in VRAM at once on a single GPU
        suffix = f"::{purpose}"
        with self._lock:
            keys = [key for key in self._objects if key.startswith("causal::") and key.endswith(suffix)]
            for key in keys:
                del self._objects[key]
        if keys:
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - defensive cleanup
                pass

    def release_causal_lm(
        self,
        model_id: str | None = None,
        adapter_id: str | None = None,
    ) -> None:
        with self._lock:
            keys = [
                key
                for key in self._objects
                if key.startswith("causal::")
                and (model_id is None or model_id in key)
                and (adapter_id is None or adapter_id in key)
            ]

            for key in keys:
                self._objects.pop(key, None)

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:  # pragma: no cover - defensive cleanup
            pass

    def release_nli(self, model_id: str | None = None, adapter_id: str | None = None) -> None:
        with self._lock:
            keys = [
                key
                for key in self._objects
                if key.startswith("nli::")
                and (model_id is None or f"nli::{model_id}::" in key)
                and (adapter_id is None or f"::{adapter_id}::" in key)
            ]
            for key in keys:
                del self._objects[key]
        if keys:
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
            except Exception:  # pragma: no cover
                pass

    def close(self) -> None:
        with self._lock:
            self._objects.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass
