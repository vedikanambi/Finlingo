from types import SimpleNamespace

import httpx
import pytest
import torch

from backend.app.core.config import Settings
from backend.training.silver_label_providers import (
    LocalQwenProvider,
    OllamaProvider,
    _parse_structured_response,
    build_silver_label_provider,
)
from backend.training.silver_labels import SimplificationBatch, SimplificationQualityBatch, RiskLabelReviewBatch


class _Tokenizer:
    pad_token_id = 0

    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def apply_chat_template(self, *args, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}

    def decode(self, *args, **kwargs):
        return next(self.outputs)


class _Model:
    device = torch.device("cpu")

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([[1, 2, 3]])


def _registry(outputs):
    model = _Model()
    handle = SimpleNamespace(
        tokenizer=_Tokenizer(outputs), model=model, device="cuda", model_id="Qwen/test", adapter_id=None
    )
    registry = SimpleNamespace(
        causal_lm=lambda *args, **kwargs: handle,
        groq_client=lambda: (_ for _ in ()).throw(AssertionError("Groq must not be called")),
    )
    return registry, model


def test_local_provider_selection_does_not_call_groq():
    registry, _ = _registry(['{"items":[]}'])
    provider = build_silver_label_provider(Settings(_env_file=None, silver_label_provider="local_qwen"), registry)
    assert isinstance(provider, LocalQwenProvider)


def test_local_generation_is_deterministic_and_repairs_malformed_json():
    registry, model = _registry(["not json", '{"items":[]}'])
    settings = Settings(
        _env_file=None,
        silver_label_provider="local_qwen",
        silver_local_do_sample=False,
        silver_local_malformed_retries=2,
        silver_local_max_new_tokens=17,
    )
    result = LocalQwenProvider(settings, registry).generate(SimplificationBatch, "system", "user")
    assert result.parsed == SimplificationBatch(items=[])
    assert result.malformed_attempts == 1
    assert len(model.calls) == 2
    assert all(call["do_sample"] is False for call in model.calls)
    assert all(call["max_new_tokens"] == 17 for call in model.calls)
    assert all("temperature" not in call for call in model.calls)
    assert result.metadata["provider"] == "local_qwen"
    assert result.metadata["generation_parameters"]["seed"] == settings.random_seed


def test_fenced_array_and_simplification_alias_are_schema_safe():
    parsed = _parse_structured_response('```json\n[{"id":"x","clause":"Plain text"}]\n```', SimplificationBatch)
    assert parsed.items[0].simplified == "Plain text"
    import pytest

    with pytest.raises(Exception):
        _parse_structured_response('[{"id":"x","clause":"Plain text"}]', SimplificationQualityBatch)


def test_stage2_adapter_is_only_used_for_simplification(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        '{"base_model_name_or_path":"Qwen/Qwen2.5-1.5B-Instruct"}', encoding="utf-8"
    )
    calls = []
    handle = SimpleNamespace(
        tokenizer=_Tokenizer(['{"items":[]}']),
        model=_Model(),
        device="cuda",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        adapter_id=str(adapter),
    )
    registry = SimpleNamespace(causal_lm=lambda *args, **kwargs: calls.append((args, kwargs)) or handle)
    settings = Settings(
        _env_file=None,
        project_root=tmp_path,
        simplifier_adapter=str(adapter),
        silver_local_model="Qwen/Qwen2.5-1.5B-Instruct",
        silver_judge_model="Qwen/Qwen2.5-1.5B-Instruct",
    )
    LocalQwenProvider(settings, registry, task="simplification")._model()
    LocalQwenProvider(settings, registry, task="judge")._model()
    assert calls[0][0][1] == str(adapter)
    assert calls[1][0][1] is None


def test_judge_generation_uses_configured_1024_tokens():
    registry, model = _registry(['{"items":[]}'])
    settings = Settings(_env_file=None, silver_local_max_new_tokens=1024)
    LocalQwenProvider(settings, registry, task="judge").generate(SimplificationQualityBatch, "system", "user")
    assert model.calls[0]["max_new_tokens"] == 1024


def test_quality_batch_accepts_singleton_object_response():
    registry, _ = _registry(
        [
            """```json
{
  "id": "row-1",
  "semantic_accuracy": 5,
  "readability": 5,
  "faithful": true,
  "rationale": "Meaning preserved."
}
```"""
        ]
    )

    settings = Settings(
        _env_file=None,
        silver_label_provider="local_qwen",
        silver_local_do_sample=False,
        silver_local_malformed_retries=1,
        silver_local_max_new_tokens=1024,
    )

    result = LocalQwenProvider(
        settings,
        registry,
        task="judge",
    ).generate(
        SimplificationQualityBatch,
        "system",
        "user",
    )

    assert result.parsed is not None
    assert len(result.parsed.items) == 1
    assert result.parsed.items[0].id == "row-1"
    assert result.parsed.items[0].semantic_accuracy == 5
    assert result.parsed.items[0].faithful is True


def test_quality_batch_rejects_ambiguous_top_level_scores():
    malformed_response = """{
      "items": [
        {
          "id": "row-1",
          "source": "Original text",
          "candidate": "Simplified text"
        }
      ],
      "scores": [3],
      "faithful": true,
      "reasoning": "Looks acceptable."
    }"""

    registry, _ = _registry(
        [
            malformed_response,
            malformed_response,
        ]
    )

    settings = Settings(
        _env_file=None,
        silver_label_provider="local_qwen",
        silver_local_do_sample=False,
        silver_local_malformed_retries=1,
        silver_local_max_new_tokens=1024,
    )

    result = LocalQwenProvider(
        settings,
        registry,
        task="judge",
    ).generate(
        SimplificationQualityBatch,
        "system",
        "user",
    )

    assert result.parsed is None
    assert result.malformed_attempts >= 1


def test_risk_review_accepts_proposed_label_alias():
    registry, _ = _registry(
        [
            """```json
[
  {
    "id": "row-1",
    "proposed_label": "Safe",
    "confidence": 5,
    "rationale": "No listed risk factor is present."
  }
]
```"""
        ]
    )

    settings = Settings(
        _env_file=None,
        silver_label_provider="local_qwen",
        silver_local_do_sample=False,
        silver_local_malformed_retries=1,
        silver_local_max_new_tokens=1024,
    )

    result = LocalQwenProvider(
        settings,
        registry,
        task="judge",
    ).generate(
        RiskLabelReviewBatch,
        "system",
        "user",
    )

    assert result.parsed is not None
    assert len(result.parsed.items) == 1
    assert result.parsed.items[0].correct_label == "Safe"
    assert result.parsed.items[0].confidence == 5


def test_ollama_provider_retries_transient_connection_error_mid_batch(monkeypatch):
    """ollama drops the connection when it restarts mid-request - retry with backoff instead of blowing up the whole run."""
    calls = {"n": 0}

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": '{"items":[]}'}, "prompt_eval_count": 1, "eval_count": 1}

    def fake_post(url, json, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("connection reset")
        return _Response()

    sleeps = []
    monkeypatch.setattr("backend.training.silver_label_providers.httpx.post", fake_post)
    monkeypatch.setattr("backend.training.silver_label_providers.time.sleep", lambda s: sleeps.append(s))

    settings = Settings(_env_file=None, ollama_connection_retries=3, ollama_connection_retry_backoff_seconds=1.0)
    result = OllamaProvider(settings).generate(schema=SimplificationBatch, system_prompt="s", user_prompt="u")

    assert calls["n"] == 3
    assert result.parsed == SimplificationBatch(items=[])
    assert len(sleeps) == 2
    assert sleeps == [1.0, 2.0]


def test_ollama_provider_raises_runtime_error_after_exhausting_connection_retries(monkeypatch):
    def fake_post(url, json, timeout):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr("backend.training.silver_label_providers.httpx.post", fake_post)
    monkeypatch.setattr("backend.training.silver_label_providers.time.sleep", lambda s: None)

    settings = Settings(_env_file=None, ollama_connection_retries=2, ollama_connection_retry_backoff_seconds=0.1)
    with pytest.raises(RuntimeError, match="Ollama connection failed after 3 attempts"):
        OllamaProvider(settings).generate(schema=SimplificationBatch, system_prompt="s", user_prompt="u")


def test_risk_review_unwraps_nested_items_batch():
    raw = """```json
[
  {
    "items": [
      {
        "id": "row-1",
        "correct_label": "Data Sharing",
        "confidence": 5,
        "rationale": "The clause explicitly describes data sharing."
      }
    ]
  }
]
```"""

    parsed = _parse_structured_response(
        raw,
        RiskLabelReviewBatch,
    )

    assert len(parsed.items) == 1
    assert parsed.items[0].id == "row-1"
    assert parsed.items[0].correct_label == "Data Sharing"
    assert parsed.items[0].confidence == 5
