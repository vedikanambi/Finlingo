"""Regression tests for a handful of preflight bugs that came back more than once."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.core.config import Settings
from backend.app.core.preflight import run_preflight
from backend.app.core.schemas import EvidenceChunk
from backend.app.services.model_registry import ModelRegistry
from backend.app.services.text_generation import TextGenerator
from backend.app.stages.stage5_risk_classifier import Stage5RiskClassifier


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = {"project_root": tmp_path, "_env_file": None}
    values.update(overrides)
    return Settings(**values)


def _write_risk_configs(tmp_path: Path) -> tuple[Path, Path]:
    taxonomy_path = tmp_path / "risk_taxonomy.yaml"
    taxonomy_path.write_text(
        json.dumps(
            {
                "labels": {
                    "Auto-Renewal": [],
                    "Hidden Fee": [],
                    "Liability Waiver": [],
                    "Data Sharing": [],
                    "Penalty Clause": [],
                },
                "safe_label": "Safe",
            }
        ),
        encoding="utf-8",
    )
    few_shot_path = tmp_path / "risk_few_shot.yaml"
    few_shot_path.write_text(
        json.dumps(
            {
                "risk_score_rubric": {1: "low", 5: "high"},
                "examples": [
                    {
                        "clause": "Example clause.",
                        "risk_label": "Safe",
                        "risk_score": 1,
                        "explanation": "No risk.",
                        "secondary_risk_labels": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return taxonomy_path, few_shot_path


def test_s2_ollama_backend_requires_no_adapter(tmp_path):
    taxonomy_path, few_shot_path = _write_risk_configs(tmp_path)
    settings = make_settings(
        tmp_path,
        s2_generation_backend="ollama",
        simplifier_adapter=None,
        allow_base_models=False,
        risk_taxonomy_path=taxonomy_path,
        risk_few_shot_path=few_shot_path,
        verifier_calibration_path=None,
    )
    result = run_preflight(settings, final=False)
    simplifier_check = next(c for c in result["checks"] if c["name"] == "simplifier_adapter")
    assert simplifier_check["ok"] is True


def test_s5_ollama_backend_never_calls_hf_causal_lm(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, s5_generation_backend="ollama")
    registry = ModelRegistry(settings)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("S5 must never load a Hugging Face causal LM")

    monkeypatch.setattr(ModelRegistry, "causal_lm", fail_if_called)

    from backend.app.services.ollama_runtime import OllamaRuntime

    monkeypatch.setattr(OllamaRuntime, "chat", lambda self, **kwargs: '{"ok": true}')

    generator = TextGenerator(settings, registry)
    output = generator.generate(
        instructions="x",
        user_input="y",
        model_id="mistral:7b-instruct",
        adapter_id=None,
        purpose="S5 risk classifier",
        max_new_tokens=16,
        temperature=0.0,
    )
    assert output == '{"ok": true}'


def test_s5_ollama_chat_receives_json_mode(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, s5_generation_backend="ollama")
    registry = ModelRegistry(settings)
    seen = {}

    from backend.app.services.ollama_runtime import OllamaRuntime

    def fake_chat(self, **kwargs):
        seen.update(kwargs)
        return "{}"

    monkeypatch.setattr(OllamaRuntime, "chat", fake_chat)
    generator = TextGenerator(settings, registry)
    generator.generate(
        instructions="x",
        user_input="y",
        model_id="mistral:7b-instruct",
        adapter_id=None,
        purpose="S5 risk classifier",
        max_new_tokens=16,
        temperature=0.0,
    )
    assert seen["json_mode"] is True


def test_missing_verifier_v3_adapter_blocks_final_preflight(tmp_path):
    taxonomy_path, few_shot_path = _write_risk_configs(tmp_path)
    settings = make_settings(
        tmp_path,
        verifier_adapter=None,
        verifier_require_adapter=True,
        risk_taxonomy_path=taxonomy_path,
        risk_few_shot_path=few_shot_path,
        verifier_calibration_path=None,
    )
    with pytest.raises(RuntimeError, match="Final preflight failed"):
        run_preflight(settings, final=True)


def test_verifier_calibration_threshold_loads_from_artifact(tmp_path):
    taxonomy_path, few_shot_path = _write_risk_configs(tmp_path)
    calibration_path = tmp_path / "verifier_faithbench_calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "systems": {
                    "deberta_fine_tuned": {
                        "calibration": {"selected_threshold": {"threshold": 0.404}},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    adapter_dir = tmp_path / "models" / "stage6_verifier_v3_financial"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    settings = make_settings(
        tmp_path,
        verifier_adapter=str(adapter_dir),
        verifier_calibration_path=calibration_path,
        risk_taxonomy_path=taxonomy_path,
        risk_few_shot_path=few_shot_path,
    )
    result = run_preflight(settings, final=False)
    calibration_check = next(c for c in result["checks"] if c["name"] == "verifier_calibration")
    assert calibration_check["ok"] is True
    assert "0.404" in calibration_check["detail"]

    record = next(c for c in result["checks"] if c["name"] == "verifier_adapter_runtime_record")
    assert record["detail"]["threshold"] == 0.404
    assert record["detail"]["base_model"] == settings.verifier_model
    assert record["detail"]["adapter_sha256"]


def test_invalid_s5_output_never_defaults_to_safe(tmp_path, monkeypatch):
    taxonomy_path, few_shot_path = _write_risk_configs(tmp_path)
    settings = make_settings(
        tmp_path,
        risk_taxonomy_path=taxonomy_path,
        risk_few_shot_path=few_shot_path,
        risk_parse_retries=1,
    )
    registry = ModelRegistry(settings)
    classifier = Stage5RiskClassifier(settings, registry)

    monkeypatch.setattr(
        TextGenerator,
        "generate",
        lambda self, **kwargs: "not valid json at all",
    )

    chunk = EvidenceChunk(
        chunk_id="c1",
        source="reg",
        jurisdiction="US",
        source_url="https://example.test",
        section="s1",
        text="Evidence text.",
    )
    prediction = classifier.classify("Original clause text.", [chunk])
    assert prediction.risk_label == "Needs Review"
    assert prediction.risk_label != "Safe"
