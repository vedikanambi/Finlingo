"""Misc unit tests: Ollama backend routing, FLB plan maths, McNemar, manifest hashing. All CPU-only, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.core.config import Settings
from backend.evaluation.metrics import mcnemar_exact


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = {"project_root": tmp_path, "_env_file": None}
    values.update(overrides)
    return Settings(**values)


def test_s5_defaults_to_ollama_backend(tmp_path):
    settings = make_settings(tmp_path)
    assert settings.backend_for("S5 risk classifier") == "ollama"
    assert settings.backend_for("S2 simplifier") == settings.model_backend


def test_backend_override_inherit_and_explicit(tmp_path):
    settings = make_settings(tmp_path, s5_generation_backend="inherit", model_backend="local")
    assert settings.backend_for("S5 risk classifier") == "local"
    settings = make_settings(tmp_path, s2_generation_backend="ollama")
    assert settings.backend_for("S2 simplifier") == "ollama"


def test_ollama_generation_model_falls_back(tmp_path):
    settings = make_settings(tmp_path, ollama_model="qwen2.5:7b-instruct")
    assert settings.ollama_generation_model_resolved == "qwen2.5:7b-instruct"
    settings = make_settings(tmp_path, ollama_generation_model="mistral:7b-instruct")
    assert settings.ollama_generation_model_resolved == "mistral:7b-instruct"


def test_ollama_tag_matching():
    from backend.app.services.ollama_runtime import OllamaRuntime

    tags = ["qwen2.5:7b-instruct-q4_K_M", "mistral:7b-instruct", "llama3:8b"]
    assert OllamaRuntime._match_tag("mistral:7b-instruct", tags) == "mistral:7b-instruct"
    assert OllamaRuntime._match_tag("qwen2.5:7b-instruct", tags) == "qwen2.5:7b-instruct-q4_K_M"
    assert OllamaRuntime._match_tag("qwen2.5", tags) == "qwen2.5:7b-instruct-q4_K_M"
    assert OllamaRuntime._match_tag("phi3", tags) is None


def test_cfpb_cache_dir_is_project_relative(tmp_path):
    settings = make_settings(tmp_path)
    resolved = settings.resolve(settings.corrected_cfpb_cache_dir)
    assert not str(settings.corrected_cfpb_cache_dir).startswith("C:")
    assert str(resolved).startswith(str(tmp_path))


def test_mcnemar_no_discordance():
    result = mcnemar_exact([True, False, True], [True, False, True])
    assert result["p_value"] == 1.0 and not result["significant_at_05"]


def test_mcnemar_strong_asymmetry_is_significant():
    a = [True] * 20
    b = [False] * 20
    result = mcnemar_exact(a, b)
    assert result["b"] == 20 and result["c"] == 0
    assert result["p_value"] < 0.001 and result["significant_at_05"]


def test_mcnemar_symmetry_not_significant():
    a = [True, False] * 10
    b = [False, True] * 10
    result = mcnemar_exact(a, b)
    assert result["b"] == result["c"] == 10
    assert result["p_value"] > 0.5


def test_mcnemar_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        mcnemar_exact([True], [True, False])


def test_flb_plan_quota_maths(tmp_path, monkeypatch):
    from backend.evaluation import flb_plan

    settings = make_settings(
        tmp_path,
        flb_size=60,
        flb_candidate_multiplier=3,
        flb_plan_min_per_class=5,
        risk_taxonomy_path=_taxonomy(tmp_path),
    )
    audit = {
        "quotas": {
            "Auto-Renewal": 10,
            "Hidden Fee": 10,
            "Liability Waiver": 10,
            "Data Sharing": 10,
            "Penalty Clause": 10,
            "Safe": 10,
        },
        "class_counts": {
            "Auto-Renewal": 300,
            "Hidden Fee": 300,
            "Liability Waiver": 300,
            "Data Sharing": 300,
            "Penalty Clause": 18,
            "Safe": 300,
        },
    }
    monkeypatch.setattr(flb_plan.FLBBuilder, "feasibility_audit", classmethod(lambda cls, s, t: audit))
    plan = flb_plan.build_flb_plan(settings, accept=False, output_path=tmp_path / "plan.json")
    assert plan["achievable_quotas"]["Penalty Clause"] == 6
    assert plan["achievable_quotas"]["Safe"] == 10
    assert plan["achievable_total"] == 56
    assert plan["plan_feasible"] and not plan["accepted"]

    accepted = flb_plan.build_flb_plan(settings, accept=True, output_path=tmp_path / "plan.json")
    assert accepted["accepted"] is True
    loaded = flb_plan.load_accepted_plan(
        make_settings(tmp_path, flb_plan_path=tmp_path / "plan.json", risk_taxonomy_path=_taxonomy(tmp_path))
    )
    assert loaded["achievable_total"] == 56


def test_flb_plan_refuses_below_floor(tmp_path, monkeypatch):
    from backend.evaluation import flb_plan

    settings = make_settings(
        tmp_path,
        flb_size=60,
        flb_candidate_multiplier=3,
        flb_plan_min_per_class=10,
        risk_taxonomy_path=_taxonomy(tmp_path),
    )
    audit = {
        "quotas": {
            label: 10
            for label in ("Auto-Renewal", "Hidden Fee", "Liability Waiver", "Data Sharing", "Penalty Clause", "Safe")
        },
        "class_counts": {
            "Auto-Renewal": 300,
            "Hidden Fee": 300,
            "Liability Waiver": 300,
            "Data Sharing": 300,
            "Penalty Clause": 9,
            "Safe": 300,
        },
    }
    monkeypatch.setattr(flb_plan.FLBBuilder, "feasibility_audit", classmethod(lambda cls, s, t: audit))
    plan = flb_plan.build_flb_plan(settings, accept=True, output_path=tmp_path / "plan.json")
    assert plan["accepted"] is False and plan["infeasible_classes"]
    with pytest.raises(RuntimeError):
        flb_plan.load_accepted_plan(
            make_settings(tmp_path, flb_plan_path=tmp_path / "plan.json", risk_taxonomy_path=_taxonomy(tmp_path))
        )


def _taxonomy(tmp_path: Path) -> Path:
    path = tmp_path / "taxonomy.yaml"
    if not path.exists():
        path.write_text(
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
    return path


def test_submission_manifest_hashing(tmp_path):
    from main import _sha256_file

    file = tmp_path / "artifact.json"
    file.write_text('{"metric": 1}', encoding="utf-8")
    digest = _sha256_file(file)
    assert len(digest) == 64
    file.write_text('{"metric": 2}', encoding="utf-8")
    assert _sha256_file(file) != digest


def test_environment_report_redacts_secrets(tmp_path):
    from backend.app.core.environment import environment_report

    settings = make_settings(tmp_path, hf_token="hf_secret_value", risk_taxonomy_path=_taxonomy(tmp_path))
    report = environment_report(settings)
    assert report["settings"]["hf_token"] == "***set***"
    assert "hf_secret_value" not in json.dumps(report)
    assert report["packages_sha256"] and report["package_count"] > 0
