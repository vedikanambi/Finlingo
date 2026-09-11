"""Tests for the reproducible RQ4 deterministic-generation mode."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.app.core.config import Settings
from backend.app.core.schemas import SimplificationResult
from backend.experiments.failure_analysis import EXPECTED_RQ4_CONDITIONS, validate_rq4_conditions


def _fake_stage2(settings: Settings, fixed_output: str = "your contract renews yearly.") -> object:
    """Fake stage2 that skips loading a real model - only the hashing/branching logic is under test here."""
    from backend.app.stages.stage2_simplification import Stage2Simplification

    stage = Stage2Simplification.__new__(Stage2Simplification)
    stage.settings = settings
    stage.generator = MagicMock()
    stage.generator.generate.return_value = fixed_output
    nli_result = MagicMock(entailment=0.95, neutral=0.03, contradiction=0.02)
    stage.nli = MagicMock()
    stage.nli.score_pairs.return_value = [nli_result]
    return stage


def test_deterministic_mode_produces_identical_output_hash_across_runs():
    settings = Settings(s2_deterministic_mode=True, fk_generation_target=100, simplifier_min_semantic_entailment=0.0)
    stage = _fake_stage2(settings)
    result_a: SimplificationResult = stage.simplify("original clause text")
    result_b: SimplificationResult = stage.simplify("original clause text")
    assert result_a.output_hash == result_b.output_hash
    assert result_a.generation_mode == "deterministic"
    assert result_a.num_retries == 0
    assert result_a.retry_temperatures == []
    assert result_a.initial_temperature == 0.0


def test_deterministic_mode_single_attempt_no_retries():
    settings = Settings(s2_deterministic_mode=True, fk_generation_target=100, simplifier_min_semantic_entailment=0.0)
    stage = _fake_stage2(settings)
    result = stage.simplify("clause")
    assert result.attempts == 1
    assert len(result.candidates) == 1
    assert result.candidates[0].temperature == 0.0
    assert result.candidates[0].accepted is True


def test_stochastic_mode_records_retry_temperatures_and_hash():
    settings = Settings(s2_deterministic_mode=False, fk_generation_target=100, simplifier_min_semantic_entailment=0.0)
    stage = _fake_stage2(settings)
    result = stage.simplify("clause")
    assert result.generation_mode == "stochastic"
    assert result.output_hash is not None
    assert result.initial_temperature == settings.simplifier_temperature
    assert result.generation_seed == settings.random_seed


def test_rq4_validation_passes_on_well_formed_seven_condition_payload():
    payload = {
        "rq4_failure_modes": {
            name: {"pipeline": {"clause_ids": ["c1", "c2"]}} for name in EXPECTED_RQ4_CONDITIONS
        }
    }
    validate_rq4_conditions(payload)  # does not raise


def test_rq4_validation_fails_on_missing_condition():
    conditions = {name: {"pipeline": {"clause_ids": ["c1"]}} for name in EXPECTED_RQ4_CONDITIONS}
    del conditions["cross_clause_dependency"]
    with pytest.raises(ValueError, match="Expected exactly 7"):
        validate_rq4_conditions({"rq4_failure_modes": conditions})


def test_rq4_validation_fails_on_unexpected_extra_condition():
    conditions = {name: {"pipeline": {"clause_ids": ["c1"]}} for name in EXPECTED_RQ4_CONDITIONS}
    conditions["surprise_condition"] = {"pipeline": {"clause_ids": ["c1"]}}
    with pytest.raises(ValueError, match="Expected exactly 7"):
        validate_rq4_conditions({"rq4_failure_modes": conditions})


def test_rq4_validation_fails_when_clause_id_sets_diverge():
    conditions = {name: {"pipeline": {"clause_ids": ["c1", "c2"]}} for name in EXPECTED_RQ4_CONDITIONS}
    conditions["very_short_or_truncated"]["pipeline"]["clause_ids"] = ["c1", "c3"]
    with pytest.raises(ValueError, match="different clause-id set"):
        validate_rq4_conditions({"rq4_failure_modes": conditions})


def test_rq4_validation_accepts_bare_conditions_dict_too():
    conditions = {name: {"pipeline": {"clause_ids": ["c1"]}} for name in EXPECTED_RQ4_CONDITIONS}
    validate_rq4_conditions(conditions)  # does not raise
