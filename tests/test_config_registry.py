"""makes sure two configs can't silently share a fingerprint or get combined when tau/reranker differ"""

from __future__ import annotations

import pytest

from backend.evaluation.config_registry import (
    REQUIRED_PROVENANCE_FIELDS,
    ConfigRegistryError,
    assert_not_combined,
    validate_provenance_record,
    validate_registry,
)


def _full_record(**overrides) -> dict:
    base = {field: None for field in REQUIRED_PROVENANCE_FIELDS}
    base.update(
        {
            "configuration_name": "example",
            "config_fingerprint": "abc123",
            "tau": 0.376,
            "reranker_enabled": False,
            "risk_backend": "trained",
            "verifier_backend": "source_clause",
            "top_k": 500,
            "top_j": 15,
        }
    )
    base.update(overrides)
    return base


def test_validate_provenance_record_passes_when_complete():
    validate_provenance_record(_full_record())  # does not raise


def test_validate_provenance_record_fails_when_field_missing():
    record = _full_record()
    del record["human_adjudication_status"]
    with pytest.raises(ConfigRegistryError, match="missing required fields"):
        validate_provenance_record(record)


def test_validate_registry_passes_for_distinct_configs():
    registry = {
        "configurations": {
            "config_a": _full_record(configuration_name="config_a", config_fingerprint="fp_a"),
            "config_b": _full_record(configuration_name="config_b", config_fingerprint="fp_b", tau=0.75),
        }
    }
    validate_registry(registry)  # does not raise


def test_validate_registry_fails_when_two_names_share_a_fingerprint():
    registry = {
        "configurations": {
            "config_a": _full_record(configuration_name="config_a", config_fingerprint="same"),
            "config_b": _full_record(configuration_name="config_b", config_fingerprint="same"),
        }
    }
    with pytest.raises(ConfigRegistryError, match="shared between"):
        validate_registry(registry)


def test_validate_registry_fails_on_empty_registry():
    with pytest.raises(ConfigRegistryError, match="no configuration entries"):
        validate_registry({"configurations": {}})


def test_assert_not_combined_passes_for_identical_headline_fields():
    a = _full_record(configuration_name="a")
    b = _full_record(configuration_name="b")
    assert_not_combined(a, b)  # does not raise


def test_assert_not_combined_fails_when_tau_differs():
    a = _full_record(configuration_name="a", tau=0.376)
    b = _full_record(configuration_name="b", tau=0.75)
    with pytest.raises(ConfigRegistryError, match="Refusing to combine"):
        assert_not_combined(a, b)


def test_assert_not_combined_fails_when_reranker_differs():
    a = _full_record(configuration_name="a", reranker_enabled=False)
    b = _full_record(configuration_name="b", reranker_enabled=True)
    with pytest.raises(ConfigRegistryError, match="reranker_enabled"):
        assert_not_combined(a, b)
