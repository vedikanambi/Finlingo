"""checks that every headline number in the results table can be traced back to a specific run config, so nothing gets silently mixed up."""

from __future__ import annotations

REQUIRED_PROVENANCE_FIELDS = (
    "evaluation_scope",
    "configuration_name",
    "dataset_sha256",
    "model_sha256",
    "adapter_sha256",
    "config_fingerprint",
    "random_seeds",
    "bootstrap_unit",
    "human_adjudication_status",
    "independent_ground_truth",
    "reranker_enabled",
    "risk_backend",
    "verifier_backend",
    "tau",
    "top_k",
    "top_j",
)


class ConfigRegistryError(ValueError):
    pass


def validate_provenance_record(record: dict) -> None:
    """Raise if a provenance record is missing any required field."""
    missing = [field for field in REQUIRED_PROVENANCE_FIELDS if field not in record]
    if missing:
        name = record.get("configuration_name", "<unnamed>")
        raise ConfigRegistryError(f"Provenance record {name!r} is missing required fields: {missing}")


def validate_registry(registry: dict) -> None:
    """checks the whole registry: no duplicate fingerprints under different names, no one name covering two different configs."""
    entries = registry.get("configurations", registry)
    if not entries:
        raise ConfigRegistryError("Registry has no configuration entries")

    seen_fingerprint_to_name: dict[str, str] = {}
    seen_name_to_fingerprint: dict[str, str] = {}
    for name, record in entries.items():
        validate_provenance_record(record)
        fingerprint = record["config_fingerprint"]
        if fingerprint is not None:
            if fingerprint in seen_fingerprint_to_name and seen_fingerprint_to_name[fingerprint] != name:
                raise ConfigRegistryError(
                    f"config_fingerprint {fingerprint!r} is shared between {seen_fingerprint_to_name[fingerprint]!r} "
                    f"and {name!r} -- these are documented as different configurations but have an identical "
                    "fingerprint, which means they are actually the same run recorded twice, or the fingerprint "
                    "is not capturing a real difference between them."
                )
            seen_fingerprint_to_name[fingerprint] = name
        if name in seen_name_to_fingerprint and seen_name_to_fingerprint[name] != fingerprint:
            raise ConfigRegistryError(
                f"Configuration name {name!r} maps to two different config_fingerprints -- "
                "a single named configuration must not silently combine two different runs."
            )
        seen_name_to_fingerprint[name] = fingerprint


def assert_not_combined(record_a: dict, record_b: dict) -> None:
    """block combining two records into one row if they differ on tau, reranker, backends, top_k or top_j."""
    headline_fields = ("tau", "reranker_enabled", "risk_backend", "verifier_backend", "top_k", "top_j")
    conflicts = [field for field in headline_fields if record_a.get(field) != record_b.get(field)]
    if conflicts:
        raise ConfigRegistryError(
            f"Refusing to combine {record_a.get('configuration_name')!r} and "
            f"{record_b.get('configuration_name')!r} into one row: they differ in {conflicts}."
        )
