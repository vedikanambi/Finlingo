import json
from types import SimpleNamespace

from backend.app.core.config import Settings
from backend.training.corrected_cfpb_cache import CorrectedCFPBCache, output_hash, source_hash, validate_preservation
from backend.training.silver_label_providers import ProviderResult


def test_hashes_are_normalized_and_distinct():
    assert source_hash("  Pay   $25 now ") == source_hash("Pay $25 now")
    assert output_hash("PLAIN TEXT") == output_hash("plain text")


def test_preservation_checks_critical_fields():
    source = "Acme Bank shall not charge $25 on January 2, 2027 or 5 percent."
    valid = "Acme Bank shall not charge $25 on January 2, 2027 or 5 percent."
    invalid = "Acme Bank may charge $20 on January 2, 2027."
    assert validate_preservation(source, valid) == (True, [])
    ok, failures = validate_preservation(source, invalid)
    assert not ok
    assert {"number", "percentage", "obligation", "negation"} <= set(failures)


def test_equivalent_obligation_modal_is_accepted():
    source = "Acme Bank shall provide the statement."
    target = "Acme Bank must provide the statement."
    assert validate_preservation(source, target) == (True, [])


def test_atomic_writer_creates_parent_and_replaces_file(tmp_path):
    target = tmp_path / "nested" / "manifest.json"
    CorrectedCFPBCache._atomic_json(target, {"valid_label_count": 1})
    CorrectedCFPBCache._atomic_json(target, {"valid_label_count": 2})
    assert target.read_text(encoding="utf-8").find('"valid_label_count": 2') > 0
    assert not target.with_suffix(".json.tmp").exists()


def _client(response: dict):
    from backend.training.silver_labels import SimplificationBatch

    parsed = SimplificationBatch.model_validate(response)
    return SimpleNamespace(
        generate=lambda *args: ProviderResult(
            parsed,
            0,
            {
                "provider": "test",
                "model": "test/model",
                "generation_seconds": 0.01,
            },
        )
    )


def test_batch_identity_validation_atomic_cache_and_manifest(tmp_path):
    source = "Acme Bank shall not charge $25 on January 2, 2027 or 5 percent."
    digest = source_hash(source)
    cache = CorrectedCFPBCache(
        Settings(_env_file=None, corrected_cfpb_cache_dir=tmp_path),
        provider=_client({"items": [{"id": digest, "simplified": source}]}),
    )
    result = cache.generate_one([{"text": source}], "train")
    assert result.accepted_labels == 1
    manifest = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
    assert manifest["valid_label_count"] == 1
    assert manifest["identity"]["selected_narrative_field"] == "consumer_narrative"
    assert manifest["raw_records_persisted"] is False
    assert len(list(cache.batch_dir.glob("*.json"))) == 1


def test_wrong_source_hash_is_malformed_and_rejected(tmp_path):
    source = "You must pay $25."
    cache = CorrectedCFPBCache(
        Settings(_env_file=None, corrected_cfpb_cache_dir=tmp_path),
        provider=_client({"items": [{"id": "wrong", "simplified": source}]}),
    )
    result = cache.generate_one([{"text": source}], "train")
    assert result.malformed_responses == 1
    assert result.rejected_labels == 1
    assert result.accepted_labels == 0
    assert not list(cache.batch_dir.glob("*.json"))
