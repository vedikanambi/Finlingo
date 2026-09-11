from types import SimpleNamespace

from backend.app.core.config import Settings
from backend.training.silver_labels import SimplificationBatch, SilverLabelGenerator
from backend.training.silver_labels import ProviderQuotaExhausted, _retry_after_seconds


class _Completions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        message = SimpleNamespace(content='{"items":[{"id":"one","simplified":"Plain text."}]}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_validated_provider_responses_are_resumable(tmp_path):
    completions = _Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    registry = SimpleNamespace(groq_client=lambda: client)
    settings = Settings(_env_file=None, silver_label_cache_dir=tmp_path)
    generator = SilverLabelGenerator(settings, registry)
    kwargs = dict(
        model="provider/model",
        instructions="Return JSON.",
        input_text="request body",
    )
    first = generator._validated_response(SimplificationBatch, **kwargs)
    second = generator._validated_response(SimplificationBatch, **kwargs)
    assert first == second
    assert completions.calls == 1
    assert len(list(tmp_path.rglob("*.json"))) == 1
    assert generator.new_labels_generated == 1
    assert generator.cached_labels_reused == 1
    assert generator.rejected_labels == 0


def test_quota_exhaustion_stops_cleanly_and_preserves_retry_after(tmp_path):
    class Response:
        status_code = 429
        headers = {"retry-after": "42"}

    class RateLimit(Exception):
        response = Response()

    class QuotaCompletions:
        def create(self, **kwargs):
            raise RateLimit("quota")

    client = SimpleNamespace(chat=SimpleNamespace(completions=QuotaCompletions()))
    registry = SimpleNamespace(groq_client=lambda: client)
    generator = SilverLabelGenerator(Settings(_env_file=None, silver_label_cache_dir=tmp_path), registry)
    import pytest

    with pytest.raises(ProviderQuotaExhausted) as error:
        generator._validated_response(
            SimplificationBatch,
            model="provider/model",
            instructions="Return JSON.",
            input_text="uncached request",
        )
    assert error.value.retry_after_seconds == 42
    assert not list(tmp_path.rglob("*.json"))


def test_retry_after_can_be_parsed_from_groq_message():
    assert _retry_after_seconds(Exception("Please try again in 6m48.5s")) == 408.5


def test_concurrent_writes_to_same_cache_key_never_corrupt(tmp_path):
    """many threads racing to fill the same cache key should still leave one valid, non-truncated file."""
    import threading

    completions = _Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    registry = SimpleNamespace(groq_client=lambda: client)
    settings = Settings(_env_file=None, silver_label_cache_dir=tmp_path)
    generator = SilverLabelGenerator(settings, registry)
    kwargs = dict(model="provider/model", instructions="Return JSON.", input_text="shared request")

    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        return generator._validated_response(SimplificationBatch, **kwargs)

    results = []
    threads = [threading.Thread(target=lambda: results.append(worker())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 8
    assert all(r == SimplificationBatch(items=[{"id": "one", "simplified": "Plain text."}]) for r in results)
    cache_files = list(tmp_path.rglob("*.json"))
    assert len(cache_files) == 1
    SimplificationBatch.model_validate_json(cache_files[0].read_text(encoding="utf-8"))
    assert not list(tmp_path.rglob("*.tmp"))
