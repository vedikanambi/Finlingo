import sys
import types
from backend.app.core.config import Settings
from backend.app.services.dataset_streams import StreamingDatasetRepository


class FakeDataset:
    def __iter__(self):
        yield {"id": "1"}

    def shuffle(self, **kwargs):
        return self


def test_huggingface_loader_always_uses_streaming(monkeypatch):
    captured = {}

    def load_dataset(**kwargs):
        captured.update(kwargs)
        return FakeDataset()

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    repo = StreamingDatasetRepository(Settings(_env_file=None, hf_token="token"))
    assert list(repo.stream_raw("example/data", split="train", limit=1)) == [{"id": "1"}]
    assert captured["streaming"] is True
    assert captured["token"] == "token"


def test_streaming_cache_is_process_temporary_and_deleted():
    from pathlib import Path

    repo = StreamingDatasetRepository(Settings(_env_file=None))
    cache = Path(repo._ephemeral_cache.name)
    assert cache.exists()
    repo.close()
    assert not cache.exists()
