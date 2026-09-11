import pytest

from backend.app.core.config import Settings
from backend.app.services.model_registry import ModelRegistry


def test_causal_adapter_requirement_fails_before_model_download():
    registry = ModelRegistry(Settings(_env_file=None, allow_base_models=False))
    with pytest.raises(RuntimeError, match="No trained adapter"):
        registry.causal_lm("example/model", None, "S2 simplifier", require_adapter=True)


def test_verifier_adapter_requirement_fails_before_model_download():
    # allow_base_models=True would bypass the gate this test is checking, so force it False here
    registry = ModelRegistry(
        Settings(_env_file=None, verifier_adapter=None, verifier_require_adapter=True, allow_base_models=False)
    )
    with pytest.raises(RuntimeError, match="No trained verifier adapter"):
        registry.nli()


def test_explicit_auxiliary_nli_does_not_inherit_final_verifier_adapter(monkeypatch):
    settings = Settings(
        _env_file=None,
        verifier_adapter="models/future-stage6-adapter",
        verifier_require_adapter=True,
    )
    registry = ModelRegistry(settings)

    class StopHere(RuntimeError):
        pass

    import transformers

    def stop(*args, **kwargs):
        assert kwargs.get("revision") is None
        raise StopHere

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", stop)
    with pytest.raises(StopHere):
        registry.nli("example/independent-nli", adapter_id=None, require_adapter=False)
