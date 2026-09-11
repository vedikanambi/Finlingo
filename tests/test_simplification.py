from types import SimpleNamespace

from backend.app.core.config import Settings
from backend.app.stages.stage2_simplification import Stage2Simplification
from backend.app.services.nli_service import NLIService


class DummyRegistry:
    pass


class FakeGenerator:
    def generate(self, **kwargs):
        return "The customer is required to discharge all obligations pursuant to the agreement."


class LowSemanticNLI:
    def score_pairs(self, pairs, batch_size, max_length):
        return [SimpleNamespace(entailment=0.60, neutral=0.30, contradiction=0.10)]


def test_failed_simplification_gate_is_explicitly_marked_needs_review():
    settings = Settings(
        _env_file=None,
        simplifier_retries=1,
        fk_generation_target=0.0,
        simplifier_min_semantic_entailment=0.75,
        verifier_require_adapter=False,
    )
    stage = Stage2Simplification(settings, DummyRegistry())
    stage.generator = FakeGenerator()
    stage.nli = LowSemanticNLI()
    result = stage.simplify("The borrower must pay ten euro each month.")
    assert result.accepted is False
    assert result.status == "needs_review"
    assert result.semantic_target_met is False
    assert result.attempts == 2
    assert len(result.candidates) == 2


def test_auxiliary_nli_can_ignore_future_verifier_calibration(tmp_path):
    settings = Settings(
        _env_file=None,
        verifier_calibration_path=tmp_path / "not-trained-yet.json",
        verifier_temperature=1.0,
    )
    registry = SimpleNamespace(settings=settings)
    service = NLIService(registry, use_verifier_calibration=False)
    assert service.temperature == 1.0
