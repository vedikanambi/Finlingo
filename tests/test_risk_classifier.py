from backend.app.core.config import Settings
from backend.app.core.schemas import EvidenceChunk
from backend.app.stages.stage5_risk_classifier import Stage5RiskClassifier


class FakeGenerator:
    def generate(self, **kwargs):
        return '{"risk_label":"Hidden Fee","risk_score":4,"confidence":0.9,"explanation":"A fee is disclosed.","evidence_chunk_ids":["e1"]}'


class DummyRegistry:
    pass


def test_structured_risk_output_is_validated():
    stage = Stage5RiskClassifier(Settings(_env_file=None, risk_classifier_backend="prompted"), DummyRegistry())
    stage.generator = FakeGenerator()
    evidence = [
        EvidenceChunk(chunk_id="e1", source="FCA", jurisdiction="UK", source_url="https://example", text="fee rule")
    ]
    result = stage.classify("A fee applies", evidence)
    assert result.risk_label == "Hidden Fee"
    assert result.evidence_chunk_ids == ["e1"]


def test_few_shot_prompt_contains_rubric_and_all_proposal_examples():
    stage = Stage5RiskClassifier(Settings(_env_file=None), DummyRegistry())
    prompt = stage._few_shot_prompt()
    assert "RUBRIC:" in prompt
    assert "EXAMPLE CLAUSE:" in prompt
    for label in ["Auto-Renewal", "Hidden Fee", "Liability Waiver", "Data Sharing", "Penalty Clause", "Safe"]:
        assert label in prompt


def test_trained_backend_uses_explicit_selected_adapter(tmp_path, monkeypatch):
    selected = tmp_path / "selected-trial"
    selected.mkdir()
    (selected / "adapter_config.json").write_text("{}", encoding="utf-8")
    settings = Settings(
        _env_file=None,
        risk_classifier_backend="trained",
        risk_classifier_adapter=selected,
        risk_classifier_adapter_output=tmp_path / "default-output",
        risk_classifier_require_adapter=True,
    )
    stage = Stage5RiskClassifier(settings, DummyRegistry())
    observed = {}

    def fake_classify(text, adapter_path):
        observed["path"] = adapter_path
        return "trained-result"

    monkeypatch.setattr(stage, "_classify_trained", fake_classify)
    assert stage.classify("A clause", []) == "trained-result"
    assert observed["path"] == selected.resolve()


def test_frozen_backend_uses_selected_artifact(tmp_path, monkeypatch):
    artifact = tmp_path / "classifier.joblib"
    artifact.write_bytes(b"fixture")
    settings = Settings(
        _env_file=None,
        risk_classifier_backend="frozen",
        risk_classifier_frozen_artifact=artifact,
        risk_classifier_require_adapter=True,
    )
    stage = Stage5RiskClassifier(settings, DummyRegistry())
    observed = {}

    def fake_classify(text, artifact_path):
        observed["path"] = artifact_path
        return "frozen-result"

    monkeypatch.setattr(stage, "_classify_frozen", fake_classify)
    assert stage.classify("A clause", []) == "frozen-result"
    assert observed["path"] == artifact.resolve()
