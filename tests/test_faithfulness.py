from types import SimpleNamespace
from backend.app.core.config import Settings
from backend.app.core.schemas import EvidenceChunk
from backend.app.stages.stage6_faithfulness import Stage6Faithfulness


class FakeNLI:
    def score_pairs(self, pairs, batch_size, max_length):
        if len(pairs) == 1:
            return [SimpleNamespace(entailment=0.9)]
        return [SimpleNamespace(entailment=0.2), SimpleNamespace(entailment=0.8)]


class DummyRegistry:
    pass


def test_source_faithfulness_and_regulatory_support_are_separate():
    stage = Stage6Faithfulness(Settings(_env_file=None), DummyRegistry())
    stage.nli = FakeNLI()
    chunks = [
        EvidenceChunk(chunk_id="a", source="FCA", jurisdiction="UK", source_url="https://a", text="unrelated"),
        EvidenceChunk(chunk_id="b", source="EU", jurisdiction="EU", source_url="https://b", text="support"),
    ]
    result = stage.verify("The fee is €10.", "The fee is €10.", chunks)[0]
    assert result.source_faithfulness == 0.9
    assert result.regulatory_support == 0.8
    assert result.attribution_chunk_id == "b"

    risk_support = stage.verify_regulatory_text("This fee may be unfair.", chunks)[0]
    assert risk_support.regulatory_support == 0.8
    assert risk_support.attribution_chunk_id == "b"


class FakeLowRegulatoryNLI:
    def score_pairs(self, pairs, batch_size, max_length):
        if len(pairs) == 1:
            return [SimpleNamespace(entailment=0.95, neutral=0.03, contradiction=0.02)]
        return [
            SimpleNamespace(entailment=0.30, neutral=0.60, contradiction=0.10),
            SimpleNamespace(entailment=0.70, neutral=0.20, contradiction=0.10),
        ]


def test_primary_fi_uses_retrieved_chunk_and_abstains_below_attribution_threshold():
    settings = Settings(
        _env_file=None,
        faithfulness_primary_premise="retrieved_chunk",
        faithfulness_tau=0.75,
        attribution_tau=0.75,
    )
    stage = Stage6Faithfulness(settings, DummyRegistry())
    stage.nli = FakeLowRegulatoryNLI()
    chunks = [
        EvidenceChunk(chunk_id="a", source="FCA", jurisdiction="UK", source_url="https://a", text="weak"),
        EvidenceChunk(chunk_id="b", source="EU", jurisdiction="EU", source_url="https://b", text="best but low"),
    ]
    result = stage.verify("The source supports this.", "The generated sentence.", chunks)[0]
    assert result.source_faithfulness == 0.95
    assert result.faithfulness_score == 0.70
    assert result.faithfulness_premise == "retrieved_chunk"
    assert result.unsupported is True
    assert result.attribution_chunk_id is None
    assert result.attribution_candidate_chunk_id == "b"
    assert result.attribution_candidate_score == 0.70
