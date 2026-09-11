"""Acceptance tests for the Stage 7 review-routing policy."""

from __future__ import annotations

from backend.app.core.config import Settings
from backend.app.core.schemas import ClauseResult, SentenceFaithfulness
from backend.app.services.review_routing import (
    ALL_REASON_CODES,
    REASON_CRITICAL_FACT_CHANGED,
    REASON_EVIDENCE_CONFLICT,
    REASON_HIGH_IMPACT_RISK,
    REASON_LOW_SOURCE_ENTAILMENT,
    REASON_MISSING_REGULATORY_ATTRIBUTION,
    REASON_RISK_ABSTENTION,
    REASON_SIMPLIFICATION_GATE_FAILED,
    route_clause,
    routing_summary,
)


def _clause(**overrides) -> ClauseResult:
    base = dict(
        clause_id="c1",
        original_text="text",
        simplified_text="text",
        risk_label="Safe",
        risk_score=1,
        simplification_status="accepted",
        semantic_target_met=True,
        faithfulness=[],
        evidence=[],
        supported_attribution_coverage=1.0,
    )
    base.update(overrides)
    return ClauseResult(**base)


def _settings(**overrides) -> Settings:
    return Settings(review_routing_policy="strict_policy", **overrides)


def test_safe_supported_clause_not_routed():
    decision = route_clause(_clause(), _settings())
    assert decision.routed_for_review is False
    assert decision.reasons == []


def test_simplification_gate_failure_routes():
    decision = route_clause(_clause(simplification_status="needs_review"), _settings())
    assert REASON_SIMPLIFICATION_GATE_FAILED in decision.reasons
    assert decision.routed_for_review is True


def test_risk_abstention_routes():
    decision = route_clause(_clause(risk_label="Needs Review"), _settings())
    assert REASON_RISK_ABSTENTION in decision.reasons


def test_low_source_entailment_routes():
    sentence = SentenceFaithfulness(
        sentence="s",
        source_faithfulness=0.1,
        source_supported=False,
        unsupported=True,
    )
    decision = route_clause(_clause(faithfulness=[sentence]), _settings(review_low_entailment_tau=0.376))
    assert REASON_LOW_SOURCE_ENTAILMENT in decision.reasons


def test_missing_regulatory_attribution_routes_when_required():
    from backend.app.core.schemas import EvidenceChunk

    chunk = EvidenceChunk(chunk_id="e1", source="FCA", text="text", jurisdiction="UK", source_url="https://example.test")
    decision = route_clause(
        _clause(evidence=[chunk], supported_attribution_coverage=0.0),
        _settings(review_require_attribution=True),
    )
    assert REASON_MISSING_REGULATORY_ATTRIBUTION in decision.reasons


def test_critical_fact_changed_routes():
    decision = route_clause(_clause(semantic_target_met=False), _settings())
    assert REASON_CRITICAL_FACT_CHANGED in decision.reasons


def test_evidence_conflict_routes():
    sentence = SentenceFaithfulness(
        sentence="s",
        source_faithfulness=0.9,
        source_supported=True,
        regulatory_support=0.1,
        unsupported=False,
    )
    decision = route_clause(_clause(faithfulness=[sentence]), _settings(review_evidence_conflict_gap=0.3))
    assert REASON_EVIDENCE_CONFLICT in decision.reasons


def test_high_impact_risk_routes():
    decision = route_clause(_clause(risk_score=5), _settings(review_high_impact_risk_score=4))
    assert REASON_HIGH_IMPACT_RISK in decision.reasons


def test_multiple_reasons_preserved():
    decision = route_clause(
        _clause(risk_label="Needs Review", simplification_status="needs_review", risk_score=5),
        _settings(review_high_impact_risk_score=4),
    )
    assert REASON_RISK_ABSTENTION in decision.reasons
    assert REASON_SIMPLIFICATION_GATE_FAILED in decision.reasons
    assert REASON_HIGH_IMPACT_RISK in decision.reasons
    assert len(decision.reasons) == 3


def test_legacy_policy_default_unchanged():
    decision = route_clause(_clause(simplification_status="needs_review"), Settings())
    assert decision.policy == "legacy"
    assert decision.reasons == [REASON_SIMPLIFICATION_GATE_FAILED]


def test_routing_summary_reports_policy_and_counts():
    settings = _settings()
    decisions = [
        route_clause(_clause(clause_id="a"), settings),
        route_clause(_clause(clause_id="b", risk_label="Needs Review"), settings),
    ]
    summary = routing_summary(decisions, settings)
    assert summary["routing_policy"] == "strict_policy"
    assert summary["routed_clause_count"] == 1
    assert summary["total_clause_count"] == 2
    assert set(summary["reason_counts"]) == set(ALL_REASON_CODES)
    assert summary["reason_counts"][REASON_RISK_ABSTENTION] == 1
