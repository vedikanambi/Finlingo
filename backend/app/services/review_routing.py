"""Stage 7 human-review routing policy."""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.app.core.config import Settings
from backend.app.core.schemas import ClauseResult

REASON_SIMPLIFICATION_GATE_FAILED = "SIMPLIFICATION_GATE_FAILED"
REASON_RISK_ABSTENTION = "RISK_ABSTENTION"
REASON_LOW_SOURCE_ENTAILMENT = "LOW_SOURCE_ENTAILMENT"
REASON_MISSING_REGULATORY_ATTRIBUTION = "MISSING_REGULATORY_ATTRIBUTION"
REASON_CRITICAL_FACT_CHANGED = "CRITICAL_FACT_CHANGED"
REASON_EVIDENCE_CONFLICT = "EVIDENCE_CONFLICT"
REASON_HIGH_IMPACT_RISK = "HIGH_IMPACT_RISK"

ALL_REASON_CODES = (
    REASON_SIMPLIFICATION_GATE_FAILED,
    REASON_RISK_ABSTENTION,
    REASON_LOW_SOURCE_ENTAILMENT,
    REASON_MISSING_REGULATORY_ATTRIBUTION,
    REASON_CRITICAL_FACT_CHANGED,
    REASON_EVIDENCE_CONFLICT,
    REASON_HIGH_IMPACT_RISK,
)


@dataclass
class RoutingDecision:
    clause_id: str
    routed_for_review: bool
    reasons: list[str] = field(default_factory=list)
    policy: str = "legacy"


def route_clause(clause: ClauseResult, settings: Settings) -> RoutingDecision:
    if settings.review_routing_policy != "strict_policy":
        reasons: list[str] = []
        if clause.simplification_status == "needs_review":
            reasons.append(REASON_SIMPLIFICATION_GATE_FAILED)
        if clause.risk_label == "Needs Review":
            reasons.append(REASON_RISK_ABSTENTION)
        return RoutingDecision(clause.clause_id, bool(reasons), reasons, policy="legacy")

    reasons = []
    if clause.simplification_status == "needs_review":
        reasons.append(REASON_SIMPLIFICATION_GATE_FAILED)
    if clause.risk_label == "Needs Review":
        reasons.append(REASON_RISK_ABSTENTION)
    if any(
        item.source_faithfulness is not None and item.source_faithfulness < settings.review_low_entailment_tau
        for item in clause.faithfulness
    ):
        reasons.append(REASON_LOW_SOURCE_ENTAILMENT)
    if (
        settings.review_require_attribution
        and clause.evidence
        and not (clause.supported_attribution_coverage or 0)
    ):
        reasons.append(REASON_MISSING_REGULATORY_ATTRIBUTION)
    if clause.semantic_target_met is False:
        reasons.append(REASON_CRITICAL_FACT_CHANGED)
    if any(
        item.source_faithfulness is not None
        and item.regulatory_support is not None
        and abs(item.source_faithfulness - item.regulatory_support) >= settings.review_evidence_conflict_gap
        for item in clause.faithfulness
    ):
        reasons.append(REASON_EVIDENCE_CONFLICT)
    if clause.risk_score >= settings.review_high_impact_risk_score:
        reasons.append(REASON_HIGH_IMPACT_RISK)
    return RoutingDecision(clause.clause_id, bool(reasons), reasons, policy="strict_policy")


def route_clauses(clauses: list[ClauseResult], settings: Settings) -> list[RoutingDecision]:
    return [route_clause(clause, settings) for clause in clauses]


def routing_summary(decisions: list[RoutingDecision], settings: Settings) -> dict:
    reason_counts: dict[str, int] = {code: 0 for code in ALL_REASON_CODES}
    for decision in decisions:
        for reason in decision.reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "routing_policy": settings.review_routing_policy,
        "routed_clause_count": sum(decision.routed_for_review for decision in decisions),
        "total_clause_count": len(decisions),
        "reason_counts": reason_counts,
    }
