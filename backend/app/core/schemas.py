from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

RiskLabel = Literal[
    "Auto-Renewal",
    "Hidden Fee",
    "Liability Waiver",
    "Data Sharing",
    "Penalty Clause",
    "Safe",
    "Needs Review",
]
ProposalRiskLabel = Literal[
    "Auto-Renewal",
    "Hidden Fee",
    "Liability Waiver",
    "Data Sharing",
    "Penalty Clause",
    "Safe",
]


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class ParsedClause(BaseModel):
    clause_id: str
    text: str
    page_number: int | None = None
    block_number: int | None = None
    section_heading: str | None = None
    extraction_method: Literal["pymupdf", "pdfplumber", "ocr", "docx", "txt"]
    bounding_box: BoundingBox | None = None


class SimplificationCandidate(BaseModel):
    text: str
    fk_grade: float
    semantic_preservation_score: float = Field(ge=0, le=1)
    semantic_entailment: float = Field(ge=0, le=1)
    semantic_neutral: float | None = Field(default=None, ge=0, le=1)
    semantic_contradiction: float | None = Field(default=None, ge=0, le=1)
    attempt: int = Field(ge=1)
    temperature: float | None = None
    accepted: bool = False


class SimplificationResult(BaseModel):
    text: str
    fk_grade: float
    semantic_preservation_score: float = Field(ge=0, le=1)
    semantic_target_met: bool
    readability_target_met: bool
    acceptance_target_met: bool
    accepted: bool
    attempts: int = Field(ge=1)
    status: Literal["accepted", "needs_review"]
    candidates: list[SimplificationCandidate] = Field(default_factory=list)
    generation_mode: Literal["stochastic", "deterministic"] = "stochastic"
    generation_seed: int | None = None
    initial_temperature: float | None = None
    retry_temperatures: list[float] = Field(default_factory=list)
    num_retries: int = 0
    output_hash: str | None = None


class EvidenceChunk(BaseModel):
    chunk_id: str
    source: str
    jurisdiction: str
    source_url: str
    section: str | None = None
    source_anchor: str | None = None
    document_version: str | None = None
    fetched_at: str | None = None
    content_sha256: str | None = None
    page_number: int | None = None
    text: str
    token_start: int | None = None
    token_end: int | None = None
    retrieval_score: float | None = None
    rerank_score: float | None = None


class SentenceFaithfulness(BaseModel):
    sentence: str
    faithfulness_score: float | None = Field(default=None, ge=0, le=1)
    faithfulness_premise: Literal["retrieved_chunk", "source_clause", "dual"] = "retrieved_chunk"
    source_faithfulness: float = Field(ge=0, le=1)
    source_supported: bool
    source_neutral: float | None = Field(default=None, ge=0, le=1)
    source_contradiction: float | None = Field(default=None, ge=0, le=1)
    regulatory_support: float | None = Field(default=None, ge=0, le=1)
    regulatory_supported: bool | None = None
    regulatory_neutral: float | None = Field(default=None, ge=0, le=1)
    regulatory_contradiction: float | None = Field(default=None, ge=0, le=1)
    unsupported: bool
    unsupported_reason: str | None = None
    attribution_chunk_id: str | None = None
    attribution: EvidenceChunk | None = None
    attribution_candidate_chunk_id: str | None = None
    attribution_candidate_score: float | None = Field(default=None, ge=0, le=1)
    attribution_candidate: EvidenceChunk | None = None

    @model_validator(mode="after")
    def fill_primary_score(self) -> "SentenceFaithfulness":
        if self.faithfulness_score is None:
            if self.faithfulness_premise == "source_clause":
                self.faithfulness_score = self.source_faithfulness
            elif self.faithfulness_premise == "retrieved_chunk":
                self.faithfulness_score = self.regulatory_support
            elif self.regulatory_support is not None:
                self.faithfulness_score = min(self.source_faithfulness, self.regulatory_support)
            else:
                self.faithfulness_score = self.source_faithfulness
        return self


class RegulatoryAttribution(BaseModel):
    sentence: str
    regulatory_support: float = Field(ge=0, le=1)
    regulatory_neutral: float | None = Field(default=None, ge=0, le=1)
    regulatory_contradiction: float | None = Field(default=None, ge=0, le=1)
    supported: bool
    attribution_chunk_id: str | None = None
    attribution: EvidenceChunk | None = None
    candidate_chunk_id: str | None = None
    candidate: EvidenceChunk | None = None


class RiskPrediction(BaseModel):
    risk_label: RiskLabel
    risk_score: int = Field(ge=1, le=5)
    confidence: float = Field(ge=0, le=1)
    explanation: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    # true only if the cited evidence is actually specific to this clause, not just generic boilerplate that happens to look similar
    evidence_relevant: bool = True
    secondary_risk_labels: list[ProposalRiskLabel] = Field(default_factory=list)
    parse_attempts: int = 1


class ClauseResult(BaseModel):
    clause_id: str
    original_text: str
    page_number: int | None = None
    section_heading: str | None = None
    extraction_method: Literal["pymupdf", "pdfplumber", "ocr", "docx", "txt"] | None = None
    simplified_text: str
    generated_simplified_text: str | None = None
    simplification_suppressed: bool = False
    readability_grade: float | None = None
    readability_target_met: bool | None = None
    semantic_preservation_score: float | None = None
    semantic_target_met: bool | None = None
    simplification_accepted: bool | None = None
    simplification_attempts: int | None = None
    simplification_status: Literal["accepted", "needs_review"] | None = None
    risk_label: RiskLabel
    risk_score: int
    risk_confidence: float = 0.0
    risk_explanation: str = ""
    secondary_risk_labels: list[ProposalRiskLabel] = Field(default_factory=list)
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    faithfulness: list[SentenceFaithfulness] = Field(default_factory=list)
    risk_explanation_support: list[RegulatoryAttribution] = Field(default_factory=list)
    mean_source_faithfulness: float | None = None
    mean_regulatory_support: float | None = None
    mean_primary_faithfulness: float | None = None
    mean_risk_regulatory_support: float | None = None
    mean_faithfulness: float | None = None
    unsupported_flag_rate: float | None = None
    hallucination_rate: float | None = None
    retrieval_coverage: float | None = None
    supported_attribution_coverage: float | None = None
    attribution_coverage: float | None = None
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    routed_for_review: bool | None = None
    routing_reasons: list[str] = Field(default_factory=list)
    routing_policy: str | None = None

    @model_validator(mode="after")
    def aliases(self) -> "ClauseResult":
        if self.mean_faithfulness is None:
            self.mean_faithfulness = self.mean_primary_faithfulness or self.mean_source_faithfulness
        if self.unsupported_flag_rate is None:
            self.unsupported_flag_rate = self.hallucination_rate
        if self.hallucination_rate is None:
            self.hallucination_rate = self.unsupported_flag_rate
        if self.supported_attribution_coverage is None:
            self.supported_attribution_coverage = self.attribution_coverage
        if self.attribution_coverage is None:
            self.attribution_coverage = self.supported_attribution_coverage
        return self


class DocumentMetrics(BaseModel):
    clause_count: int = 0
    sentence_count: int = 0
    risky_clause_count: int = 0
    needs_review_count: int = 0
    simplification_review_count: int = 0
    unsupported_sentence_count: int = 0
    mean_source_faithfulness: float | None = None
    mean_regulatory_support: float | None = None
    mean_primary_faithfulness: float | None = None
    mean_risk_regulatory_support: float | None = None
    unsupported_flag_rate: float | None = None
    hallucination_rate: float | None = None
    retrieval_coverage: float | None = None
    supported_attribution_coverage: float | None = None
    attribution_coverage: float | None = None
    mean_fk_grade: float | None = None
    pct_fk_le_8: float | None = None
    pct_fk_le_9: float | None = None
    pct_semantic_target_met: float | None = None
    routing_policy: str | None = None
    routed_clause_count: int = 0
    routing_reason_counts: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def aliases(self) -> "DocumentMetrics":
        if self.unsupported_flag_rate is None:
            self.unsupported_flag_rate = self.hallucination_rate
        if self.hallucination_rate is None:
            self.hallucination_rate = self.unsupported_flag_rate
        if self.supported_attribution_coverage is None:
            self.supported_attribution_coverage = self.attribution_coverage
        if self.attribution_coverage is None:
            self.attribution_coverage = self.supported_attribution_coverage
        return self


class ModelProvenance(BaseModel):
    simplifier_model: str
    simplifier_adapter: str | None = None
    simplifier_adapter_sha256: str | None = None
    embedding_model: str
    embedding_dimensions: int
    reranker_model: str
    risk_model: str
    risk_adapter: str | None = None
    risk_adapter_sha256: str | None = None
    verifier_model: str
    verifier_adapter: str | None = None
    verifier_adapter_sha256: str | None = None
    faithfulness_tau: float
    attribution_tau: float
    faithfulness_primary_premise: str
    document_sha256: str | None = None
    regulatory_corpus_sha256: str | None = None
    regulatory_source_manifests: list[dict[str, Any]] = Field(default_factory=list)
    runtime_manifest: dict[str, Any] = Field(default_factory=dict)
    config_fingerprint: str


class DocumentResult(BaseModel):
    filename: str
    clauses: list[ClauseResult] = Field(default_factory=list)
    metrics: DocumentMetrics = Field(default_factory=DocumentMetrics)
    models: ModelProvenance
    regulatory_sources: list[str] = Field(default_factory=list)
    total_runtime_ms: float | None = None
    warnings: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    disclaimer: str = (
        "FinLingo++ is an academic research prototype. It does not provide legal or financial advice. "
        "A low support score is a review signal, not a legal conclusion. Verify important terms against "
        "the original document and consult a qualified professional."
    )
