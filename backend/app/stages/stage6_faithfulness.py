"""Stage 6 - checks simplified sentences against retrieved regulatory chunks via NLI to catch unsupported claims.
Base DeBERTa-v3 is deployed since the fine-tuned adapter underperformed on long premises."""

from __future__ import annotations

from backend.app.core.config import Settings
from backend.app.core.schemas import EvidenceChunk, RegulatoryAttribution, SentenceFaithfulness
from backend.app.services.model_registry import ModelRegistry
from backend.app.services.nli_service import NLIService, NLIScore
from backend.app.services.text_utils import split_sentences


class Stage6Faithfulness:
    """Sentence-level NLI verification with dual grounding (source clause + regulatory chunk);
    attribution is emitted only above ``ATTRIBUTION_TAU``."""

    def __init__(self, settings: Settings, registry: ModelRegistry) -> None:
        self.settings = settings
        if settings.verifier_use_base_model:
            # the fine-tuned verifier adapter barely separated supported from unsupported pairs on long
            # regulatory premises - the un-adapted base model does much better here, so that's what's deployed
            self.nli = NLIService(registry, model_id=settings.verifier_model, adapter_id=None, require_adapter=False)
        else:
            self.nli = NLIService(registry, require_adapter=settings.verifier_require_adapter)

    def verify(
        self,
        original_clause: str,
        simplified_text: str,
        chunks: list[EvidenceChunk],
    ) -> list[SentenceFaithfulness]:
        sentences = split_sentences(simplified_text)
        if not sentences:
            return []
        source_scores = self.nli.score_pairs(
            [(original_clause, sentence) for sentence in sentences],
            self.settings.verifier_batch_size,
            self.settings.verifier_max_length,
        )
        regulatory_scores = (
            self.nli.score_pairs(
                [(chunk.text, sentence) for sentence in sentences for chunk in chunks],
                self.settings.verifier_batch_size,
                self.settings.verifier_max_length,
            )
            if chunks
            else []
        )

        output: list[SentenceFaithfulness] = []
        for index, (sentence, source) in enumerate(zip(sentences, source_scores)):
            best_chunk: EvidenceChunk | None = None
            best_reg: NLIScore | None = None
            if chunks:
                subset = regulatory_scores[index * len(chunks) : (index + 1) * len(chunks)]
                best_index = max(range(len(subset)), key=lambda item: subset[item].entailment)
                best_chunk = chunks[best_index]
                best_reg = subset[best_index]

            source_supported = source.entailment >= self.settings.faithfulness_tau
            regulatory_supported = (
                best_reg.entailment >= self.settings.faithfulness_tau if best_reg is not None else False
            )
            primary_score, primary_supported, reason = self._primary_decision(
                source.entailment,
                best_reg.entailment if best_reg is not None else None,
            )
            attribution_supported = (
                best_chunk is not None and best_reg is not None and best_reg.entailment >= self.settings.attribution_tau
            )
            output.append(
                SentenceFaithfulness(
                    sentence=sentence,
                    faithfulness_score=round(primary_score, 6),
                    faithfulness_premise=self.settings.faithfulness_primary_premise,
                    source_faithfulness=round(source.entailment, 6),
                    source_supported=source_supported,
                    source_neutral=_round_optional(getattr(source, "neutral", None)),
                    source_contradiction=_round_optional(getattr(source, "contradiction", None)),
                    regulatory_support=_round_optional(best_reg.entailment if best_reg else None),
                    regulatory_supported=regulatory_supported if best_reg is not None else False,
                    regulatory_neutral=_round_optional(getattr(best_reg, "neutral", None) if best_reg else None),
                    regulatory_contradiction=_round_optional(
                        getattr(best_reg, "contradiction", None) if best_reg else None
                    ),
                    unsupported=not primary_supported,
                    unsupported_reason=reason if not primary_supported else None,
                    attribution_chunk_id=best_chunk.chunk_id if attribution_supported else None,
                    attribution=best_chunk if attribution_supported else None,
                    attribution_candidate_chunk_id=best_chunk.chunk_id if best_chunk else None,
                    attribution_candidate_score=_round_optional(best_reg.entailment if best_reg else None),
                    attribution_candidate=best_chunk,
                )
            )
        return output

    def verify_regulatory_text(
        self,
        text: str,
        chunks: list[EvidenceChunk],
    ) -> list[RegulatoryAttribution]:
        """Attribute each risk-explanation sentence to regulatory evidence."""
        sentences = split_sentences(text)
        if not sentences:
            return []
        if not chunks:
            return [
                RegulatoryAttribution(
                    sentence=sentence,
                    regulatory_support=0.0,
                    supported=False,
                )
                for sentence in sentences
            ]
        scores = self.nli.score_pairs(
            [(chunk.text, sentence) for sentence in sentences for chunk in chunks],
            self.settings.verifier_batch_size,
            self.settings.verifier_max_length,
        )
        output: list[RegulatoryAttribution] = []
        for index, sentence in enumerate(sentences):
            subset = scores[index * len(chunks) : (index + 1) * len(chunks)]
            best_index = max(range(len(subset)), key=lambda item: subset[item].entailment)
            best_chunk = chunks[best_index]
            best_score = subset[best_index]
            supported = best_score.entailment >= self.settings.attribution_tau
            output.append(
                RegulatoryAttribution(
                    sentence=sentence,
                    regulatory_support=round(best_score.entailment, 6),
                    regulatory_neutral=_round_optional(getattr(best_score, "neutral", None)),
                    regulatory_contradiction=_round_optional(getattr(best_score, "contradiction", None)),
                    supported=supported,
                    attribution_chunk_id=best_chunk.chunk_id if supported else None,
                    attribution=best_chunk if supported else None,
                    candidate_chunk_id=best_chunk.chunk_id,
                    candidate=best_chunk,
                )
            )
        return output

    def _primary_decision(
        self,
        source_score: float,
        regulatory_score: float | None,
    ) -> tuple[float, bool, str]:
        mode = self.settings.faithfulness_primary_premise
        if mode == "source_clause":
            return (
                source_score,
                source_score >= self.settings.faithfulness_tau,
                "source clause does not entail sentence",
            )
        if mode == "dual":
            score = min(source_score, regulatory_score if regulatory_score is not None else 0.0)
            if source_score < self.settings.faithfulness_tau:
                reason = "source clause does not entail sentence"
            elif regulatory_score is None:
                reason = "no regulatory evidence was retrieved"
            else:
                reason = "retrieved evidence does not entail sentence"
            return score, score >= self.settings.faithfulness_tau, reason
        score = regulatory_score if regulatory_score is not None else 0.0
        reason = (
            "no regulatory evidence was retrieved"
            if regulatory_score is None
            else "retrieved evidence does not entail sentence"
        )
        return score, score >= self.settings.faithfulness_tau, reason


def _round_optional(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None
