"""Stage 2 - rewrites a clause in plain English, checking readability and meaning preservation;
retries and falls back to a mechanical sentence-split if the target isn't hit."""

import hashlib

from backend.app.core.config import Settings
from backend.app.core.schemas import SimplificationCandidate, SimplificationResult
from backend.app.services.model_registry import ModelRegistry
from backend.app.services.nli_service import NLIService
from backend.app.services.text_generation import TextGenerator
from backend.app.services.text_utils import flesch_kincaid_grade, normalise_text, split_long_clauses


class Stage2Simplification:
    """Never silently accept a rewrite that fails readability or semantic checks."""

    def __init__(self, settings: Settings, registry: ModelRegistry) -> None:
        self.settings = settings
        self.generator = TextGenerator(settings, registry)
        self.nli = NLIService(
            registry,
            model_id=settings.s2_semantic_model,
            adapter_id=settings.s2_semantic_adapter,
            require_adapter=settings.s2_semantic_require_adapter,
            use_verifier_calibration=False,
        )

    def simplify(self, text: str) -> SimplificationResult:
        if self.settings.s2_deterministic_mode:
            return self._simplify_deterministic(text)
        return self._simplify_stochastic(text)

    def _simplify_deterministic(self, text: str) -> SimplificationResult:
        """Greedy, no retries - same input always gives the same output_hash."""
        instructions = (
            "Simplify consumer financial/legal clauses into accurate plain English. Preserve every amount, date, "
            "condition, exception, obligation, party, negation and legal effect. Add no facts or advice. Return only text."
        )
        prompt = f"Rewrite at Flesch-Kincaid grade {self.settings.fk_generation_target:g} or lower.\n\nCLAUSE:\n{text}"
        candidate_text = normalise_text(
            self.generator.generate(
                instructions=instructions,
                user_input=prompt,
                model_id=self.settings.simplifier_model,
                adapter_id=self.settings.simplifier_adapter,
                purpose="S2 simplifier",
                max_new_tokens=self.settings.simplifier_max_new_tokens,
                temperature=0.0,
                max_input_tokens=self.settings.simplifier_max_input_tokens,
            )
            .strip()
            .strip('"')
        )
        if not candidate_text:
            raise RuntimeError("S2 (deterministic mode) returned empty text")

        grade = flesch_kincaid_grade(candidate_text)
        nli = self.nli.score_pairs(
            [(text, candidate_text)],
            self.settings.verifier_batch_size,
            self.settings.verifier_max_length,
        )[0]
        row = SimplificationCandidate(
            text=candidate_text,
            fk_grade=grade,
            semantic_preservation_score=nli.entailment,
            semantic_entailment=nli.entailment,
            semantic_neutral=nli.neutral,
            semantic_contradiction=nli.contradiction,
            attempt=1,
            temperature=0.0,
            accepted=True,
        )
        semantic_met = row.semantic_preservation_score >= self.settings.simplifier_min_semantic_entailment
        generation_met = row.fk_grade <= self.settings.fk_generation_target
        acceptance_met = row.fk_grade <= self.settings.fk_acceptance_target
        final_accepted = bool(semantic_met and generation_met)
        return SimplificationResult(
            text=row.text,
            fk_grade=row.fk_grade,
            semantic_preservation_score=row.semantic_preservation_score,
            semantic_target_met=semantic_met,
            readability_target_met=generation_met,
            acceptance_target_met=acceptance_met,
            accepted=final_accepted,
            attempts=1,
            status="accepted" if final_accepted else "needs_review",
            candidates=[row],
            generation_mode="deterministic",
            generation_seed=self.settings.random_seed,
            initial_temperature=0.0,
            retry_temperatures=[],
            num_retries=0,
            output_hash=hashlib.sha256(row.text.encode("utf-8")).hexdigest(),
        )

    def _simplify_stochastic(self, text: str) -> SimplificationResult:
        instructions = (
            "Simplify consumer financial/legal clauses into accurate plain English. Preserve every amount, date, "
            "condition, exception, obligation, party, negation and legal effect. Add no facts or advice. Return only text."
        )
        prompt = f"Rewrite at Flesch-Kincaid grade {self.settings.fk_generation_target:g} or lower.\n\nCLAUSE:\n{text}"
        candidates: list[SimplificationCandidate] = []
        attempt_temperatures: list[float] = []

        for attempt in range(1, self.settings.simplifier_retries + 2):
            # bump temperature each retry so it doesn't just regenerate the same rejected phrasing
            attempt_temperature = min(
                self.settings.simplifier_temperature
                + (attempt - 1) * self.settings.simplifier_retry_temperature_step,
                self.settings.simplifier_retry_temperature_cap,
            )
            attempt_temperatures.append(attempt_temperature)
            candidate = normalise_text(
                self.generator.generate(
                    instructions=instructions,
                    user_input=prompt,
                    model_id=self.settings.simplifier_model,
                    adapter_id=self.settings.simplifier_adapter,
                    purpose="S2 simplifier",
                    max_new_tokens=self.settings.simplifier_max_new_tokens,
                    temperature=attempt_temperature,
                    max_input_tokens=self.settings.simplifier_max_input_tokens,
                )
                .strip()
                .strip('"')
            )
            if not candidate:
                raise RuntimeError("S2 returned empty text")

            grade = flesch_kincaid_grade(candidate)
            nli = self.nli.score_pairs(
                [(text, candidate)],
                self.settings.verifier_batch_size,
                self.settings.verifier_max_length,
            )[0]
            row = SimplificationCandidate(
                text=candidate,
                fk_grade=grade,
                semantic_preservation_score=nli.entailment,
                semantic_entailment=nli.entailment,
                semantic_neutral=nli.neutral,
                semantic_contradiction=nli.contradiction,
                attempt=attempt,
                temperature=attempt_temperature,
            )
            candidates.append(row)

            if self._accepted(row):
                return self._result(row, candidates, accepted=True, attempt_temperatures=attempt_temperatures)

            prompt = (
                f"The previous rewrite had FK grade {grade:.2f} and semantic-entailment score {nli.entailment:.3f}. "
                "Rewrite again with shorter sentences and common words, but restore every legal fact, amount, "
                f"condition, exception, and negation.\n\nORIGINAL:\n{text}\n\nPREVIOUS:\n{candidate}"
            )

        semantically_valid = [
            item
            for item in candidates
            if item.semantic_preservation_score >= self.settings.simplifier_min_semantic_entailment
        ]
        if semantically_valid:
            best = min(semantically_valid, key=lambda item: (item.fk_grade, -item.semantic_preservation_score))
        else:
            best = max(candidates, key=lambda item: (item.semantic_preservation_score, -item.fk_grade))

        if best.fk_grade > self.settings.fk_acceptance_target:
            split = split_long_clauses(best.text)
            if split != best.text:
                split_grade = flesch_kincaid_grade(split)
                if split_grade < best.fk_grade:
                    split_nli = self.nli.score_pairs(
                        [(text, split)],
                        self.settings.verifier_batch_size,
                        self.settings.verifier_max_length,
                    )[0]
                    if split_nli.entailment >= self.settings.simplifier_min_semantic_entailment:
                        best = SimplificationCandidate(
                            text=split,
                            fk_grade=split_grade,
                            semantic_preservation_score=split_nli.entailment,
                            semantic_entailment=split_nli.entailment,
                            semantic_neutral=split_nli.neutral,
                            semantic_contradiction=split_nli.contradiction,
                            attempt=best.attempt,
                            temperature=None,
                        )
                        candidates.append(best)

        result = self._result(best, candidates, accepted=False, attempt_temperatures=attempt_temperatures)
        if self.settings.simplifier_fail_on_unaccepted:
            raise RuntimeError(
                "S2 could not meet both semantic and readability constraints after "
                f"{result.attempts} attempts (best FK={result.fk_grade:.2f}, "
                f"entailment={result.semantic_preservation_score:.3f})."
            )
        return result

    def _accepted(self, item: SimplificationCandidate) -> bool:
        return (
            item.fk_grade <= self.settings.fk_generation_target
            and item.semantic_preservation_score >= self.settings.simplifier_min_semantic_entailment
        )

    def _result(
        self,
        selected: SimplificationCandidate,
        candidates: list[SimplificationCandidate],
        *,
        accepted: bool,
        attempt_temperatures: list[float],
    ) -> SimplificationResult:
        semantic_met = selected.semantic_preservation_score >= self.settings.simplifier_min_semantic_entailment
        generation_met = selected.fk_grade <= self.settings.fk_generation_target
        acceptance_met = selected.fk_grade <= self.settings.fk_acceptance_target
        final_accepted = bool(accepted and semantic_met and generation_met)
        for item in candidates:
            item.accepted = item is selected
        return SimplificationResult(
            text=selected.text,
            fk_grade=selected.fk_grade,
            semantic_preservation_score=selected.semantic_preservation_score,
            semantic_target_met=semantic_met,
            readability_target_met=generation_met,
            acceptance_target_met=acceptance_met,
            accepted=final_accepted,
            attempts=len(candidates),
            status="accepted" if final_accepted else "needs_review",
            candidates=candidates,
            generation_mode="stochastic",
            generation_seed=self.settings.random_seed,
            initial_temperature=attempt_temperatures[0] if attempt_temperatures else self.settings.simplifier_temperature,
            retry_temperatures=attempt_temperatures[1:],
            num_retries=max(len(attempt_temperatures) - 1, 0),
            output_hash=hashlib.sha256(selected.text.encode("utf-8")).hexdigest(),
        )
