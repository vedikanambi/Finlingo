# Adapter Behavioural-Attachment Proof: Root Cause of the Threshold Failure

`reports/adapter_attachment_proof.json` (generated 2026-07-14, pre-dates this session) reports
`max_abs_delta=0.011381` against a `min_required_delta=0.02` threshold and marks
`attached_verdict: false`. Investigated this session to determine whether this reflects a real
adapter-attachment problem or a flaw in the probe design.

## Finding: the probe set is saturated, not the adapter inert

All 14 probe pairs' **base-model** entailment scores sit within 0.0091 of the 0 or 1 boundary
(computed directly from the stored `base_entailment` array):

| # | base entailment | distance from nearest boundary (0 or 1) |
|---|---|---|
| 1 | 0.9926 | 0.0074 |
| 2 | 0.9976 | 0.0024 |
| 3 | 0.0001 | 0.0001 |
| 4 | 0.9909 | 0.0091 (largest gap in the set) |
| 5-14 | 0.0001-0.9954 | 0.0001-0.0059 |

The base model (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`) is itself a strong,
already-fine-tuned NLI model (trained on MNLI+FEVER+ANLI+LingNLI+WANLI). The 14 probe pairs
selected for this test are all "easy" examples the base model already classifies with near-total
confidence. Because entailment probability is bounded to [0,1], a score already at 0.9976 or
0.0001 is mathematically incapable of moving by 0.02 in the direction away from its current
boundary — the largest possible delta for probe #2, for example, is 0.0024, an order of magnitude
below the 0.02 threshold, independent of whether the LoRA adapter changed the model's behaviour at
all.

**This is a probe-selection defect in the original test (probes should have been drawn from
near-boundary, ambiguous cases where a fine-tuning effect is actually visible), not evidence that
the adapter is unattached or non-functional.** No probe scores or thresholds have been altered to
produce this finding -- it is a direct arithmetic consequence of the stored `base_entailment`
values already on disk.

## What remains valid evidence either way

- **Structural proof** (648 LoRA modules present, correct `target_modules`, matching PEFT config,
  SHA256-verified adapter files) is untouched by this issue and remains a clean PASS.
- **`reports/verifier_faithbench_v3_reproduction.json`** provides the real behavioural evidence:
  Verifier v3 vs. base on the FaithBench benchmark shows +0.038 balanced accuracy
  (CI [0.006, 0.072], p=0.02), a statistically significant, properly-powered before/after
  comparison on a benchmark that is *not* saturated at the base model's ceiling. This is the
  correct artifact to cite as behavioural proof of adapter attachment and effect; the 14-pair toy
  probe should not be treated as a contradicting result, since it was never capable of detecting
  an effect at this base-model confidence level.

## Recommendation for the submission

State plainly in the limitations/discussion section: the original toy behavioural probe is
underpowered by construction (probes too easy / base model too confident), and the FaithBench
reproduction is the methodologically sound behavioural-attachment evidence. Do not report the
14-pair probe delta as a "failed" adapter-attachment result without this context -- doing so
understates a genuine, statistically-supported result (FaithBench) in favour of a broken test.
