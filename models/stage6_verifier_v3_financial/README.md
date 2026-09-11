---
base_model: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
library_name: peft
tags:
- lora
- transformers
- text-classification
- natural-language-inference
---

# FinLingo++ Stage 6 Faithfulness Verifier (v3, financial domain)

LoRA adapter fine-tuned on top of `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
for entailment scoring: given a premise (regulatory chunk or source clause) and a
hypothesis (a simplified sentence), predict entailment/neutral/contradiction.

## Model description

- **Base model:** DeBERTa-v3-large, already pretrained on MNLI/FEVER/ANLI/LingNLI/WANLI.
- **Fine-tuning method:** LoRA (PEFT) on SNLI, ContractNLI, and synthetic
  financial-domain entailment pairs.
- **Training data:** general NLI datasets plus contract-specific pairs, not the
  full FLB benchmark itself (to avoid train/test leakage).

## Intended use and an important deployment note

This adapter is used for the **FaithBench comparison** reported in
`reports/final_results.json` (fine-tuned model beats the base model by +0.038
balanced accuracy, 95% CI [0.006, 0.072], p=0.02 — a real, statistically
significant improvement on short-premise NLI pairs).

**It is not the default verifier in the deployed pipeline.** Direct testing
found this fine-tuned adapter shows almost no discrimination between
genuinely-supported and genuinely-unsupported pairs when the premise is a long
regulatory chunk (the RQ3 grounding task) — entailment scores for both classes
cluster around 0.35–0.40 regardless of the true label. The base (un-adapted)
model discriminates better on this specific long-premise/short-hypothesis input
shape, so `verifier_use_base_model=True` in `backend/app/core/config.py` runs
the base model for RQ3 scoring by default. Full comparison and root cause in
`reports/proposal_gap_analysis.md`, section 8.

In short: this adapter helps on short NLI pairs, hurts on long regulatory-chunk
premises. Both results are real and both are reported — it is kept in this
submission because it produced a genuine, disclosed comparative finding, not
because it's what actually runs by default.

### Framework versions

- PEFT 0.19.1
