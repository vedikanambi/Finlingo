---
base_model: nlpaueb/legal-bert-base-uncased
library_name: peft
tags:
- lora
- transformers
- text-classification
---

# FinLingo++ Stage 5 Risk Classifier (v5)

LoRA adapter fine-tuned on top of `nlpaueb/legal-bert-base-uncased` to classify a
financial/legal contract clause into one of six risk categories: Auto-Renewal,
Hidden Fee, Liability Waiver, Data Sharing, Penalty Clause, or Safe.

## Model description

- **Base model:** `nlpaueb/legal-bert-base-uncased` (BERT pretrained on contracts
  and legislation, chosen over `ProsusAI/finbert` because FinBERT is tuned for
  financial sentiment, a domain mismatch for clause-risk classification).
- **Fine-tuning method:** LoRA (PEFT), with full-parity minority-class
  oversampling during training to address class imbalance across the six risk
  categories.
- **Training data:** clauses drawn from CUAD (CC BY 4.0) and an external
  supplementary pool for classes CUAD underrepresents (Data Sharing, Penalty
  Clause), labelled by risk category.

## Intended use

This adapter is the deployed backend for Stage 5 of the FinLingo++ pipeline
(`RISK_CLASSIFIER_BACKEND=trained` in `.env`). It takes a single clause's text
as input (no retrieved evidence) and outputs a risk label.

## Evaluation results

Real FLB benchmark evaluation (91 clauses), reported in `reports/final_results.json`:

| Metric | Value |
|---|---|
| Macro-F1 | 0.79 |
| Accuracy | 0.82 |
| Macro precision | 0.88 |
| Macro recall | 0.81 |

This beats the target of 0.75 macro-F1, and beats both comparator baselines
tested on the same benchmark: a prompted LLM (0.56) and a frozen-embedding +
logistic-regression classifier (0.69). Full per-class breakdown and confusion
matrix in `reports/final_results.json` and `reports/proposal_gap_analysis.md`.

## Limitations

Per-class recall is uneven: Data Sharing recall is comparatively low (0.29)
while Auto-Renewal and Hidden Fee recall are near-perfect (1.0). See
`reports/final_results.json`'s per-class table for the full breakdown.

### Framework versions

- PEFT 0.19.1
