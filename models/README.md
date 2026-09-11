# Trained model weights

Only the weights actually used to produce the reported final numbers are kept here,
to keep the submission a reasonable size.

- `stage5_risk_classifier_v5/` — LoRA adapter for the S5 risk classifier
  (legal-bert-base-uncased backbone). This is the deployed model behind the
  reported RQ2 result (macro-F1 = 0.79). Referenced by `RISK_CLASSIFIER_ADAPTER`
  in `.env.example`.
- `stage6_verifier_v3_financial/` — LoRA adapter for the S6 faithfulness verifier
  (DeBERTa-v3-large backbone), used for the FaithBench reproduction comparison
  reported in the results (fine-tuned model beats the base model by +0.038
  balanced accuracy on short-premise NLI pairs). Note: the *deployed* pipeline
  configuration (`verifier_use_base_model=True` in `backend/app/core/config.py`)
  actually runs the base model by default for the long-premise RQ3 grounding
  task, because the fine-tuned adapter was found to under-perform there — see
  `reports/proposal_gap_analysis.md` section 8 for the full, disclosed reasoning.
  This adapter is kept because it is a real, reported result, not because it's
  the active default.

- `stage6_modernbert_verifier/` — LoRA adapter for the ModernBERT-large
  faithfulness-verifier comparator. Its headline result (precision/recall
  both 0.0 at the deployed threshold) is a degenerate, always-predicts
  "unsupported" outcome, not a promising model — see
  `reports/multiverifier.json`. Kept (rather than cut for space) because
  `scripts/diagnose_modernbert_verifier.py` uses these exact weights to run
  a controlled-pair diagnostic that shows the label mapping and classification
  head are wired correctly (predictions are not constant on 14 hand-written
  test pairs) and the failure is a genuine training-data-budget limitation,
  not a broken adapter; see `reports/modernbert_controlled_pair_diagnosis.json`.

Earlier training attempts (`stage5_risk_classifier_v1`–`v4` and assorted
training caches) produced weaker, fully-superseded results and were left out
of this submission copy to avoid shipping redundant weights. Their outcomes
are still fully documented in `reports/proposal_gap_analysis.md` and
`reports/final_summary.txt` even though those particular weight files aren't
included.
