# Which config produced which number

This project retains several deliberately different experimental
configurations. Configuration B is the submitted runtime and the source of the
submitted frozen RQ1, trained-S5 RQ2, and source-premise RQ3 figures.
Configurations A, C, and D are retained for historical comparisons, direct
component evaluation, ablations, and the exploratory RQ4 pilot. This file maps
each reported number to its exact configuration and artefact.

## Config A — Historical canonical comparison configuration

Used for: the historical prompted-classifier result, regulatory-chunk premise
comparison, FaithBench comparison, multi-verifier comparison, and relevant
ablation experiments. Its saved results are retained for provenance and
comparison; it is not the source of the submitted Configuration B RQ1,
trained-S5 RQ2, or source-premise RQ3 headline figures.

| Setting | Value |
|---|---|
| Retrieval | FAISS (dense), query mode = simplified |
| Reranker | On (L-12 cross-encoder) |
| S2 backend | Ollama |
| S5 backend | Prompted (few-shot, six-class) |
| S6 verifier | Base DeBERTa-v3-large |
| τ (faithfulness / attribution) | 0.75 |
| Source file | `reports/evaluation_achievable_v2.json` |
| Command | `python main.py evaluate` (default config as pinned in `reports/pinned_revisions.json`) |

Note: this table previously (incorrectly) cited `reports/evaluation_final5.json` as the
source and listed BM25 retrieval. Verified while building the configuration
registry: `evaluation_final5.json` is a different, earlier run (macro-F1 0.544, source-premise
precision/recall 0.962/0.560 -- neither matches the reported headline). The file that actually
reproduces every headline RQ2/RQ3 number exactly (macro-F1 0.5628, chunk-premise 1.0/0.00556,
source-premise 0.90625/0.6374) is `evaluation_achievable_v2.json`, whose own recorded
`configuration` block shows `retrieval_method="faiss"`, `retrieval_query_mode="simplified"`,
`use_reranker=true` -- corrected here to match.

## Config B — "Live deployment" (what `.env` actually runs today)

Used for: the submitted report's frozen RQ1 figures, trained-S5 RQ2 figure,
and source-premise RQ3 precision/recall. The authoritative run artefact is
`reports/evaluation_trained_s5_frozen_e2e.json`.

| Setting | Value |
|---|---|
| Retrieval | BM25, query mode = combined |
| Reranker | Off |
| S2 backend | Local Qwen2.5-1.5B-Instruct, no deployed S2 adapter |
| S5 backend | Trained (`models/stage5_risk_classifier_v5`) |
| S6 verifier | Base DeBERTa-v3-large |
| τ | 0.376 |
| Source file | `reports/evaluation_trained_s5_frozen_e2e.json` |
| Command | Frozen evaluator using Configuration B |

## Config C — Trained risk-classifier component (direct evaluation, no pipeline)

Used for: macro-F1 0.790 (RQ2 component result).

| Setting | Value |
|---|---|
| Input | Clause text only, no retrieval, no S1/S3/S4 |
| Model | `models/stage5_risk_classifier_v5` (legal-bert-base-uncased + LoRA) |
| Source file | `reports/quick_rq2_v5.log`, `reports/final_results.json` |
| Command | `python scripts/score_existing_flb_export.py` (direct classifier scoring path) |

## Config D — RQ4 document-condition pilot

Used for: Table 6/7 (risk macro-F1 / FK grade / faithfulness recall per condition).

| Setting | Value |
|---|---|
| Sample | 10 clauses (2 Auto-Renewal, 8 Data Sharing) from `reports/benchmark_dataset/flb_final_check3_export.json` |
| τ | 0.376 (live-deployment value) |
| Retrieval | FAISS (see `backend/experiments/failure_analysis.py::run_failure_analysis`) |
| Source file | `reports/failure_analysis_rq4.json` |
| Command | `python scripts/run_rq4_from_export.py` |

## Why the historical configurations remain

Configuration A is retained for historical prompted-path, ablation, and
comparison experiments. Configuration B is the submitted runtime and the
configuration behind the frozen evaluator artefact used for the submitted
RQ1, trained-S5 RQ2, and source-premise RQ3 headline figures.

Configuration C remains a direct component-only classifier evaluation, while
Configuration D remains the limited RQ4 stress-test configuration. Results
from these configurations must not be combined as though they came from one
identical run.

## Exact commands and artefacts for reproduction

```powershell
# Submitted Configuration B frozen RQ1, trained-S5 RQ2,
# and source-premise RQ3 results
python main.py evaluate --output reports\rerun_configuration_b.json

# Trained classifier direct component result (Configuration C)
python scripts\score_existing_flb_export.py

# RQ4 pilot (Configuration D)
python scripts\run_rq4_from_export.py

# Test suite
python -m pytest -q
```

The commands use saved data and locally available model artefacts. Ollama is
required only for workflows that explicitly use the historical prompted path;
first-time Hugging Face model downloads require internet access when the
relevant checkpoints are not already cached.
