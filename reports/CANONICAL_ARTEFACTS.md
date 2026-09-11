# Canonical artefacts for reported RQ1/RQ2/RQ3 figures

`reports/evaluation_trained_s5_frozen_e2e.json` is the canonical source for the
reported RQ1 (SARI, BERTScore F1, mean FK grade), RQ2 (risk macro-F1), and
source-premise RQ3 (precision/recall) figures.

`reports/final_results.json` retains its `rq1_simplification_quality` and
`rq2_risk_classification` blocks from the earlier `evaluation_final5.json` run
for provenance. Those two blocks are superseded for RQ1 and RQ2 by
`evaluation_trained_s5_frozen_e2e.json` and are not deleted or edited.

This file adds no new claims and changes no existing artefact; it restates,
as a standalone pointer, the supersession note already present at
`reports/final_results.json`'s `superseded_for_rq1_rq2` field.

## Benchmark export format

The following compatibility-named `.json` files use JSON Lines/NDJSON format,
with one complete JSON object per line:

- `reports/benchmark_dataset/flb_final_check3_export.json`
- `reports/benchmark_dataset/flb_combined_export.json`
- `reports/benchmark_dataset/flb_ccqa_supplement.json`

The `.json` suffix is retained because existing loaders, tests, hashes, and
saved manifests reference these filenames. They should not be parsed as one
single JSON array or object.

Do not rename or edit those benchmark files.
