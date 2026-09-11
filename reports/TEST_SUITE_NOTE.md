# Test suite count: note on current archive state

`reports/test_suite_summary.json` records 183 collected / 181 passed / 2 skipped /
0 failed, generated earlier in the project.

A later cleanup pass, done after that summary was generated, removed some
exploratory/diagnostic artefacts from the archive. Running `pytest -q` against
the current archive collects 178 tests (168 passed, 10 skipped, 0 failed):

- 5 fewer collected tests: `tests/test_final_evidence_workflow.py` was removed
  together with the internal-audit script it tested
  (`scripts/run_final_evidence_workflow.py`), which was excluded as
  non-essential tooling, not part of the deployed pipeline.
- 8 additional skips, all `pytest.mark.skipif` guards that were always present
  in the test files, now triggered because their target diagnostic files are
  no longer in `reports/`: `test_modernbert_controlled_pairs.py` (3 skips,
  needs `reports/modernbert_controlled_pair_diagnosis.json`) and
  `test_retrieval_reranker_deep_dive.py` (5 skips, needs
  `reports/retrieval_reranker_deep_dive.json`).

No test fails in either state. No metric, model, threshold, dataset, or
evaluation-logic file is affected by this. `reports/test_suite_summary.json`
is left unmodified as a historical artefact; this note records the current,
real count for anyone reconciling the two.
