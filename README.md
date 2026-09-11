# FinLingo++

> **AI-powered financial and legal contract simplifier with risk classification and regulatory faithfulness verification.**

---

## What is FinLingo++?

Financial and legal contracts are notoriously difficult to understand — dense language, buried obligations, and regulatory jargon that most people can't parse without a lawyer. FinLingo++ addresses this by automatically breaking a contract into its individual clauses, rewriting each one in plain English, flagging risk categories (e.g. liability, privacy, indemnification), and verifying that the simplified version is actually faithful to the underlying regulatory text.

The system combines a retrieval-augmented pipeline with fine-tuned transformer adapters for risk classification (Stage 5) and faithfulness verification (Stage 6), achieving **macro-F1 0.793** on risk classification and **precision 0.828 / recall 0.791** on source-premise verification. It runs as both a CLI tool and a React web app backed by a FastAPI server.

---

## Pipeline overview

Takes a financial/legal contract, breaks it into clauses, rewrites each one in
plain English, tags a risk category, and checks the rewrite against real
regulatory text to see if it's actually faithful. 7 stages:

1. parse the document into clauses (OCR fallback for scanned PDFs)
2. simplify each clause
3. retrieve relevant regulatory text
4. rerank (off by default, hurt recall on this corpus)
5. classify risk category
6. check faithfulness against the regulatory text
7. build the final report

## Folders

```
backend/        FastAPI app + the 7 stages + training scripts
frontend-react/ React/Vite UI
main.py         CLI entry point, same pipeline as the UI
scripts/        scripts RUN_CONFIGS.md points to for reproducing specific numbers
configs/        few-shot prompts, regulatory sources list
models/         the deployed S5/S6 adapters, plus a ModernBERT comparator that didn't work out (kept for the diagnostic)
tests/          pytest
figures/        plots + the script that makes them
reports/        results, write-up, benchmark data
sample_documents/  a couple of test contracts
```

## Running the submitted runtime

Python 3.11 is recommended. Node.js is required only for the optional React
front end. Tesseract is required only for scanned-PDF OCR. Ollama is required
only for the prompted-classifier and other explicitly documented optional
workflows; it is not required by the trained S5 path used in submitted
Configuration B.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python main.py preflight --final
```

Run on one document:

```bash
python main.py analyse sample_documents/some_contract.pdf --output result.json
```

Or run the full app:

```bash
python main.py serve
```

Frontend dev:

```bash
cd frontend-react
npm install
npm run dev
```

`main.py --help` for everything else (training/eval/ablation commands — a lot
of them, this codebase also ran all the thesis experiments, not just the app).

Tests:

```bash
pytest -q
```

## Submitted results and authoritative artefact

The authoritative source for the submitted Configuration B RQ1, RQ2, and
source-premise RQ3 figures is:

`reports/evaluation_trained_s5_frozen_e2e.json`

| Component | Submitted Configuration B result | Scope |
|---|---:|---|
| Simplification | SARI 54.03; BERTScore F1 0.899; mean FK grade 6.15 | Accepted frozen FLB set |
| Risk classification | macro-F1 0.793 | Trained S5 inside the frozen evaluator chain |
| Source-premise verification | precision 0.828; recall 0.791 | Threshold τ = 0.376 |
| Regulatory retrieval | recall@15 0.099 | No independently adjudicated regulatory gold |
| Retrieval configuration | BM25, combined query, reranker disabled | Submitted Configuration B |

`reports/final_results.json` retains historical and consolidated experiment
blocks for provenance. Where its RQ1 or RQ2 values differ, the frozen
Configuration B artefact named above is authoritative for the submitted report.

Further provenance and configuration mapping are documented in
`reports/CANONICAL_ARTEFACTS.md` and `RUN_CONFIGS.md`.

Do not edit the result JSON files themselves.

Full write-up (what worked, what didn't, why) is in `reports/final_summary.txt`
and `reports/proposal_gap_analysis.md`.

## Note on model choices

Used Qwen2.5 instead of Mistral-7B for a couple of stages — GPU memory reasons,
this ran on one consumer GPU not a cloud box. Mentioned here and in the reports
so it's not a surprise later.
