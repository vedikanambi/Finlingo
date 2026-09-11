"""CLI entry point for the pipeline plus all thesis training/eval/ablation commands.
Run `python main.py --help` for the list."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import configure_logging


def emit(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def stream_check(settings: Settings, limit: int = 1) -> dict[str, dict[str, Any]]:
    from backend.app.services.dataset_streams import StreamingDatasetRepository

    repository = StreamingDatasetRepository(settings)
    checks: dict[str, dict[str, Any]] = {}
    sources = {
        "cuad": lambda: repository.cuad_clauses(limit=limit),
        "contractnli": lambda: repository.contractnli_pairs(limit=limit),
        "snli": lambda: repository.snli_pairs(limit=limit),
        "cfpb": lambda: repository.cfpb_narratives(limit=limit),
        "edgar": lambda: repository.edgar_texts(limit=limit),
        "financebench": lambda: repository.financebench(limit=limit),
    }
    try:
        for name, factory in sources.items():
            try:
                rows_read = len(list(factory()))
                checks[name] = {
                    "status": "ok" if rows_read else "error",
                    "rows_read": rows_read,
                    "error": None if rows_read else "stream returned no usable rows",
                    "mode": "streaming",
                }
            except Exception as exc:  # each source is reported independently
                checks[name] = {"status": "error", "error": str(exc), "mode": "streaming"}
    finally:
        repository.close()
    return checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FinLingo++ seven-stage thesis pipeline")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Start FastAPI and the built React application")
    serve.add_argument("--reload", action="store_true")

    analyse = subparsers.add_parser("analyse", help="Run S1-S7 for one PDF, DOCX, or TXT document")
    analyse.add_argument("file", type=Path)
    analyse.add_argument("--output", type=Path)

    subparsers.add_parser("stream-check", help="Read one live row from every configured dataset stream")

    preflight = subparsers.add_parser("preflight", help="Validate proposal compliance and runtime prerequisites")
    preflight.add_argument(
        "--final", action="store_true", help="Treat missing adapters, revisions, and calibration as blocking"
    )

    calibration = subparsers.add_parser(
        "calibrate-verifier", help="Fit verifier temperature on a held-out NLI validation stream"
    )
    calibration.add_argument("--max-examples", type=int, default=2_000)
    calibration.add_argument("--dataset", choices=["contractnli", "snli"], default="contractnli")
    calibration.add_argument("--output", type=Path, default=Path("models/stage6_calibration.json"))

    leakage = subparsers.add_parser(
        "leakage-check", help="Audit train/validation/test document overlap without persisting source text"
    )
    leakage.add_argument("--sample-size", type=int, default=5_000)
    leakage.add_argument("--output", type=Path, default=Path("reports/leakage_check.json"))

    parser_eval = subparsers.add_parser(
        "parser-eval", help="Evaluate S1 clause boundaries against a manually labelled manifest"
    )
    parser_eval.add_argument("manifest", type=Path)
    parser_eval.add_argument("--threshold", type=float, default=0.85)
    parser_eval.add_argument("--output", type=Path, default=Path("reports/parser_evaluation.json"))

    eda = subparsers.add_parser("eda", help="Run aggregate EDA without saving source records")
    eda.add_argument("--sample-size", type=int, default=2_000)
    eda.add_argument("--output", type=Path, default=Path("reports/streaming_eda.json"))

    subparsers.add_parser("train-domain-warmup", help="QLoRA EDGAR domain warm-up for S2")
    subparsers.add_parser("train-simplifier", help="Quality-aware Qwen QLoRA training for S2")

    verifier = subparsers.add_parser("train-verifier", help="QLoRA fine-tune the S6 DeBERTa verifier")
    verifier.add_argument("--model")
    verifier.add_argument("--output", type=Path)

    modernbert = subparsers.add_parser("train-modernbert", help="QLoRA fine-tune the ModernBERT comparator")
    modernbert.add_argument("--output", type=Path, default=Path("models/modernbert_nli_adapter"))

    risk_clf = subparsers.add_parser(
        "train-risk-classifier", help="LoRA fine-tune a trained S5 six-class risk classifier"
    )
    risk_clf.add_argument("--model")
    risk_clf.add_argument("--output", type=Path)

    evaluate = subparsers.add_parser("evaluate", help="Build FLB in memory and evaluate RQ1-RQ3")
    evaluate.add_argument(
        "--max-examples", type=int, help="Number of unique FLB source clauses (two NLI rows per clause)"
    )
    evaluate.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    evaluate.add_argument("--export-flb", type=Path)
    evaluate.add_argument("--review-template", type=Path, default=Path("reports/flb_human_review.csv"))
    evaluate.add_argument("--review-file", type=Path, help="Completed adjudication CSV to apply before scoring")
    evaluate.add_argument(
        "--review-all", action="store_true", help="Export every FLB row for independent human adjudication"
    )
    evaluate.add_argument(
        "--final", action="store_true", help="Enforce final-evaluation safeguards and human-only labels"
    )

    for name, default in [
        ("ablations", "ablations.json"),
        ("multiverifier", "multiverifier.json"),
        ("tau-sweep", "tau_sweep.json"),
    ]:
        command = subparsers.add_parser(name)
        command.add_argument("--max-examples", type=int, help="Number of unique FLB source clauses")
        command.add_argument("--output", type=Path, default=Path("reports") / default)
        if name in {"multiverifier", "ablations"}:
            command.add_argument(
                "--skip-gpt",
                action="store_true",
                help="Skip the optional GPT-4 zero-shot baseline (requires OPENAI_API_KEY)",
            )
        if name == "ablations":
            command.add_argument(
                "--resume",
                action="store_true",
                help="Skip variants/baselines already present in --output's existing checkpoint",
            )

    reranker = subparsers.add_parser("reranker-comparison", help="Compare MiniLM L-6 and L-12 on shared FLB queries")
    reranker.add_argument("--max-examples", type=int, help="Number of unique FLB source clauses")
    reranker.add_argument("--output", type=Path, default=Path("reports/reranker_comparison.json"))

    retrieval_query = subparsers.add_parser(
        "retrieval-query-ablation", help="Compare original, simplified, and combined retrieval queries"
    )
    retrieval_query.add_argument("--max-examples", type=int, help="Number of unique FLB source clauses")
    retrieval_query.add_argument("--output", type=Path, default=Path("reports/retrieval_query_ablation.json"))

    failure = subparsers.add_parser("failure-analysis", help="Run RQ4 controlled failure-mode evaluation")
    failure.add_argument(
        "--max-examples", type=int, help="Defaults to FLB_SIZE; use an explicit smaller value only for smoke testing"
    )
    failure.add_argument(
        "--manifest", type=Path, help="CSV/JSON manifest of real documents; runs each file through S1-S7"
    )
    failure.add_argument("--output", type=Path, default=Path("reports/failure_analysis.json"))

    financebench = subparsers.add_parser("financebench", help="Run the external FinanceBench RAG evaluation")
    financebench.add_argument("--max-examples", type=int)
    financebench.add_argument("--output", type=Path, default=Path("reports/financebench.json"))
    financebench.add_argument("--retrieval-only", action="store_true")
    financebench.add_argument("--gpt-judge", action="store_true")

    review = subparsers.add_parser("score-review", help="Calculate agreement from a completed FLB review CSV")
    review.add_argument("file", type=Path)

    consolidate = subparsers.add_parser("consolidate-results", help="Create CSV/Markdown tables from run artifacts")
    consolidate.add_argument("--output-dir", type=Path, default=Path("reports/tables"))

    research = subparsers.add_parser(
        "research-run",
        help="Single-command training, calibration, evaluation, ablations, and optional document analysis",
    )
    research.add_argument("--file", type=Path)
    research.add_argument(
        "--max-examples", type=int, help="Defaults to FLB_SIZE; use an explicit smaller value only for smoke testing"
    )
    research.add_argument("--output-dir", type=Path, default=Path("reports"))
    research.add_argument(
        "--train", action="store_true", help="Train missing S2/S6/ModernBERT adapters before evaluation"
    )
    research.add_argument("--skip-domain-warmup", action="store_true")
    research.add_argument("--skip-modernbert-training", action="store_true")
    research.add_argument("--skip-calibration", action="store_true")
    research.add_argument("--include-gpt-judge", action="store_true")
    research.add_argument("--skip-multiverifier", action="store_true")
    research.add_argument("--skip-financebench", action="store_true")
    research.add_argument("--financebench-generation", action="store_true")
    research.add_argument("--review-file", type=Path)
    research.add_argument("--review-all", action="store_true")
    research.add_argument("--final", action="store_true")

    proof = subparsers.add_parser(
        "adapter-proof", help="Prove Verifier-v3 adapter attachment (structural + behavioural)"
    )
    proof.add_argument("--output", type=Path)

    repro = subparsers.add_parser(
        "faithbench-reproduce", help="Reproduce the frozen Verifier-v3 FaithBench result with CIs and a tolerance gate"
    )
    repro.add_argument("--max-examples", type=int)
    repro.add_argument("--output", type=Path)

    plan = subparsers.add_parser("flb-plan", help="Audit live supply and write the achievable FLB quota plan")
    plan.add_argument("--accept", action="store_true", help="Freeze and accept the plan for FLB_MODE=achievable builds")
    plan.add_argument("--output", type=Path)

    err = subparsers.add_parser(
        "error-analysis", help="Confusion matrices, boundary errors, and McNemar tests from run artifacts"
    )
    err.add_argument("--evaluation", type=Path, default=Path("reports/evaluation.json"))
    err.add_argument("--multiverifier", type=Path, default=Path("reports/multiverifier.json"))
    err.add_argument("--output", type=Path, default=Path("reports/error_analysis.json"))

    env = subparsers.add_parser("environment-report", help="Freeze interpreter, GPU, package, and settings state")
    env.add_argument("--output", type=Path, default=Path("reports/environment_report.json"))

    pins = subparsers.add_parser(
        "pin-revisions", help="Resolve current HF dataset/model revisions and emit .env pin lines"
    )
    pins.add_argument("--output", type=Path, default=Path("reports/pinned_revisions.json"))

    submission = subparsers.add_parser(
        "submission-run",
        help="One-command final thesis run: train, prove, evaluate --final, experiments, statistics, consolidated manifest",
    )
    submission.add_argument("--file", type=Path)
    submission.add_argument("--max-examples", type=int, help="Defaults to the FLB plan / FLB_SIZE")
    submission.add_argument("--output-dir", type=Path, help="Defaults to SUBMISSION_DIR")
    submission.add_argument("--train", action="store_true", help="Train missing adapters first")
    submission.add_argument("--skip-domain-warmup", action="store_true")
    submission.add_argument("--skip-modernbert-training", action="store_true")
    submission.add_argument("--skip-calibration", action="store_true")
    submission.add_argument("--include-gpt-judge", action="store_true")
    submission.add_argument("--skip-multiverifier", action="store_true")
    submission.add_argument("--skip-financebench", action="store_true")
    submission.add_argument("--financebench-generation", action="store_true")
    submission.add_argument("--review-file", type=Path, help="Completed FLB-Gold adjudication CSV")
    submission.add_argument("--review-all", action="store_true")

    for command_name in ("evaluate-trained", "full"):
        help_text = (
            "Evaluate an already-trained pipeline; does not train missing adapters"
            if command_name == "evaluate-trained"
            else "Deprecated alias for evaluate-trained"
        )
        full = subparsers.add_parser(command_name, help=help_text)
        full.add_argument("--file", type=Path)
        full.add_argument("--max-examples", type=int, help="Defaults to FLB_SIZE")
        full.add_argument("--output-dir", type=Path, default=Path("reports"))
        full.add_argument("--include-gpt-judge", action="store_true")
        full.add_argument("--skip-multiverifier", action="store_true")
        full.add_argument("--skip-financebench", action="store_true")
        full.add_argument("--financebench-generation", action="store_true")
        full.add_argument("--review-file", type=Path, help="Completed FLB adjudication CSV")
        full.add_argument("--review-all", action="store_true", help="Export all FLB rows for review")
        full.add_argument("--final", action="store_true", help="Run strict final preflight and require human labels")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    command = args.command or "serve"
    settings = get_settings()
    configure_logging(settings.log_level)

    if command == "serve":
        import uvicorn

        uvicorn.run(
            "backend.app.api_sse:app",
            host=settings.host,
            port=settings.port,
            reload=getattr(args, "reload", False),
        )
        return 0

    if command == "stream-check":
        emit(stream_check(settings))
        return 0

    if command == "preflight":
        from backend.app.core.preflight import run_preflight

        emit(run_preflight(settings, final=args.final))
        return 0

    if command == "calibrate-verifier":
        from backend.evaluation.calibration import calibrate_verifier

        emit(
            calibrate_verifier(
                settings,
                max_examples=args.max_examples or settings.flb_size,
                output_path=args.output,
                dataset=args.dataset,
            )
        )
        return 0

    if command == "leakage-check":
        from backend.evaluation.leakage import audit_split_contamination

        emit(
            audit_split_contamination(
                settings,
                sample_per_split=args.sample_size,
                output_path=args.output,
            )
        )
        return 0

    if command == "parser-eval":
        from backend.evaluation.parser_eval import evaluate_parser_boundaries

        emit(
            evaluate_parser_boundaries(
                settings,
                args.manifest,
                output_path=args.output,
                match_threshold=args.threshold,
            )
        )
        return 0

    if command == "analyse":
        from backend.app.pipeline import FinLingoPipeline

        if not args.file.exists():
            parser.error(f"File not found: {args.file}")
        result = FinLingoPipeline(settings).run(args.file).model_dump(mode="json")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        emit(result)
        return 0

    if command == "eda":
        from backend.evaluation.eda import run_streaming_eda

        emit(run_streaming_eda(settings, args.sample_size, args.output))
        return 0

    if command == "train-domain-warmup":
        from backend.training.train_domain_warmup import train_domain_warmup

        print(train_domain_warmup(settings))
        return 0

    if command == "train-simplifier":
        from backend.training.train_simplifier import train_simplifier

        print(train_simplifier(settings))
        return 0

    if command in {"train-verifier", "train-modernbert"}:
        from backend.training.train_verifier import train_verifier

        if command == "train-modernbert":
            print(
                train_verifier(
                    settings,
                    model_id=settings.modernbert_training_base_model,
                    output_dir=args.output,
                )
            )
        else:
            print(train_verifier(settings, model_id=args.model, output_dir=args.output))
        return 0

    if command == "train-risk-classifier":
        from backend.training.train_risk_classifier import train_risk_classifier

        print(train_risk_classifier(settings, model_id=args.model, output_dir=args.output))
        return 0

    if command == "evaluate":
        from backend.evaluation.evaluator import evaluate_finlingo

        emit(
            evaluate_finlingo(
                settings,
                max_examples=args.max_examples or settings.flb_size,
                output_path=args.output,
                export_flb_path=args.export_flb,
                review_template_path=args.review_template,
                review_file_path=args.review_file,
                review_all=args.review_all,
                final_evaluation=args.final,
            )
        )
        return 0

    if command == "ablations":
        from backend.experiments.ablation_studies import run_ablations

        emit(
            run_ablations(
                settings,
                args.max_examples or settings.flb_size,
                args.output,
                include_gpt=not args.skip_gpt,
                resume=args.resume,
            )
        )
        return 0

    if command == "multiverifier":
        from backend.experiments.multiverifier_comparison import run_multiverifier

        emit(
            run_multiverifier(
                settings,
                args.max_examples,
                args.output,
                include_gpt=not args.skip_gpt,
            )
        )
        return 0

    if command == "tau-sweep":
        from backend.experiments.tau_sweep import run_tau_sweep

        emit(run_tau_sweep(settings, args.max_examples or settings.flb_size, args.output))
        return 0

    if command == "reranker-comparison":
        from backend.experiments.reranker_comparison import run_reranker_comparison

        emit(run_reranker_comparison(settings, args.max_examples or settings.flb_size, args.output))
        return 0

    if command == "retrieval-query-ablation":
        from backend.experiments.retrieval_query_ablation import run_retrieval_query_ablation

        emit(run_retrieval_query_ablation(settings, args.max_examples or settings.flb_size, args.output))
        return 0

    if command == "failure-analysis":
        from backend.experiments.failure_analysis import run_document_failure_analysis, run_failure_analysis

        if args.manifest:
            emit(run_document_failure_analysis(settings, args.manifest, args.output))
        else:
            emit(run_failure_analysis(settings, args.max_examples or settings.flb_size, args.output))
        return 0

    if command == "financebench":
        from backend.evaluation.financebench_eval import evaluate_financebench

        emit(
            evaluate_financebench(
                settings,
                max_examples=args.max_examples or settings.flb_size,
                output_path=args.output,
                include_generation=not args.retrieval_only,
                include_gpt_judge=args.gpt_judge,
            )
        )
        return 0

    if command == "score-review":
        from backend.evaluation.human_review import score_review_file

        emit(score_review_file(args.file))
        return 0

    if command == "consolidate-results":
        from backend.evaluation.consolidate_results import consolidate_results

        emit(consolidate_results(settings, args.output_dir))
        return 0

    if command == "research-run":
        emit(_run_research(settings, args))
        return 0

    if command in {"evaluate-trained", "full"}:
        if command == "full":
            print("WARNING: 'full' is deprecated; use 'evaluate-trained'.", file=sys.stderr)
        emit(_run_full(settings, args))
        return 0

    if command == "adapter-proof":
        from backend.evaluation.adapter_proof import prove_adapter_attachment

        emit(prove_adapter_attachment(settings, args.output))
        return 0

    if command == "faithbench-reproduce":
        from backend.evaluation.faithbench_reproduce import reproduce_faithbench

        emit(reproduce_faithbench(settings, args.max_examples, args.output))
        return 0

    if command == "flb-plan":
        from backend.evaluation.flb_plan import build_flb_plan

        emit(build_flb_plan(settings, accept=args.accept, output_path=args.output))
        return 0

    if command == "error-analysis":
        from backend.evaluation.error_analysis import run_error_analysis

        emit(run_error_analysis(settings, args.evaluation, args.multiverifier, args.output))
        return 0

    if command == "environment-report":
        from backend.app.core.environment import environment_report

        emit(environment_report(settings, args.output))
        return 0

    if command == "pin-revisions":
        emit(_pin_revisions(settings, args.output))
        return 0

    if command == "submission-run":
        emit(_run_submission(settings, args))
        return 0

    parser.error(f"Unknown command: {command}")
    return 2


def _run_research(settings: Settings, args: argparse.Namespace) -> dict[str, Any]:
    """Run training (if --train), calibration, and evaluation in one command.
    Any failed phase raises and leaves a phase-status manifest on disk."""
    import traceback
    from datetime import datetime, timezone

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "research_run_status.json"
    status: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "train_requested": bool(args.train),
        "phases": {},
    }

    def phase(name: str, fn):
        started = datetime.now(timezone.utc).isoformat()
        try:
            value = fn()
            status["phases"][name] = {"status": "ok", "started_at": started}
            status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
            return value
        except Exception as exc:
            status["phases"][name] = {
                "status": "error",
                "started_at": started,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=12),
            }
            status["finished_at"] = datetime.now(timezone.utc).isoformat()
            status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
            raise

    outputs: dict[str, Any] = {}
    if args.train:
        if not args.skip_domain_warmup:
            from backend.training.train_domain_warmup import train_domain_warmup

            outputs["domain_warmup"] = phase("domain_warmup", lambda: train_domain_warmup(settings))
            settings.domain_warmup_adapter = str(outputs["domain_warmup"])
        from backend.training.train_simplifier import train_simplifier

        outputs["simplifier_training"] = phase("simplifier_training", lambda: train_simplifier(settings))
        settings.simplifier_adapter = str(outputs["simplifier_training"])
        from backend.training.train_verifier import train_verifier

        outputs["verifier_training"] = phase("verifier_training", lambda: train_verifier(settings))
        settings.verifier_adapter = str(outputs["verifier_training"])
        if not args.skip_modernbert_training:
            outputs["modernbert_training"] = phase(
                "modernbert_training",
                lambda: train_verifier(
                    settings,
                    model_id=settings.modernbert_training_base_model,
                    output_dir=Path("models/modernbert_nli_adapter"),
                ),
            )
            settings.modernbert_adapter = str(outputs["modernbert_training"])
        if not args.skip_calibration:
            from backend.evaluation.calibration import calibrate_verifier

            calibration_path = Path("models/stage6_calibration.json")
            outputs["calibration"] = phase(
                "calibration",
                lambda: calibrate_verifier(
                    settings,
                    max_examples=2_000,
                    output_path=calibration_path,
                    dataset="contractnli",
                ),
            )
            settings.verifier_calibration_path = calibration_path

    outputs["research"] = phase("research", lambda: _run_full(settings, args))
    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    status["status"] = "ok"
    status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    outputs["status_manifest"] = str(status_path)
    return outputs


def _run_full(settings: Settings, args: argparse.Namespace) -> dict[str, Any]:
    from backend.app.pipeline import FinLingoPipeline
    from backend.evaluation.eda import run_streaming_eda
    from backend.evaluation.evaluator import evaluate_records
    from backend.evaluation.flb_builder import FLBBuilder
    from backend.app.services.model_registry import ModelRegistry
    from backend.experiments.ablation_studies import run_ablations
    from backend.experiments.failure_analysis import run_failure_analysis
    from backend.app.core.preflight import run_preflight

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Fail fast before preflight, stream checks, EDA, FinanceBench,
    # model loading, or provider generation.
    target_size = args.max_examples or settings.flb_size
    feasibility = FLBBuilder.feasibility_audit(
        settings,
        target_size=target_size,
    )

    if not feasibility.get("feasible", False):
        raise RuntimeError(
            "FLB feasibility failed before external evaluation work. "
            f"Available class counts={feasibility.get('class_counts')}; "
            f"required quotas={feasibility.get('quotas')}; "
            f"insufficient classes={feasibility.get('shortages')}. "
            "No external evaluation datasets, models, or Ollama generation "
            "were started."
        )

    payload: dict[str, Any] = {
        "flb_feasibility": feasibility,
        "preflight": run_preflight(settings, final=args.final),
        "stream_check": stream_check(settings),
        "eda": run_streaming_eda(
            settings,
            1_000,
            args.output_dir / "streaming_eda.json",
        ),
    }

    if args.file:
        payload["document"] = FinLingoPipeline(settings).run(args.file).model_dump(mode="json")

    # Build FLB exactly once so every experiment uses the same clauses,
    # generated references, NLI items, and provisional/human evidence labels.
    builder = FLBBuilder(settings, ModelRegistry(settings))
    shared_records = builder.build(target_size)
    review_status = None
    if args.review_file:
        from backend.evaluation.human_review import apply_adjudicated_reviews

        review_status = apply_adjudicated_reviews(shared_records, args.review_file)
    payload["evaluation"] = evaluate_records(
        settings,
        shared_records,
        adjudicate_evidence=not args.final,
        final_evaluation=args.final,
    )
    payload["evaluation"]["human_review"] = {
        "review_file": str(args.review_file) if args.review_file else None,
        "status": review_status,
        "human_reviewed_records": sum(record.human_reviewed for record in shared_records),
    }
    payload["evaluation"]["shared_benchmark"] = {
        "unique_clauses": len({record.source_id for record in shared_records}),
        "verifier_rows": len(shared_records),
        "reused_across_full_workflow": True,
    }
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(payload["evaluation"], indent=2, default=str),
        encoding="utf-8",
    )
    builder.export_review_template(
        shared_records,
        args.output_dir / "flb_human_review.csv",
        len(shared_records) if args.review_all else settings.flb_human_review_size,
        settings.random_seed,
    )
    payload["ablations"] = run_ablations(
        settings,
        args.max_examples,
        args.output_dir / "ablations.json",
        records=shared_records,
    )
    from backend.experiments.reranker_comparison import run_reranker_comparison

    payload["reranker_comparison"] = run_reranker_comparison(
        settings,
        args.max_examples,
        args.output_dir / "reranker_comparison.json",
        records=shared_records,
    )
    from backend.experiments.retrieval_query_ablation import run_retrieval_query_ablation

    payload["retrieval_query_ablation"] = run_retrieval_query_ablation(
        settings,
        args.max_examples,
        args.output_dir / "retrieval_query_ablation.json",
        records=shared_records,
    )
    payload["rq4"] = run_failure_analysis(
        settings,
        min(args.max_examples, 100),
        args.output_dir / "failure_analysis.json",
        records=shared_records,
    )
    if not args.skip_multiverifier:
        from backend.experiments.multiverifier_comparison import run_multiverifier

        payload["multiverifier"] = run_multiverifier(
            settings,
            args.max_examples,
            args.output_dir / "multiverifier.json",
            include_gpt=args.include_gpt_judge,
            flb_records=shared_records,
        )
    if not args.skip_financebench:
        from backend.evaluation.financebench_eval import evaluate_financebench

        payload["financebench"] = evaluate_financebench(
            settings,
            max_examples=min(args.max_examples, settings.financebench_max_examples),
            output_path=args.output_dir / "financebench.json",
            include_generation=args.financebench_generation,
            include_gpt_judge=args.include_gpt_judge and args.financebench_generation,
        )
    return payload


def _pin_revisions(settings: Settings, output_path: Path) -> dict[str, Any]:
    """Resolve current HF revisions for every dataset/model, write them to JSON,
    and print ready-to-paste .env lines for revision-pinning future runs."""
    from huggingface_hub import HfApi

    api = HfApi(token=settings.hf_token)
    datasets = [
        settings.cuad_dataset,
        settings.contractnli_dataset,
        settings.snli_dataset,
        settings.financebench_dataset,
        settings.cfpb_dataset,
        settings.edgar_dataset,
    ]
    models = sorted(
        {
            settings.simplifier_model,
            settings.risk_model or settings.simplifier_model,
            settings.s2_semantic_model,
            settings.verifier_model,
            settings.verifier_training_base_model,
            settings.modernbert_model,
            settings.modernbert_training_base_model,
            settings.reranker_model,
            *settings.reranker_models,
            settings.bertscore_model,
            settings.finbert_model,
            settings.embedding_model,
            settings.silver_local_model,
            settings.silver_judge_model,
        }
    )
    dataset_pins, model_pins, errors = {}, {}, {}
    for dataset in datasets:
        try:
            dataset_pins[dataset] = api.dataset_info(dataset).sha
        except Exception as exc:
            errors[dataset] = str(exc)
    for model in models:
        try:
            model_pins[model] = api.model_info(model).sha
        except Exception as exc:
            errors[model] = str(exc)
    payload = {
        "dataset_revisions": dataset_pins,
        "model_revisions": model_pins,
        "unresolved": errors,
        "env_lines": [
            f"DATASET_REVISIONS_JSON={json.dumps(dataset_pins)}",
            f"MODEL_REVISIONS_JSON={json.dumps(model_pins)}",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Paste these two lines into .env to pin every run:")
    for line in payload["env_lines"]:
        print(line)
    return payload


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_submission(settings: Settings, args: argparse.Namespace) -> dict[str, Any]:
    """One-command final thesis run: env freeze, optional training, attachment
    proof, FaithBench reproduction, --final evaluation, error analysis, tables,
    and a SHA-256 manifest. A failed phase halts the run; nothing is reported
    complete unless its artefact exists and is hashed."""
    import traceback
    from datetime import datetime, timezone

    output_dir = args.output_dir or settings.resolve(settings.submission_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir
    args.final = True

    status_path = output_dir / "submission_run_status.json"
    status: dict[str, Any] = {"started_at": datetime.now(timezone.utc).isoformat(), "phases": {}}

    def phase(name: str, fn):
        started = datetime.now(timezone.utc).isoformat()
        try:
            value = fn()
            status["phases"][name] = {"status": "ok", "started_at": started}
            status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
            return value
        except Exception as exc:
            status["phases"][name] = {
                "status": "error",
                "started_at": started,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=12),
            }
            status["finished_at"] = datetime.now(timezone.utc).isoformat()
            status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
            raise

    outputs: dict[str, Any] = {}
    from backend.app.core.environment import environment_report

    outputs["environment"] = phase(
        "environment_report",
        lambda: environment_report(settings, output_dir / "environment_report.json"),
    )

    if args.train:
        if not args.skip_domain_warmup:
            from backend.training.train_domain_warmup import train_domain_warmup

            outputs["domain_warmup"] = phase("domain_warmup", lambda: train_domain_warmup(settings))
            settings.domain_warmup_adapter = str(outputs["domain_warmup"])
        from backend.training.train_simplifier import train_simplifier

        outputs["simplifier_training"] = phase("simplifier_training", lambda: train_simplifier(settings))
        settings.simplifier_adapter = str(outputs["simplifier_training"])
        from backend.training.train_verifier import train_verifier

        outputs["verifier_training"] = phase("verifier_training", lambda: train_verifier(settings))
        settings.verifier_adapter = str(outputs["verifier_training"])
        if not args.skip_modernbert_training:
            outputs["modernbert_training"] = phase(
                "modernbert_training",
                lambda: train_verifier(
                    settings,
                    model_id=settings.modernbert_training_base_model,
                    output_dir=Path("models/modernbert_nli_adapter"),
                ),
            )
            settings.modernbert_adapter = str(outputs["modernbert_training"])
        if not args.skip_calibration:
            from backend.evaluation.calibration import calibrate_verifier

            calibration_path = Path("models/stage6_calibration.json")
            outputs["calibration"] = phase(
                "calibration",
                lambda: calibrate_verifier(
                    settings, max_examples=2_000, output_path=calibration_path, dataset="contractnli"
                ),
            )
            settings.verifier_calibration_path = calibration_path

    from backend.evaluation.adapter_proof import prove_adapter_attachment

    outputs["adapter_proof"] = phase(
        "adapter_attachment_proof",
        lambda: prove_adapter_attachment(settings, output_dir / "adapter_attachment_proof.json"),
    )
    from backend.evaluation.faithbench_reproduce import reproduce_faithbench

    outputs["faithbench_reproduction"] = phase(
        "faithbench_reproduction",
        lambda: reproduce_faithbench(
            settings, args.max_examples, output_dir / "verifier_faithbench_v3_reproduction.json"
        ),
    )
    if settings.flb_mode.value == "achievable":
        from backend.evaluation.flb_plan import load_accepted_plan

        outputs["flb_plan"] = phase("flb_plan_check", lambda: load_accepted_plan(settings))

    outputs["evaluation_suite"] = phase("final_evaluation_suite", lambda: _run_full(settings, args))

    from backend.evaluation.error_analysis import run_error_analysis

    outputs["error_analysis"] = phase(
        "error_analysis",
        lambda: run_error_analysis(
            settings,
            output_dir / "evaluation.json",
            output_dir / "multiverifier.json",
            output_dir / "error_analysis.json",
        ),
    )
    from backend.evaluation.consolidate_results import consolidate_results

    outputs["consolidated_tables"] = phase(
        "consolidate_results", lambda: consolidate_results(settings, output_dir / "tables")
    )

    artifacts = {
        str(path.relative_to(output_dir)): {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "submission_manifest.json"
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": settings.random_seed,
        "flb_mode": settings.flb_mode.value,
        "dataset_revisions": settings.dataset_revisions,
        "model_revisions": settings.model_revisions,
        "verifier_adapter": settings.verifier_adapter,
        "simplifier_adapter": settings.simplifier_adapter,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    (output_dir / "submission_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    status["status"] = "ok"
    status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    outputs["submission_manifest"] = str(output_dir / "submission_manifest.json")
    return outputs


if __name__ == "__main__":
    raise SystemExit(main())
