from __future__ import annotations

import json
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class FLBMode(str, Enum):
    """Controls FLB construction behaviour."""

    pilot = "pilot"
    final = "final"
    achievable = "achievable"


class Settings(BaseSettings):
    """Single canonical configuration for the seven-stage FinLingo++ pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False, env_ignore_empty=True
    )

    # App / API
    app_name: str = "FinLingo++"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    host: str = "127.0.0.1"
    port: int = 8000
    max_upload_mb: int = 30
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000"
    persist_reports: bool = False
    output_dir: Path = Path("reports")
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3])

    # Credentials
    hf_token: str | None = None
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    azure_endpoint: str | None = None
    azure_api_version: str | None = None
    deployment_name: str | None = None

    model_backend: Literal["local", "hf_endpoint", "ollama"] = "local"
    hf_generation_endpoint_url: str | None = None
    hf_simplifier_endpoint_url: str | None = None
    hf_risk_endpoint_url: str | None = None
    hf_endpoint_supports_routing: bool = False
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    # S5 stays on ollama so it doesn't fight S2/S6 for VRAM on one GPU
    s5_generation_backend: Literal["inherit", "ollama", "local", "hf_endpoint"] = "ollama"
    s2_generation_backend: Literal["inherit", "ollama", "local", "hf_endpoint"] = "inherit"
    ollama_generation_model: str | None = None
    ollama_generation_json_mode: bool = True
    ollama_health_timeout_seconds: float = 10.0
    model_cache_dir: Path | None = None
    allow_base_models: bool = True
    model_trust_remote_code: bool = False
    release_causal_models_between_stages: bool = True
    pipeline_ollama_concurrency: int = 4

    # S1 parser
    parser_ocr_min_chars_per_page: int = 40
    parser_ocr_dpi: int = 250
    parser_min_clause_words: int = 3
    parser_max_pages: int = 500
    parser_max_clauses: int = 2_000

    # S2 Qwen + QLoRA
    simplifier_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    simplifier_adapter: str | None = None
    domain_warmup_adapter: str | None = None
    simplifier_max_input_tokens: int = 1024
    simplification_source_max_words: int = 600
    simplifier_max_new_tokens: int = 384
    simplifier_temperature: float = 0.1
    simplifier_retry_temperature_step: float = 0.15
    simplifier_retry_temperature_cap: float = 0.7
    ask_max_new_tokens: int = 300
    ask_temperature: float = 0.2
    simplifier_retries: int = 7
    s2_deterministic_mode: bool = False
    simplifier_min_semantic_entailment: float = 0.75
    simplifier_fail_on_unaccepted: bool = False
    suppress_unaccepted_simplification: bool = True
    fk_generation_target: float = 7.0
    fk_acceptance_target: float = 9.0
    qlora_rank: int = 16
    qlora_alpha: int = 32
    qlora_dropout: float = 0.05
    qlora_use_double_quant: bool = True
    qlora_quant_type: Literal["nf4", "fp4"] = "nf4"
    simplifier_batch_size: int = 1
    simplifier_gradient_accumulation_steps: int = 16
    simplifier_fp16: bool = True
    simplifier_bf16: bool = False

    # S3 in-memory FAISS / BM25
    # Ollama is the default local embedding provider; the legacy local model remains available.
    embedding_provider: Literal["ollama", "local"] = "ollama"
    ollama_embedding_model: str = "all-minilm"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    # raised from 20 - dense retrieval alone barely found anything (recall@20 ~0.05), BM25 at this k does much better
    top_k_retrieval: int = 500
    embedding_batch_size: int = 64
    regulatory_request_timeout: float = 45.0
    regulatory_max_chars_per_source: int = 5_000_000
    regulatory_max_pages_per_source: int = 20
    regulatory_min_chunk_tokens: int = 40
    regulatory_sources_path: Path = Path("configs/regulatory_sources.yaml")
    regulatory_urls_json: str = ""

    # S4
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    reranker_comparison_models: str = "cross-encoder/ms-marco-MiniLM-L-6-v2,cross-encoder/ms-marco-MiniLM-L-12-v2"
    reranker_batch_size: int = 16
    top_j_reranked: int = 15
    reranker_min_score: float | None = None
    retrieval_query_mode: Literal["original", "simplified", "combined"] = "combined"

    # S5
    risk_model: str | None = None
    risk_adapter: str | None = None
    risk_max_input_tokens: int = 4096
    risk_temperature: float = 0.0
    risk_max_new_tokens: int = 256
    risk_parse_retries: int = 2
    risk_abstain_label: str = "Needs Review"
    risk_require_adapter: bool = False
    risk_require_evidence_for_risk: bool = False
    risk_few_shot_path: Path = Path("configs/risk_few_shot.yaml")
    risk_taxonomy_path: Path = Path("configs/risk_taxonomy.yaml")
    risk_max_secondary_labels: int = 3

    risk_classifier_backend: Literal["prompted", "trained", "frozen"] = "trained"
    # legal-bert instead of finbert - finbert is tuned for market sentiment, not clause semantics
    risk_classifier_model: str = "nlpaueb/legal-bert-base-uncased"
    risk_classifier_adapter: Path | None = Path("models/stage5_risk_classifier_v5")
    risk_classifier_adapter_output: Path = Path("models/stage5_risk_classifier_v3")
    risk_classifier_frozen_artifact: Path = Path("models/stage5_finbert_frozen/classifier.joblib")
    risk_classifier_max_length: int = 512
    risk_classifier_train_samples: int = 10_000
    risk_classifier_validation_samples: int = 800
    risk_classifier_learning_rate: float = 2e-5
    risk_classifier_num_train_epochs: float = 10.0
    risk_classifier_early_stopping_patience: int = 4
    risk_classifier_use_class_weights: bool = True
    risk_classifier_require_adapter: bool = False

    # S6
    verifier_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    verifier_training_base_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    verifier_adapter: str | None = None
    modernbert_model: str = "tasksource/ModernBERT-large-nli"
    modernbert_training_base_model: str = "answerdotai/ModernBERT-large"
    modernbert_adapter: str | None = None
    verifier_batch_size: int = 16
    verifier_max_length: int = 512
    faithfulness_tau: float = 0.376
    faithfulness_taus: str = "0.3,0.376,0.5,0.65"
    attribution_tau: float = 0.376
    faithfulness_primary_premise: Literal["retrieved_chunk", "source_clause", "dual"] = "retrieved_chunk"
    verifier_require_adapter: bool = True
    verifier_use_base_model: bool = True
    verifier_temperature: float = 1.0
    verifier_calibration_path: Path | None = None
    verifier_feedback_enabled: bool = False
    verifier_feedback_retries: int = 1

    # S2 semantic-preservation gate is intentionally independent of the final
    # S6 thesis verifier to avoid circular candidate selection/evaluation.
    s2_semantic_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    s2_semantic_adapter: str | None = None
    s2_semantic_require_adapter: bool = False

    review_routing_policy: Literal["legacy", "strict_policy"] = "legacy"
    review_low_entailment_tau: float = 0.376
    review_evidence_conflict_gap: float = 0.3
    review_high_impact_risk_score: int = 4
    review_require_attribution: bool = False

    bootstrap_unit: Literal["record", "source_clause"] = "source_clause"
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 42

    evaluation_seeds: str = "42"

    groq_silver_label_model: str = "llama-3.3-70b-versatile"
    groq_max_retries: int = 3
    silver_label_provider: Literal["local_qwen", "gemini", "groq", "ollama"] = "local_qwen"
    silver_judge_provider: Literal["local_qwen", "gemini", "groq", "ollama"] | None = None
    silver_regeneration_provider: Literal["local_qwen", "gemini", "groq", "ollama"] | None = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_timeout_seconds: int = 300
    ollama_temperature: float = 0.0
    ollama_num_ctx: int = 8192
    ollama_num_predict: int = 1536
    ollama_keep_alive: str = "10m"
    ollama_connection_retries: int = 3
    ollama_connection_retry_backoff_seconds: float = 2.0
    silver_local_model: str = "Qwen/Qwen2.5-3B-Instruct"
    silver_judge_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    silver_local_model_revision: str | None = None
    silver_local_batch_size: int = 1
    silver_local_max_new_tokens: int = 1024
    silver_local_temperature: float = 0.0
    silver_local_do_sample: bool = False
    silver_local_use_4bit: bool = True
    silver_local_malformed_retries: int = 2
    gemini_silver_label_model: str = "gemini-2.5-flash"
    gemini_api_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    judge_model: str = "gpt-4o"
    silver_label_model: str = "llama-3.3-70b-versatile"
    silver_label_batch_size: int = 8
    silver_label_cache_dir: Path | None = None
    corrected_cfpb_cache_dir: Path = Path("models/silver_cache/cfpb_corrected_v1")
    corrected_cfpb_narrative_field: str = "consumer_narrative"
    corrected_cfpb_preprocessing_version: str = "cfpb_narrative_normalise_v1"
    corrected_cfpb_prompt_template_version: str = "finlingo_simplify_v2"
    corrected_cfpb_schema_version: str = "corrected_cfpb_cache_v1"
    simplifier_smoke_train_samples: int = 16
    simplifier_smoke_validation_samples: int = 8
    simplifier_smoke_max_sequence_length: int = 512
    simplifier_smoke_output: Path = Path("models/local_smoke/simplifier_corrected_cfpb")
    synthetic_nli_batch_size: int = 6
    synthetic_nli_concurrency: int = 4
    openai_max_retries: int = 3
    gpt_input_cost_per_million: float | None = None
    gpt_output_cost_per_million: float | None = None
    gpt_cost_currency: str = "USD"

    cuad_dataset: str = "dvgodoy/CUAD_v1_Contract_Understanding_clause_classification"
    cuad_config: str | None = None
    contractnli_dataset: str = "urialon/converted_contract_nli"
    contractnli_configs: str = "default"
    snli_dataset: str = "stanfordnlp/snli"
    financebench_dataset: str = "PatronusAI/financebench"
    faithbench_url: str = "https://raw.githubusercontent.com/vectara/FaithBench/main/FaithBench.csv"
    faithbench_timeout: float = 60.0
    faithbench_max_examples: int = 500
    cfpb_dataset: str = "Mouwiya/cfpb-consumer-complaints"
    legalbench_dataset: str = "nguha/legalbench"
    legalbench_data_sharing_config: str = "opp115_third_party_sharing_collection"
    edgar_dataset: str = "gbharti/finance-alpaca"
    edgar_config: str | None = None
    edgar_trust_remote_code: bool = False
    dataset_shuffle_buffer: int = 10_000
    random_seed: int = 42
    dataset_revisions_json: str = "{}"
    model_revisions_json: str = "{}"

    # QLoRA training
    domain_warmup_samples: int = 20_000
    domain_warmup_validation_samples: int = 1_000
    domain_warmup_adapter_output: Path = Path("models/stage2_domain_warmup_adapter")
    simplifier_train_samples: int = 30_000
    simplifier_validation_samples: int = 2_000
    simplifier_min_training_pairs: int = 100
    simplifier_min_validation_pairs: int = 20
    simplifier_quality_objective: bool = True
    simplifier_quality_min_weight: float = 0.25
    simplifier_quality_sari_weight: float = 0.5
    simplifier_quality_nli_weight: float = 0.5
    verifier_train_samples: int = 30_000
    verifier_validation_samples: int = 2_000
    verifier_training_mode: Literal["fp16_lora"] = "fp16_lora"
    verifier_learning_rate: float = 1e-5
    verifier_num_train_epochs: float = 2.0
    verifier_gradient_checkpointing: bool = False
    verifier_early_stopping_patience: int = 3
    verifier_use_class_weights: bool = True
    verifier_snli_ratio: float = 0.30
    verifier_contractnli_ratio: float = 0.55
    verifier_synthetic_ratio: float = 0.15
    verifier_validation_snli_ratio: float = 0.40
    verifier_validation_contractnli_ratio: float = 0.60
    verifier_min_synthetic_fraction: float = 0.10
    verifier_fail_on_source_shortage: bool = True
    train_batch_size: int = 1
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    max_train_steps: int | None = None
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 250
    eval_steps: int = 250
    simplifier_adapter_output: Path = Path("models/stage2_qlora_adapter")
    verifier_adapter_output: Path = Path("models/stage6_qlora_adapter")

    # Evaluation / FLB
    flb_mode: FLBMode = FLBMode.final
    flb_size: int = 500
    flb_pilot_size: int = 6
    flb_human_review_size: int = 100
    final_evaluation_require_all_human_review: bool = True
    final_evaluation_min_review_fraction: float = 1.0
    final_evaluation_allow_provisional_evidence: bool = False
    final_evaluation_require_risk_scores: bool = True
    flb_min_bertscore: float = 0.88
    bertscore_model: str = "roberta-large"
    flb_min_sari: float = 40.0
    flb_candidate_multiplier: int = 3
    flb_judge_min_score: int = 4
    flb_quality_regeneration_retries: int = 4
    flb_synthetic_nli_repair_retries: int = 12
    flb_external_gold_path: Path = Path("configs/flb_external_gold.yaml")
    flb_external_silver_path: Path = Path("configs/flb_external_silver.yaml")
    flb_safe_candidates_path: Path = Path("configs/flb_safe_candidates.yaml")
    flb_external_gold_enabled: bool = False
    flb_external_silver_enabled: bool = False
    target_risk_macro_f1: float = 0.75
    target_faithfulness_precision: float = 0.90
    target_faithfulness_recall: float = 0.80
    target_attribution_accuracy: float = 0.75
    target_hallucination_rate: float = 0.08
    finbert_model: str = "ProsusAI/finbert"
    finbert_train_samples: int = 1500
    selfcheck_samples: int = 5
    selfcheck_temperature: float = 0.8
    baseline_max_examples: int = 500
    bootstrap_samples: int = 1000
    financebench_max_examples: int = 150
    financebench_distractors: int = 7
    financebench_generation_model: str | None = None
    financebench_generation_adapter: str | None = None
    financebench_max_input_tokens: int = 4096
    financebench_temperature: float = 0.0

    # Adapter attachment proof (Verifier v3)
    adapter_probe_pairs_path: Path = Path("configs/adapter_probe_pairs.yaml")
    adapter_proof_min_delta: float = 0.02
    adapter_proof_output: Path = Path("reports/adapter_attachment_proof.json")

    # FaithBench reproduction gate
    faithbench_reference_path: Path = Path("reports/verifier_faithbench_v3_financial.json")
    faithbench_reproduction_tolerance: float = 0.03
    faithbench_reproduction_output: Path = Path("reports/verifier_faithbench_v3_reproduction.json")

    # Achievable FLB plan
    flb_plan_path: Path = Path("reports/benchmark_dataset/flb_plan.json")
    flb_plan_min_per_class: int = 10

    # Submission run
    submission_dir: Path = Path("reports/submission")

    enable_tf32: bool = True
    use_length_grouped_batching: bool = True
    dataloader_num_workers: int = 2

    @field_validator(
        "faithfulness_tau",
        "attribution_tau",
        "simplifier_temperature",
        "selfcheck_temperature",
        "simplifier_min_semantic_entailment",
        "qlora_dropout",
        "warmup_ratio",
        "weight_decay",
        "final_evaluation_min_review_fraction",
        "simplifier_quality_min_weight",
        "simplifier_quality_sari_weight",
        "simplifier_quality_nli_weight",
        "verifier_snli_ratio",
        "verifier_contractnli_ratio",
        "verifier_synthetic_ratio",
        "verifier_validation_snli_ratio",
        "verifier_validation_contractnli_ratio",
        "verifier_min_synthetic_fraction",
    )
    @classmethod
    def unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("value must be in [0,1]")
        return value

    @model_validator(mode="after")
    def validate_pipeline(self) -> "Settings":
        if self.top_j_reranked > self.top_k_retrieval:
            raise ValueError("TOP_J_RERANKED cannot exceed TOP_K_RETRIEVAL")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS")
        if self.embedding_dimensions <= 0:
            raise ValueError("EMBEDDING_DIMENSIONS must be positive")
        if self.parser_ocr_dpi < 72:
            raise ValueError("PARSER_OCR_DPI must be at least 72")
        if self.parser_max_pages <= 0 or self.parser_max_clauses <= 0:
            raise ValueError("Parser page and clause limits must be positive")
        if self.verifier_temperature <= 0:
            raise ValueError("VERIFIER_TEMPERATURE must be positive")
        if self.risk_temperature < 0 or self.financebench_temperature < 0:
            raise ValueError("Generation temperatures cannot be negative")
        if self.verifier_feedback_retries < 0:
            raise ValueError("VERIFIER_FEEDBACK_RETRIES cannot be negative")
        if self.risk_max_secondary_labels < 0:
            raise ValueError("RISK_MAX_SECONDARY_LABELS cannot be negative")
        if self.reranker_min_score is not None and not isinstance(self.reranker_min_score, (int, float)):
            raise ValueError("RERANKER_MIN_SCORE must be numeric when set")
        if self.simplifier_quality_sari_weight + self.simplifier_quality_nli_weight <= 0:
            raise ValueError("At least one simplifier quality-objective weight must be positive")
        if self.simplifier_min_training_pairs <= 0:
            raise ValueError("SIMPLIFIER_MIN_TRAINING_PAIRS must be positive")
        if self.simplifier_min_validation_pairs <= 0:
            raise ValueError("SIMPLIFIER_MIN_VALIDATION_PAIRS must be positive")
        if abs(self.verifier_snli_ratio + self.verifier_contractnli_ratio + self.verifier_synthetic_ratio - 1.0) > 1e-6:
            raise ValueError("Verifier training source ratios must sum to 1.0")
        if abs(self.verifier_validation_snli_ratio + self.verifier_validation_contractnli_ratio - 1.0) > 1e-6:
            raise ValueError("Verifier validation source ratios must sum to 1.0")
        if self.verifier_learning_rate <= 0 or self.verifier_num_train_epochs <= 0:
            raise ValueError("Verifier learning rate and epochs must be positive")
        if self.verifier_early_stopping_patience < 1:
            raise ValueError("VERIFIER_EARLY_STOPPING_PATIENCE must be at least 1")
        if self.flb_quality_regeneration_retries < 0:
            raise ValueError("FLB_QUALITY_REGENERATION_RETRIES cannot be negative")
        return self

    @property
    def configured_risk_labels(self) -> tuple[str, ...]:
        import yaml

        path = self.resolve(self.risk_taxonomy_path)
        if not path.exists():
            raise ValueError(f"Risk taxonomy file not found: {path}")

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        mapped = payload.get("labels")
        safe_label = str(payload.get("safe_label") or "").strip()

        if not isinstance(mapped, dict) or not mapped:
            raise ValueError("Risk taxonomy must define a non-empty labels mapping")
        if not safe_label:
            raise ValueError("Risk taxonomy must define safe_label")

        labels = tuple(str(label).strip() for label in mapped if str(label).strip())
        if safe_label in labels:
            raise ValueError("safe_label must not duplicate a mapped risk label")
        return (*labels, safe_label)

    @property
    def safe_risk_label(self) -> str:
        return self.configured_risk_labels[-1]

    @property
    def cors_origins(self) -> list[str]:
        return [x.strip() for x in self.allowed_origins.split(",") if x.strip()]

    @property
    def tau_values(self) -> tuple[float, ...]:
        values = tuple(float(x.strip()) for x in self.faithfulness_taus.split(",") if x.strip())
        if not values or any(not 0 <= x <= 1 for x in values):
            raise ValueError("FAITHFULNESS_TAUS must contain values in [0,1]")
        return values

    @property
    def evaluation_seed_list(self) -> tuple[int, ...]:
        values = tuple(int(x.strip()) for x in self.evaluation_seeds.split(",") if x.strip())
        if not values:
            raise ValueError("EVALUATION_SEEDS must contain at least one integer seed")
        return values

    @property
    def reranker_models(self) -> tuple[str, ...]:
        values = tuple(item.strip() for item in self.reranker_comparison_models.split(",") if item.strip())
        if not values:
            raise ValueError("RERANKER_COMPARISON_MODELS must contain at least one model")
        return values

    @property
    def regulatory_sources(self) -> list[dict[str, str]]:
        if self.regulatory_urls_json.strip():
            raw = json.loads(self.regulatory_urls_json)
        else:
            import yaml

            path = self.resolve(self.regulatory_sources_path)
            if not path.exists():
                raise ValueError(f"Regulatory sources file not found: {path}")
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            raw = payload.get("sources") or []
        required = {"source", "jurisdiction", "url"}
        if not isinstance(raw, list):
            raise ValueError("Regulatory sources must be a list")
        parsed = []
        for item in raw:
            if not isinstance(item, dict) or not required.issubset(item):
                raise ValueError(f"Every regulatory source must contain {sorted(required)}")
            parsed.append({str(key): str(value) for key, value in item.items()})
        return parsed

    @property
    def dataset_revisions(self) -> dict[str, str]:
        raw = json.loads(self.dataset_revisions_json)
        if not isinstance(raw, dict):
            raise ValueError("DATASET_REVISIONS_JSON must be a JSON object")
        return {str(dataset): str(revision) for dataset, revision in raw.items() if str(revision).strip()}

    def dataset_revision(self, dataset_id: str) -> str | None:
        return self.dataset_revisions.get(dataset_id)

    @property
    def model_revisions(self) -> dict[str, str]:
        raw = json.loads(self.model_revisions_json)
        if not isinstance(raw, dict):
            raise ValueError("MODEL_REVISIONS_JSON must be a JSON object")
        return {str(model): str(revision) for model, revision in raw.items() if str(revision).strip()}

    def model_revision(self, model_id: str) -> str | None:
        if model_id == self.silver_local_model and self.silver_local_model_revision:
            return self.silver_local_model_revision
        return self.model_revisions.get(model_id)

    def backend_for(self, purpose: str) -> str:
        override = None
        if purpose.startswith("S5"):
            override = self.s5_generation_backend
        elif purpose.startswith("S2"):
            override = self.s2_generation_backend
        if override and override != "inherit":
            return override
        return self.model_backend

    @property
    def ollama_generation_model_resolved(self) -> str:
        return self.ollama_generation_model or self.ollama_model

    def endpoint_for(self, purpose: str) -> str | None:
        if purpose.startswith("S2"):
            return self.hf_simplifier_endpoint_url or self.hf_generation_endpoint_url
        if purpose.startswith("S5"):
            return self.hf_risk_endpoint_url or self.hf_generation_endpoint_url
        return self.hf_generation_endpoint_url

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.project_root / path).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
