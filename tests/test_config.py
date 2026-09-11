from backend.app.core.config import Settings


def test_canonical_proposal_values_and_no_aws():
    settings = Settings(_env_file=None)
    assert settings.qlora_rank == 16
    assert settings.qlora_alpha == 32
    assert settings.qlora_quant_type == "nf4"
    assert settings.chunk_size_tokens == 512
    assert settings.chunk_overlap_tokens == 64
    # these got widened from the proposal's original 8/5/0.75 once I saw how low retrieval recall was at 8
    assert settings.top_k_retrieval == 500
    assert settings.top_j_reranked == 15
    assert settings.regulatory_max_chars_per_source == 5_000_000
    assert settings.faithfulness_tau == 0.376
    assert settings.tau_values == (0.3, 0.376, 0.5, 0.65)
    assert settings.simplifier_min_training_pairs == 100
    assert settings.simplifier_batch_size == 1
    assert settings.simplifier_gradient_accumulation_steps == 16
    assert settings.simplifier_fp16 is True
    assert settings.cfpb_dataset == "Mouwiya/cfpb-consumer-complaints"
    assert settings.verifier_model == "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    assert settings.verifier_training_base_model == settings.verifier_model
    assert settings.verifier_training_mode == "fp16_lora"
    assert settings.verifier_learning_rate == 1e-5
    assert settings.verifier_num_train_epochs == 2
    assert settings.verifier_use_class_weights is True
    assert settings.verifier_early_stopping_patience == 3
    assert not hasattr(settings, "aws_access_key_id")
    assert not hasattr(settings, "s3_bucket")


def test_task_specific_endpoint_routing_and_primary_premise():
    settings = Settings(
        _env_file=None,
        model_backend="hf_endpoint",
        hf_generation_endpoint_url="https://legacy",
        hf_simplifier_endpoint_url="https://simplifier",
        hf_risk_endpoint_url="https://risk",
    )
    assert settings.endpoint_for("S2 simplifier") == "https://simplifier"
    assert settings.endpoint_for("S5 risk classifier") == "https://risk"
    assert settings.faithfulness_primary_premise == "retrieved_chunk"
    assert settings.attribution_tau == 0.376


def test_simplifier_minimum_pairs_must_be_positive():
    import pytest

    with pytest.raises(ValueError, match="SIMPLIFIER_MIN_TRAINING_PAIRS"):
        Settings(_env_file=None, simplifier_min_training_pairs=0)
