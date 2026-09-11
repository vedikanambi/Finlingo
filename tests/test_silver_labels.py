from backend.training.silver_labels import SilverLabelGenerator, SimplificationQualityBatch, _critical_tokens_preserved


def test_silver_quality_filter_protects_numbers_and_negation():
    assert _critical_tokens_preserved("You must not pay €10.", "You do not have to pay €10.")
    assert not _critical_tokens_preserved("You must pay €10.", "You must pay €100.")
    assert not _critical_tokens_preserved("You must not pay €10.", "You must pay €10.")


def test_silver_quality_filter_allows_faithful_compression_of_incidental_numbers():
    source = (
        "If Distributor does not pay within 30 days, Company may charge 1.5% interest "
        "under Section 4.2. This Agreement, dated 1 January 2015, has a 3 year term."
    )
    target = "If you do not pay within 30 days, the company can charge 1.5% interest."
    assert _critical_tokens_preserved(source, target)


def test_silver_quality_filter_still_catches_hallucinated_numbers():
    source = "Distributor shall pay within 30 days, Company may charge 1.5% interest."
    hallucinated = "If you do not pay within 30 days, the company can charge 5% interest."
    assert not _critical_tokens_preserved(source, hallucinated)


def test_silver_quality_filter_ignores_trailing_punctuation_artifacts():
    # "2015," shouldn't be treated as a different number than "2015" elsewhere
    source = "This Agreement, dated 1 January 2015, remains in effect."
    target = "This agreement started in 2015."
    assert _critical_tokens_preserved(source, target)


def test_judge_prompt_forbids_echoing_source_and_candidate():
    generator = SilverLabelGenerator.__new__(SilverLabelGenerator)
    from backend.app.core.config import Settings

    generator.settings = Settings(_env_file=None)
    captured = {}
    generator._validated_response = lambda schema, **kwargs: (
        captured.update(kwargs) or SimplificationQualityBatch(items=[])
    )
    generator._quality_batch([{"source_id": "x", "source": "source", "candidate": "candidate"}])
    combined = captured["instructions"] + captured["input_text"]
    assert "Never echo or return source or candidate text" in combined
