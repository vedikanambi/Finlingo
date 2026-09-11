from backend.app.core.config import Settings
from backend.app.core.schemas import ClauseResult, SentenceFaithfulness
from backend.app.stages.stage7_report_generator import Stage7ReportGenerator


def test_document_metrics_weight_sentences_not_clause_means():
    settings = Settings(_env_file=None)
    clauses = [
        ClauseResult(
            clause_id="C1",
            original_text="x",
            simplified_text="Simple.",
            readability_grade=7,
            risk_label="Safe",
            risk_score=1,
            faithfulness=[
                SentenceFaithfulness(sentence="a", source_faithfulness=0.9, source_supported=True, unsupported=False),
                SentenceFaithfulness(sentence="b", source_faithfulness=0.3, source_supported=False, unsupported=True),
            ],
        )
    ]
    result = Stage7ReportGenerator(settings).build(
        "test.txt", clauses, total_runtime_ms=1, regulatory_sources=[], warnings=[]
    )
    assert result.metrics.sentence_count == 2
    assert result.metrics.hallucination_rate == 0.5
    assert result.metrics.mean_source_faithfulness == 0.6
