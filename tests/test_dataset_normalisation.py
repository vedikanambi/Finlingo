from backend.app.core.config import Settings
from backend.app.services.dataset_streams import StreamingDatasetRepository, _normalise_nli_label


def test_contractnli_published_integer_label_order():
    assert _normalise_nli_label(0) == "contradiction"
    assert _normalise_nli_label(1) == "entailment"
    assert _normalise_nli_label(2) == "neutral"


def test_claritystorm_cfpb_consumer_narrative_is_detected(monkeypatch):
    repository = StreamingDatasetRepository(Settings(_env_file=None))
    row = {
        "complaint_id": "123",
        "consumer_narrative": "The lender charged an unexpected fee after I paid the loan balance in full.",
        "product": "Consumer Loan",
        "issue": "Fees",
        "sub_issue": "Unexpected fee",
    }
    monkeypatch.setattr(repository, "stream_raw", lambda *args, **kwargs: iter([row]))
    try:
        records = list(repository.cfpb_narratives(limit=1))
    finally:
        repository.close()
    assert len(records) == 1
    assert records[0]["text"] == row["consumer_narrative"]
    assert records[0]["dataset"] == "Mouwiya/cfpb-consumer-complaints"
