from backend.evaluation.metrics import binary_faithfulness_metrics, attribution_accuracy, retrieval_metrics


def test_faithfulness_reports_both_supported_and_unsupported_classes():
    result = binary_faithfulness_metrics([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], 0.75)
    assert result.precision == 1
    assert result.recall == 1
    assert result.unsupported_precision == 1
    assert result.unsupported_recall == 1
    assert result.predicted_hallucination_rate == 0.5


def test_attribution_accuracy_is_not_coverage():
    result = attribution_accuracy(["a", "wrong", None], ["a", "b", "c"])
    assert result["coverage"] == 2 / 3
    assert result["accuracy"] == 1 / 3


def test_retrieval_metrics():
    result = retrieval_metrics([["x", "gold"], ["none"]], ["gold", "gold2"], 2)
    assert result["recall@2"] == 0.5
    assert result["mrr"] == 0.25


def test_risk_abstention_is_not_silently_mapped_to_safe():
    from backend.evaluation.metrics import multiclass_metrics

    labels = ["Auto-Renewal", "Hidden Fee", "Liability Waiver", "Data Sharing", "Penalty Clause", "Safe"]
    result = multiclass_metrics(["Safe", "Hidden Fee"], ["Needs Review", "Hidden Fee"], labels)
    assert result["n_invalid_or_abstained"] == 1
    assert result["invalid_or_abstention_rate"] == 0.5
    assert "__OTHER__" in result["labels"]
    assert result["accuracy"] == 0.5


def test_missed_unsupported_rate_uses_independent_ground_truth():
    result = binary_faithfulness_metrics([1, 0, 0], [0.9, 0.8, 0.1], 0.75)
    assert result.missed_unsupported_rate == 0.5
