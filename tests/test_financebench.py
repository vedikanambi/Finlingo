from backend.evaluation.financebench_eval import normalise_financebench_row
from backend.evaluation.metrics import retrieval_metrics_multi


def test_financebench_normalises_list_of_structs():
    row = {
        "financebench_id": "fb-1",
        "question": "What was revenue?",
        "answer": "$10 million",
        "evidence": [
            {"evidence_text": "Revenue was $10 million.", "doc_name": "annual-report.pdf", "page_number": 7},
            {"evidence_text": "Prior revenue was $8 million.", "doc_name": "annual-report.pdf", "page_number": 6},
        ],
    }
    example = normalise_financebench_row(row)
    assert example is not None
    assert example.example_id == "fb-1"
    assert len(example.evidence) == 2
    assert example.evidence[0].page_number == 7
    assert example.evidence[0].chunk_id.startswith("FB-")


def test_financebench_normalises_struct_of_lists():
    row = {
        "id": "fb-2",
        "question": "Question",
        "answer": "Answer",
        "evidence": {
            "evidence_text": ["First evidence", "Second evidence"],
            "doc_name": ["a.pdf", "b.pdf"],
            "page_number": [1, 2],
        },
    }
    example = normalise_financebench_row(row)
    assert example is not None
    assert [item.document_name for item in example.evidence] == ["a.pdf", "b.pdf"]


def test_multi_gold_retrieval_metrics():
    result = retrieval_metrics_multi(
        rankings=[["x", "g2", "g1"], ["z"]],
        expected_sets=[{"g1", "g2"}, {"g3"}],
        k=3,
    )
    assert result["recall@3"] == 0.5
    assert result["mrr"] == 0.25
    assert result["n_examples"] == 2
