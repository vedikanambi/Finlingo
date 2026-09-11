import pytest

from backend.training.silver_label_providers import (
    _parse_structured_response,
)
from backend.training.silver_labels import (
    RiskLabelReviewBatch,
    SimplificationQualityBatch,
)


@pytest.mark.parametrize(
    "raw",
    [
        """
        {
          "items": [
            {
              "id": "row-1",
              "semantic_accuracy": 5,
              "readability": 4,
              "faithful": true,
              "rationale": "Meaning preserved."
            }
          ]
        }
        """,
        """
        [
          {
            "id": "row-1",
            "semantic_accuracy": 5,
            "readability": 4,
            "faithful": true,
            "rationale": "Meaning preserved."
          }
        ]
        """,
        """
        {
          "id": "row-1",
          "semantic_accuracy": 5,
          "readability": 4,
          "faithful": true,
          "rationale": "Meaning preserved."
        }
        """,
        """
        [
          {
            "items": [
              {
                "id": "row-1",
                "semantic_accuracy": 5,
                "readability": 4,
                "faithful": true,
                "rationale": "Meaning preserved."
              }
            ]
          }
        ]
        """,
        """
        {
          "items": [
            {
              "items": [
                {
                  "id": "row-1",
                  "semantic_accuracy": 5,
                  "readability": 4,
                  "faithful": true,
                  "rationale": "Meaning preserved."
                }
              ]
            }
          ]
        }
        """,
    ],
)
def test_quality_batch_normalizes_all_observed_envelopes(raw):
    parsed = _parse_structured_response(
        raw,
        SimplificationQualityBatch,
    )

    assert len(parsed.items) == 1
    assert parsed.items[0].id == "row-1"
    assert parsed.items[0].semantic_accuracy == 5
    assert parsed.items[0].readability == 4
    assert parsed.items[0].faithful is True


def test_risk_batch_unwraps_nested_envelope():
    raw = """
    [
      {
        "items": [
          {
            "id": "row-1",
            "correct_label": "Data Sharing",
            "confidence": 5,
            "rationale": "The clause describes data sharing."
          }
        ]
      }
    ]
    """

    parsed = _parse_structured_response(
        raw,
        RiskLabelReviewBatch,
    )

    assert len(parsed.items) == 1
    assert parsed.items[0].id == "row-1"
    assert parsed.items[0].correct_label == "Data Sharing"
    assert parsed.items[0].confidence == 5


def test_risk_batch_maps_proposed_label_alias():
    raw = """
    [
      {
        "id": "row-1",
        "proposed_label": "Safe",
        "confidence": 5,
        "rationale": "No listed risk is present."
      }
    ]
    """

    parsed = _parse_structured_response(
        raw,
        RiskLabelReviewBatch,
    )

    assert parsed.items[0].correct_label == "Safe"


def test_quality_batch_rejects_missing_required_scores():
    raw = """
    {
      "items": [
        {
          "id": "row-1",
          "source": "Original",
          "candidate": "Simplified"
        }
      ],
      "scores": [3],
      "faithful": true
    }
    """

    with pytest.raises(Exception):
        _parse_structured_response(
            raw,
            SimplificationQualityBatch,
        )
