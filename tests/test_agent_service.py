"""Tests for the structured output schema of the agent service."""

from services.agent_service import ClinicalAnalysis


def test_json_items_are_flattened_to_text():
    """JSON-encoded list items from the LLM are converted into readable sentences."""
    result = ClinicalAnalysis(
        risk_factors=[
            '{"name": "ibuprofen_nephropathy", "source": "Kim et al. (2019)", "description": "Erhöhtes Risiko für Nierenschäden."}',
            "Erhöhtes Blutungsrisiko bei gleichzeitiger Einnahme von Aspirin.",
        ],
        contraindications=[{"name": "ibuprofen_hemophilia_a", "description": "Hämophilie A."}],
        summary="Kurze Zusammenfassung.",
        confidence_score=0.8,
    )

    assert result.risk_factors == [
        "Erhöhtes Risiko für Nierenschäden. (Kim et al. (2019))",
        "Erhöhtes Blutungsrisiko bei gleichzeitiger Einnahme von Aspirin.",
    ]
    assert result.contraindications == ["Hämophilie A."]
