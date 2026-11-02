"""Tests for the statistical anomaly detection service."""

from unittest.mock import AsyncMock

import pandas as pd
import pytest

from services.anomaly_service import AnomalyDetectionService


@pytest.mark.asyncio
async def test_detect_anomalies_insufficient_data():
    """Empty data returns a non-anomalous 'Insufficient data' result."""
    service = AnomalyDetectionService()
    # Mock the data service to return an empty DataFrame
    service.data_service.fetch_adverse_events_timeseries = AsyncMock(return_value=pd.DataFrame())

    result = await service.detect_anomalies("test_drug")
    assert result["is_anomaly"] is False
    assert result["status"] == "Insufficient data"


@pytest.mark.asyncio
async def test_detect_anomalies_critical_spike():
    """A sharp spike above the baseline is flagged as an anomaly."""
    service = AnomalyDetectionService()
    service.window_size = 4
    service.z_threshold = 2.0

    # 4 weeks of baseline (10 cases each), 1 week of anomaly (50 cases)
    mock_data = pd.DataFrame({
        'Adverse_Events': [10, 10, 10, 10, 50]
    })
    service.data_service.fetch_adverse_events_timeseries = AsyncMock(return_value=mock_data)

    result = await service.detect_anomalies("test_drug")
    assert result["is_anomaly"] is True
    assert result["current_count"] == 50
    assert result["baseline_mean"] == 10.0
    assert result["z_score"] > 2.0
