"""Statistical anomaly detection for adverse drug events.

Replaces the previous LSTM model. This service provides a robust, statistical
approach to determining if the current week's adverse event count is anomalous
compared to a historical baseline window.
"""

import logging
import numpy as np

from config import settings
from services.openfda_service import MedicalDataService

logger = logging.getLogger(__name__)

class AnomalyDetectionService:
    """Detects statistical anomalies in adverse event reporting rates."""

    def __init__(self):
        self.data_service = MedicalDataService()
        self.window_size = settings.ANOMALY_LOOKBACK_WEEKS
        self.z_threshold = settings.ANOMALY_Z_THRESHOLD

    async def detect_anomalies(self, drug_name: str) -> dict:
        """Calculate if the most recent week is a statistical anomaly asynchronously.

        Args:
            drug_name: The active ingredient to analyze.

        Returns:
            Dictionary containing the anomaly status, z-score, and historical baseline.
        """
        logger.info(f"Running anomaly detection for {drug_name}")
        df = await self.data_service.fetch_adverse_events_timeseries(drug_name)

        if df.empty or len(df) < self.window_size + 1:
            logger.warning(f"Insufficient data for {drug_name}. Need {self.window_size + 1} weeks, got {len(df)}.")
            return {
                "is_anomaly": False,
                "current_count": 0,
                "baseline_mean": 0.0,
                "z_score": 0.0,
                "status": "Insufficient data"
            }

        counts = df['Adverse_Events'].values
        current_week_count = counts[-1]

        # Historical baseline excludes the current week
        historical_baseline = counts[-(self.window_size + 1):-1]

        mean = np.mean(historical_baseline)
        std = np.std(historical_baseline)
        std = std if std > 0 else 1.0  # Prevent division by zero

        z_score = (current_week_count - mean) / std
        is_anomaly = bool(z_score > self.z_threshold)

        return {
            "is_anomaly": is_anomaly,
            "current_count": int(current_week_count),
            "baseline_mean": round(float(mean), 2),
            "z_score": round(float(z_score), 2),
            "status": "Critical Spike Detected" if is_anomaly else "Normal Reporting Volume"
        }
