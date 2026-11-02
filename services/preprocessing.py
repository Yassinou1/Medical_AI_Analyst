"""Shared feature engineering pipeline for adverse event time series.

This module is the single source of truth for transforming raw weekly
adverse-event counts into the feature matrix consumed by the LSTM model.
Any change here automatically applies to training, evaluation, and inference.
"""

import numpy as np
import pandas as pd


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling statistics and rate-of-change from raw event counts.

    Features generated:
        - SMA_4:  4-week simple moving average  (short-term trend)
        - SMA_12: 12-week simple moving average (seasonal trend)
        - ROC:    week-over-week percentage change (momentum)

    Rows containing NaN or ±inf (from rolling window warm-up or zero-division)
    are dropped so the output is model-ready.

    Args:
        df: DataFrame with at least an ``Adverse_Events`` column,
            indexed by datetime.

    Returns:
        A copy of the input with the three new feature columns appended
        and incomplete rows removed.
    """
    data = df[["Adverse_Events"]].copy()

    data["SMA_4"] = data["Adverse_Events"].rolling(window=4).mean()
    data["SMA_12"] = data["Adverse_Events"].rolling(window=12).mean()
    data["ROC"] = data["Adverse_Events"].pct_change()

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    return data
