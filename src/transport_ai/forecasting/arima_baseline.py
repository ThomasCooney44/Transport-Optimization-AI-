"""Light-weight ARIMA style baseline forecaster."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .base import Forecaster, ForecasterFactory


class ARIMAForecaster(Forecaster):
    """Simple baseline emulating ARIMA behaviour using moving averages."""

    name = "arima"

    def __init__(self, window: int = 24) -> None:
        self.window = window
        self.train_: Optional[pd.DataFrame] = None
        self.target_col_ = "validations"

    def fit(self, train: pd.DataFrame, target_col: str = "validations") -> None:
        self.train_ = train.sort_values("timestamp").copy()
        self.target_col_ = target_col

    def predict(self, horizon: int) -> pd.DataFrame:
        if self.train_ is None:
            raise RuntimeError("Model must be fit before predicting")
        history = self.train_.copy()
        preds = []
        timestamps = []
        diffs = history["timestamp"].diff().dropna()
        freq = diffs.mode().iloc[0] if not diffs.empty else pd.Timedelta(hours=1)
        last_ts = history["timestamp"].iloc[-1]
        for i in range(1, horizon + 1):
            window = history[self.target_col_].tail(self.window)
            pred = window.mean()
            preds.append(float(pred))
            last_ts = last_ts + freq
            timestamps.append(last_ts)
            history.loc[len(history)] = history.iloc[-1]
            history.at[len(history) - 1, "timestamp"] = last_ts
            history.at[len(history) - 1, self.target_col_] = pred
        preds = np.array(preds)
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "yhat": preds,
                "yhat_lower": preds * 0.95,
                "yhat_upper": preds * 1.05,
            }
        )


ForecasterFactory.register("arima", ARIMAForecaster)
