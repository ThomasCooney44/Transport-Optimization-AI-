"""Prophet forecaster implementation."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import Forecaster, ForecasterFactory

try:  # pragma: no cover
    from prophet import Prophet
except Exception:  # pragma: no cover
    Prophet = None


class ProphetForecaster(Forecaster):
    name = "prophet"

    def __init__(self, extra_regressors: Optional[list[str]] = None, **kwargs) -> None:
        if Prophet is None:
            raise ImportError("prophet is required for ProphetForecaster")
        self.model = Prophet(**kwargs)
        self.extra_regressors = extra_regressors or []
        for reg in self.extra_regressors:
            self.model.add_regressor(reg)
        self.train_: Optional[pd.DataFrame] = None

    def fit(self, train: pd.DataFrame, target_col: str = "validations") -> None:
        df = train.rename(columns={"timestamp": "ds", target_col: "y"})
        self.train_ = df
        self.model.fit(df)

    def predict(self, horizon: int) -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=horizon, freq="H")
        for reg in self.extra_regressors:
            future[reg] = self.train_[reg].iloc[-len(future) :].values
        forecast = self.model.predict(future)
        return forecast.rename(columns={"ds": "timestamp", "yhat_lower": "yhat_lower", "yhat_upper": "yhat_upper"})[
            ["timestamp", "yhat", "yhat_lower", "yhat_upper"]
        ]


ForecasterFactory.register("prophet", ProphetForecaster)
