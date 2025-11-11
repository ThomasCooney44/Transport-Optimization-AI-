"""XGBoost forecaster implementation."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import Forecaster, ForecasterFactory

try:  # pragma: no cover - heavy dependency is optional
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


class XGBoostForecaster(Forecaster):
    name = "xgboost"

    def __init__(self, features: Optional[list[str]] = None, **kwargs) -> None:
        if XGBRegressor is None:
            raise ImportError("xgboost is required for XGBoostForecaster")
        self.model = XGBRegressor(**kwargs)
        self.features = features
        self.train_: Optional[pd.DataFrame] = None

    def fit(self, train: pd.DataFrame, target_col: str = "validations") -> None:
        self.train_ = train.copy()
        y = train[target_col]
        feature_cols = self.features or [col for col in train.columns if col not in {target_col, "timestamp"}]
        X = train[feature_cols]
        self.model.fit(X, y)
        self.feature_cols_ = feature_cols
        self.target_col_ = target_col

    def predict(self, horizon: int) -> pd.DataFrame:
        if self.train_ is None:
            raise RuntimeError("Model must be fit before predicting")
        future = self.train_.tail(horizon)
        preds = self.model.predict(future[self.feature_cols_])
        return pd.DataFrame(
            {
                "timestamp": future["timestamp"],
                "yhat": preds,
                "yhat_lower": preds * 0.9,
                "yhat_upper": preds * 1.1,
            }
        )


ForecasterFactory.register("xgboost", XGBoostForecaster)
