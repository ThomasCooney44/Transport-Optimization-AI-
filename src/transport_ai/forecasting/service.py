"""Utilities to train and generate forecasts."""

from __future__ import annotations

import pathlib
from typing import Dict, Tuple

import pandas as pd

from .arima_baseline import ARIMAForecaster  # noqa: F401
from .base import ForecasterFactory
from .prophet_model import ProphetForecaster  # noqa: F401
from .xgboost_model import XGBoostForecaster  # noqa: F401


def train_model(
    train: pd.DataFrame,
    model_name: str,
    target_col: str = "validations",
    model_kwargs: Dict | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float], str]:
    model_kwargs = model_kwargs or {}
    forecasts_list = []
    metrics_accum = []
    for (route_id, stop_id), group in train.groupby(["route_id", "stop_id"]):
        group = group.sort_values("timestamp")
        model = ForecasterFactory.create(model_name, **model_kwargs)
        horizon = min(24, max(len(group) // 5, 1))
        history = group.iloc[:-horizon] if len(group) > horizon else group
        validation = group.iloc[-horizon:]
        model.fit(history, target_col=target_col)
        preds = model.predict(horizon)
        metrics = model.evaluate(validation[target_col].values, preds["yhat"].values)
        metrics_accum.append(metrics)
        forecasts_list.append(preds.assign(route_id=route_id, stop_id=stop_id))
    if forecasts_list:
        forecasts = pd.concat(forecasts_list, ignore_index=True)
        metrics_df = pd.DataFrame(metrics_accum)
        metrics_mean = metrics_df.mean().to_dict()
    else:
        forecasts = pd.DataFrame(columns=["timestamp", "yhat", "yhat_lower", "yhat_upper", "route_id", "stop_id"])
        metrics_mean = {"rmse": float("nan"), "mape": float("nan")}
    return forecasts, metrics_mean, model_name



def save_forecasts(forecasts: pd.DataFrame, path: str | pathlib.Path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(path, index=False)
