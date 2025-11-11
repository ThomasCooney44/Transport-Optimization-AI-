"""Metrics computation helpers."""

from __future__ import annotations

import pandas as pd


def compute_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    error = actual - predicted
    rmse = float((error**2).mean() ** 0.5)
    mape = float((error.abs() / actual.replace(0, pd.NA)).dropna().mean())
    return {"rmse": rmse, "mape": mape}
