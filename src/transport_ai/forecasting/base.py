"""Forecasting interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass
class ForecastResult:
    predictions: pd.DataFrame
    metrics: Dict[str, float]


class Forecaster(ABC):
    name: str

    @abstractmethod
    def fit(self, train: pd.DataFrame, target_col: str = "validations") -> None:
        ...

    @abstractmethod
    def predict(self, horizon: int) -> pd.DataFrame:
        ...

    @staticmethod
    def evaluate(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        error = y_true - y_pred
        rmse = float((error**2).mean() ** 0.5)
        mape = float((error.abs() / y_true.replace(0, pd.NA)).dropna().mean())
        return {"rmse": rmse, "mape": mape}


class ForecasterFactory:
    registry: Dict[str, type[Forecaster]] = {}

    @classmethod
    def register(cls, name: str, estimator: type[Forecaster]) -> None:
        cls.registry[name] = estimator

    @classmethod
    def create(cls, name: str, **kwargs) -> Forecaster:
        if name not in cls.registry:
            raise KeyError(f"Unknown forecaster: {name}")
        return cls.registry[name](**kwargs)

    @classmethod
    def available(cls) -> Tuple[str, ...]:
        return tuple(cls.registry)
