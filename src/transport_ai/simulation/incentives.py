"""Demand shifting simulation using elasticity assumptions."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ElasticityConfig:
    price_elasticity: float
    time_elasticity: float
    peak_hours: tuple[int, int] = (7, 9)
    shoulder_hours: tuple[int, int] = (10, 15)


@dataclass
class SimulationResult:
    baseline: pd.DataFrame
    simulated: pd.DataFrame
    peak_reduction_pct: float
    revenue_delta_pct: float


def simulate_incentives(
    demand: pd.DataFrame,
    elasticity: ElasticityConfig,
    fare: float,
) -> SimulationResult:
    baseline = demand.copy()
    baseline["hour"] = baseline["timestamp"].dt.hour
    baseline_revenue = baseline["validations"].sum() * fare

    simulated = baseline.copy()
    peak_mask = simulated["hour"].between(elasticity.peak_hours[0], elasticity.peak_hours[1])
    shoulder_mask = simulated["hour"].between(elasticity.shoulder_hours[0], elasticity.shoulder_hours[1])

    peak_demand = simulated.loc[peak_mask, "validations"]
    shoulder_demand = simulated.loc[shoulder_mask, "validations"]

    total_elasticity = elasticity.price_elasticity + elasticity.time_elasticity
    shifted = peak_demand * (-total_elasticity)
    simulated.loc[peak_mask, "validations"] = peak_demand + shifted
    shoulder_count = max(shoulder_mask.sum(), 1)
    simulated.loc[shoulder_mask, "validations"] = shoulder_demand - shifted.sum() / shoulder_count
    simulated["validations"] = simulated["validations"].clip(lower=0)

    simulated_revenue = simulated["validations"].sum() * fare
    peak_reduction = 1 - (simulated.loc[peak_mask, "validations"].sum() / peak_demand.sum())
    revenue_delta = (simulated_revenue - baseline_revenue) / baseline_revenue if baseline_revenue else 0

    return SimulationResult(
        baseline=baseline,
        simulated=simulated,
        peak_reduction_pct=float(peak_reduction * 100),
        revenue_delta_pct=float(revenue_delta * 100),
    )
