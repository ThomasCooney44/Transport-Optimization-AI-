"""Optimization of demand allocations using a linear program."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

try:  # pragma: no cover
    import pulp
except Exception:  # pragma: no cover
    pulp = None


@dataclass
class OptimizationConfig:
    revenue_tolerance: float = 0.0
    min_service_factor: float = 0.8


def optimize_demand(
    demand: pd.DataFrame,
    capacity: pd.DataFrame,
    fares: Dict[str, float],
    config: OptimizationConfig,
) -> pd.DataFrame:
    if pulp is None:
        raise ImportError("PuLP must be installed to run optimization")

    demand = demand.copy()
    demand["timestamp"] = pd.to_datetime(demand["timestamp"])
    capacity = capacity.copy()
    capacity["timestamp"] = pd.to_datetime(capacity["timestamp"])

    routes = demand["route_id"].unique()
    timestamps = sorted(demand["timestamp"].unique())
    fares = {route: fares.get(route, 0.0) for route in routes}

    model = pulp.LpProblem("DemandOptimization", pulp.LpMinimize)

    decision = {
        (route, ts): pulp.LpVariable(f"x_{route}_{i}", lowBound=0)
        for i, ts in enumerate(timestamps)
        for route in routes
    }

    peak_var = pulp.LpVariable("peak_load", lowBound=0)
    model += peak_var

    for route in routes:
        for ts in timestamps:
            cap = capacity.loc[
                (capacity["route_id"] == route) & (capacity["timestamp"] == ts),
                "capacity",
            ]
            cap_value = cap.iloc[0] if not cap.empty else 1e6
            model += decision[(route, ts)] <= cap_value

    for ts in timestamps:
        model += pulp.lpSum(decision[(route, ts)] for route in routes) <= peak_var

    baseline_revenue = (demand["validations"] * demand["route_id"].map(fares)).sum()
    model += (
        pulp.lpSum(decision[(route, ts)] * fares[route] for route in routes for ts in timestamps)
        >= baseline_revenue * (1 - config.revenue_tolerance)
    )

    for _, row in demand.iterrows():
        key = (row["route_id"], row["timestamp"])
        baseline = float(row["validations"])
        model += decision[key] >= baseline * config.min_service_factor

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    optimized = demand.copy()
    optimized["optimized_validations"] = [
        decision[(row.route_id, row.timestamp)].value() for row in optimized.itertuples()
    ]
    optimized["capacity"] = [
        capacity.loc[
            (capacity["route_id"] == row.route_id) & (capacity["timestamp"] == row.timestamp),
            "capacity",
        ].iloc[0]
        if not capacity.loc[
            (capacity["route_id"] == row.route_id) & (capacity["timestamp"] == row.timestamp)
        ].empty
        else 1e6
        for row in optimized.itertuples()
    ]
    optimized["peak_load"] = peak_var.value()
    revenue = (optimized["optimized_validations"] * optimized["route_id"].map(fares)).sum()
    if baseline_revenue:
        optimized["revenue_delta_pct"] = (revenue - baseline_revenue) / baseline_revenue * 100
    else:
        optimized["revenue_delta_pct"] = 0.0
    return optimized
