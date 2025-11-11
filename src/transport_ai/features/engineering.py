"""Feature engineering for transport demand forecasting."""

from __future__ import annotations

import pandas as pd


def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_peak"] = df["hour"].between(7, 9) | df["hour"].between(16, 18)
    return df


def _merge_weather(demand: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    weather = weather.copy()
    weather = weather.sort_values("timestamp").set_index("timestamp").resample("H").ffill()
    merged = demand.merge(
        weather,
        left_on=["timestamp"],
        right_index=True,
        how="left",
    )
    return merged


def _merge_events(demand: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    if events.empty:
        demand["event_weight"] = 0.0
        return demand
    expanded_events = []
    for _, row in events.iterrows():
        rng = pd.date_range(row["start_ts"], row["end_ts"], freq="H")
        expanded_events.append(pd.DataFrame({"timestamp": rng, "event_weight": row.get("event_weight", 1.0)}))
    expanded = pd.concat(expanded_events, ignore_index=True)
    expanded = expanded.groupby("timestamp", as_index=False).agg({"event_weight": "sum"})
    demand = demand.merge(expanded, on="timestamp", how="left")
    demand["event_weight"] = demand["event_weight"].fillna(0.0)
    return demand


def add_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(["route_id", "stop_id"])["validations"].shift(lag)
    df["rolling_24h_mean"] = (
        df.groupby(["route_id", "stop_id"])["validations"].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)
    )
    df["rolling_7d_mean"] = (
        df.groupby(["route_id", "stop_id"])["validations"]
        .rolling(window=24 * 7, min_periods=1)
        .mean()
        .reset_index(0, drop=True)
    )
    return df


def engineer_features(
    ridership: pd.DataFrame,
    weather: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Return a dataframe with engineered features."""

    features = _time_features(ridership)
    features = _merge_weather(features, weather)
    features = _merge_events(features, events)
    features = add_lag_features(features, lags=[1, 24])
    features = features.fillna(method="ffill").fillna(0)
    return features


def compute_capacity(gtfs: dict[str, pd.DataFrame], vehicle_capacity: int = 80) -> pd.DataFrame:
    stop_times = gtfs["stop_times"].copy()
    trips = gtfs["trips"][['trip_id', 'route_id']].copy()
    merged = stop_times.merge(trips, on="trip_id")
    merged["timestamp"] = pd.to_timedelta(merged["arrival_time"].fillna("00:00:00"))
    base_date = pd.Timestamp('2023-01-01')
    merged["timestamp"] = (base_date + merged["timestamp"]).dt.floor("H")
    capacity = (
        merged.groupby(["route_id", "timestamp"], as_index=False)["trip_id"].count().rename(columns={"trip_id": "trips"})
    )
    capacity["capacity"] = capacity["trips"] * vehicle_capacity
    return capacity[["route_id", "timestamp", "capacity"]]
