from __future__ import annotations

from pathlib import Path

import pytest

pandas = pytest.importorskip('pandas')
pd = pandas

from transport_ai.config import Config
from transport_ai.data.ingest import ingest
from transport_ai.features.engineering import compute_capacity, engineer_features
from transport_ai.forecasting.service import train_model
from transport_ai.optimization.optimizer import OptimizationConfig, optimize_demand, pulp
from transport_ai.simulation.incentives import ElasticityConfig, simulate_incentives


@pytest.fixture()
def tmp_config(tmp_path: Path) -> Config:
    gtfs_dir = tmp_path / "gtfs"
    gtfs_dir.mkdir()
    (gtfs_dir / "stops.txt").write_text("stop_id,stop_name\nSTOP1,Stop 1\n", encoding="utf-8")
    (gtfs_dir / "routes.txt").write_text("route_id,route_short_name\nROUTE1,1\n", encoding="utf-8")
    (gtfs_dir / "trips.txt").write_text(
        "route_id,service_id,trip_id\nROUTE1,WEEKDAY,TRIP1\nROUTE1,WEEKDAY,TRIP2\n", encoding="utf-8"
    )
    (gtfs_dir / "stop_times.txt").write_text(
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
        "TRIP1,08:00:00,08:05:00,STOP1,1\n"
        "TRIP1,09:00:00,09:05:00,STOP1,2\n"
        "TRIP2,17:00:00,17:05:00,STOP1,1\n",
        encoding="utf-8",
    )
    (gtfs_dir / "calendar.txt").write_text(
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
        "WEEKDAY,1,1,1,1,1,0,0,20230101,20231231\n",
        encoding="utf-8",
    )
    (gtfs_dir / "calendar_dates.txt").write_text("service_id,date,exception_type\n", encoding="utf-8")

    weather_path = tmp_path / "weather.csv"
    weather = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=48, freq="H"),
            "station_id": "DUBLIN",
            "temp_c": 10.0,
            "precipitation_mm": 0.1,
            "wind_m_s": 3.0,
            "weather_code": "clear",
        }
    )
    weather.to_csv(weather_path, index=False)

    events_path = tmp_path / "events.csv"
    events = pd.DataFrame(
        {
            "start_ts": ["2023-01-01 18:00:00"],
            "end_ts": ["2023-01-01 20:00:00"],
            "event_name": ["Concert"],
            "event_weight": [1.5],
        }
    )
    events.to_csv(events_path, index=False)

    cfg = Config(
        raw={
            "paths": {
                "gtfs_dir": str(gtfs_dir),
                "ridership": str(tmp_path / "ridership.csv"),
                "weather": str(weather_path),
                "events": str(events_path),
                "processed_ridership": str(tmp_path / "ridership.parquet"),
                "features": str(tmp_path / "features.parquet"),
                "forecasts": str(tmp_path / "forecasts.parquet"),
                "metrics_json": str(tmp_path / "metrics.json"),
                "simulation": str(tmp_path / "simulation.parquet"),
                "optimization": str(tmp_path / "optimization.parquet"),
                "capacity": str(tmp_path / "capacity.parquet"),
                "report": str(tmp_path / "report.html"),
            },
            "modeling": {"model": "arima", "params": {"window": 12}, "vehicle_capacity": 80},
            "simulation": {"fare": 2.0, "price_elasticity": 0.05, "time_elasticity": 0.05},
            "optimization": {"revenue_tolerance": 0.0, "min_service_factor": 0.8},
        }
    )
    return cfg


def test_end_to_end_pipeline(tmp_config: Config) -> None:
    ingested = ingest(tmp_config["paths"], generate_if_missing=True, synthetic_range=("2023-01-01", "2023-01-03"))
    assert not ingested.ridership.empty

    features = engineer_features(ingested.ridership, ingested.weather, ingested.events)
    assert {"lag_1", "lag_24"}.issubset(features.columns)

    forecasts, metrics, model_name = train_model(features, model_name="arima", model_kwargs={"window": 12})
    assert model_name == "arima"
    assert set(metrics) == {"rmse", "mape"}

    sim_result = simulate_incentives(
        ingested.ridership,
        ElasticityConfig(price_elasticity=0.05, time_elasticity=0.05),
        fare=2.0,
    )
    assert sim_result.peak_reduction_pct >= 0

    capacity = compute_capacity(ingested.gtfs, vehicle_capacity=80)
    capacity["timestamp"] = pd.to_datetime(capacity["timestamp"])
    if pulp is not None:
        optimized = optimize_demand(
            ingested.ridership.assign(timestamp=pd.to_datetime(ingested.ridership["timestamp"])),
            capacity,
            {route: 2.0 for route in ingested.ridership["route_id"].unique()},
            OptimizationConfig(revenue_tolerance=0.0, min_service_factor=0.5),
        )
        assert "optimized_validations" in optimized.columns
    else:
        pytest.skip("PuLP not available")
