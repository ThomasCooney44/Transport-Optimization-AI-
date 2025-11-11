"""Data ingestion utilities for GTFS, ridership, weather and events."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class IngestedData:
    """Container with aligned datasets."""

    ridership: pd.DataFrame
    weather: pd.DataFrame
    events: pd.DataFrame
    gtfs: Dict[str, pd.DataFrame]


def _read_csv(path: pathlib.Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required data file missing: {path}")
    return pd.read_csv(path, **kwargs)


def load_gtfs(gtfs_dir: str | pathlib.Path) -> Dict[str, pd.DataFrame]:
    dir_path = pathlib.Path(gtfs_dir)
    gtfs_tables = {}
    for name in [
        "stops",
        "stop_times",
        "trips",
        "routes",
        "calendar",
        "calendar_dates",
    ]:
        file_path = dir_path / f"{name}.txt"
        if file_path.exists():
            gtfs_tables[name] = _read_csv(file_path)
        else:
            raise FileNotFoundError(f"GTFS file missing: {file_path}")
    return gtfs_tables


def load_optional_csv(path: str | pathlib.Path, **kwargs) -> Optional[pd.DataFrame]:
    file_path = pathlib.Path(path)
    if not file_path.exists():
        return None
    return pd.read_csv(file_path, **kwargs)


def generate_synthetic_ridership(
    gtfs_tables: Dict[str, pd.DataFrame],
    start: str,
    end: str,
    freq: str = "H",
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic ridership dataset using schedule information."""

    rng = np.random.default_rng(seed)
    trips = gtfs_tables["trips"]["route_id"].unique()
    stops = gtfs_tables["stop_times"]["stop_id"].unique()
    date_range = pd.date_range(start=start, end=end, freq=freq, inclusive="left")

    records = []
    for route_id in trips:
        base_demand = rng.integers(20, 100)
        for stop_id in stops:
            profile = rng.normal(loc=1.0, scale=0.1, size=len(date_range))
            daily_cycle = 1 + 0.5 * np.sin(2 * np.pi * date_range.hour / 24)
            weekly_cycle = 1 + 0.2 * np.sin(2 * np.pi * date_range.dayofweek / 7)
            demand = np.clip(base_demand * profile * daily_cycle * weekly_cycle, 5, None)
            records.append(
                pd.DataFrame(
                    {
                        "route_id": route_id,
                        "stop_id": stop_id,
                        "timestamp": date_range,
                        "validations": demand.astype(int),
                    }
                )
            )
    ridership = pd.concat(records, ignore_index=True)
    return ridership


def ingest(
    config: Dict[str, str],
    generate_if_missing: bool = True,
    synthetic_range: Optional[Iterable[str]] = None,
) -> IngestedData:
    """Ingest raw data according to *config* paths."""

    gtfs_tables = load_gtfs(config["gtfs_dir"])
    ridership_path = pathlib.Path(config["ridership"]) if "ridership" in config else None
    if ridership_path and ridership_path.exists():
        ridership = pd.read_csv(ridership_path, parse_dates=["timestamp"])
    elif generate_if_missing:
        start, end = synthetic_range or ("2023-01-01", "2023-02-01")
        ridership = generate_synthetic_ridership(gtfs_tables, start=start, end=end)
    else:
        raise FileNotFoundError("Ridership data missing and synthetic generation disabled.")

    weather = _read_csv(pathlib.Path(config["weather"]), parse_dates=["timestamp"])
    events = _read_csv(pathlib.Path(config["events"]), parse_dates=["start_ts", "end_ts"])

    return IngestedData(ridership=ridership, weather=weather, events=events, gtfs=gtfs_tables)
