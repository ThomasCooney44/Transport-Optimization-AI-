"""Command line interface for the Transport Optimization AI toolkit."""

from __future__ import annotations

import argparse
import json
import pathlib

import pandas as pd

from transport_ai.config import Config, load_config
from transport_ai.data.ingest import ingest
from transport_ai.features.engineering import compute_capacity, engineer_features
from transport_ai.forecasting.service import save_forecasts, train_model
from transport_ai.simulation.incentives import ElasticityConfig, simulate_incentives
from transport_ai.optimization.optimizer import OptimizationConfig, optimize_demand
from transport_ai.reporting.report import ReportContext, generate_report


def _load_config(path: str | None) -> Config:
    cfg_path = pathlib.Path(path) if path else pathlib.Path("conf/config.yaml")
    return load_config(cfg_path)


def command_ingest(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    ingested = ingest(config["paths"], generate_if_missing=True)
    out_path = pathlib.Path(config["paths"]["processed_ridership"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ingested.ridership.to_parquet(out_path, index=False)
    print(json.dumps({"rows": len(ingested.ridership)}, indent=2))


def command_features(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    ingested = ingest(config["paths"], generate_if_missing=True)
    features = engineer_features(ingested.ridership, ingested.weather, ingested.events)
    out_path = pathlib.Path(config["paths"]["features"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    capacity = compute_capacity(ingested.gtfs, vehicle_capacity=config["modeling"].get("vehicle_capacity", 80))
    capacity.to_parquet(pathlib.Path(config["paths"]["capacity"]), index=False)
    print(json.dumps({"features": len(features.columns)}, indent=2))


def command_train(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    features_path = pathlib.Path(config["paths"]["features"])
    if not features_path.exists():
        raise FileNotFoundError("Run features command before training")
    features = pd.read_parquet(features_path)
    forecasts, metrics, model_name = train_model(
        features,
        model_name=config["modeling"]["model"],
        model_kwargs=config["modeling"].get("params", {}),
    )
    metrics_path = pathlib.Path(config["paths"]["metrics_json"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_forecasts(forecasts, config["paths"]["forecasts"])
    print(json.dumps({"model": model_name, **metrics}, indent=2))


def command_forecast(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    forecasts_path = pathlib.Path(config["paths"]["forecasts"])
    if not forecasts_path.exists():
        raise FileNotFoundError("No forecasts available, run train first")
    forecasts = pd.read_parquet(forecasts_path)
    horizon = args.horizon
    print(json.dumps({"records": len(forecasts), "horizon": horizon}, indent=2))


def command_simulate(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    ridership_path = pathlib.Path(config["paths"]["processed_ridership"])
    if not ridership_path.exists():
        raise FileNotFoundError("Run ingest command first")
    ridership = pd.read_parquet(ridership_path)
    elasticity_cfg = ElasticityConfig(
        price_elasticity=config["simulation"]["price_elasticity"],
        time_elasticity=config["simulation"].get("time_elasticity", 0.05),
    )
    result = simulate_incentives(ridership, elasticity_cfg, fare=config["simulation"]["fare"])
    sim_path = pathlib.Path(config["paths"]["simulation"])
    sim_path.parent.mkdir(parents=True, exist_ok=True)
    result.simulated.to_parquet(sim_path, index=False)
    summary = {
        "peak_reduction_pct": result.peak_reduction_pct,
        "revenue_delta_pct": result.revenue_delta_pct,
    }
    print(json.dumps(summary, indent=2))


def command_optimize(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    ridership = pd.read_parquet(config["paths"]["processed_ridership"])
    capacity = pd.read_parquet(config["paths"]["capacity"])
    fares = config["modeling"].get("fares", {route: config["simulation"]["fare"] for route in ridership["route_id"].unique()})
    optimized = optimize_demand(
        ridership,
        capacity,
        fares,
        OptimizationConfig(
            revenue_tolerance=config["optimization"].get("revenue_tolerance", 0.0),
            min_service_factor=config["optimization"].get("min_service_factor", 0.8),
        ),
    )
    opt_path = pathlib.Path(config["paths"]["optimization"])
    opt_path.parent.mkdir(parents=True, exist_ok=True)
    optimized.to_parquet(opt_path, index=False)
    print(json.dumps({"peak_load": optimized["peak_load"].iloc[0]}, indent=2))


def command_report(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    forecasts = pd.read_parquet(config["paths"]["forecasts"])
    simulation = pd.read_parquet(config["paths"]["simulation"])
    optimization = pd.read_parquet(config["paths"]["optimization"])
    metrics = json.loads(pathlib.Path(config["paths"]["metrics_json"]).read_text())
    context = ReportContext(
        metrics=metrics,
        simulation={"peak_reduction_pct": simulation["validations"].sum()},
        optimization={"revenue_delta_pct": optimization["revenue_delta_pct"].iloc[0]},
    )
    generate_report(
        forecasts,
        simulation,
        optimization,
        context,
        template_dir="templates",
        output_path=config["paths"]["report"],
    )
    print(json.dumps({"report": config["paths"]["report"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Transport Optimization AI CLI")
    parser.add_argument("--config", help="Path to YAML configuration", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest")
    subparsers.add_parser("features")
    subparsers.add_parser("train")

    forecast_parser = subparsers.add_parser("forecast")
    forecast_parser.add_argument("--horizon", type=int, default=24)

    subparsers.add_parser("simulate")
    subparsers.add_parser("optimize")
    subparsers.add_parser("report")

    args = parser.parse_args()

    commands = {
        "ingest": command_ingest,
        "features": command_features,
        "train": command_train,
        "forecast": command_forecast,
        "simulate": command_simulate,
        "optimize": command_optimize,
        "report": command_report,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
