# Transport Optimization AI

A reference Python toolkit for ingesting transport data, generating demand forecasts, simulating incentive schemes, and optimizing public transport loads. The project targets GTFS feeds from Ireland's National Transport Authority (NTA) with optional ridership, weather, and event feeds.

## Features

- **ETL** for GTFS, weather, events, and optional ridership CSVs. Synthetic demand generation fills gaps during development.
- **Feature engineering** for temporal, calendar, weather, and lagged demand signals plus GTFS-derived capacity estimates.
- **Forecasting** abstractions with interchangeable ARIMA baseline, XGBoost, and Prophet implementations.
- **Incentive simulation** that applies configurable price/time elasticities to shift demand from peak to shoulder periods.
- **Optimization** via linear programming (PuLP) to minimize peak loads subject to capacity and revenue constraints.
- **CLI** with subcommands for ingest, feature generation, training, forecasting, simulation, optimization, and reporting.
- **Automated HTML reports** rendered with Jinja2 and matplotlib charts.
- **Docker & Makefile** for reproducible execution.
- **Pytest suite** with synthetic data generator to validate the end-to-end workflow.

## Quickstart

```bash
# Optional: create a virtual environment
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt  # or use poetry/pip-tools if preferred

# Run ETL + feature engineering
python cli.py ingest
python cli.py features

# Train baseline model and produce forecasts
python cli.py train

# Simulate incentives and optimize allocations
python cli.py simulate
python cli.py optimize

# Build HTML report
python cli.py report
```

Generated artefacts are stored under `data/processed/`, models in `models/`, and reports in `reports/`.

## Configuration

All paths and modelling parameters are configured via YAML (`conf/config.yaml`). Override fields by pointing the CLI to a different configuration file:

```bash
python cli.py --config conf/alt_config.yaml train
```

Key sections include:

- `paths`: Raw data locations and processed artefact destinations.
- `modeling`: Forecast model selection (`arima`, `xgboost`, or `prophet`), hyperparameters, vehicle capacity, and optional fares per route.
- `simulation`: Elasticity parameters and default fare.
- `optimization`: Revenue tolerance and minimum service guarantees.

## Data Expectations

Place source files under `data/raw/`:

- `gtfs/`: `stops.txt`, `stop_times.txt`, `trips.txt`, `routes.txt`, `calendar.txt`, `calendar_dates.txt`
- `ridership/ridership.csv` (optional): `route_id,stop_id,timestamp,validations`
- `weather/weather.csv`: `timestamp,station_id,temp_c,precipitation_mm,wind_m_s,weather_code`
- `events/events.csv`: `start_ts,end_ts,event_name,event_weight`

If ridership data is missing the CLI synthesises realistic panel data per route-stop-hour for experimentation.

## Metrics & Reports

- Forecast evaluation: RMSE and MAPE stored in `data/processed/metrics.json`.
- Simulation: peak-to-off-peak demand shift percentage and revenue deltas.
- Optimization: peak load after optimization, capacity adherence, and revenue impact.
- Reports: `reports/summary.html` bundles metrics with plots for forecast, simulation, and optimization outputs.

## Development

```bash
make setup    # install Python dependencies
make lint     # run formatting/linting hooks (configurable)
make test     # run pytest suite
make train    # execute CLI train pipeline
make simulate # run incentive simulation
make optimize # run optimization workflow
```

## Notebook

`notebooks/demo.ipynb` demonstrates an end-to-end workflow using synthetic data and mirrors the CLI steps for exploratory analysis.

## License

MIT
