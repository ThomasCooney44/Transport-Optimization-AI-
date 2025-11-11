# offpeak-simulator-lite

## Running in GitHub Codespaces
1. Open `notebooks/offpeak_simulator.ipynb`.
2. Choose **Run All** from the notebook toolbar.

## Running in Google Colab
1. Upload `notebooks/offpeak_simulator.ipynb` and `data/ridership.csv` to Colab.
2. Select **Runtime → Run all**.

## Data format
`data/ridership.csv` must contain two columns: `timestamp` (ISO 8601) and `riders` (integer counts at 15- or 60-minute intervals).

## Outputs
Running the notebook saves results to `data/processed/forecast.csv` and `data/processed/simulation.csv`.
