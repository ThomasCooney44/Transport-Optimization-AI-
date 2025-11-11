Build a minimal repo called "offpeak-simulator-lite" with:

1) A single Jupyter notebook: notebooks/offpeak_simulator.ipynb
   It must:
   - Read a CSV at data/ridership.csv with schema:
       timestamp (ISO 8601), riders (int)
     Example rows:
       2025-09-01 07:00:00, 132
       2025-09-01 07:15:00, 158
     Assume 15-minute or 60-minute intervals; auto-detect and proceed.
   - Create simple features: hour, day_of_week, is_weekend.
   - Baseline forecast:
       For each hour×day_of_week, use the recent N weeks’ mean to predict next 7 days.
       Provide a parameter N=4 (rolling 4-week average).
   - Elasticity simulation (one-slider design):
       Inputs:
         • offpeak_start (HH:MM), offpeak_end (HH:MM)
         • discount_pct (e.g., 0.10 for –10%)
         • price_elasticity (default –0.30)
       Logic:
         • For off-peak periods, adjusted demand y' = y * (1 + price_elasticity * discount_pct)
         • Demand “freed” from peak is the increase observed in off-peak versus baseline.
         • Also subtract a proportional share from peak windows so total daily riders stays constant
           (keep it simple: reduce peak windows uniformly until daily total is unchanged).
   - Outputs:
       • data/processed/forecast.csv (timestamp, yhat)
       • data/processed/simulation.csv (timestamp, baseline, after_incentive, is_peak, is_offpeak)
   - Plots (matplotlib only):
       • Baseline vs after_incentive time series for one representative week
       • Bar chart of riders by hour (before vs after)
       • Summary printout:
           - Total riders unchanged (check)
           - Peak hour reduction (%)
           - Off-peak increase (%)
           - Approx revenue change = Σ (price * qty), assume price=1.00 baseline; off-peak price = 1 – discount_pct
             (report % change)
   - Include a small UI block at the top using ipywidgets:
       offpeak_start, offpeak_end, discount_pct slider, elasticity slider, and a “Run” button.
       When clicked, recompute and redraw.
   - If data/ridership.csv is missing, fabricate 12 weeks of synthetic data with clear peaks at 08:00 and 18:00, plus weekend effects and light weather noise (no external API).

2) A tiny README.md that says:
   - How to run in GitHub Codespaces: open notebooks/offpeak_simulator.ipynb, then "Run All".
   - How to run in Google Colab: upload the notebook and the CSV; then “Run All”.
   - Data format required (two columns).
   - Where outputs are saved.

3) requirements.txt with only:
   pandas
   numpy
   matplotlib
   ipywidgets
   pyarrow

4) Put an example CSV at data/ridership.csv (synthetic, 8–12 weeks).

Make everything work by just opening the notebook and clicking “Run All”. No CLI, no Docker, no complex config.

