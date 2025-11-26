# Offpeak Simulator Lite

A simple model that forecasts public transport demand across the day and tests whether price discounts can shift people from peak to off-peak travel. Built for our Digital & AI Strategy project on improving capacity planning in Irish public transport.

---

## What the Model Does
- Creates a daily demand profile (baseline + morning/evening peaks).
- Tests different discount levels for off-peak or peak hours.
- Estimates price elasticity of demand (PED).
- Compares baseline vs incentive scenarios using clear plots.
- Shows that **peak-hour demand is highly inelastic** – pricing does not meaningfully shift commuters.

---

## How to Run It (GitHub Codespaces)
1. Open this repo in **Codespaces**.
2. Go to `notebooks/offpeak_simulator.ipynb`.
3. Click **Run All**.
4. Scroll to the bottom for PED results and scenario comparisons.

---

## Why This Model Matters
This notebook acts as the technical add-on to our project.  
It provides evidence that:
- AI can forecast demand well, **but**
- Pricing alone will not fix peak-hour congestion in Ireland,
because commuters have fixed schedules.

The model supports our argument that real impact will come from:
- dynamic capacity planning,
- better data integration,
- and redesigned operational processes.

---

## Repo Structure
- `notebooks/offpeak_simulator.ipynb` — main model.
- `figures/` — optional output plots.
- `data/` — optional input data.

---

## Next Steps (Optional)
Future versions could:
- use real TFI/GTFS data,
- include weather/events,
- or simulate dispatching extra buses based on forecasts.

---
