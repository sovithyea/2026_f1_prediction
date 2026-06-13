# 2026 F1 Race Predictor

Machine learning predictions for every round of the 2026 Formula 1 World Championship, built on real 2026 FastF1 data.
---

## How it works

Each race file follows a 5-stage live workflow updated across the race weekend:

```
STAGE 1 — After FP1      → enter FP1 best laps
STAGE 2 — After FP2      → enter FP2 long-run pace  ← most important
STAGE 3 — After FP3      → enter FP3 best laps
STAGE 4 — After qualifying → enter real qualifying times
STAGE 5 — After race     → fill ACTUAL_RESULT for accuracy tracking
```

Run any race file at any stage — the model uses the best available data automatically:

```
qualifying_2026  →  FP3 pace  →  FP2 pace  →  FP1 pace  →  synthetic
```

Missing FP sessions are auto-fetched from FastF1 for completed rounds, or imputed from qualifying when not yet available.

---

## Project structure

```
2026_f1_prediction/
│
├── core/                       shared modules (don't touch once stable)
│   ├── config.py               drivers, calendar, team scores, model params
│   ├── data.py                 FastF1 fetchers (qualifying, FP1/2/3, race positions)
│   ├── features.py             feature engineering + normalization
│   ├── model.py                GBR train / predict
│   ├── viz.py                  charts + console output
│   └── predictor.py            main pipeline — called by every race file
│
├── races/                      one file per GP
│   ├── template.py            copy this for each new race
│   ├── australia.py            R1  ✅  RUS wins
│   ├── china.py                R2  ✅  ANT wins
│   ├── japan.py                R3  ✅  ANT wins
│   ├── miami.py                R4  ✅  ANT wins
│   ├── canada.py               R5  ✅  ANT wins
│   ├── monaco.py               R6  ✅  ANT wins
│   └── spain.py                R7  ⏳  race weekend
│
├── results/
│   ├── actual.csv              ground truth — update after each race
│   ├── tracker.py              season accuracy summary
│   └── *_prediction.csv        auto-saved after each run
│
├── charts/                     prediction PNGs (auto-saved)
└── fastf1_cache/               FastF1 API cache (auto-created)
```

---

## Setup

```bash
pip install fastf1 scikit-learn pandas numpy matplotlib
```

Python 3.11+ recommended.

---

## Running a prediction

```bash
# From project root
python races/spain.py

# Or from inside races/
cd races
python spain.py
```

Each run:
1. Auto-fetches 2026 training data from FastF1 for all completed rounds
2. Fetches FP1/FP2/FP3 for the target race if not manually entered
3. Trains a Gradient Boosting Regressor on completed race positions
4. Predicts finishing order for the target race
5. Saves chart to `charts/` and CSV to `results/`

---

## Season accuracy tracker

After running a few race files:

```bash
python results/tracker.py
```

Shows a full season table — P1 accuracy, top-5 hits, average position error per race.

Update `results/actual.csv` with each race result to keep it current.

---

## Adding a new race

1. Copy `races/_template.py` → `races/country.py`
2. Fill in the round number, circuit, base lap time and weather
3. Add the round numbers of all completed races to `training_rounds`
4. Run it — before qualifying you'll get a team score + FP pace prediction

```python
RACE_CONFIG = {
    "round_2026":       8,
    "training_rounds":  [1, 2, 3, 4, 5, 6, 7],   # all completed rounds
    "race_name":        "Austrian GP",
    "base_lap_time":    67.0,                       # Red Bull Ring ~1:07
    "RainProbability":  0.30,
    "Temperature":      24.0,
    ...
}
```

---

## The model

**Algorithm:** Gradient Boosting Regressor (scikit-learn)

**Training data:** All completed 2026 rounds pulled from FastF1 — qualifying times, FP1/FP2/FP3 best laps, official race finishing positions.

**Features:**

| Feature | Description |
|---|---|
| `QualifyingTime` | Gap from pole (seconds) — normalized per session |
| `FP1Time` | FP1 best lap gap — early baseline |
| `FP2Time` | FP2 best lap / long-run gap — race pace signal |
| `FP3Time` | FP3 best lap gap — setup confirmation |
| `TeamPerformanceScore` | Constructor baseline (0–10) + driver modifier |
| `RainProbability` | Race-day rain forecast (0–1) |
| `Temperature` | Air temperature (°C) |
| `AverageSeasonPerformance` | Rolling season form (updates mid-season) |

**Target:** Race finishing position (1–22)

All time-based features are **normalized to gap from fastest** within each session, making 2026 data comparable across circuits regardless of regulation-era lap time differences.

**Fallback (no training data):** Ranking by qualifying gap weighted against team performance score.

---

## Team performance scores (2026 baseline)

| Team | Score | Notes |
|---|---|---|
| McLaren | 9.5 | Reigning constructors' champion |
| Ferrari | 9.0 | |
| Red Bull | 8.8 | |
| Mercedes | 8.7 | |
| Aston Martin | 7.8 | Adrian Newey era |
| Williams | 7.5 | |
| Racing Bulls | 7.2 | |
| Haas | 6.8 | |
| Alpine | 6.5 | |
| Audi | 6.0 | New manufacturer, regulation reset |
| Cadillac | 5.5 | New 11th constructor |

Scores update automatically as `AverageSeasonPerformance` accumulates real race data.

---

## 2026 grid

| Team | Driver 1 | Driver 2 |
|---|---|---|
| McLaren | Lando Norris | Oscar Piastri |
| Ferrari | Lewis Hamilton | Charles Leclerc |
| Red Bull | Max Verstappen | Isack Hadjar |
| Mercedes | George Russell | Kimi Antonelli |
| Williams | Alex Albon | Carlos Sainz |
| Audi | Nico Hulkenberg | Gabriel Bortoleto |
| Aston Martin | Fernando Alonso | Lance Stroll |
| Alpine | Pierre Gasly | Franco Colapinto |
| Haas | Esteban Ocon | Oliver Bearman |
| Racing Bulls | Liam Lawson | Arvid Lindblad |
| Cadillac | Sergio Perez | Valtteri Bottas |

---

## Notes

- **Bahrain and Saudi Arabia were cancelled** (2026 Iran war) — the season runs 22 rounds instead of 24, with round numbers shifted from R4 onward
- **FP1 is often unreliable** — regulations require each team to run a rookie driver twice per season, meaning key drivers frequently sit out
- **Sprint weekends** (China, Miami, Canada, Great Britain, Netherlands, Qatar, São Paulo) have no FP3 — FP2 is the primary race pace session
- FastF1 cache is stored in `fastf1_cache/` — first run per session downloads data, subsequent runs use cache (much faster)

---

*Unofficial. Not associated with Formula 1 or the FIA.*