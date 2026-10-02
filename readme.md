# RRsystem — Elo Ranking & Rating Engine

A Python implementation of an advanced **Elo ranking system** for ranking teams, players, or competitors from historical match results.

## Features

- Standard Elo expected-score calculation
- Configurable **K-factor**
- **Home advantage** adjustment
- **Margin-of-victory** weighting
- Draw support
- Sequential historical match processing
- Off-season regression toward the mean
- Parameter tuning using prediction error
- Jupyter notebook for experimentation

## Project Structure

```text
RRsystem/
├── data/
├── notebooks/
│   ├── advanced_elo_ranking.ipynb            # base Elo workflow
│   ├── basic_seasonal_rating (2).ipynb       # core seasonal update logic
│   ├── four_dimensional_elo_ratings.ipynb    # the four-rating framework (prototype)
│   ├── seasonal_rating_explained.ipynb       # 1. Seasonal  — in depth
│   ├── overall_rating_explained.ipynb        # 2. Overall   — in depth
│   ├── prosperity_rating_explained.ipynb     # 3. Prosperity — in depth
│   └── thinkers_rating_explained.ipynb       # 4. Thinkers  — in depth
├── src/
│   ├── __init__.py
│   └── elo.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

```python
from src.elo import Team, EloCalculator

home = Team("Team A")
away = Team("Team B")

elo = EloCalculator(home_advantage=50, k=40)

result = elo.update(
    home,
    away,
    home_score=1,
    home_points=30,
    away_points=20,
)

print(home.elo_rating)
print(away.elo_rating)
```

## Elo Model

The expected probability for the home team is calculated from the relative team strengths:

```text
Expected Score = Home Strength / (Home Strength + Away Strength)
```

The model optionally adds home advantage to the home team's rating before calculating expected probabilities.

Ratings are then updated based on:

```text
New Rating = Old Rating + K × Margin Coefficient × (Actual - Expected)
```

The margin coefficient increases the impact of more decisive victories while accounting for the rating difference between competitors.

## Notebooks

### Getting started

```text
notebooks/advanced_elo_ranking.ipynb
```

Demonstrates the complete base workflow: match processing, ranking generation, season regression,
and K/home-advantage parameter tuning.

### The four-dimensional rating system

`four_dimensional_elo_ratings.ipynb` sketches four complementary ratings. Each then gets a
notebook that builds it properly, validates it against ground truth from a simulator, and states
what it can and cannot support.

| # | Rating | Question it answers | Reads from | Status |
|---|---|---|---|---|
| 1 | **Seasonal** | How are you doing *now*? | `match_events` + `season_id` | ranked |
| 2 | **Overall** | How strong are you? | `match_events` | ranked |
| 3 | **Prosperity** | Which way are you going? | `overall_rating_events` | ranked, with a confidence band |
| 4 | **Thinkers** | Are your decisions good? | `match_events.strategy_id` *(new capture)* | **diagnostic only** |

The framework comes out as **three ratings and one diagnostic**. Thinkers does not clear the
reliability bar the other notebooks set for a ranked component, and
`thinkers_rating_explained.ipynb` shows why rather than reweighting until something emerges.

Each notebook is self-contained and executable, and each clears a trap specific to its rating:

| Rating | The trap | The fix |
|---|---|---|
| Seasonal | the season boundary is the only real design decision | store an append-only event log, not final ratings |
| Overall | "won't results from years ago pollute it forever?" | measure Elo's memory — it is already a forgetting algorithm |
| Prosperity | apparent improvement is mostly convergence from 1500 | a measurement window, plus a field-relative target |
| Thinkers | outcomes after a decision are selected on form | a within-player matched control — which then was not enough |

### Method

The later notebooks share four habits worth knowing about before reading them:

- **Placebo runs** — build the world where the effect is zero and check what the estimator reports.
  This is what invalidated the Thinkers prototype rule.
- **Split-half reliability**, Spearman–Brown corrected — a statistic that cannot agree with itself
  across two halves of the same career cannot support a leaderboard.
- **Multi-seed replication** — a rank correlation over a couple of dozen competitors has error bars
  of roughly ±0.2. An earlier draft of the Prosperity notebook shipped a confident finding that did
  not survive a second seed, and says so.
- **Power curves** — the only way to tell "not enough data yet" from "this will never work".

## Reference

The advanced notebook structure was adapted from the uploaded Elo implementation article, which demonstrates team initialization, home advantage, margin-of-victory weighting, seasonal regression, and parameter evaluation. 
