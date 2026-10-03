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
│   ├── elo.py           # expected_score, update_ratings, EloCalculator, draw model
│   ├── matches.py       # the Match fact + a loader that validates before processing
│   ├── policies.py      # every tunable, as data -- loadable from rating_policies
│   ├── ranking.py       # 1. Seasonal   — K schedule, boundary rule, standings
│   ├── overall.py       # 2. Overall    — career K, career stats, confidence bands
│   ├── prosperity.py    # 3. Prosperity — trajectory, coherence, dispersion
│   ├── thinkers.py      # 4. Thinkers   — discipline, adaptation value, diagnostic
│   └── reliability.py   # split-half, Spearman–Brown, power curves, the rankable gate
├── tests/
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Package layout

Each rating module corresponds to the notebook that derives it, and the notebook is
the place to look for *why* it is shaped that way.

```python
from src.matches import load_matches
from src.overall import run_overall, build_overall_board
from src.prosperity import prepare_window, measure_components, build_prosperity

matches = load_matches("data/match_events.csv")       # validates, then sorts
_, _, overall_events = run_overall(matches)           # append-only event log
board = build_overall_board(overall_events)           # materialised view

window = prepare_window(overall_events)
prosperity = build_prosperity(measure_components(window))
```

Three conventions are worth knowing before using the package:

- **Parameters are data.** Every module takes an optional policy object from
  `src.policies`; nothing reads a module-level constant. `save_policies` /
  `load_policies` round-trip them through a `rating_policies` table, so a parameter
  change is a data change and historical recomputes stay reproducible.
- **Event logs are append-only, and boards are views over them.** `run_season` and
  `run_overall` return a rating-event log; `build_standings(events, as_of=...)` and
  `build_overall_board` materialise a leaderboard from it. That `as_of` replay is
  what makes a season auditable and rebuildable at any date.
- **`is_rankable` is computed, never configured.** `src.reliability` measures a
  component's split-half reliability and gates ranking on it. This is why
  `build_thinkers_diagnostic` returns `status = "provisional"` rather than a rating.

Two name pairs are easy to confuse and are kept distinct on purpose:

| | Means | Note |
|---|---|---|
| `elo.regress_to_mean(r, mean, fraction)` | move `fraction` of the way *toward* the mean | used by the `EloCalculator` workflow |
| `ranking.carry_over(r, lam, mean)` | *keep* `lam` of the deviation | used by the seasonal boundary; `fraction == 1 - lam` |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite encodes the invariants the notebooks established rather than just
exercising the code — the Elo update is zero-sum, `delta / k_used` recovers the
expected score, the draw model is unbiased, `theil_sen_slope` is reproducible
across calls, and a placebo world where strategy does nothing still fools the
naive before/after estimator while the matched-control estimator survives it.

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
