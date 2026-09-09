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
│   └── advanced_elo_ranking.ipynb
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

## Notebook

Open:

```text
notebooks/advanced_elo_ranking.ipynb
```

The notebook demonstrates the complete workflow, including match processing, ranking generation, season regression, and K/home-advantage parameter tuning.

## Reference

The advanced notebook structure was adapted from the uploaded Elo implementation article, which demonstrates team initialization, home advantage, margin-of-victory weighting, seasonal regression, and parameter evaluation. 
