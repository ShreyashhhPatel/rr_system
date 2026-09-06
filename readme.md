# RRsystem

## A Python Framework for Dynamic Ranking and Rating Using the Elo Algorithm

RRsystem is a Python-based ranking and rating framework that implements an extended Elo rating model for evaluating teams, players, or other competitive entities from sequential match results.

The project is designed as both an analytical implementation and a foundation for a larger ranking and prediction system. It combines traditional Elo rating principles with practical extensions for home-field advantage, margin of victory, seasonal rating regression, and parameter evaluation.

---

## Abstract

Ranking systems are commonly based on aggregate statistics such as total wins, points, or winning percentage. These approaches do not necessarily account for the relative strength of opponents.

The Elo rating system addresses this limitation by estimating the expected outcome of a contest from the current ratings of the competitors. A victory against a highly rated opponent therefore has a different effect from a victory against a lower-rated opponent.

RRsystem extends the standard Elo framework by incorporating:

- Configurable initial ratings
- Expected outcome probabilities
- Configurable K-factors
- Home advantage adjustments
- Margin-of-victory weighting
- Draw handling
- Sequential processing of historical results
- Seasonal regression toward the population mean
- Parameter evaluation using root mean squared error
- A Jupyter notebook for experimentation and analysis

The resulting framework can be adapted to sports analytics, esports, competitive games, ranking experiments, and other domains involving repeated pairwise comparisons.

---

## Methodology

### 1. Initial Rating

Each competitor begins with a baseline Elo rating:

```text
Initial Rating = 1500
```

New competitors can therefore enter the system without requiring historical performance data.

---

### 2. Expected Outcome

The probability of a home competitor achieving a positive result is calculated from the relative strengths of the two competitors.

The home rating may include an additional home-advantage parameter:

```text
Home Strength = 10 ^ ((Home Rating + Home Advantage) / 400)

Away Strength = 10 ^ (Away Rating / 400)
```

The expected score for the home competitor is:

```text
Expected Home Score =
Home Strength / (Home Strength + Away Strength)
```

The expected score for the away competitor is:

```text
Expected Away Score = 1 - Expected Home Score
```

This allows the model to estimate an expected probability before the match result is observed.

---

### 3. Elo Rating Update

After each contest, ratings are updated according to the difference between the actual result and the predicted result:

```text
New Rating =
Old Rating + K × Margin Coefficient × (Actual Score - Expected Score)
```

Where:

- `K` controls the sensitivity of rating changes
- `Actual Score` represents the observed result
- `Expected Score` represents the model prediction
- `Margin Coefficient` optionally accounts for the decisiveness of the result

The outcome values are represented as:

| Outcome | Score |
|---|---:|
| Home win | 1.0 |
| Draw | 0.5 |
| Away win | 0.0 |

---

### 4. Home Advantage

In many competitive environments, the location of an event may influence the probability of success.

RRsystem supports a configurable home-advantage value that is added to the home competitor's rating when calculating expected probabilities.

The default configuration used by the implementation is:

```python
home_advantage = 50
```

This value is not assumed to be universally optimal. It can be evaluated against historical data through the parameter-tuning workflow included in the project.

---

### 5. Margin of Victory

The extended model can increase the magnitude of rating changes when a contest is won by a larger margin.

The margin coefficient is calculated from the difference in points and the relative ratings of the winner and loser:

```text
Margin Coefficient =
(log(|Point Difference| + 1) × 2.2)
/
((Winner Rating - Loser Rating) × 0.001 + 2.2)
```

This approach allows decisive victories to have a greater influence while reducing the effect of rating differences on the weighting mechanism.

Draws use a neutral coefficient of `1.0`.

---

### 6. Seasonal Regression

Ratings may become increasingly separated after long periods of competition. To account for changes between seasons, RRsystem supports regression toward the population mean.

The general operation is:

```text
New Rating = Rating + (Mean Rating - Rating) × Regression Fraction
```

The default notebook demonstrates a regression fraction of:

```text
1 / 3
```

This moves each rating one-third of the way toward the baseline rating of 1500.

---

## Parameter Evaluation

The model includes an experimental workflow for evaluating combinations of:

- K-factor
- Home advantage

Each parameter combination is evaluated by comparing predicted probabilities with observed match outcomes.

The notebook uses Root Mean Squared Error (RMSE):

```text
RMSE = √(Σ(Prediction - Actual Outcome)² / N)
```

The parameter combination with the lowest error can be used as a candidate configuration for the dataset being analyzed.

Parameter optimization should be performed on representative historical data and validated on separate data where possible.

---

## Project Structure

```text
RRsystem/
│
├── data/
│   └── Match datasets and future data sources
│
├── notebooks/
│   └── advanced_elo_ranking.ipynb
│
├── src/
│   ├── __init__.py
│   └── elo.py
│
├── requirements.txt
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ShreyashhhPatel/RRsystem.git
cd RRsystem
```

Create and activate a virtual environment.

### macOS and Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

## Basic Usage

The core implementation is located in:

```text
src/elo.py
```

Example:

```python
from src.elo import Team, EloCalculator

home = Team("Team A")
away = Team("Team B")

elo = EloCalculator(
    home_advantage=50,
    k=40
)

result = elo.update(
    home,
    away,
    home_score=1,
    home_points=30,
    away_points=20
)

print(result)
print(home.elo_rating)
print(away.elo_rating)
```

---

## Core Components

### `Team`

Represents a competitive entity and its current Elo rating.

```python
Team("Team A")
```

Each new team begins with the default rating of `1500`.

---

### `EloCalculator`

Responsible for:

- Calculating expected probabilities
- Applying home advantage
- Calculating margin-of-victory weighting
- Updating ratings after results

Example configuration:

```python
elo = EloCalculator(
    home_advantage=50,
    k=40
)
```

---

### `regress_to_mean`

Moves ratings toward the baseline value between seasons.

```python
new_rating = regress_to_mean(
    rating,
    mean=1500,
    fraction=1/3
)
```

---

## Jupyter Notebook

The project includes:

```text
notebooks/advanced_elo_ranking.ipynb
```

The notebook demonstrates the complete analytical workflow:

1. Import required libraries
2. Create or load match data
3. Initialize competitors
4. Configure the Elo model
5. Process matches chronologically
6. Update ratings after each event
7. Generate a ranking table
8. Apply seasonal regression
9. Evaluate K-factor and home-advantage combinations
10. Visualize parameter performance

Run Jupyter with:

```bash
jupyter notebook
```

Then open:

```text
notebooks/advanced_elo_ranking.ipynb
```

---

## Match Data Requirements

The current notebook expects data containing information equivalent to:

| Column | Description |
|---|---|
| `date` | Date of the contest |
| `home_team` | Home competitor |
| `away_team` | Away competitor |
| `home_points` | Points scored by the home competitor |
| `away_points` | Points scored by the away competitor |
| `home_win` | Result encoded as 1, 0.5, or 0 |

Matches should be processed chronologically because each rating update affects future predictions.

---

## Example Ranking Workflow

```text
Historical Match Data
        |
        v
Sort Matches Chronologically
        |
        v
Initialize Competitor Ratings
        |
        v
Calculate Expected Outcome
        |
        v
Observe Actual Result
        |
        v
Calculate Margin Coefficient
        |
        v
Update Elo Ratings
        |
        v
Store Rating History
        |
        v
Generate Current Rankings
```

---

## Current Scope

RRsystem currently focuses on the ranking engine and analytical evaluation of Elo parameters.

The implementation is intended to serve as a foundation for future extensions involving persistent storage, APIs, predictive services, and visualization.

---

## Future Development

Planned directions for the project include:

### Data Engineering

- Automated ingestion of historical match data
- CSV and API-based data sources
- Data validation and preprocessing pipelines
- Database persistence for teams, fixtures, and ratings
- Historical rating snapshots

### Analytics

- Rating-history visualization
- Ranking movement over time
- Prediction accuracy analysis
- Cross-validation for parameter selection
- Alternative evaluation metrics

### Application Development

- REST API using FastAPI or Django
- PostgreSQL integration
- Real-time ranking updates
- Prediction endpoints for future fixtures
- Authentication and user management

### Visualization

- Interactive ranking dashboard
- Team and competitor profiles
- Historical rating charts
- Match prediction displays
- Parameter comparison views

---

## Research Considerations

Elo ratings provide a useful framework for modeling relative competitive strength, but their performance depends on the assumptions and configuration used for a particular dataset.

Important areas for future experimentation include:

- Selection of the K-factor
- Home advantage calibration
- Margin-of-victory weighting
- Rating initialization
- Seasonal regression
- Handling of inactive competitors
- Dataset-specific validation

The optimal configuration should therefore be treated as an empirical question rather than a fixed universal setting.

---

## Technologies

- Python
- NumPy
- pandas
- Matplotlib
- Jupyter Notebook

---

## Author

Shreyash Patel

GitHub: https://github.com/ShreyashhhPatel

Repository: https://github.com/ShreyashhhPatel/RRsystem

---

## License

A license has not yet been specified for this repository.
