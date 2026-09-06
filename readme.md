# 🏆 RRsystem — Ranking & Rating System

A data-driven **Ranking and Rating System** that uses the **Elo Rating Algorithm** to calculate dynamic rankings based on match outcomes, comparisons, or competitive events.

The project includes a Jupyter Notebook for experimenting with the Elo algorithm, analyzing rating changes, and generating rankings.

---

## 📌 Overview

Traditional ranking systems often rely only on simple metrics such as:

* Total wins
* Total points
* Win percentage

This project uses the **Elo Rating System**, which provides a more intelligent approach to ranking.

Instead of treating every win equally, Elo considers:

* The current rating of each participant
* The expected probability of winning
* Whether the result was an upset
* Historical performance through continuously updated ratings

For example:

> Beating a highly ranked player should increase your rating more than beating a lower-ranked player.

---

# 🧠 Elo Rating Algorithm

The Elo system calculates the expected probability that one participant will defeat another.

## Expected Score

For Player A:

```text
EA = 1 / (1 + 10^((RB - RA) / 400))
```

For Player B:

```text
EB = 1 / (1 + 10^((RA - RB) / 400))
```

Where:

* `RA` = Current rating of Player A
* `RB` = Current rating of Player B
* `EA` = Expected score for Player A
* `EB` = Expected score for Player B

---

## Rating Update

After a match, ratings are updated using:

```text
New Rating = Old Rating + K × (Actual Score - Expected Score)
```

Where:

* `K` = Rating sensitivity factor
* `Actual Score`:

  * Win = `1`
  * Draw = `0.5`
  * Loss = `0`

---

# 📊 Example

Suppose:

```text
Player A Rating: 1600
Player B Rating: 1400
```

Player A is expected to win because they have a higher rating.

However, if Player B wins, Player B receives a larger rating increase because the result was unexpected.

This allows the ranking system to naturally reward:

* Upsets
* Strong performances
* Consistent winning
* Wins against highly ranked opponents

---

# 🏗️ Project Structure

```text
RRsystem/
│
├── notebooks/
│   └── elo_ranking.ipynb
│
├── data/
│   └── matches.csv
│
├── src/
│   ├── elo.py
│   ├── ranking.py
│   └── utils.py
│
├── requirements.txt
│
└── README.md
```

---

# 📓 Notebook

The main notebook:

```text
notebooks/elo_ranking.ipynb
```

contains:

1. Loading match data
2. Initializing participant ratings
3. Calculating expected scores
4. Updating Elo ratings
5. Processing historical matches
6. Generating final rankings
7. Visualizing rating changes

---

# 🚀 Elo Setup

## Initial Rating

Every participant starts with a default rating:

```python
INITIAL_RATING = 1500
```

---

## K-Factor

The `K-factor` controls how quickly ratings change.

```python
K_FACTOR = 32
```

Higher values mean:

* Faster rating changes
* More volatility

Lower values mean:

* More stable rankings
* Slower rating changes

---

# 💻 Elo Implementation

```python
INITIAL_RATING = 1500
K_FACTOR = 32
```

### Expected Score

```python
def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
```

### Update Ratings

```python
def update_ratings(rating_a, rating_b, score_a, k=32):

    expected_a = expected_score(rating_a, rating_b)
    expected_b = expected_score(rating_b, rating_a)

    score_b = 1 - score_a

    new_rating_a = rating_a + k * (score_a - expected_a)
    new_rating_b = rating_b + k * (score_b - expected_b)

    return new_rating_a, new_rating_b
```

---

# 🥊 Example Match

```python
player_a = 1500
player_b = 1500

new_a, new_b = update_ratings(
    player_a,
    player_b,
    score_a=1
)

print(new_a, new_b)
```

Expected result:

```text
Player A → 1516
Player B → 1484
```

Since both players had equal ratings, Player A was expected to have a `50%` probability of winning.

---

# 📈 Processing Multiple Matches

The system can process historical match data.

Example:

```python
matches = [
    ("Alice", "Bob", 1),
    ("Bob", "Charlie", 1),
    ("Alice", "Charlie", 0),
]
```

Where:

```text
1   = Player A wins
0.5 = Draw
0   = Player B wins
```

Ratings are updated sequentially after every match.

---

# 🏅 Generate Rankings

```python
ratings = {
    "Alice": 1620,
    "Bob": 1510,
    "Charlie": 1370
}
```

Rank participants:

```python
ranking = sorted(
    ratings.items(),
    key=lambda x: x[1],
    reverse=True
)

for rank, (player, rating) in enumerate(ranking, start=1):
    print(f"{rank}. {player}: {rating:.2f}")
```

Example output:

```text
1. Alice: 1620.00
2. Bob: 1510.00
3. Charlie: 1370.00
```

---

# 📊 Match Data Format

The system expects match data in a format similar to:

| player_a | player_b | result |
| -------- | -------- | -----: |
| Alice    | Bob      |      1 |
| Bob      | Charlie  |      1 |
| Alice    | Charlie  |      0 |
| Alice    | Bob      |    0.5 |

Where:

```text
1   → Player A wins
0.5 → Draw
0   → Player B wins
```

---

# 🔄 Ranking Workflow

```text
Match Data
     │
     ▼
Initialize Ratings
     │
     ▼
Calculate Expected Score
     │
     ▼
Compare Actual Result
     │
     ▼
Update Elo Ratings
     │
     ▼
Store New Ratings
     │
     ▼
Generate Rankings
```

---

# 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/ShreyashhhPatel/RRsystem.git
cd RRsystem
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

Example `requirements.txt`:

```text
pandas
numpy
matplotlib
jupyter
```

---

# 📈 Future Improvements

* [ ] Dynamic K-factor
* [ ] Player confidence scores
* [ ] Match importance weighting
* [ ] Home/away advantage
* [ ] Time-based rating decay
* [ ] Ranking history visualization
* [ ] REST API using FastAPI or Django
* [ ] Database integration
* [ ] Real-time ranking updates
* [ ] Machine Learning ranking models

---

# 🔮 Potential Use Cases

The ranking system can be adapted for:

* 🏆 Sports players and teams
* 🎮 Competitive gaming
* 🧑‍💻 Coding competitions
* 📚 Recommendation systems
* ⭐ Product ranking
* 🏢 Employee performance ranking
* 🤖 AI model evaluation
* 📊 Algorithmic ranking systems

---

# 🧑‍💻 Author

**Shreyash Patel**

GitHub: [ShreyashhhPatel](https://github.com/ShreyashhhPatel)

---

## ⭐ Future Vision

The goal of **RRsystem** is to evolve from a simple Elo implementation into a flexible and scalable **Ranking & Rating Engine** capable of processing large datasets and generating intelligent, dynamic rankings.

The system will provide a foundation for experimenting with ranking algorithms, statistical models, and machine learning approaches.

⭐ If you find this project useful, consider starring the repository!
