"""Core Elo primitives shared by every rating in the system.

Two APIs live here, deliberately:

* ``expected_score`` / ``update_ratings`` -- the plain, symmetric Elo used by the
  seasonal, overall, prosperity and thinkers pipelines. These are the functions the
  four ``*_rating_explained`` notebooks are built on.
* ``Team`` / ``EloCalculator`` -- the original head-to-head calculator with home
  advantage and margin-of-victory weighting, kept for the workflow documented in
  the README and in ``notebooks/advanced_elo_ranking.ipynb``.

They are not interchangeable. ``EloCalculator`` adjusts the home side's rating
before computing an expectation and scales the update by margin of victory, so it
is **not** zero-sum; the plain functions are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

INITIAL_RATING = 1500.0
RATING_MEAN = 1500.0
ELO_DIVISOR = 400.0

__all__ = [
    "INITIAL_RATING", "RATING_MEAN", "ELO_DIVISOR",
    "expected_score", "update_ratings", "draw_probability", "sample_result",
    "Team", "EloCalculator", "regress_to_mean",
]


# --------------------------------------------------------------------------- #
# Plain Elo -- the basis of all four ratings
# --------------------------------------------------------------------------- #

def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that A scores against B, given current ratings.

    >>> expected_score(1500.0, 1500.0)
    0.5
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / ELO_DIVISOR))


def update_ratings(rating_a: float, rating_b: float, score_a: float,
                   k: float) -> tuple[float, float]:
    """Return ``(new_a, new_b)`` after one match. Strictly zero-sum.

    ``score_a`` is 1.0 / 0.5 / 0.0 for a win / draw / loss by A.
    """
    adjustment = k * (score_a - expected_score(rating_a, rating_b))
    return rating_a + adjustment, rating_b - adjustment


def draw_probability(p: float, draw_base: float) -> float:
    """Draw probability that peaks for even matchups and vanishes in blowouts.

    ``draw_base`` is the draw rate between perfectly matched competitors. Pairing
    this with :func:`sample_result` leaves ``E[S] == p`` *exactly*, for any
    ``draw_base`` and any ``p``.

    This matters more than it looks. A flat draw rate -- the same probability
    regardless of matchup -- forces draws into lopsided pairings where the
    favourite should win ~90% of the time, which drags strong ratings down and
    weak ones up for hundreds of matches. A trajectory estimator reads that as a
    genuine decline for every strong competitor. See
    ``notebooks/prosperity_rating_explained.ipynb`` section 3.
    """
    return min(draw_base * 4.0 * p * (1.0 - p), 2.0 * min(p, 1.0 - p))


def sample_result(p: float, draw_base: float, uniform: float) -> float:
    """Draw a 1.0 / 0.5 / 0.0 result with ``E[S] == p``.

    ``uniform`` is a sample from U(0, 1), passed in so the caller owns the RNG.
    """
    p_draw = draw_probability(p, draw_base)
    p_win = p - p_draw / 2.0
    if uniform < p_win:
        return 1.0
    return 0.5 if uniform < p_win + p_draw else 0.0


# --------------------------------------------------------------------------- #
# Head-to-head calculator with home advantage and margin of victory
# --------------------------------------------------------------------------- #

@dataclass
class Team:
    name: str
    elo_rating: float = INITIAL_RATING


class EloCalculator:
    """Elo with optional home advantage and margin-of-victory weighting."""

    def __init__(self, home_advantage: float = 50.0, k: float = 40.0):
        self.home_advantage = home_advantage
        self.k = k

    def expected_home_score(self, home_rating: float, away_rating: float) -> float:
        home_strength = 10 ** ((home_rating + self.home_advantage) / 400)
        away_strength = 10 ** (away_rating / 400)
        return home_strength / (home_strength + away_strength)

    def margin_coefficient(
        self,
        home_points: float,
        away_points: float,
        winner_rating: float,
        loser_rating: float,
    ) -> float:
        rating_diff = winner_rating - loser_rating
        return (
            math.log(abs(home_points - away_points) + 1) * 2.2
        ) / (rating_diff * 0.001 + 2.2)

    def update(
        self,
        home: Team,
        away: Team,
        home_score: float,
        home_points: float | None = None,
        away_points: float | None = None,
    ):
        expected_home = self.expected_home_score(home.elo_rating, away.elo_rating)
        expected_away = 1 - expected_home
        away_score = 1 - home_score

        coefficient = 1.0
        if home_points is not None and away_points is not None and home_score != 0.5:
            winner = home if home_score == 1 else away
            loser = away if home_score == 1 else home
            coefficient = self.margin_coefficient(
                home_points, away_points,
                winner.elo_rating, loser.elo_rating
            )

        home.elo_rating += self.k * coefficient * (home_score - expected_home)
        away.elo_rating += self.k * coefficient * (away_score - expected_away)

        return {
            "home_expected": expected_home,
            "away_expected": expected_away,
            "margin_coefficient": coefficient,
            "home_rating": home.elo_rating,
            "away_rating": away.elo_rating,
        }


def regress_to_mean(rating: float, mean: float = RATING_MEAN,
                    fraction: float = 1 / 3) -> float:
    """Move an off-season rating fractionally *toward* the population mean.

    ``fraction`` is how far the rating travels toward ``mean``: 0.0 leaves it
    untouched, 1.0 collapses it onto the mean.

    .. warning::
       This is the **complement** of :func:`src.ranking.carry_over`, which the
       seasonal notebook parameterises by how much of the deviation is *kept*
       (``lam``). The two are related by ``fraction == 1 - lam``, so passing a
       lambda here silently inverts the intended regression. The seasonal
       pipeline uses ``carry_over``; this function is kept for the original
       ``EloCalculator`` workflow and the README example.
    """
    return rating + (mean - rating) * fraction
