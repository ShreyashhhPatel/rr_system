import math
from dataclasses import dataclass

INITIAL_RATING = 1500.0


@dataclass
class Team:
    name: str
    elo_rating: float = INITIAL_RATING


class EloCalculator:
    """
    Elo rating calculator with optional home advantage and
    margin-of-victory weighting.
    """

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


def regress_to_mean(rating: float, mean: float = INITIAL_RATING, fraction: float = 1/3) -> float:
    """Move an off-season rating fractionally toward the population mean."""
    return rating + (mean - rating) * fraction
