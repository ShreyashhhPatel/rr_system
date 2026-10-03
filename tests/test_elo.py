import numpy as np
import pytest

from src.elo import (ELO_DIVISOR, INITIAL_RATING, EloCalculator, Team,
                     draw_probability, expected_score, regress_to_mean,
                     sample_result, update_ratings)
from src.ranking import carry_over


def test_expected_score_even_match():
    assert expected_score(1500.0, 1500.0) == pytest.approx(0.5)


def test_expected_score_complementary_and_monotonic():
    a, b = 1700.0, 1480.0
    assert expected_score(a, b) + expected_score(b, a) == pytest.approx(1.0)
    assert expected_score(a, b) > 0.5 > expected_score(b, a)
    # A 400-point gap is the classic 10:1 odds.
    assert expected_score(1900.0, 1500.0) == pytest.approx(10 / 11, abs=1e-9)


def test_update_is_zero_sum():
    for score in (1.0, 0.5, 0.0):
        a, b = update_ratings(1600.0, 1450.0, score, k=24.0)
        assert (a + b) == pytest.approx(1600.0 + 1450.0)


def test_update_draw_moves_favourite_down():
    a, b = update_ratings(1700.0, 1500.0, 0.5, k=32.0)
    assert a < 1700.0 and b > 1500.0


def test_residual_identity_recovers_expected_score():
    """delta / k == S - E, so the expected score needs no column of its own."""
    ra, rb, score, k = 1623.0, 1481.0, 1.0, 16.0
    new_a, _ = update_ratings(ra, rb, score, k=k)
    delta = new_a - ra
    assert score - delta / k == pytest.approx(expected_score(ra, rb))


def test_draw_probability_vanishes_in_blowouts():
    assert draw_probability(0.5, 0.16) == pytest.approx(0.16)
    assert draw_probability(0.99, 0.16) < 0.01
    assert draw_probability(0.5, 0.16) > draw_probability(0.8, 0.16)


def test_draw_probability_never_exceeds_feasible_mass():
    for p in np.linspace(0.001, 0.999, 99):
        d = draw_probability(p, 0.9)
        assert d <= 2 * min(p, 1 - p) + 1e-12
        assert p - d / 2 >= -1e-12          # win probability stays non-negative


@pytest.mark.parametrize("p", [0.1, 0.35, 0.5, 0.72, 0.9])
def test_draw_model_is_unbiased(p):
    """E[S] == p exactly, which is what keeps strong ratings from drifting down."""
    rng = np.random.default_rng(3)
    draws = [sample_result(p, 0.16, u) for u in rng.random(40_000)]
    assert float(np.mean(draws)) == pytest.approx(p, abs=0.01)


def test_regress_to_mean_and_carry_over_are_complements():
    """Same operation, inverted parameter. The two names exist to prevent a mixup."""
    for rating in (1800.0, 1500.0, 1200.0):
        for lam in (0.0, 0.3, 0.7, 1.0):
            assert carry_over(rating, lam) == pytest.approx(
                regress_to_mean(rating, fraction=1.0 - lam))


def test_regress_to_mean_default_signature_unchanged():
    """The README documents this call; it must keep working."""
    assert regress_to_mean(1800.0) == pytest.approx(1800.0 + (1500.0 - 1800.0) / 3)


def test_elo_calculator_matches_documented_usage():
    home, away = Team("Team A"), Team("Team B")
    calc = EloCalculator(home_advantage=50, k=40)
    result = calc.update(home, away, home_score=1, home_points=30, away_points=20)
    assert home.elo_rating > INITIAL_RATING > away.elo_rating
    assert result["home_expected"] > 0.5
    assert result["margin_coefficient"] > 0


def test_elo_calculator_margin_scales_update():
    """A bigger win moves the rating further."""
    def gain(points_for, points_against):
        home, away = Team("H"), Team("A")
        EloCalculator().update(home, away, 1, points_for, points_against)
        return home.elo_rating - INITIAL_RATING

    assert gain(40, 10) > gain(21, 20)
