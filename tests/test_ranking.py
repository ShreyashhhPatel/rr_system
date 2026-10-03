import numpy as np
import pytest

from src.policies import SeasonalPolicy
from src.ranking import (apply_boundary, build_standings, carry_over,
                         flag_inactive, k_factor, match_k, run_season)


def test_k_factor_steps_down_and_never_up():
    ks = [k_factor(n) for n in range(0, 60)]
    assert ks == sorted(ks, reverse=True)
    assert k_factor(0) == 48.0 and k_factor(9) == 48.0
    assert k_factor(10) == 32.0 and k_factor(29) == 32.0
    assert k_factor(30) == 20.0


def test_match_k_takes_the_lower_of_the_two():
    """Required for the update to stay zero-sum."""
    assert match_k(0, 100) == k_factor(100)
    assert match_k(100, 0) == k_factor(100)


def test_carry_over_endpoints():
    assert carry_over(1800.0, lam=1.0) == pytest.approx(1800.0)   # full carry
    assert carry_over(1800.0, lam=0.0) == pytest.approx(1500.0)   # hard reset


def test_carry_over_preserves_order_for_any_lambda():
    """Regression changes the spread, never the ranking."""
    ratings = [1900.0, 1700.0, 1550.0, 1500.0, 1430.0, 1200.0]
    for lam in (0.0, 0.25, 0.5, 0.7, 1.0):
        moved = [carry_over(r, lam) for r in ratings]
        assert moved == sorted(moved, reverse=True)


def test_carry_over_shrinks_spread():
    spread = lambda xs: max(xs) - min(xs)
    ratings = [1900.0, 1500.0, 1200.0]
    assert spread([carry_over(r, 0.7) for r in ratings]) < spread(ratings)


def test_apply_boundary_applies_to_whole_field():
    out = apply_boundary({"a": 1800.0, "b": 1200.0}, lam=0.5)
    assert out == {"a": pytest.approx(1650.0), "b": pytest.approx(1350.0)}


def test_run_season_is_zero_sum(matches):
    _, _, events = run_season(matches)
    assert events.delta.sum() == pytest.approx(0.0, abs=1e-9)


def test_run_season_event_log_has_two_rows_per_match(matches):
    _, _, events = run_season(matches)
    assert len(events) == 2 * len(matches)
    assert (events.groupby("match_id").size() == 2).all()


def test_run_season_residual_identity(matches):
    """E is recoverable from the log, so it needs no stored column."""
    _, _, events = run_season(matches)
    recovered = events.result - events.delta / events.k_used
    assert recovered.between(-1e-9, 1 + 1e-9).all()


def test_standings_are_derivable_from_the_log(matches):
    _, meta, events = run_season(matches)
    board = build_standings(events)
    for player, count in meta["counts"].items():
        assert board.loc[player, "matches_played"] == count
    assert (board.wins + board.draws + board.losses == board.matches_played).all()


def test_standings_rank_only_eligible_and_stay_contiguous(matches):
    _, _, events = run_season(matches)
    board = build_standings(events, policy=SeasonalPolicy(min_matches_for_ranking=10))
    ranked = board[~board.is_provisional]
    assert board[board.is_provisional]["rank"].isna().all()
    assert sorted(ranked["rank"].tolist()) == list(range(1, len(ranked) + 1))


def test_standings_replay_as_of_is_a_prefix(matches):
    """The as_of replay is the whole GET /standings?as_of= endpoint."""
    _, _, events = run_season(matches)
    midpoint = events.played_at.quantile(0.5)
    early = build_standings(events, as_of=midpoint)
    full = build_standings(events)
    assert early.matches_played.sum() < full.matches_played.sum()
    assert (early.matches_played <= full.matches_played.reindex(early.index)).all()


def test_standings_empty_log_returns_empty():
    import pandas as pd
    assert build_standings(pd.DataFrame(columns=["played_at"])).empty


def test_second_season_starts_from_regressed_ratings(matches):
    half = len(matches) // 2
    end_ratings, _, _ = run_season(matches[:half], season_id="S1")
    carried = apply_boundary(end_ratings, lam=0.7)
    _, _, events2 = run_season(matches[half:], starting_ratings=carried,
                               season_id="S2")
    assert (events2.season_id == "S2").all()
    best = max(end_ratings, key=end_ratings.get)
    assert abs(carried[best] - 1500.0) < abs(end_ratings[best] - 1500.0)


def test_flag_inactive_leaves_rating_untouched(matches):
    _, _, events = run_season(matches)
    board = build_standings(events)
    as_of = events.played_at.max()
    flagged = flag_inactive(board, as_of)
    assert (flagged.rating == board.rating).all()
    assert flagged.is_inactive.dtype == bool
