import numpy as np
import pytest

from src.overall import (build_overall_board, career_k, career_stats,
                         rating_deviation, run_overall)
from src.policies import OverallPolicy


def test_career_k_floor_is_stiffer_than_seasonal():
    from src.ranking import k_factor
    assert career_k(10_000) == 10.0
    assert career_k(10_000) < k_factor(10_000)


def test_career_k_steps_down_monotonically():
    ks = [career_k(n) for n in range(0, 500, 5)]
    assert ks == sorted(ks, reverse=True)


def test_run_overall_is_zero_sum(matches):
    _, _, events = run_overall(matches)
    assert events.delta.sum() == pytest.approx(0.0, abs=1e-9)


def test_run_overall_has_no_boundary_discontinuity(matches):
    """One continuous run: rating_before always equals the previous rating_after."""
    _, _, events = run_overall(matches)
    for _, d in events.groupby("player_id"):
        d = d.sort_values("played_at")
        assert np.allclose(d.rating_before.to_numpy()[1:],
                           d.rating_after.to_numpy()[:-1])


def test_career_matches_counter_is_dense(matches):
    _, counts, events = run_overall(matches)
    for player, d in events.groupby("player_id"):
        seq = sorted(d.career_matches_after)
        assert seq == list(range(1, counts[player] + 1))


def test_overall_recovers_true_skill_order(matches, overall_events):
    """The point of the rating: strong competitors end up ranked above weak ones."""
    from tests.conftest import TRUE_SKILL
    from src.reliability import spearman
    final = overall_events.sort_values("played_at").groupby("player_id").rating_after.last()
    truth = {p: TRUE_SKILL[p] for p in final.index}
    assert spearman(final.values, [truth[p] for p in final.index]) > 0.8


def test_career_stats_bounds(overall_events):
    stats = career_stats(overall_events)
    assert (stats.trough <= stats.current).all()
    assert (stats.current <= stats.peak).all()
    assert (stats.prime <= stats.peak).all()
    assert (stats.wins + stats.draws + stats.losses == stats.career_matches).all()


def test_prime_is_a_rolling_mean_not_the_peak(overall_events):
    """A single peak is a max over a noisy series, so it is biased upward."""
    stats = career_stats(overall_events, OverallPolicy(prime_window=50))
    assert (stats.prime < stats.peak).all()


def test_rating_deviation_shrinks_with_evidence():
    assert rating_deviation(5, 0) > rating_deviation(50, 0) > rating_deviation(500, 0)


def test_rating_deviation_grows_while_idle():
    assert rating_deviation(100, 365) > rating_deviation(100, 0)


def test_rating_deviation_respects_floor_and_ceiling():
    policy = OverallPolicy()
    assert rating_deviation(10 ** 9, 0) == pytest.approx(policy.rd_floor)
    assert rating_deviation(1, 10 ** 9) == pytest.approx(policy.rd_ceiling)


def test_board_marks_provisional_and_ranks_only_eligible(overall_events):
    board = build_overall_board(overall_events,
                               policy=OverallPolicy(min_career_matches=50))
    assert board[board.is_provisional]["rank"].isna().all()
    ranked = board[~board.is_provisional]
    if len(ranked):
        assert sorted(ranked["rank"]) == list(range(1, len(ranked) + 1))


def test_board_confidence_band_brackets_the_rating(overall_events):
    board = build_overall_board(overall_events)
    assert (board.lo < board.current).all()
    assert (board.current < board.hi).all()
