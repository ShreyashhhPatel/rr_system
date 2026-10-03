"""Overall rating: one continuous career run, no boundary regression.

Not "the seasonal rating with the reset removed". Three things differ: there is
no boundary regression, K is driven by *career* matches rather than this
season's, and the K floor is stiffer (10, not 20).

The overall rating is a **second, independent materialisation** over the same
bronze match facts. It cannot be derived from the seasonal ``rating_events``,
because those numbers already have the boundary regressions baked in. Two silver
tables, one bronze source.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from .elo import INITIAL_RATING, update_ratings
from .policies import OverallPolicy, k_from_tiers

__all__ = ["career_k", "match_k", "run_overall", "career_stats",
           "rating_deviation", "build_overall_board"]


def career_k(career_matches: int, policy: OverallPolicy | None = None) -> float:
    """Step-down K based on career experience. Never steps back up."""
    policy = policy or OverallPolicy()
    return k_from_tiers(career_matches, policy.k_tiers)


def match_k(count_a: int, count_b: int,
            policy: OverallPolicy | None = None) -> float:
    """Lower of the two, keeping the system zero-sum."""
    return min(career_k(count_a, policy), career_k(count_b, policy))


def run_overall(matches, policy: OverallPolicy | None = None):
    """Continuous career Elo. No season boundaries, ever.

    Returns ``(ratings, counts, overall_rating_events)``.

    Note what is *not* here: explicit time decay. Elo is already a forgetting
    algorithm -- a fixed K makes the rating an exponentially-weighted moving
    average with a measurable memory length -- so the real trade-off is not
    remembering versus forgetting but *stability versus tracking speed*. That is
    what the K floor controls. See ``notebooks/overall_rating_explained.ipynb``
    sections 5-6, where decay is measured and found unnecessary.
    """
    policy = policy or OverallPolicy()
    ratings: dict = {}
    counts: dict = defaultdict(int)
    events = []

    for m in sorted(matches, key=lambda x: x.played_at):
        a, b, s_a = m.player_a, m.player_b, m.result
        ra = ratings.get(a, INITIAL_RATING)
        rb = ratings.get(b, INITIAL_RATING)

        k = match_k(counts[a], counts[b], policy)
        new_a, new_b = update_ratings(ra, rb, s_a, k=k)
        ratings[a], ratings[b] = new_a, new_b
        counts[a] += 1
        counts[b] += 1

        for player, before, after in ((a, ra, new_a), (b, rb, new_b)):
            events.append({
                "match_id": m.match_id,
                "season_id": m.season_id,
                "played_at": m.played_at,
                "player_id": player,
                "opponent_id": b if player == a else a,
                "result": s_a if player == a else 1.0 - s_a,
                "rating_before": before,
                "rating_after": after,
                "delta": after - before,
                "k_used": k,
                "career_matches_after": counts[player],
            })

    return ratings, dict(counts), pd.DataFrame(events)


def career_stats(events: pd.DataFrame,
                 policy: OverallPolicy | None = None) -> pd.DataFrame:
    """Current, peak, prime and volatility per competitor, from the log alone.

    ``prime`` is the best rolling mean over ``prime_window`` matches -- a more
    robust "how good were they at their best" than a single peak, which is a
    maximum over a noisy series and therefore biased upward.
    """
    policy = policy or OverallPolicy()
    window = policy.prime_window

    rows = []
    for player, d in events.groupby("player_id"):
        d = d.sort_values("played_at")
        series = d.rating_after

        prime = (series.rolling(window).mean().max() if len(series) >= window
                 else series.mean())

        rows.append({
            "player_id": player,
            "current": series.iloc[-1],
            "peak": series.max(),
            "prime": prime,
            "trough": series.min(),
            "career_matches": len(d),
            "wins": int((d.result == 1.0).sum()),
            "draws": int((d.result == 0.5).sum()),
            "losses": int((d.result == 0.0).sum()),
            "first_played": d.played_at.min(),
            "last_played": d.played_at.max(),
            "volatility": d.delta.std(),
        })
    return pd.DataFrame(rows).set_index("player_id")


def rating_deviation(n_matches: int, days_idle: float,
                     policy: OverallPolicy | None = None) -> float:
    """How much uncertainty is left in a rating. Heuristic, not derived.

    Shrinks with evidence and grows back while idle.
    """
    policy = policy or OverallPolicy()
    n = max(int(n_matches), 1)
    rd = (policy.rd_base / math.sqrt(n)
          + policy.rd_idle_beta * math.sqrt(max(days_idle, 0)))
    return float(np.clip(rd, policy.rd_floor, policy.rd_ceiling))


def build_overall_board(events: pd.DataFrame, as_of=None,
                        policy: OverallPolicy | None = None) -> pd.DataFrame:
    """The overall leaderboard, with confidence bands and eligibility."""
    policy = policy or OverallPolicy()
    df = events if as_of is None else events[events.played_at <= as_of]
    if df.empty:
        return pd.DataFrame()
    as_of = as_of or df.played_at.max()

    board = career_stats(df, policy)
    board["days_idle"] = (as_of - board.last_played).dt.days
    board["rd"] = [rating_deviation(n, d, policy)
                   for n, d in zip(board.career_matches, board.days_idle)]
    board["lo"] = board.current - policy.confidence_z * board.rd
    board["hi"] = board.current + policy.confidence_z * board.rd
    board["is_provisional"] = board.career_matches < policy.min_career_matches

    board = board.sort_values(["is_provisional", "current"], ascending=[True, False])
    eligible = ~board.is_provisional
    board["rank"] = np.nan
    board.loc[eligible, "rank"] = board.loc[eligible, "current"].rank(
        ascending=False, method="min")
    return board
