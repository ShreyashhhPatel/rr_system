"""Seasonal rating: ordinary Elo inside a window, plus a boundary rule.

Everything *inside* a season is plain Elo. The only thing that makes it seasonal
is what happens at the boundary between seasons, so that is where the design
work is.

The other argument this module encodes: store an append-only ``rating_events``
log, not final ratings. The leaderboard is a materialised view over that log,
which is what makes a season replayable and auditable as of any date -- see
:func:`build_standings`.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from .elo import INITIAL_RATING, RATING_MEAN, update_ratings
from .policies import SeasonalPolicy, k_from_tiers

__all__ = [
    "k_factor", "match_k", "carry_over", "apply_boundary",
    "run_season", "build_standings", "flag_inactive",
]


def k_factor(matches_played_this_season: int,
             policy: SeasonalPolicy | None = None) -> float:
    """Step-down K based on how much evidence we have this season."""
    policy = policy or SeasonalPolicy()
    return k_from_tiers(matches_played_this_season, policy.k_tiers)


def match_k(count_a: int, count_b: int,
            policy: SeasonalPolicy | None = None) -> float:
    """Symmetric, zero-sum choice: the lower K of the two competitors."""
    return min(k_factor(count_a, policy), k_factor(count_b, policy))


def carry_over(rating: float, lam: float = 0.70,
               mean: float = RATING_MEAN) -> float:
    """Season-boundary carry-over: ``lam`` is the fraction of the deviation kept.

    ``lam=1.0`` carries the rating over untouched; ``lam=0.0`` is a hard reset to
    ``mean``. Ordering is preserved for any ``lam`` in [0, 1], because the map is
    monotonic -- regression changes the spread, never the ranking.

    .. note::
       :func:`src.elo.regress_to_mean` is parameterised the other way round, by
       how far the rating *travels toward* the mean (``fraction == 1 - lam``).
       Two names for two conventions, so neither can be passed the other's
       argument by accident.
    """
    return mean + lam * (rating - mean)


def apply_boundary(end_ratings: dict, lam: float = 0.70,
                   mean: float = RATING_MEAN) -> dict:
    """Carry a whole field's end-of-season ratings into the next season."""
    return {p: carry_over(r, lam, mean) for p, r in end_ratings.items()}


def run_season(matches, starting_ratings: dict | None = None,
               season_id: str = "S1",
               policy: SeasonalPolicy | None = None):
    """Process one season chronologically.

    Returns ``(ratings, meta, rating_events)``. Win/draw/loss counts, peak rating
    and rank are deliberately **not** tracked here -- they are all derivable from
    the event log by :func:`build_standings`. One source of truth.
    """
    policy = policy or SeasonalPolicy()
    ratings = dict(starting_ratings or {})
    counts: dict = defaultdict(int)
    last_played: dict = {}
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
        last_played[a] = last_played[b] = m.played_at

        for player, before, after, n in ((a, ra, new_a, counts[a]),
                                         (b, rb, new_b, counts[b])):
            events.append({
                "season_id": season_id,
                "match_id": m.match_id,
                "played_at": m.played_at,
                "player_id": player,
                "opponent_id": b if player == a else a,
                "result": s_a if player == a else 1.0 - s_a,
                "rating_before": before,
                "rating_after": after,
                "delta": after - before,
                "k_used": k,
                "matches_played_after": n,
            })

    meta = {"counts": dict(counts), "last_played": last_played}
    return ratings, meta, pd.DataFrame(events)


def build_standings(events: pd.DataFrame, as_of: datetime | None = None,
                    policy: SeasonalPolicy | None = None) -> pd.DataFrame:
    """Materialise the seasonal leaderboard from the rating-event log.

    Passing ``as_of`` rebuilds the table as it stood on that date. That replay is
    the whole of a ``GET /standings?season_id=&as_of=`` endpoint, and it only
    works because the log is append-only.
    """
    policy = policy or SeasonalPolicy()
    min_matches = policy.min_matches_for_ranking

    df = events if as_of is None else events[events.played_at <= as_of]
    if df.empty:
        return pd.DataFrame()

    last = (df.sort_values("played_at")
              .groupby("player_id")
              .tail(1)
              .set_index("player_id"))

    agg = df.groupby("player_id").agg(
        matches_played=("match_id", "count"),
        peak_rating=("rating_after", "max"),
        wins=("result", lambda s: (s == 1.0).sum()),
        draws=("result", lambda s: (s == 0.5).sum()),
        losses=("result", lambda s: (s == 0.0).sum()),
        last_played=("played_at", "max"),
    )

    out = agg.join(last[["rating_after"]].rename(columns={"rating_after": "rating"}))
    out["is_provisional"] = out.matches_played < min_matches
    out = out.sort_values(["is_provisional", "rating"], ascending=[True, False])

    # Rank only among eligible competitors, so numbering stays contiguous.
    eligible = ~out.is_provisional
    out["rank"] = np.nan
    out.loc[eligible, "rank"] = out.loc[eligible, "rating"].rank(
        ascending=False, method="min")

    return out[["rank", "rating", "matches_played", "wins", "draws", "losses",
                "peak_rating", "is_provisional", "last_played"]]


def flag_inactive(standings: pd.DataFrame, as_of: datetime,
                  policy: SeasonalPolicy | None = None) -> pd.DataFrame:
    """Mark competitors with no match inside the inactivity window.

    Inactive competitors come off the *active* board; their rating is left
    untouched. Decaying an idle rating is a different policy choice and is not
    the default here -- absence is not evidence of having got worse.
    """
    policy = policy or SeasonalPolicy()
    out = standings.copy()
    out["days_idle"] = (as_of - out.last_played).dt.days
    out["is_inactive"] = out.days_idle > policy.inactivity_days
    return out
