"""Thinkers: a diagnostic, not a ranked rating.

The fourth dimension, and the only one that does not survive as a rating. Two
problems, one of them fatal:

1. **It cannot be a query.** Nothing in any existing table implies which
   approach a competitor chose. ``match_events.strategy_id`` has to be captured
   at match time and cannot be back-filled -- a strategy label inferred from
   results cannot be used to explain those results.
2. **You cannot grade a decision by what happened after it.** Competitors change
   approach *after bad results*; bad results are partly bad luck; luck reverts.
   So "they switched and then improved" happens reliably even when switching does
   nothing at all.

What is measurable is the **policy, not the outcome**. ``switch_discipline`` --
how often someone changes approach when nothing is wrong -- reaches validity
~+0.77 against known decision quality. ``adaptation_value`` does not work and is
retained deliberately: it is the regression test that stops the next person
re-deriving it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .policies import ThinkersPolicy
from .reliability import is_rankable, spearman, spearman_brown

__all__ = [
    "window_mean", "iter_players", "naive_before_after", "adaptation_value",
    "switch_discipline", "measure_components", "build_thinkers_diagnostic",
]


def window_mean(career_counts, resid, at: int, lo: int, hi: int,
                min_events: int = 5) -> float:
    """Mean residual over ``(at+lo, at+hi]`` of a competitor's match sequence."""
    mask = (career_counts > at + lo) & (career_counts <= at + hi)
    return float(resid[mask].mean()) if mask.sum() >= min_events else float("nan")


def iter_players(events: pd.DataFrame, decisions: pd.DataFrame,
                 policy: ThinkersPolicy | None = None):
    """Yield ``(player, career_counts, resid, decisions)`` past convergence."""
    policy = policy or ThinkersPolicy()
    ev = events[events.career_matches_after > policy.convergence_matches]
    dec = decisions[decisions.career_matches_after > policy.convergence_matches]
    for player, d in ev.groupby("player_id"):
        d = d.sort_values("career_matches_after")
        yield (player,
               d.career_matches_after.to_numpy(),
               d.resid.to_numpy(),
               dec[dec.player_id == player].sort_values("career_matches_after"))


def naive_before_after(events, decisions,
                       policy: ThinkersPolicy | None = None) -> pd.Series:
    """The obvious estimator, kept **only** to demonstrate that it is broken.

    In a placebo world where strategy has exactly zero effect, this reports
    roughly **+0.06 of expected score per match** "from switching", for nearly
    every competitor, on every seed. Do not ship it.

    Two things cause the bias, both working as designed: competitors are not
    randomly assigned to switch (they switch *because* they are slumping, and
    slumps partly revert), and Elo lowers their expected score during the slump,
    so recovery beats a bar the slump itself lowered.
    """
    policy = policy or ThinkersPolicy()
    out = {}
    for player, cm, resid, dec in iter_players(events, decisions, policy):
        sw = dec[dec.switched]
        deltas = [
            window_mean(cm, resid, c, 0, policy.post_window, policy.min_window_events)
            - window_mean(cm, resid, c, -policy.pre_window, 0, policy.min_window_events)
            for c in sw.career_matches_after
        ]
        out[player] = float(np.nanmean(deltas)) if len(deltas) else float("nan")
    return pd.Series(out, name="before_after")


def adaptation_value(events, decisions,
                     policy: ThinkersPolicy | None = None) -> pd.DataFrame:
    """Within-player matched comparison: slumped-and-switched vs slumped-and-stayed.

    This is the *correct* estimator. Restricting to slump episodes means the same
    regression-to-the-mean pull acts on both arms, so it cancels: placebo bias
    drops from +0.060 to within +/-0.02 of zero.

    **And it is still not usable.** Validity against known decision quality
    swings from -0.23 to +0.30 around a mean of ~0, reliability sits near 0.26,
    and eight times the career length does not improve it. Beyond that, the
    quantity is misaligned with judgement even when measured well: the benefit of
    a switch depends on the position switched *from*, and good decision-makers
    switch from good positions, so their upside is compressed by their own
    competence (headroom correlates ~-0.61 with true decision quality).

    Identifiability note: this needs **control episodes** -- slumps the
    competitor did not act on. If everyone always switches when slumping, the
    quantity is not identifiable from observational data at all.
    """
    policy = policy or ThinkersPolicy()
    rows = {}
    for player, cm, resid, dec in iter_players(events, decisions, policy):
        slumps = dec[dec.in_slump]
        treated = [window_mean(cm, resid, c, 0, policy.post_window,
                               policy.min_window_events)
                   for c in slumps[slumps.switched].career_matches_after]
        control = [window_mean(cm, resid, c, 0, policy.post_window,
                               policy.min_window_events)
                   for c in slumps[~slumps.switched].career_matches_after]
        n_t = int(np.sum(~np.isnan(treated)))
        n_c = int(np.sum(~np.isnan(control)))
        enough = n_t >= policy.min_episodes and n_c >= policy.min_episodes
        rows[player] = {
            "adaptation_value": (float(np.nanmean(treated) - np.nanmean(control))
                                 if enough else float("nan")),
            "n_treated": n_t,
            "n_control": n_c,
        }
    return pd.DataFrame(rows).T


def switch_discipline(events, decisions,
                      policy: ThinkersPolicy | None = None) -> pd.DataFrame:
    """Share of calm decision points at which the competitor did *not* switch.

    If recent results are at or above expectation, nothing was wrong, so changing
    approach is churn. This asks nothing about what happened *after* the switch,
    which is exactly what makes it immune to the regression-to-the-mean problem
    that sinks :func:`adaptation_value`.

    Caveat worth carrying: this measures the **policy** facet of decision
    quality. Whether real competitors' judgement shows up as restlessness
    (measurable) or as choice quality (not) is an empirical question about the
    domain, and needs ``strategy_id`` history to answer.
    """
    policy = policy or ThinkersPolicy()
    rows = {}
    for player, cm, resid, dec in iter_players(events, decisions, policy):
        calm = dec[~dec.in_slump]
        rows[player] = {
            "discipline": (1.0 - float(calm.switched.mean()) if len(calm)
                           else float("nan")),
            "n_calm_points": int(len(calm)),
            "n_slump_points": int(dec.in_slump.sum()),
        }
    return pd.DataFrame(rows).T


def measure_components(events, decisions, parity: int | None = None,
                       policy: ThinkersPolicy | None = None) -> pd.DataFrame:
    """Both components per competitor.

    ``parity`` keeps only every other decision point (0 or 1), which is what
    :func:`src.reliability.split_half_reliability` needs. The interleaved split
    leaves both halves spanning the whole career.
    """
    policy = policy or ThinkersPolicy()
    dec = decisions
    if parity is not None:
        dec = decisions.sort_values(["player_id", "career_matches_after"]).copy()
        # cumcount keeps player_id a column; groupby.apply would move it into
        # the index, which silently breaks every lookup downstream.
        dec = dec[dec.groupby("player_id").cumcount() % 2 == parity]
    return (switch_discipline(events, dec, policy)
            .join(adaptation_value(events, dec, policy)))


def build_thinkers_diagnostic(events, decisions,
                              policy: ThinkersPolicy | None = None):
    """A diagnostic that will not promote itself to a rating.

    Reliability is **measured from this very dataset** and ``is_rankable`` is
    computed from it, never configured. At realistic volumes it comes back False,
    and the right response is to keep capturing ``strategy_id`` and re-evaluate
    at roughly 36 recorded switches per competitor -- not to lower the bar.

    Returns ``(diagnostic, measured_reliability)``.
    """
    policy = policy or ThinkersPolicy()

    disc = switch_discipline(events, decisions, policy)
    adapt = adaptation_value(events, decisions, policy)

    h1 = measure_components(events, decisions, parity=0, policy=policy)
    h2 = measure_components(events, decisions, parity=1, policy=policy)
    measured = {
        c: spearman_brown(spearman(h1[c], h2[c]))
        for c in ("discipline", "adaptation_value")
    }

    out = disc.join(adapt)
    pct = out.discipline.rank(pct=True)
    out["discipline_index"] = 1500.0 + 120.0 * ((pct - 0.5) * 2.0)
    out["reliability"] = measured["discipline"]
    rankable = is_rankable(measured["discipline"], policy.reliability_bar)
    out["is_rankable"] = rankable
    out["status"] = "ranked" if rankable else "provisional - diagnostic only"

    half = 120.0 * (1.0 - max(measured["discipline"] if
                              measured["discipline"] == measured["discipline"]
                              else 0.0, 0.0))
    out["band_lo"] = out.discipline_index - half
    out["band_hi"] = out.discipline_index + half
    out["adaptation_value_rankable"] = is_rankable(measured["adaptation_value"],
                                                   policy.reliability_bar)
    return out.sort_values("discipline", ascending=False), measured
