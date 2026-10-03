"""Prosperity: a derivative over the overall rating's event log.

Prosperity is a different kind of object from the first two ratings. It measures
a *direction* rather than a level, it is **recomputed for the whole field at
once** rather than updated per match, and its scale is **cohort-relative** -- a
number is only comparable within a single recompute.

What survived measurement: one ranked component (trajectory), one confidence
(coherence), one flag (dispersion). Two candidates were rejected, and the
rejected ones are kept here on purpose:

* ``retention`` -- reliable, but largely a restatement of trajectory;
* ``block_dispersion`` as a *ranked* component -- valid against ground truth but
  far too noisy to order a leaderboard, so it ships as a threshold flag.

Coherence is a **confidence, not a virtue**. It does not make the trajectory
estimate more accurate; it identifies who has a real trend at all. Rewarding it
directly would pay a competitor for declining tidily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .policies import ProsperityPolicy

__all__ = [
    "theil_sen_slope", "block_means", "path_coherence", "block_dispersion",
    "retention", "prepare_window", "measure_components",
    "confidence_from_coherence", "consistency_flag", "build_prosperity",
]


def theil_sen_slope(y, max_pairs: int = 40_000, seed: int = 0) -> float:
    """Median of all pairwise slopes of ``y`` against its own index.

    Robust where OLS is not: one hot streak puts a cluster of outliers in a
    corner of a rating path and levers a least-squares line noticeably.

    The subsample for long careers uses its **own seeded generator**, not a
    module-level or global RNG. With a shared generator the estimate depends on
    how many times it has been called before, which makes the whole rating
    non-reproducible -- it drifted by ~2 rating points before this was fixed.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return float("nan")
    x = np.arange(n, dtype=float)
    i, j = np.triu_indices(n, k=1)
    if len(i) > max_pairs:
        sel = np.random.default_rng(seed).choice(len(i), max_pairs, replace=False)
        i, j = i[sel], j[sel]
    dx = x[j] - x[i]
    ok = dx != 0
    return float(np.median((y[j] - y[i])[ok] / dx[ok]))


def block_means(y, block: int):
    """Average ``y`` within consecutive blocks; ``None`` if too few blocks."""
    y = np.asarray(y, dtype=float)
    nb = len(y) // block
    return y[:nb * block].reshape(nb, block).mean(axis=1) if nb >= 3 else None


def path_coherence(y, block: int = 25) -> float:
    """Net displacement over total distance travelled, on block means.

    A straight line scores 1; pure churn scores near 0. Blocking first is
    essential -- at the single-match level every path is jagged, and the
    statistic would just measure K.

    Note the structural consequence: a genuinely flat competitor scores near 0
    too, because there is no net displacement to divide by. Coherence cannot
    separate "stable" from "no signal". That is acceptable only because both
    should produce the same output -- a neutral rating with a wide band.
    """
    b = block_means(y, block)
    if b is None:
        return float("nan")
    travelled = float(np.abs(np.diff(b)).sum())
    return float(abs(b[-1] - b[0]) / travelled) if travelled > 0 else float("nan")


def block_dispersion(expected, resid, block: int = 10) -> float:
    """Observed variance of block-summed residuals over the variance predicted.

    ~1.0 means results scatter exactly as much as the ratings imply; above 1
    means the competitor runs in streaks the rating did not anticipate.

    Blocks are required. Independent per-match noise is *invisible* -- Elo
    converges to the noisy competitor's average and prices it in. Only
    persistent form leaves a trace, because a hot stretch produces a run of
    correlated surprises, and runs inflate the variance of block sums without
    inflating single-match variance.

    The predicted variance uses the Bernoulli form ``E(1-E)``. Draws make the
    true variance slightly smaller, which biases this low by roughly the same
    amount for everyone -- harmless for a cohort-relative comparison.
    """
    e = np.asarray(expected, dtype=float)
    r = np.asarray(resid, dtype=float)
    nb = len(r) // block
    if nb < 3:
        return float("nan")
    observed = r[:nb * block].reshape(nb, block).sum(axis=1)
    predicted = (e * (1 - e))[:nb * block].reshape(nb, block).sum(axis=1)
    denom = float(np.mean(predicted))
    if denom <= 0:
        return float("nan")
    return float(np.mean(observed ** 2) / denom)


def retention(y, min_gain: float = 25.0) -> float:
    """Of the rating gained above the window start, how much is still held?

    **Rejected as a ranked component** -- reliable, but correlated ~+0.6 with
    trajectory and with no independent ground truth for the remainder. Kept so
    that the next person to think of it finds the measurement instead of
    re-deriving it.
    """
    y = np.asarray(y, dtype=float)
    gain = y.max() - y[0]
    if gain < min_gain:
        return float("nan")       # never gained anything -- nothing to hold
    return float(np.clip((y[-1] - y[0]) / gain, 0.0, 1.0))


def prepare_window(overall_events: pd.DataFrame,
                   policy: ProsperityPolicy | None = None) -> pd.DataFrame:
    """Restrict to the measurement window and add the K-free residual columns.

    The window exists because ratings sprint from 1500 to their true level early
    on, and that motion is **convergence, not development**. Measured from the
    starting point, "improvement" correlates ~+0.9 with how *strong* a competitor
    is and ~0.0 with how much they actually improved.

    ``resid = delta / k_used`` recovers the score residual ``S - E`` exactly, so
    the expected score never needs its own column.
    """
    policy = policy or ProsperityPolicy()
    out = (overall_events[overall_events.career_matches_after
                          > policy.convergence_matches]
           .sort_values(["player_id", "played_at"])
           .copy())
    out["resid"] = out.delta / out.k_used
    out["expected"] = out.result - out.resid
    return out


def measure_components(window: pd.DataFrame,
                       policy: ProsperityPolicy | None = None) -> pd.DataFrame:
    """All four candidate components, per competitor, over the window."""
    policy = policy or ProsperityPolicy()
    rows = {}
    counts = window.player_id.value_counts()
    for player, d in window.groupby("player_id"):
        if counts[player] < policy.min_window_matches:
            continue
        d = d.sort_values("played_at")
        y = d.rating_after.to_numpy()
        rows[player] = {
            "trajectory": theil_sen_slope(y) * 100.0,     # points per 100 matches
            "coherence": path_coherence(y, policy.coherence_block),
            "dispersion": block_dispersion(d.expected.to_numpy(), d.resid.to_numpy(),
                                           policy.dispersion_block),
            "retention": retention(y),
            "window_events": int(len(d)),
        }
    return pd.DataFrame(rows).T


def confidence_from_coherence(coherence: float,
                              policy: ProsperityPolicy | None = None) -> float:
    """Linear ramp from coherence to a shrinkage weight, floored."""
    policy = policy or ProsperityPolicy()
    if coherence != coherence:
        return policy.min_confidence
    ramp = ((coherence - policy.coherence_floor)
            / (policy.coherence_ref - policy.coherence_floor))
    return float(np.clip(ramp, policy.min_confidence, 1.0))


def consistency_flag(dispersion: float,
                     policy: ProsperityPolicy | None = None) -> str:
    """Threshold flag on dispersion. A flag, never a weight."""
    policy = policy or ProsperityPolicy()
    if dispersion != dispersion:
        return "unknown"
    if dispersion >= policy.dispersion_erratic:
        return "erratic"
    if dispersion <= policy.dispersion_steady:
        return "steady"
    return "typical"


def build_prosperity(components: pd.DataFrame,
                     policy: ProsperityPolicy | None = None) -> pd.DataFrame:
    """Compose the prosperity rating: a trajectory, shrunk by its confidence.

    ``prosperity = base + scale * z(trajectory) * confidence``

    A competitor whose rating churned is not *punished* -- they are moved toward
    the middle, which is the honest reading of their data: there is little
    evidence they are going anywhere, so an ordering of noise should not be
    presented as a ranking.

    The band is the honest part of the output. A wide band does **not** mean
    "average", it means "not established", and a client that renders the point
    estimate alone presents "we cannot tell" as "mid-table".
    """
    policy = policy or ProsperityPolicy()
    out = components.copy()
    if "retention" in out.columns:
        out = out.drop(columns=["retention"])

    mu = out.trajectory.mean()
    sd = out.trajectory.std(ddof=0)
    out["z_trajectory"] = (np.clip((out.trajectory - mu) / sd,
                                   -policy.z_clip, policy.z_clip)
                           if sd and sd > 0 else 0.0)
    out["confidence"] = [confidence_from_coherence(c, policy) for c in out.coherence]
    out["prosperity_rating"] = (policy.base
                                + policy.scale * out.z_trajectory * out.confidence)

    half = policy.scale * ((1.0 - out.confidence) + policy.min_band_frac)
    out["band_lo"] = out.prosperity_rating - half
    out["band_hi"] = out.prosperity_rating + half
    out["consistency_flag"] = [consistency_flag(d, policy) for d in out.dispersion]
    return out.sort_values("prosperity_rating", ascending=False)
