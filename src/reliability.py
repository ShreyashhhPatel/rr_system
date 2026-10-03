"""Reliability machinery, shared by every rating.

This module exists because two candidate components in the prosperity notebook
and both in the thinkers notebook looked reasonable and were not. None of those
failures is visible by reading a formula -- only by measuring. So the measuring
belongs in the package, runnable in CI over the live cohort, rather than in a
notebook somebody runs once.

The rule the notebooks arrive at: a component's ``is_rankable`` should be
**computed from its measured reliability**, never configured by hand.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "spearman", "spearman_brown", "split_half_reliability",
    "is_rankable", "reliability_report", "power_curve",
    "DEFAULT_RELIABILITY_BAR",
]

DEFAULT_RELIABILITY_BAR = 0.70


def spearman(x, y) -> float:
    """Rank correlation, without pulling in scipy.

    Returns NaN rather than raising when there is nothing to correlate -- fewer
    than three usable pairs, or no variance in either input.
    """
    x = pd.Series(np.asarray(x, dtype=float))
    y = pd.Series(np.asarray(y, dtype=float))
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return float("nan")
    rx, ry = x[ok].rank(), y[ok].rank()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_brown(split_half_r: float) -> float:
    """Correct a split-half correlation up to full-test reliability.

    A split-half correlation measures a *half-length* test and therefore
    understates the whole thing:

    .. math:: r_{full} = \\frac{2 r_{half}}{1 + r_{half}}

    This is not cosmetic at the low end. The thinkers ``discipline`` component
    reads 0.40 uncorrected -- which looks like a dead end -- and ~0.57
    corrected, which is merely early. Skipping the correction retires a usable
    component.
    """
    if split_half_r != split_half_r or split_half_r <= -1.0:
        return float("nan")
    return 2.0 * split_half_r / (1.0 + split_half_r)


def split_half_reliability(per_unit: Callable[[int], pd.DataFrame],
                           components: Sequence[str]) -> pd.DataFrame:
    """Odd/even split-half reliability for each named component.

    ``per_unit(parity)`` must recompute the components using only the events
    whose index within each subject has ``index % 2 == parity``, returning a
    DataFrame indexed by subject with one column per component.

    The *interleaved* split is what makes this valid for a slope or a trend: both
    halves still span the full window, so the estimand is unchanged. A
    first-half/second-half split would instead compare two different windows.
    """
    h1, h2 = per_unit(0), per_unit(1)
    rows = []
    for c in components:
        half = spearman(h1[c], h2[c])
        rows.append({
            "component": c,
            "split_half": half,
            "spearman_brown": spearman_brown(half),
            "n_subjects": int((h1[c].notna() & h2[c].notna()).sum()),
        })
    return pd.DataFrame(rows).set_index("component")


def is_rankable(reliability: float, bar: float = DEFAULT_RELIABILITY_BAR) -> bool:
    """Whether a component's measured reliability supports a ranked ordering.

    NaN is not rankable. An unmeasured component is not a passing one.
    """
    if reliability != reliability:
        return False
    return bool(reliability >= bar)


def reliability_report(per_unit: Callable[[int], pd.DataFrame],
                       components: Sequence[str],
                       bar: float = DEFAULT_RELIABILITY_BAR) -> pd.DataFrame:
    """:func:`split_half_reliability` plus the computed rankable gate."""
    out = split_half_reliability(per_unit, components)
    out["bar"] = bar
    out["is_rankable"] = [is_rankable(r, bar) for r in out.spearman_brown]
    return out


def power_curve(sizes: Iterable[int],
                build: Callable[[int, int], tuple[Callable[[int], pd.DataFrame], float]],
                components: Sequence[str],
                seeds: Sequence[int] = (7, 11),
                bar: float = DEFAULT_RELIABILITY_BAR) -> pd.DataFrame:
    """Reliability as a function of how much data there is.

    ``build(size, seed)`` must return ``(per_unit, exposure)`` where ``per_unit``
    is as in :func:`split_half_reliability` and ``exposure`` is the mean number
    of relevant events per subject at that size (switches per competitor,
    matches per competitor, whatever the component consumes).

    This is the plot to reach for whenever someone asks to ship a metric that is
    "nearly there". It separates two answers that lead to opposite decisions:

    * **rising** -- the component is data-limited, so wait and keep capturing;
    * **flat** -- no amount of patience helps, so stop building it.
    """
    rows = []
    for size in sizes:
        acc = []
        for seed in seeds:
            per_unit, exposure = build(size, seed)
            rel = split_half_reliability(per_unit, components)
            acc.append([exposure] + [rel.loc[c, "spearman_brown"] for c in components])
        mean = np.nanmean(np.array(acc, dtype=float), axis=0)
        row = {"size": size, "exposure_per_subject": mean[0]}
        for c, value in zip(components, mean[1:]):
            row[f"reliability_{c}"] = value
            row[f"rankable_{c}"] = is_rankable(value, bar)
        rows.append(row)
    return pd.DataFrame(rows).set_index("size")
