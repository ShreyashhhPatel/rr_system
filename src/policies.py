"""Rating policies: every tunable, in one place, loadable from a table.

The four notebooks each keep their parameters in a configuration cell and then
argue that those belong in a ``rating_policies`` table instead -- so that a
parameter change is a *data* change and historical recomputes stay reproducible.
This module is that table's in-process form.

The defaults below are the values the notebooks were validated at. Changing a
default silently invalidates every number in them, so prefer loading a policy
row over editing one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields

import pandas as pd

__all__ = [
    "SeasonalPolicy", "OverallPolicy", "ProsperityPolicy", "ThinkersPolicy",
    "POLICY_TYPES", "load_policies", "save_policies", "default_policies",
]


@dataclass(frozen=True)
class SeasonalPolicy:
    """Seasonal rating: Elo inside a window, plus a boundary rule."""

    policy_id: str = "seasonal_v1"
    applies_to: str = "seasonal"
    # Step-down K: (exclusive upper bound on matches this season, K)
    k_tiers: tuple[tuple[int | None, float], ...] = (
        (10, 48.0), (30, 32.0), (None, 20.0),
    )
    lambda_regression: float = 0.70   # 1.0 = full carry-over, 0.0 = hard reset
    min_matches_for_ranking: int = 10
    inactivity_days: int = 30


@dataclass(frozen=True)
class OverallPolicy:
    """Overall rating: one continuous career run, no boundary regression."""

    policy_id: str = "overall_v1"
    applies_to: str = "overall"
    k_tiers: tuple[tuple[int | None, float], ...] = (
        (25, 40.0), (75, 24.0), (200, 16.0), (None, 10.0),
    )
    min_career_matches: int = 50
    confidence_z: float = 1.96
    rd_base: float = 600.0
    rd_idle_beta: float = 3.0
    rd_floor: float = 20.0
    rd_ceiling: float = 350.0
    prime_window: int = 50


@dataclass(frozen=True)
class ProsperityPolicy:
    """Prosperity: a derivative over the overall rating's event log."""

    policy_id: str = "prosperity_v1"
    applies_to: str = "prosperity"
    source_table: str = "overall_rating_events"
    convergence_matches: int = 60
    min_window_matches: int = 120
    coherence_block: int = 25
    dispersion_block: int = 10
    # Coherence -> confidence ramp
    coherence_floor: float = 0.05
    coherence_ref: float = 0.30
    min_confidence: float = 0.20
    min_band_frac: float = 0.15
    base: float = 1500.0
    scale: float = 120.0
    z_clip: float = 2.5
    dispersion_erratic: float = 1.40
    dispersion_steady: float = 0.80


@dataclass(frozen=True)
class ThinkersPolicy:
    """Thinkers: a diagnostic, not a ranked rating.

    ``ranked_components`` is empty on purpose. Nothing clears
    ``reliability_bar`` at realistic data volumes, and recording that explicitly
    is the difference between "we measured and nothing qualified" and "nobody
    looked". See ``notebooks/thinkers_rating_explained.ipynb`` sections 13-16.
    """

    policy_id: str = "thinkers_v1"
    applies_to: str = "thinkers"
    source_tables: str = "match_events(strategy_id) + overall_rating_events"
    requires_new_capture: bool = True
    convergence_matches: int = 60
    decision_every: int = 15
    trigger_window: int = 15
    trigger_level: float = -0.05
    pre_window: int = 20
    post_window: int = 20
    min_window_events: int = 5
    min_episodes: int = 2
    reliability_bar: float = 0.70
    ranked_components: tuple[str, ...] = ()
    diagnostic_components: tuple[str, ...] = ("discipline",)
    rejected_components: tuple[tuple[str, str], ...] = (
        ("adaptation_value", "reliability ~0.26 and no better with 8x the data; "
                             "and it tracks headroom, not judgement"),
        ("prototype_rule", "ranks decision quality backwards (rho ~ -0.35); "
                           "tracks switch volume at +0.42"),
    )


POLICY_TYPES = {
    "seasonal": SeasonalPolicy,
    "overall": OverallPolicy,
    "prosperity": ProsperityPolicy,
    "thinkers": ThinkersPolicy,
}


def default_policies() -> dict[str, object]:
    """One policy per rating, at the values the notebooks were validated at."""
    return {name: cls() for name, cls in POLICY_TYPES.items()}


def _to_row(policy) -> dict:
    """Flatten a policy to a table row, JSON-encoding anything non-scalar."""
    row = {}
    for f in fields(policy):
        value = getattr(policy, f.name)
        if isinstance(value, (tuple, list, dict)):
            row[f.name] = json.dumps(value)
        else:
            row[f.name] = value
    return row


def _from_row(cls, row: dict):
    """Rebuild a policy from a table row, decoding the JSON columns."""
    kwargs = {}
    for f in fields(cls):
        if f.name not in row or pd.isna(row[f.name]):
            continue
        value = row[f.name]
        default = getattr(cls(), f.name)
        if isinstance(default, tuple):
            decoded = json.loads(value) if isinstance(value, str) else value
            kwargs[f.name] = tuple(
                tuple(x) if isinstance(x, list) else x for x in decoded
            )
        elif isinstance(default, bool):
            kwargs[f.name] = bool(value)
        else:
            kwargs[f.name] = type(default)(value)
    return cls(**kwargs)


def save_policies(policies: dict, path: str) -> pd.DataFrame:
    """Write the ``rating_policies`` table. One row per rating."""
    df = pd.DataFrame([_to_row(p) for p in policies.values()])
    df.to_csv(path, index=False)
    return df


def load_policies(path: str) -> dict[str, object]:
    """Read the ``rating_policies`` table back into policy objects.

    Dispatches on ``applies_to``, so an unrecognised rating raises rather than
    being silently skipped.
    """
    df = pd.read_csv(path)
    if "applies_to" not in df.columns:
        raise ValueError(f"{path}: rating_policies needs an 'applies_to' column")

    out = {}
    for _, row in df.iterrows():
        applies_to = str(row.applies_to)
        if applies_to not in POLICY_TYPES:
            raise ValueError(
                f"{path}: unknown applies_to={applies_to!r}; "
                f"expected one of {sorted(POLICY_TYPES)}"
            )
        out[applies_to] = _from_row(POLICY_TYPES[applies_to], row.to_dict())
    return out


def k_from_tiers(count: int, tiers: tuple[tuple[int | None, float], ...]) -> float:
    """Step-down K lookup shared by the seasonal and overall schedules."""
    for cutoff, k in tiers:
        if cutoff is None or count < cutoff:
            return float(k)
    return float(tiers[-1][1])
