"""Thinkers tests.

The important ones reproduce the notebook's central finding in miniature: in a
world where switching does nothing, the before/after estimator still reports a
gain, and the matched-control estimator does not.
"""

import numpy as np
import pandas as pd
import pytest

from src.policies import ThinkersPolicy
from src.reliability import spearman
from src.thinkers import (adaptation_value, build_thinkers_diagnostic,
                          measure_components, naive_before_after,
                          switch_discipline, window_mean)

POLICY = ThinkersPolicy(convergence_matches=20)
PLAYERS = [f"p{i:02d}" for i in range(24)]


def synth(restlessness=None, matches_per=700, seed=0, act_on_slump=0.7):
    """A placebo world: residuals are pure noise, so switching cannot help.

    Switches are still *selected on form* -- competitors act on slumps -- which
    is the only ingredient the regression-to-the-mean artifact needs.
    """
    rng = np.random.default_rng(seed)
    restlessness = restlessness or {p: float(rng.uniform(0.0, 0.5)) for p in PLAYERS}

    events, decisions = [], []
    for player in PLAYERS:
        resid = rng.normal(0.0, 0.45, matches_per)      # pure noise, mean zero
        for i, r in enumerate(resid, start=1):
            events.append({"player_id": player, "career_matches_after": i,
                           "resid": float(r)})
        for at in range(POLICY.trigger_window, matches_per + 1, POLICY.decision_every):
            recent = float(resid[at - POLICY.trigger_window:at].mean())
            in_slump = recent < POLICY.trigger_level
            if in_slump:
                switched = rng.random() < act_on_slump
            else:
                switched = rng.random() < restlessness[player]
            decisions.append({"player_id": player, "career_matches_after": at,
                              "in_slump": bool(in_slump), "switched": bool(switched)})
    return (pd.DataFrame(events), pd.DataFrame(decisions), restlessness)


@pytest.fixture(scope="module")
def placebo():
    return synth(seed=11)


# --------------------------------------------------------------------------- #
# the central finding
# --------------------------------------------------------------------------- #

def test_before_after_is_biased_in_a_placebo_world(placebo):
    """Switching does nothing here, and the naive estimator still reports a gain.

    This is why the prototype rule cannot ship: its `performance_change` term is
    this estimator.
    """
    events, decisions, _ = placebo
    biased = naive_before_after(events, decisions, POLICY)
    assert biased.mean() > 0.02
    assert (biased > 0).mean() > 0.7          # not one or two unlucky competitors


def test_matched_control_removes_the_bias(placebo):
    """Restricting to slump episodes makes the regression pull cancel."""
    events, decisions, _ = placebo
    matched = adaptation_value(events, decisions, POLICY).adaptation_value.astype(float)
    assert abs(matched.mean()) < 0.02
    biased = naive_before_after(events, decisions, POLICY)
    assert abs(matched.mean()) < abs(biased.mean())


def test_matched_control_needs_control_episodes(placebo):
    """If everyone always acts on a slump, the quantity is unidentifiable."""
    events, decisions, _ = synth(seed=3, act_on_slump=1.0)
    matched = adaptation_value(events, decisions, POLICY)
    assert matched.adaptation_value.astype(float).isna().all()
    assert (matched.n_control.astype(float) == 0).all()


# --------------------------------------------------------------------------- #
# switch discipline -- the component that survives
# --------------------------------------------------------------------------- #

def test_discipline_recovers_restlessness(placebo):
    """The policy facet is observable, unlike the choice-quality facet."""
    events, decisions, restlessness = placebo
    disc = switch_discipline(events, decisions, POLICY)
    truth = [restlessness[p] for p in disc.index]
    assert spearman(disc.discipline, truth) < -0.8        # more restless -> less disciplined


def test_discipline_is_a_rate_in_zero_one(placebo):
    events, decisions, _ = placebo
    disc = switch_discipline(events, decisions, POLICY)
    assert disc.discipline.between(0.0, 1.0).all()
    assert (disc.n_calm_points > 0).all()


def test_discipline_ignores_what_happened_after_the_switch(placebo):
    """Immunity to sections 6-10 comes from not looking at outcomes at all."""
    events, decisions, _ = placebo
    before = switch_discipline(events, decisions, POLICY).discipline
    scrambled = events.copy()
    scrambled["resid"] = -scrambled["resid"] * 3.0        # mangle every outcome
    after = switch_discipline(scrambled, decisions, POLICY).discipline
    assert (before == after).all()


def test_perfect_discipline_when_nobody_churns():
    events, decisions, _ = synth(restlessness={p: 0.0 for p in PLAYERS}, seed=5)
    disc = switch_discipline(events, decisions, POLICY)
    assert (disc.discipline == 1.0).all()


# --------------------------------------------------------------------------- #
# the diagnostic refuses to promote itself
# --------------------------------------------------------------------------- #

def test_adaptation_value_never_clears_the_bar(placebo):
    """The component the notebook rejected stays rejected, by measurement."""
    events, decisions, _ = placebo
    diag, measured = build_thinkers_diagnostic(events, decisions, POLICY)
    assert measured["adaptation_value"] < POLICY.reliability_bar
    assert not bool(diag.adaptation_value_rankable.iloc[0])


def test_diagnostic_not_rankable_when_competitors_barely_differ():
    """Low between-competitor variance -> nothing to rank, and the gate says so."""
    flat = {p: 0.10 for p in PLAYERS}
    events, decisions, _ = synth(restlessness=flat, seed=9)
    diag, measured = build_thinkers_diagnostic(events, decisions, POLICY)
    assert measured["discipline"] < POLICY.reliability_bar
    assert not bool(diag.is_rankable.iloc[0])
    assert diag.status.iloc[0].startswith("provisional")


def test_diagnostic_reports_its_own_reliability(placebo):
    events, decisions, _ = placebo
    diag, measured = build_thinkers_diagnostic(events, decisions, POLICY)
    assert diag.reliability.nunique() == 1                 # one number for the cohort
    assert diag.reliability.iloc[0] == pytest.approx(measured["discipline"])


def test_band_width_tracks_measured_reliability(placebo):
    """The band is the honest part of the output: it is set by the measurement."""
    events, decisions, _ = placebo
    diag, measured = build_thinkers_diagnostic(events, decisions, POLICY)
    expected_width = 2 * 120.0 * (1.0 - max(measured["discipline"], 0.0))
    assert (diag.band_hi - diag.band_lo).unique() == pytest.approx(expected_width)
    assert (diag.band_lo < diag.discipline_index).all()
    assert (diag.discipline_index < diag.band_hi).all()


def test_band_is_wider_when_reliability_is_lower():
    flat = {p: 0.10 for p in PLAYERS}
    noisy, _ = build_thinkers_diagnostic(*synth(restlessness=flat, seed=9)[:2],
                                         policy=POLICY)
    spread, _ = build_thinkers_diagnostic(*synth(seed=11)[:2], policy=POLICY)
    assert (noisy.band_hi - noisy.band_lo).iloc[0] > (
        spread.band_hi - spread.band_lo).iloc[0]


def test_gate_is_computed_not_configured(placebo):
    """Both directions: the gate follows the bar, and is never hardcoded."""
    events, decisions, _ = placebo
    permissive, _ = build_thinkers_diagnostic(
        events, decisions, ThinkersPolicy(convergence_matches=20, reliability_bar=0.0))
    assert bool(permissive.is_rankable.iloc[0])
    assert permissive.status.iloc[0] == "ranked"

    impossible, _ = build_thinkers_diagnostic(
        events, decisions, ThinkersPolicy(convergence_matches=20, reliability_bar=1.01))
    assert not bool(impossible.is_rankable.iloc[0])


def test_discipline_index_is_ordered_like_discipline(placebo):
    events, decisions, _ = placebo
    diag, _ = build_thinkers_diagnostic(events, decisions, POLICY)
    assert spearman(diag.discipline, diag.discipline_index) == pytest.approx(1.0)


def test_split_half_parity_partitions_decision_points(placebo):
    events, decisions, _ = placebo
    h1 = measure_components(events, decisions, parity=0, policy=POLICY)
    h2 = measure_components(events, decisions, parity=1, policy=POLICY)
    full = measure_components(events, decisions, policy=POLICY)
    assert set(h1.index) == set(h2.index) == set(full.index)
    assert (h1.n_calm_points + h2.n_calm_points == full.n_calm_points).all()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def test_window_mean_respects_minimum_events():
    cm = np.arange(1, 11)
    resid = np.ones(10)
    assert window_mean(cm, resid, 0, 0, 10, min_events=5) == pytest.approx(1.0)
    assert np.isnan(window_mean(cm, resid, 0, 0, 3, min_events=5))


def test_window_mean_is_half_open():
    cm = np.arange(1, 11)
    resid = np.arange(1, 11, dtype=float)
    # (2, 7] -> matches 3..7
    assert window_mean(cm, resid, 2, 0, 5, min_events=1) == pytest.approx(5.0)
