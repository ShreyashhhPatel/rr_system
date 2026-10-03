import numpy as np
import pandas as pd
import pytest

from src.policies import ProsperityPolicy
from src.prosperity import (block_dispersion, block_means, build_prosperity,
                            confidence_from_coherence, consistency_flag,
                            measure_components, path_coherence, prepare_window,
                            retention, theil_sen_slope)

SMALL = ProsperityPolicy(convergence_matches=20, min_window_matches=40,
                         coherence_block=10)


# --------------------------------------------------------------------------- #
# theil_sen_slope
# --------------------------------------------------------------------------- #

def test_theil_sen_recovers_a_known_slope():
    y = 1500.0 + 0.4 * np.arange(300)
    assert theil_sen_slope(y) == pytest.approx(0.4)


def test_theil_sen_is_robust_to_a_spike():
    y = 1500.0 + 0.4 * np.arange(300)
    spiked = y.copy()
    spiked[150:160] += 400.0                      # a hot streak
    ols = np.polyfit(np.arange(300), spiked, 1)[0]
    robust = theil_sen_slope(spiked)
    assert abs(robust - 0.4) < abs(ols - 0.4)


def test_theil_sen_is_deterministic_under_subsampling():
    """Regression test: a shared RNG made the whole rating non-reproducible.

    With >40k pairs the estimator subsamples. Drawing that subsample from the
    global RNG made the result depend on how many times it had been called
    before, drifting the published rating by ~2 points between rebuilds.
    """
    y = np.cumsum(np.random.default_rng(0).normal(size=400))
    assert len(y) * (len(y) - 1) // 2 > 40_000    # subsampling really is active

    np.random.seed(1)
    first = theil_sen_slope(y)
    np.random.seed(999)
    _ = np.random.random(1000)                    # perturb global state
    second = theil_sen_slope(y)
    assert first == second


def test_theil_sen_seed_argument_is_honoured():
    y = np.cumsum(np.random.default_rng(0).normal(size=400))
    assert theil_sen_slope(y, seed=0) != theil_sen_slope(y, seed=1)


def test_theil_sen_needs_three_points():
    assert np.isnan(theil_sen_slope([1.0, 2.0]))


# --------------------------------------------------------------------------- #
# path_coherence
# --------------------------------------------------------------------------- #

def test_coherence_is_one_for_a_straight_line():
    assert path_coherence(np.arange(100.0), block=10) == pytest.approx(1.0)


def test_coherence_is_near_zero_for_a_round_trip():
    up = np.arange(50.0)
    y = np.concatenate([up, up[::-1]])            # out and back
    assert path_coherence(y, block=10) < 0.05


def test_coherence_is_near_zero_for_a_flat_noisy_path():
    """Structural: a genuinely flat competitor is indistinguishable from no signal."""
    y = 1500 + np.random.default_rng(2).normal(0, 30, 200)
    assert path_coherence(y, block=10) < 0.4


def test_coherence_needs_three_blocks():
    assert np.isnan(path_coherence(np.arange(10.0), block=10))


def test_coherence_is_scale_invariant():
    y = np.cumsum(np.random.default_rng(3).normal(size=200))
    assert path_coherence(y, 10) == pytest.approx(path_coherence(y * 7.0, 10))


# --------------------------------------------------------------------------- #
# block_dispersion
# --------------------------------------------------------------------------- #

def test_dispersion_is_about_one_for_calibrated_independent_results():
    """If results scatter exactly as the ratings imply, dispersion is ~1."""
    rng = np.random.default_rng(5)
    expected = rng.uniform(0.25, 0.75, 4000)
    scores = (rng.random(4000) < expected).astype(float)
    assert block_dispersion(expected, scores - expected, block=10) == pytest.approx(
        1.0, abs=0.15)


def test_dispersion_rises_for_streaky_results():
    """Persistent form inflates block sums without inflating single-match variance."""
    rng = np.random.default_rng(6)
    n = 4000
    expected = np.full(n, 0.5)
    # A slow-moving latent advantage: runs of wins then runs of losses.
    latent = np.repeat(rng.normal(0, 0.35, n // 50), 50)
    scores = (rng.random(n) < np.clip(expected + latent, 0.02, 0.98)).astype(float)
    assert block_dispersion(expected, scores - expected, block=10) > 1.3


def test_dispersion_needs_three_blocks():
    assert np.isnan(block_dispersion(np.full(10, 0.5), np.zeros(10), block=10))


# --------------------------------------------------------------------------- #
# retention (a rejected component, kept as a regression test)
# --------------------------------------------------------------------------- #

def test_retention_full_and_zero():
    assert retention(np.array([1500.0, 1600.0, 1700.0])) == pytest.approx(1.0)
    assert retention(np.array([1500.0, 1700.0, 1500.0])) == pytest.approx(0.0)


def test_retention_undefined_without_a_gain():
    assert np.isnan(retention(np.array([1500.0, 1505.0, 1495.0])))


# --------------------------------------------------------------------------- #
# composition
# --------------------------------------------------------------------------- #

def test_confidence_ramp_endpoints():
    policy = ProsperityPolicy()
    assert confidence_from_coherence(0.0, policy) == policy.min_confidence
    assert confidence_from_coherence(1.0, policy) == pytest.approx(1.0)
    assert confidence_from_coherence(float("nan"), policy) == policy.min_confidence


def test_confidence_is_monotonic_in_coherence():
    vals = [confidence_from_coherence(c) for c in np.linspace(0, 1, 25)]
    assert vals == sorted(vals)


def test_consistency_flag_thresholds():
    assert consistency_flag(2.0) == "erratic"
    assert consistency_flag(1.0) == "typical"
    assert consistency_flag(0.5) == "steady"
    assert consistency_flag(float("nan")) == "unknown"


def test_prepare_window_adds_k_free_residual(overall_events):
    window = prepare_window(overall_events, SMALL)
    assert (window.career_matches_after > SMALL.convergence_matches).all()
    recovered = window.result - window.resid
    assert np.allclose(recovered, window.expected)
    assert window.resid.abs().max() <= 1.0 + 1e-9


def test_build_prosperity_is_a_pure_function(overall_events):
    """Same log plus same policy must give the same ratings, byte for byte."""
    window = prepare_window(overall_events, SMALL)
    comps = measure_components(window, SMALL)
    first = build_prosperity(comps, SMALL)
    second = build_prosperity(measure_components(prepare_window(overall_events, SMALL),
                                                SMALL), SMALL)
    assert (first.prosperity_rating - second.prosperity_rating).abs().max() == 0.0


def test_prosperity_band_is_never_zero_width(overall_events):
    """A finite sample is never certain, so no competitor gets a bare point."""
    window = prepare_window(overall_events, SMALL)
    out = build_prosperity(measure_components(window, SMALL), SMALL)
    assert ((out.band_hi - out.band_lo) > 0).all()
    assert (out.band_lo < out.prosperity_rating).all()
    assert (out.prosperity_rating < out.band_hi).all()


def test_low_coherence_shrinks_toward_neutral():
    """Churn is not punished; it is moved to the middle."""
    comps = pd.DataFrame({
        "trajectory": [40.0, 40.0, -40.0, -40.0],
        "coherence": [0.80, 0.02, 0.80, 0.02],     # confident / churning pairs
        "dispersion": [1.0, 1.0, 1.0, 1.0],
    }, index=["up_clear", "up_churn", "down_clear", "down_churn"])
    out = build_prosperity(comps)
    base = ProsperityPolicy().base
    assert out.loc["up_churn", "prosperity_rating"] < out.loc["up_clear", "prosperity_rating"]
    assert out.loc["down_churn", "prosperity_rating"] > out.loc["down_clear", "prosperity_rating"]
    for churner in ("up_churn", "down_churn"):
        assert abs(out.loc[churner, "prosperity_rating"] - base) < 30.0
        # ...and says so, with a wider band than the confident competitors.
        assert (out.loc[churner, "band_hi"] - out.loc[churner, "band_lo"]) > (
            out.loc["up_clear", "band_hi"] - out.loc["up_clear", "band_lo"])


def test_retention_is_dropped_from_the_output():
    comps = pd.DataFrame({"trajectory": [1.0, -1.0], "coherence": [0.5, 0.5],
                          "dispersion": [1.0, 1.0], "retention": [0.9, 0.1]},
                         index=["a", "b"])
    assert "retention" not in build_prosperity(comps).columns


def test_degenerate_field_does_not_divide_by_zero():
    """Everyone identical -> no spread -> z of 0, not NaN."""
    comps = pd.DataFrame({"trajectory": [5.0, 5.0, 5.0], "coherence": [0.5] * 3,
                          "dispersion": [1.0] * 3}, index=list("abc"))
    out = build_prosperity(comps)
    assert out.prosperity_rating.notna().all()
    assert (out.z_trajectory == 0).all()
