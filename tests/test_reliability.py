import numpy as np
import pandas as pd
import pytest

from src.reliability import (DEFAULT_RELIABILITY_BAR, is_rankable, power_curve,
                             reliability_report, spearman, spearman_brown,
                             split_half_reliability)


def test_spearman_perfect_and_inverse():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_is_rank_based_not_linear():
    """Monotone but wildly non-linear still correlates perfectly."""
    assert spearman([1, 2, 3, 4], [1, 4, 900, 10_000]) == pytest.approx(1.0)


def test_spearman_handles_degenerate_input():
    assert np.isnan(spearman([1, 2], [1, 2]))              # too few pairs
    assert np.isnan(spearman([1, 1, 1, 1], [1, 2, 3, 4]))  # no variance
    assert spearman([1, 2, 3, np.nan], [1, 2, 3, 99]) == pytest.approx(1.0)


def test_spearman_brown_known_values():
    assert spearman_brown(0.5) == pytest.approx(2 / 3)
    assert spearman_brown(1.0) == pytest.approx(1.0)
    assert spearman_brown(0.0) == pytest.approx(0.0)


def test_spearman_brown_rescues_a_usable_component():
    """0.40 raw reads like a dead end; corrected it is merely early."""
    assert spearman_brown(0.40) == pytest.approx(0.5714, abs=1e-3)


def test_spearman_brown_always_at_least_the_raw_value():
    for r in np.linspace(0.0, 1.0, 21):
        assert spearman_brown(r) >= r - 1e-12


def test_spearman_brown_nan_in_nan_out():
    assert np.isnan(spearman_brown(float("nan")))
    assert np.isnan(spearman_brown(-1.0))


def test_is_rankable_gate():
    assert is_rankable(0.70) and is_rankable(0.95)
    assert not is_rankable(0.69)
    assert not is_rankable(0.0)


def test_unmeasured_is_not_rankable():
    """An unmeasured component is not a passing one."""
    assert not is_rankable(float("nan"))


def test_split_half_detects_a_reliable_and_an_unreliable_component():
    rng = np.random.default_rng(0)
    subjects = [f"p{i}" for i in range(40)]
    latent = pd.Series(rng.normal(size=40), index=subjects)

    def per_unit(parity):
        # 'solid' barely moves between halves; 'noise' is redrawn each time.
        return pd.DataFrame({
            "solid": latent + rng.normal(0, 0.05, 40),
            "noise": rng.normal(0, 1, 40),
        }, index=subjects)

    out = split_half_reliability(per_unit, ["solid", "noise"])
    assert out.loc["solid", "spearman_brown"] > 0.9
    assert abs(out.loc["noise", "spearman_brown"]) < 0.5
    assert (out.n_subjects == 40).all()


def test_reliability_report_computes_the_gate():
    subjects = [f"p{i}" for i in range(30)]
    values = pd.Series(np.arange(30.0), index=subjects)

    def per_unit(parity):
        return pd.DataFrame({"stable": values}, index=subjects)

    out = reliability_report(per_unit, ["stable"])
    assert bool(out.loc["stable", "is_rankable"]) is True
    assert out.loc["stable", "bar"] == DEFAULT_RELIABILITY_BAR


def test_power_curve_separates_data_limited_from_hopeless():
    """A rising curve means wait for data; a flat one means stop building."""
    subjects = [f"p{i}" for i in range(40)]
    latent = np.random.default_rng(1).normal(size=40)

    def build(size, seed):
        rng = np.random.default_rng(seed)

        def per_unit(parity):
            return pd.DataFrame({
                # noise shrinks as 'size' grows -> reliability climbs
                "data_limited": latent + rng.normal(0, 4.0 / np.sqrt(size), 40),
                # pure noise regardless of size -> reliability stays flat
                "hopeless": rng.normal(0, 1, 40),
            }, index=subjects)

        return per_unit, float(size)

    curve = power_curve([4, 64, 1024], build, ["data_limited", "hopeless"], seeds=(1, 2))
    assert curve.reliability_data_limited.is_monotonic_increasing
    assert curve.reliability_data_limited.iloc[-1] > 0.9
    assert curve.reliability_hopeless.abs().max() < 0.6
    assert bool(curve.rankable_data_limited.iloc[-1]) is True
    assert bool(curve.rankable_hopeless.iloc[-1]) is False
