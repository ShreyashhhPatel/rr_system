import pytest

from src.policies import (OverallPolicy, ProsperityPolicy, SeasonalPolicy,
                          ThinkersPolicy, default_policies, k_from_tiers,
                          load_policies, save_policies)


def test_defaults_match_the_validated_notebook_values():
    """These are the values every number in the notebooks was measured at."""
    assert SeasonalPolicy().lambda_regression == 0.70
    assert OverallPolicy().k_tiers[-1][1] == 10.0      # stiffer floor than seasonal
    assert SeasonalPolicy().k_tiers[-1][1] == 20.0
    assert ProsperityPolicy().convergence_matches == 60
    assert ThinkersPolicy().reliability_bar == 0.70


def test_thinkers_ships_with_no_ranked_components():
    """Recorded explicitly: measured and nothing qualified, not 'nobody looked'."""
    policy = ThinkersPolicy()
    assert policy.ranked_components == ()
    assert "discipline" in policy.diagnostic_components
    rejected = dict(policy.rejected_components)
    assert "adaptation_value" in rejected and rejected["adaptation_value"]


def test_k_from_tiers_steps_down_and_never_up():
    tiers = OverallPolicy().k_tiers
    ks = [k_from_tiers(n, tiers) for n in range(0, 400, 5)]
    assert ks == sorted(ks, reverse=True)
    assert k_from_tiers(0, tiers) == 40.0
    assert k_from_tiers(10_000, tiers) == 10.0


def test_k_from_tiers_boundaries_are_exclusive():
    tiers = ((25, 40.0), (None, 10.0))
    assert k_from_tiers(24, tiers) == 40.0
    assert k_from_tiers(25, tiers) == 10.0


def test_policy_table_round_trip(tmp_path):
    path = str(tmp_path / "rating_policies.csv")
    before = default_policies()
    save_policies(before, path)
    after = load_policies(path)
    assert set(after) == set(before)
    for name in before:
        assert after[name] == before[name], name


def test_load_rejects_unknown_rating(tmp_path):
    import pandas as pd
    path = str(tmp_path / "p.csv")
    pd.DataFrame([{"policy_id": "x", "applies_to": "vibes"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unknown applies_to"):
        load_policies(path)


def test_load_rejects_table_without_applies_to(tmp_path):
    import pandas as pd
    path = str(tmp_path / "p.csv")
    pd.DataFrame([{"policy_id": "x"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="applies_to"):
        load_policies(path)


def test_policies_are_immutable():
    """A policy is data. Mutating one mid-run would desync the event log."""
    with pytest.raises(Exception):
        SeasonalPolicy().lambda_regression = 0.5
