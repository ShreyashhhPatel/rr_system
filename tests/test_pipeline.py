"""End-to-end: bronze CSV -> four materialisations, composed from the package."""

import pandas as pd
import pytest

from src.matches import load_matches, to_frame
from src.overall import build_overall_board, run_overall
from src.policies import default_policies, load_policies, save_policies
from src.prosperity import build_prosperity, measure_components, prepare_window
from src.ranking import apply_boundary, build_standings, run_season

SMALL_PROSPERITY = dict(convergence_matches=20, min_window_matches=40,
                        coherence_block=10)


@pytest.fixture(scope="module")
def bronze(tmp_path_factory, matches):
    """Round-trip the match log through CSV, the way production would."""
    path = tmp_path_factory.mktemp("data") / "match_events.csv"
    to_frame(matches).to_csv(path, index=False)
    return str(path)


def test_bronze_round_trips_through_the_validating_loader(bronze, matches):
    reloaded = load_matches(bronze, require_season=True)
    assert len(reloaded) == len(matches)
    assert [m.match_id for m in reloaded] == [m.match_id for m in matches]


def test_seasonal_and_overall_are_independent_materialisations(bronze):
    """Two silver tables over one bronze source, not one derived from the other."""
    loaded = load_matches(bronze)
    seasons = sorted({m.season_id for m in loaded})

    carried, seasonal_events = {}, []
    for season in seasons:
        subset = [m for m in loaded if m.season_id == season]
        end, _, events = run_season(subset, starting_ratings=carried, season_id=season)
        seasonal_events.append(events)
        carried = apply_boundary(end, lam=0.70)
    seasonal_events = pd.concat(seasonal_events, ignore_index=True)

    _, _, overall_events = run_overall(loaded)

    assert len(seasonal_events) == len(overall_events)
    # The overall rating carries no boundary regression, so the two diverge.
    final_seasonal = (seasonal_events.sort_values("played_at")
                      .groupby("player_id").rating_after.last())
    final_overall = (overall_events.sort_values("played_at")
                     .groupby("player_id").rating_after.last())
    assert not final_seasonal.round(6).equals(final_overall.round(6))


def test_prosperity_builds_on_the_overall_log(bronze):
    from src.policies import ProsperityPolicy
    policy = ProsperityPolicy(**SMALL_PROSPERITY)

    _, _, overall_events = run_overall(load_matches(bronze))
    window = prepare_window(overall_events, policy)
    board = build_prosperity(measure_components(window, policy), policy)

    assert len(board) > 0
    assert board.prosperity_rating.notna().all()
    assert set(["band_lo", "band_hi", "confidence", "consistency_flag"]) <= set(board.columns)
    # Cohort-relative by construction: the field centres on the base.
    assert board.prosperity_rating.mean() == pytest.approx(1500.0, abs=60.0)


def test_standings_and_board_agree_on_who_played_most(bronze):
    loaded = load_matches(bronze)
    _, _, seasonal = run_season(loaded)
    _, _, overall = run_overall(loaded)
    busiest_seasonal = build_standings(seasonal).matches_played.idxmax()
    busiest_overall = build_overall_board(overall).career_matches.idxmax()
    assert busiest_seasonal == busiest_overall


def test_policy_table_survives_a_round_trip_through_the_pipeline(tmp_path, bronze):
    """A parameter change should be a data change, and reproducible."""
    path = str(tmp_path / "rating_policies.csv")
    save_policies(default_policies(), path)
    policies = load_policies(path)

    loaded = load_matches(bronze)
    _, _, events_a = run_overall(loaded, policies["overall"])
    _, _, events_b = run_overall(loaded, policies["overall"])
    assert events_a.rating_after.equals(events_b.rating_after)


def test_changing_a_policy_changes_the_output(bronze):
    """...and the policy is actually wired through, not quietly ignored."""
    from src.policies import OverallPolicy
    loaded = load_matches(bronze)
    _, _, stiff = run_overall(loaded, OverallPolicy(k_tiers=((None, 8.0),)))
    _, _, loose = run_overall(loaded, OverallPolicy(k_tiers=((None, 40.0),)))
    assert stiff.rating_after.std() < loose.rating_after.std()
