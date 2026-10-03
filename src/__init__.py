"""RRsystem -- Elo ranking and rating engine.

Four ratings over one bronze table of match facts. Three of them rank; the
fourth is a diagnostic, because that is what the measurements supported.

    from src.matches import load_matches
    from src.overall import run_overall, build_overall_board
    from src.prosperity import prepare_window, measure_components, build_prosperity

Each module corresponds to a notebook in ``notebooks/`` that derives it and
shows what it can and cannot support.
"""

from .elo import (
    INITIAL_RATING, RATING_MEAN, ELO_DIVISOR,
    Team, EloCalculator, regress_to_mean,
    expected_score, update_ratings, draw_probability, sample_result,
)
from .matches import Match, load_matches, to_frame
from .policies import (
    SeasonalPolicy, OverallPolicy, ProsperityPolicy, ThinkersPolicy,
    default_policies, load_policies, save_policies,
)
from .reliability import (
    spearman, spearman_brown, split_half_reliability, reliability_report,
    is_rankable, power_curve, DEFAULT_RELIABILITY_BAR,
)

__all__ = [
    "INITIAL_RATING", "RATING_MEAN", "ELO_DIVISOR",
    "Team", "EloCalculator", "regress_to_mean",
    "expected_score", "update_ratings", "draw_probability", "sample_result",
    "Match", "load_matches", "to_frame",
    "SeasonalPolicy", "OverallPolicy", "ProsperityPolicy", "ThinkersPolicy",
    "default_policies", "load_policies", "save_policies",
    "spearman", "spearman_brown", "split_half_reliability", "reliability_report",
    "is_rankable", "power_curve", "DEFAULT_RELIABILITY_BAR",
]
