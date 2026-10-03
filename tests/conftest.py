"""Shared fixtures. Deliberately tiny and deterministic."""

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.elo import expected_score, sample_result          # noqa: E402
from src.matches import Match                              # noqa: E402

PLAYERS = ["Alice", "Bob", "Charlie", "Dana", "Eve", "Frank"]
TRUE_SKILL = {"Alice": 1700, "Bob": 1620, "Charlie": 1550,
              "Dana": 1480, "Eve": 1420, "Frank": 1350}


def make_matches(n=900, seed=7, draw_base=0.16, seasons=2, days=360):
    """A deterministic match log drawn from fixed true skills."""
    rng = random.Random(seed)
    start = datetime(2024, 1, 1)
    rows = []
    for _ in range(n):
        day = rng.uniform(0, days)
        a, b = rng.sample(PLAYERS, 2)
        p = expected_score(TRUE_SKILL[a], TRUE_SKILL[b])
        result = sample_result(p, draw_base, rng.random())
        rows.append((day, a, b, result))
    rows.sort(key=lambda r: r[0])
    season_len = days / seasons
    return [
        Match(match_id=f"M{i:05d}",
              played_at=start + timedelta(days=day),
              player_a=a, player_b=b, result=result,
              season_id=f"S{int(day // season_len) + 1}")
        for i, (day, a, b, result) in enumerate(rows)
    ]


@pytest.fixture(scope="session")
def matches():
    return make_matches()


@pytest.fixture(scope="session")
def overall_events(matches):
    from src.overall import run_overall
    _, _, events = run_overall(matches)
    return events


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)
