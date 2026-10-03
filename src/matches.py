"""The match fact, and a loader that refuses to process a bad one.

Every rating in the system is a materialisation over the same bronze table. If
that table is wrong, all four are wrong in ways that are hard to see, so the
loader validates rather than trusting.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime

import warnings

import pandas as pd

__all__ = ["Match", "REQUIRED_COLUMNS", "load_matches", "to_frame"]

REQUIRED_COLUMNS = ("match_id", "played_at", "player_a", "player_b", "result")
VALID_RESULTS = (0.0, 0.5, 1.0)


@dataclass
class Match:
    """One match, from the perspective of ``player_a``.

    ``result`` is 1.0 / 0.5 / 0.0 for a win / draw / loss by ``player_a``.
    ``strategy_a`` / ``strategy_b`` are the Thinkers inputs and are optional --
    but they can only ever be populated at match time. See
    ``notebooks/thinkers_rating_explained.ipynb`` section 3.
    """

    match_id: str
    played_at: datetime
    player_a: str
    player_b: str
    result: float
    season_id: str | None = None
    strategy_a: int | None = None
    strategy_b: int | None = None


def load_matches(path: str, require_season: bool = False,
                 require_strategy: bool = False) -> list[Match]:
    """Load matches from CSV, validating before returning anything.

    Checks, in order:

    1. required columns are present;
    2. ``played_at`` parses and is **never null** -- a match with no timestamp
       cannot be placed in a sequence, and every rating here is sequential;
    3. ``match_id`` is unique;
    4. no competitor plays themselves;
    5. ``result`` is one of 1.0 / 0.5 / 0.0.

    The returned list is sorted by ``played_at``, so a total order is guaranteed
    downstream even if the file was not ordered. Ties are broken by ``match_id``
    to keep the ordering deterministic across runs.
    """
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required column(s): {missing}")

    with warnings.catch_warnings():
        # Mixed/!unparseable formats are expected here -- nulls are validated
        # explicitly on the next line rather than warned about.
        warnings.simplefilter("ignore", UserWarning)
        df["played_at"] = pd.to_datetime(df["played_at"], errors="coerce")
    if df["played_at"].isna().any():
        bad = df.index[df["played_at"].isna()].tolist()[:5]
        raise ValueError(
            f"{path}: played_at missing or unparseable on row(s) {bad}. "
            "Every rating in this system is sequential; a match with no "
            "timestamp cannot be ordered and must not be silently dropped."
        )

    dupes = df.match_id[df.match_id.duplicated()].unique().tolist()[:5]
    if dupes:
        raise ValueError(f"{path}: duplicate match_id(s): {dupes}")

    self_play = df[df.player_a == df.player_b]
    if not self_play.empty:
        raise ValueError(
            f"{path}: {len(self_play)} match(es) where a competitor plays "
            f"themselves, e.g. match_id={self_play.match_id.iloc[0]!r}"
        )

    bad_results = sorted(set(df.result.dropna()) - set(VALID_RESULTS))
    if bad_results:
        raise ValueError(
            f"{path}: result must be one of {VALID_RESULTS}; found {bad_results}"
        )

    if require_season and ("season_id" not in df.columns or df.season_id.isna().any()):
        raise ValueError(f"{path}: season_id is required but missing or null")

    if require_strategy:
        for col in ("strategy_a", "strategy_b"):
            if col not in df.columns or df[col].isna().any():
                raise ValueError(
                    f"{path}: {col} is required for the Thinkers rating but is "
                    "missing or null. It cannot be back-filled -- a strategy "
                    "label inferred from results cannot explain those results."
                )

    df = df.sort_values(["played_at", "match_id"], kind="stable")

    present = {c: c in df.columns
               for c in ("season_id", "strategy_a", "strategy_b")}

    def optional(row, column, cast):
        """None when the column is absent *or* null on this row.

        Both cases happen in practice: a log written before the column existed,
        and a log where only some matches carry it.
        """
        if not present[column]:
            return None
        value = row[column]
        return None if pd.isna(value) else cast(value)

    out = []
    for _, row in df.iterrows():
        out.append(Match(
            match_id=str(row.match_id),
            played_at=row.played_at.to_pydatetime(),
            player_a=str(row.player_a),
            player_b=str(row.player_b),
            result=float(row.result),
            season_id=optional(row, "season_id", str),
            strategy_a=optional(row, "strategy_a", int),
            strategy_b=optional(row, "strategy_b", int),
        ))
    return out


def to_frame(matches: list[Match]) -> pd.DataFrame:
    """The bronze ``match_events`` table as a DataFrame."""
    return pd.DataFrame([asdict(m) for m in matches])
