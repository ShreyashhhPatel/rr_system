import pandas as pd
import pytest

from src.matches import REQUIRED_COLUMNS, load_matches, to_frame

GOOD = pd.DataFrame({
    "match_id": ["M2", "M1", "M3"],
    "played_at": ["2024-01-05 10:00", "2024-01-01 09:00", "2024-01-09 11:00"],
    "player_a": ["Alice", "Bob", "Alice"],
    "player_b": ["Bob", "Charlie", "Charlie"],
    "result": [1.0, 0.5, 0.0],
    "season_id": ["S1", "S1", "S1"],
})


def write(df, tmp_path, name="m.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def test_loads_and_sorts_by_time(tmp_path):
    out = load_matches(write(GOOD, tmp_path))
    assert [m.match_id for m in out] == ["M1", "M2", "M3"]
    assert out[0].played_at < out[1].played_at < out[2].played_at


def test_round_trips_to_frame(tmp_path):
    frame = to_frame(load_matches(write(GOOD, tmp_path)))
    assert len(frame) == 3
    assert set(REQUIRED_COLUMNS) <= set(frame.columns)


def test_rejects_missing_column(tmp_path):
    with pytest.raises(ValueError, match="missing required column"):
        load_matches(write(GOOD.drop(columns=["result"]), tmp_path))


def test_rejects_null_timestamp(tmp_path):
    """Every rating here is sequential, so an unorderable match must not pass."""
    bad = GOOD.copy()
    bad.loc[1, "played_at"] = None
    with pytest.raises(ValueError, match="played_at missing"):
        load_matches(write(bad, tmp_path))


def test_rejects_unparseable_timestamp(tmp_path):
    bad = GOOD.copy()
    bad.loc[0, "played_at"] = "not a date"
    with pytest.raises(ValueError, match="played_at missing"):
        load_matches(write(bad, tmp_path))


def test_rejects_duplicate_match_id(tmp_path):
    bad = GOOD.copy()
    bad.loc[2, "match_id"] = "M1"
    with pytest.raises(ValueError, match="duplicate match_id"):
        load_matches(write(bad, tmp_path))


def test_rejects_self_play(tmp_path):
    bad = GOOD.copy()
    bad.loc[0, "player_b"] = bad.loc[0, "player_a"]
    with pytest.raises(ValueError, match="plays"):
        load_matches(write(bad, tmp_path))


def test_rejects_invalid_result(tmp_path):
    bad = GOOD.copy()
    bad.loc[0, "result"] = 3.0
    with pytest.raises(ValueError, match="result must be one of"):
        load_matches(write(bad, tmp_path))


def test_require_strategy_rejects_absent_column(tmp_path):
    """Thinkers needs strategy_id, and it cannot be back-filled."""
    with pytest.raises(ValueError, match="strategy_a"):
        load_matches(write(GOOD, tmp_path), require_strategy=True)


def test_require_strategy_accepts_populated_column(tmp_path):
    ok = GOOD.assign(strategy_a=[0, 1, 2], strategy_b=[1, 2, 0])
    out = load_matches(write(ok, tmp_path), require_strategy=True)
    assert all(m.strategy_a is not None for m in out)


def test_require_season_rejects_null(tmp_path):
    bad = GOOD.copy()
    bad.loc[0, "season_id"] = None
    with pytest.raises(ValueError, match="season_id"):
        load_matches(write(bad, tmp_path), require_season=True)


def test_optional_columns_may_be_present_but_null(tmp_path):
    """A log written before strategy capture existed still loads."""
    partial = GOOD.assign(strategy_a=[0, None, 2], strategy_b=[None, None, 0])
    out = load_matches(write(partial, tmp_path))
    by_id = {m.match_id: m for m in out}
    assert by_id["M1"].strategy_a is None          # null on this row
    assert by_id["M3"].strategy_a == 2             # populated on this one
    assert by_id["M2"].strategy_b is None


def test_all_null_optional_column_is_not_an_error(tmp_path):
    allnull = GOOD.assign(strategy_a=[None] * 3, strategy_b=[None] * 3)
    out = load_matches(write(allnull, tmp_path))
    assert all(m.strategy_a is None for m in out)


def test_require_strategy_rejects_partially_null(tmp_path):
    """Partial capture is not capture: Thinkers needs it on every match."""
    partial = GOOD.assign(strategy_a=[0, None, 2], strategy_b=[1, 2, 0])
    with pytest.raises(ValueError, match="strategy_a"):
        load_matches(write(partial, tmp_path), require_strategy=True)
