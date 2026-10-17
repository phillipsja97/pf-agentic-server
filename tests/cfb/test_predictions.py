import polars as pl
import pytest
from unittest.mock import MagicMock, patch

import workflows.cfb.predictions as p
from schemas.models import CfbSlateGame


def _game(game_id=1, week=3, home="Georgia", away="Alabama", neutral=False,
          prob=0.7, home_pts=31.0, away_pts=21.0):
    return CfbSlateGame(game_id=game_id, week=week, start_date="2026-09-19",
                        home_team=home, away_team=away, neutral_site=neutral,
                        home_win_prob=prob, projected_home_points=home_pts,
                        projected_away_points=away_pts,
                        home_color="#ba0c2f", away_color="#9e1b32")


def _model_row(fitted_at="2026-10-01T00:00:00"):
    return {"fitted_at": fitted_at}


def test_rows_for_slate_stamps_identity_and_derives_margin():
    rows = p.rows_for_slate([_game()], season=2026, through_week=2, model_row=_model_row())
    assert len(rows) == 1
    r = rows[0]
    assert r["game_id"] == 1 and r["season"] == 2026 and r["week"] == 3
    assert r["through_week"] == 2
    assert r["model_fitted_at"] == "2026-10-01T00:00:00"
    assert r["expected_margin"] == pytest.approx(10.0)
    assert r["generated_at"].startswith("20")
    assert set(r) == set(p.SCHEMA.names)


def test_model_fitted_at_normalises_non_strings():
    from datetime import datetime
    assert p.model_fitted_at({"fitted_at": datetime(2026, 10, 1)}) == "2026-10-01 00:00:00"
    assert p.model_fitted_at({}) == "None"


def test_write_appends_only_unstored_rows(monkeypatch):
    rows = p.rows_for_slate([_game(game_id=1), _game(game_id=2)],
                            season=2026, through_week=2, model_row=_model_row())
    stored = pl.DataFrame({"game_id": [1], "model_fitted_at": ["2026-10-01T00:00:00"],
                           "through_week": [2]})
    monkeypatch.setattr(p, "read", lambda: stored)
    with patch("workflows.cfb.predictions.write_deltalake") as w:
        assert p.write(rows) == 1
    table = w.call_args[0][1]
    assert table.column("game_id").to_pylist() == [2]
    assert table.schema == p.SCHEMA


def test_write_is_a_noop_when_everything_is_stored(monkeypatch):
    rows = p.rows_for_slate([_game(game_id=1)], season=2026, through_week=2,
                            model_row=_model_row())
    monkeypatch.setattr(p, "read", lambda: pl.DataFrame(
        {"game_id": [1], "model_fitted_at": ["2026-10-01T00:00:00"], "through_week": [2]}))
    with patch("workflows.cfb.predictions.write_deltalake") as w:
        assert p.write(rows) == 0
    w.assert_not_called()


def test_write_creates_the_table_on_the_first_call(monkeypatch):
    rows = p.rows_for_slate([_game()], season=2026, through_week=2, model_row=_model_row())
    monkeypatch.setattr(p, "read", lambda: None)
    with patch("workflows.cfb.predictions.write_deltalake") as w:
        assert p.write(rows) == 1
    assert w.call_args.kwargs["mode"] == "append"


def test_write_dedupes_within_a_single_batch(monkeypatch):
    rows = p.rows_for_slate([_game(game_id=1), _game(game_id=1)],
                            season=2026, through_week=2, model_row=_model_row())
    monkeypatch.setattr(p, "read", lambda: None)
    with patch("workflows.cfb.predictions.write_deltalake") as w:
        assert p.write(rows) == 1
    assert w.call_args[0][1].num_rows == 1


def test_a_refit_model_is_stored_alongside_the_old_prediction(monkeypatch):
    rows = p.rows_for_slate([_game()], season=2026, through_week=2,
                            model_row=_model_row("2026-10-08T00:00:00"))
    monkeypatch.setattr(p, "read", lambda: pl.DataFrame(
        {"game_id": [1], "model_fitted_at": ["2026-10-01T00:00:00"], "through_week": [2]}))
    with patch("workflows.cfb.predictions.write_deltalake") as w:
        assert p.write(rows) == 1
    assert w.call_args[0][1].column("model_fitted_at").to_pylist() == ["2026-10-08T00:00:00"]


def test_write_never_raises(monkeypatch):
    rows = p.rows_for_slate([_game()], season=2026, through_week=2, model_row=_model_row())
    monkeypatch.setattr(p, "read", lambda: None)
    with patch("workflows.cfb.predictions.write_deltalake", side_effect=OSError("minio down")):
        assert p.write(rows) == 0


def test_write_of_nothing_touches_nothing():
    with patch("workflows.cfb.predictions.write_deltalake") as w:
        assert p.write([]) == 0
    w.assert_not_called()


def test_read_returns_none_when_the_table_is_absent():
    with patch("workflows.cfb.predictions.DeltaTable", side_effect=Exception("not a delta table")):
        assert p.read() is None


def test_read_returns_the_table(monkeypatch):
    import pyarrow as pa
    monkeypatch.setattr(p.settings, "minio_bucket", "b")
    dt = MagicMock()
    dt.to_pyarrow_table.return_value = pa.table({"game_id": [1]})
    with patch("workflows.cfb.predictions.DeltaTable", return_value=dt):
        assert p.read()["game_id"].to_list() == [1]


def test_a_stored_table_missing_the_key_columns_does_not_block_writes(monkeypatch):
    rows = p.rows_for_slate([_game()], season=2026, through_week=2, model_row=_model_row())
    monkeypatch.setattr(p, "read", lambda: pl.DataFrame({"something_else": [1]}))
    with patch("workflows.cfb.predictions.write_deltalake") as w:
        assert p.write(rows) == 1
    w.assert_called_once()
