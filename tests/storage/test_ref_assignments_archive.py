import pandas as pd
from nba_ou.postgre_db.injuries_refs import ref_assignments_archive as archive
from psycopg import sql


def _scrape():
    return pd.DataFrame(
        {
            "Game": ["Boston @ New York", "  "],
            "Crew Chief": ["Tony Brothers (#25)", "x"],
            "Referee": ["Karl Lane (#77)", "y"],
            "Umpire": ["Chelisa Painter (#79)", "z"],
            "Alternate": [None, None],
        }
    )


def test_rows_keep_roles_and_use_the_eastern_assignment_date():
    # 03:30 UTC on the 15th is still the 14th in New York.
    rows = archive.assignments_to_rows(
        _scrape(), pd.Timestamp("2026-01-15 03:30", tz="UTC")
    )
    assert len(rows) == 1
    row = rows.iloc[0]
    assert str(row["assignment_date"]) == "2026-01-14"
    assert (row["crew_chief"], row["referee"], row["umpire"]) == (
        "Tony Brothers (#25)",
        "Karl Lane (#77)",
        "Chelisa Painter (#79)",
    )
    assert row["alternate"] is None
    assert list(rows.columns) == archive._COLUMNS


def test_archive_is_a_no_op_without_rows(monkeypatch):
    def fail():
        raise AssertionError("must not connect")

    monkeypatch.setattr(archive, "connect_nba_db", fail)
    assert archive.archive_referee_assignments(pd.DataFrame()) == 0


def test_archive_inserts_first_seen_crews_without_overwriting(monkeypatch):
    executed = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, query, params=None):
            executed.append(("execute", query))

        def executemany(self, query, params):
            executed.append(("executemany", query, list(params)))

    class Conn:
        closed = False

        def cursor(self):
            return Cursor()

        def commit(self):
            pass

        def close(self):
            Conn.closed = True

    monkeypatch.setattr(archive, "connect_nba_db", Conn)
    monkeypatch.setattr(archive, "get_schema_name_refs", lambda: "refs")
    sent = archive.archive_referee_assignments(
        _scrape(), pd.Timestamp("2026-01-14 14:00", tz="UTC")
    )

    assert sent == 1
    assert Conn.closed
    kind, query, params = executed[-1]
    assert kind == "executemany"
    assert isinstance(query, sql.Composed)
    assert "DO NOTHING" in repr(query)
    assert len(params) == 1 and params[0][2] == "Tony Brothers (#25)"
