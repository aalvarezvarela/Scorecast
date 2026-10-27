import pytest
from nba_ou.fetch_data.live_games import live_games


def _fail(message: str):
    return lambda *args: (_ for _ in ()).throw(ValueError(message))


def test_get_live_game_ids_falls_back_to_scoreboard_v3(monkeypatch) -> None:
    # 2026-10-26: the CDN live scoreboard answers 403 from the runner.
    monkeypatch.setattr(live_games, "_current_nba_date", lambda: "2026-05-12")
    monkeypatch.setattr(
        live_games,
        "_get_live_game_ids_from_live_scoreboard",
        _fail("bad live response"),
    )
    monkeypatch.setattr(
        live_games,
        "_get_live_game_ids_from_scoreboard_v3",
        lambda game_date: ["0022500002"],
    )

    assert live_games.get_live_game_ids() == ["0022500002"]


def test_get_live_game_ids_raises_when_no_scoreboard_answers(monkeypatch) -> None:
    # The final-score update falls back on the tip-off window only on a raise.
    monkeypatch.setattr(live_games, "_current_nba_date", lambda: "2026-05-12")
    monkeypatch.setattr(live_games.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        live_games,
        "_get_live_game_ids_from_live_scoreboard",
        _fail("bad live response"),
    )
    monkeypatch.setattr(
        live_games, "_get_live_game_ids_from_scoreboard_v3", _fail("bad v3 response")
    )

    with pytest.raises(RuntimeError, match="ScoreboardV3"):
        live_games.get_live_game_ids()


def test_get_live_game_ids_retries_a_round_where_both_fail(monkeypatch) -> None:
    # 2026-10-26: a ScoreboardV3 read timeout stopped the predictor once.
    monkeypatch.setattr(live_games, "_current_nba_date", lambda: "2026-05-12")
    pauses: list[float] = []
    monkeypatch.setattr(live_games.time, "sleep", pauses.append)
    monkeypatch.setattr(
        live_games,
        "_get_live_game_ids_from_live_scoreboard",
        _fail("bad live response"),
    )
    v3_answers = iter([ValueError("read timeout"), ["0022600001"]])

    def flaky_v3(game_date):
        answer = next(v3_answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(live_games, "_get_live_game_ids_from_scoreboard_v3", flaky_v3)

    assert live_games.get_live_game_ids() == ["0022600001"]
    assert pauses == [live_games.RETRY_PAUSE_S]
