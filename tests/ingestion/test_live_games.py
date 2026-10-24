from nba_ou.fetch_data.live_games import live_games


def test_get_live_game_ids_falls_back_to_scoreboard_v2(monkeypatch) -> None:
    monkeypatch.setattr(live_games, "_current_nba_date", lambda: "2026-05-12")
    monkeypatch.setattr(
        live_games,
        "_get_live_game_ids_from_live_scoreboard",
        lambda: (_ for _ in ()).throw(ValueError("bad live response")),
    )
    monkeypatch.setattr(
        live_games,
        "_get_live_game_ids_from_scoreboard_v2",
        lambda game_date: ["0022500001"],
    )
    monkeypatch.setattr(
        live_games,
        "_get_live_game_ids_from_scoreboard_v3",
        lambda game_date: (_ for _ in ()).throw(AssertionError("should not call v3")),
    )

    assert live_games.get_live_game_ids() == ["0022500001"]


def test_get_live_game_ids_falls_back_to_scoreboard_v3(monkeypatch) -> None:
    monkeypatch.setattr(live_games, "_current_nba_date", lambda: "2026-05-12")
    monkeypatch.setattr(
        live_games,
        "_get_live_game_ids_from_live_scoreboard",
        lambda: (_ for _ in ()).throw(ValueError("bad live response")),
    )
    monkeypatch.setattr(
        live_games,
        "_get_live_game_ids_from_scoreboard_v2",
        lambda game_date: (_ for _ in ()).throw(ValueError("bad v2 response")),
    )
    monkeypatch.setattr(
        live_games,
        "_get_live_game_ids_from_scoreboard_v3",
        lambda game_date: ["0022500002"],
    )

    assert live_games.get_live_game_ids() == ["0022500002"]
