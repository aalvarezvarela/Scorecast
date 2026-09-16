import numpy as np
import pandas as pd
import pytest
from nba_ou.config.odds_columns import is_odds_shaped_column, spread_col
from nba_ou.data_processing.referees.referee_tendencies import (
    DEFAULT_REFEREE_TENDENCY_SPECS,
    FTA_X_ABS_SPREAD_FEATURE,
    FTA_X_EXPECTED_TOTAL_FTA_FEATURE,
    FTA_X_HOME_FTA_RATE_EDGE_FEATURE,
    POSS_X_ABS_SPREAD_FEATURE,
    PRIOR_GAMES_FEATURE,
    UNKNOWN_COUNT_FEATURE,
    RefereeTendencySpec,
    add_referee_interaction_features,
    add_referee_tendency_features,
    aggregate_crew_features,
    build_official_crews,
    build_official_name_index,
    build_referee_game_history,
    compute_official_tendencies,
    referee_tendency_feature_columns,
)

BOOK = "bet365"
SPEC = RefereeTendencySpec(
    "Q",
    "REF_CREW_Q_TENDENCY_BEFORE",
    half_life_days=10.0,
    shrinkage_k=2.0,
    track="totals",
)


def _history(rows):
    """rows: (game_id, date, value) -> minimal game history with quantity Q."""
    return pd.DataFrame(
        {
            "GAME_ID": [r[0] for r in rows],
            "GAME_DATE": pd.to_datetime([r[1] for r in rows]),
            "SEASON_YEAR": 2025,
            "Q": [r[2] for r in rows],
        }
    )


def _crews(pairs, history):
    crews = pd.DataFrame(pairs, columns=["GAME_ID", "OFFICIAL_ID"])
    return crews.merge(history[["GAME_ID", "GAME_DATE"]], on="GAME_ID")


def _estimate(estimates, game_id, official):
    row = estimates[
        (estimates.GAME_ID == game_id) & (estimates.OFFICIAL_ID == official)
    ]
    return float(row[SPEC.feature].iloc[0])


# ---------------------------------------------------------------- estimator


def test_estimate_matches_hand_computed_decay_and_shrinkage():
    history = _history(
        [
            ("g1", "2025-11-01", 4.0),
            ("g2", "2025-11-11", -2.0),
            ("g3", "2025-11-21", 0.0),
        ]
    )
    crews = _crews([("g1", "A"), ("g2", "A"), ("g3", "A")], history)
    est = compute_official_tendencies(
        crews, history, specs=(SPEC,), max_history_days=365
    )

    w1, w2 = 0.5 ** (20 / 10), 0.5 ** (10 / 10)
    assert _estimate(est, "g3", "A") == pytest.approx(
        (w1 * 4.0 + w2 * -2.0) / (w1 + w2 + 2.0)
    )
    assert _estimate(est, "g2", "A") == pytest.approx(0.5 * 4.0 / (0.5 + 2.0))
    assert _estimate(est, "g1", "A") == 0.0


def test_own_result_never_enters_its_own_estimate():
    base = _history([("g1", "2025-11-01", 4.0), ("g2", "2025-11-05", 1.0)])
    changed = base.assign(Q=[4.0, 999.0])
    crews = _crews([("g1", "A"), ("g2", "A")], base)
    a = compute_official_tendencies(crews, base, specs=(SPEC,), max_history_days=365)
    b = compute_official_tendencies(crews, changed, specs=(SPEC,), max_history_days=365)
    assert _estimate(a, "g2", "A") == _estimate(b, "g2", "A")


def test_games_on_the_same_date_do_not_see_each_other():
    history = _history([("g1", "2025-11-01", 10.0), ("g2", "2025-11-01", -10.0)])
    crews = _crews([("g1", "A"), ("g2", "A")], history)
    est = compute_official_tendencies(
        crews, history, specs=(SPEC,), max_history_days=365
    )
    assert _estimate(est, "g1", "A") == 0.0
    assert _estimate(est, "g2", "A") == 0.0
    assert (est["PRIOR_GAMES"] == 0).all()


def test_games_outside_the_window_are_ignored_so_extra_history_changes_nothing():
    full = _history(
        [
            ("old", "2020-01-01", 50.0),
            ("g1", "2025-11-01", 4.0),
            ("g2", "2025-11-05", 0.0),
        ]
    )
    trimmed = full[full.GAME_ID != "old"]
    # A near-flat decay, so only the window (not the decay) can exclude "old".
    spec = RefereeTendencySpec(
        "Q", SPEC.feature, half_life_days=1e9, shrinkage_k=2.0, track="totals"
    )
    a = compute_official_tendencies(
        _crews([("old", "A"), ("g1", "A"), ("g2", "A")], full),
        full,
        specs=(spec,),
        max_history_days=365,
    )
    b = compute_official_tendencies(
        _crews([("g1", "A"), ("g2", "A")], trimmed),
        trimmed,
        specs=(spec,),
        max_history_days=365,
    )
    assert _estimate(a, "g2", "A") == pytest.approx(_estimate(b, "g2", "A"))
    assert _estimate(a, "g2", "A") == pytest.approx(4.0 / 3.0)


def test_missing_quantities_are_skipped_not_counted_as_zero():
    history = _history(
        [
            ("g1", "2025-11-01", np.nan),
            ("g2", "2025-11-02", 6.0),
            ("g3", "2025-11-03", 0.0),
        ]
    )
    crews = _crews([("g1", "A"), ("g2", "A"), ("g3", "A")], history)
    spec = RefereeTendencySpec(
        "Q", SPEC.feature, half_life_days=1e9, shrinkage_k=0.0, track="totals"
    )
    est = compute_official_tendencies(
        crews, history, specs=(spec,), max_history_days=365
    )
    assert float(est.loc[est.GAME_ID == "g3", spec.feature].iloc[0]) == pytest.approx(
        6.0
    )
    assert float(est.loc[est.GAME_ID == "g3", "PRIOR_GAMES"].iloc[0]) == 2


def test_same_season_variant_ignores_previous_season():
    history = _history([("g1", "2025-03-01", 8.0), ("g2", "2025-11-01", 0.0)])
    history.loc[0, "SEASON_YEAR"] = 2024
    crews = _crews([("g1", "A"), ("g2", "A")], history)
    est = compute_official_tendencies(
        crews,
        history,
        specs=(SPEC,),
        max_history_days=365,
        include_same_season_variants=True,
    )
    row = est[est.GAME_ID == "g2"].iloc[0]
    assert row[SPEC.feature] != 0.0
    assert row[SPEC.same_season_feature()] == 0.0


# ---------------------------------------------------------------- crews


def _refs(rows):
    return pd.DataFrame(
        rows, columns=["GAME_ID", "OFFICIAL_ID", "FIRST_NAME", "LAST_NAME", "GAME_DATE"]
    )


def test_crew_order_does_not_change_features():
    history = _history(
        [
            ("g1", "2025-11-01", 3.0),
            ("g2", "2025-11-02", -1.0),
            ("g3", "2025-11-04", 0.0),
        ]
    )
    rows = [
        ("g1", "A", "a", "a", "2025-11-01"),
        ("g1", "B", "b", "b", "2025-11-01"),
        ("g1", "C", "c", "c", "2025-11-01"),
        ("g2", "B", "b", "b", "2025-11-02"),
        ("g2", "C", "c", "c", "2025-11-02"),
        ("g2", "D", "d", "d", "2025-11-02"),
        ("g3", "A", "a", "a", "2025-11-04"),
        ("g3", "C", "c", "c", "2025-11-04"),
        ("g3", "D", "d", "d", "2025-11-04"),
    ]
    refs = _refs(rows)
    shuffled = refs.sample(frac=1.0, random_state=3)

    def features(r):
        crews = build_official_crews(r, history[["GAME_ID", "GAME_DATE"]])
        est = compute_official_tendencies(
            crews, history, specs=(SPEC,), max_history_days=365
        )
        return (
            aggregate_crew_features(est, specs=(SPEC,))
            .set_index("GAME_ID")
            .sort_index()
        )

    pd.testing.assert_frame_equal(features(refs), features(shuffled))


def test_scheduled_names_resolve_to_official_ids_through_the_history():
    refs = _refs(
        [
            ("g1", "1146", "Tony", "Brothers", "2025-11-01"),
            ("g1", "2", "C.J.", "Washington", "2025-11-01"),
        ]
    )
    index = build_official_name_index(refs)
    assert index["Tony Brothers"] == "1146"
    assert index["CJ Washington"] == "2"

    scheduled = pd.DataFrame(
        {
            "GAME_ID": ["g9"],
            "REF_1": ["Tony Brothers (#25)"],
            "REF_2": ["C.J. Washington"],
            "REF_3": [None],
        }
    )
    dates = pd.DataFrame(
        {
            "GAME_ID": ["g1", "g9"],
            "GAME_DATE": pd.to_datetime(["2025-11-01", "2025-11-09"]),
        }
    )
    crews = build_official_crews(refs, dates, scheduled)
    assert set(crews.loc[crews.GAME_ID == "g9", "OFFICIAL_ID"]) == {"1146", "2"}


def test_unknown_scheduled_official_is_neutral_and_counted_without_raising():
    history = _history([("g1", "2025-11-01", 6.0)])
    refs = _refs([("g1", "A", "Known", "Ref", "2025-11-01")])
    scheduled = pd.DataFrame(
        {
            "GAME_ID": ["g9"],
            "REF_1": ["Known Ref"],
            "REF_2": ["Brand New"],
            "REF_3": ["Also New"],
        }
    )
    dates = pd.DataFrame(
        {
            "GAME_ID": ["g1", "g9"],
            "GAME_DATE": pd.to_datetime(["2025-11-01", "2025-11-02"]),
        }
    )
    with pytest.warns(UserWarning, match="Brand New"):
        crews = build_official_crews(refs, dates, scheduled)
    est = compute_official_tendencies(
        crews, history, specs=(SPEC,), max_history_days=365
    )
    crew = aggregate_crew_features(est, specs=(SPEC,)).set_index("GAME_ID").loc["g9"]

    known = 0.5 ** (1 / 10) * 6.0 / (0.5 ** (1 / 10) + 2.0)
    assert crew[SPEC.feature] == pytest.approx(known)  # 3 * mean(known, 0, 0)
    assert crew[UNKNOWN_COUNT_FEATURE] == 2
    assert crew[PRIOR_GAMES_FEATURE] == 0


def test_scheduled_crew_replaces_a_stored_crew_for_the_same_game():
    refs = _refs(
        [
            ("g9", "OLD", "Old", "Ref", "2025-11-09"),
            ("g1", "NEW", "New", "Ref", "2025-11-01"),
        ]
    )
    scheduled = pd.DataFrame({"GAME_ID": ["g9"], "REF_1": ["New Ref"]})
    dates = pd.DataFrame(
        {
            "GAME_ID": ["g1", "g9"],
            "GAME_DATE": pd.to_datetime(["2025-11-01", "2025-11-09"]),
        }
    )
    crews = build_official_crews(refs, dates, scheduled)
    assert list(crews.loc[crews.GAME_ID == "g9", "OFFICIAL_ID"]) == ["NEW"]


def test_four_official_crews_are_normalised_to_three():
    history = _history([("g1", "2025-11-01", 0.0)])
    est = pd.DataFrame(
        {
            "GAME_ID": ["g1"] * 4,
            SPEC.feature: [1.0, 1.0, 1.0, 1.0],
            "PRIOR_GAMES": [5, 5, 5, 5],
        }
    )
    crew = aggregate_crew_features(est, specs=(SPEC,))
    assert float(crew[SPEC.feature].iloc[0]) == pytest.approx(3.0)
    assert history is not None


# ---------------------------------------------------------------- game history


def _team_games(games):
    """games: dicts with id, date, home, away, pts_h, pts_a, fta_h, fta_a, pf_h, pf_a, min (optional)."""
    rows = []
    for g in games:
        for side, team, pts, fta, pf in (
            (True, g["home"], g["pts_h"], g["fta_h"], g["pf_h"]),
            (False, g["away"], g["pts_a"], g["fta_a"], g["pf_a"]),
        ):
            rows.append(
                {
                    "GAME_ID": g["id"],
                    "GAME_DATE": g["date"],
                    "SEASON_YEAR": 2025,
                    "SEASON_TYPE": "Regular Season",
                    "TEAM_ID": team,
                    "HOME": side,
                    "PTS": pts,
                    "FTA": fta,
                    "PF": pf,
                    "POSS": g.get("poss", 100.0),
                    "MIN": g.get("min", 240),
                }
            )
    return pd.DataFrame(rows)


def _odds(rows):
    return pd.DataFrame(
        rows, columns=["game_id", f"total_{BOOK}_line_over", f"spread_{BOOK}_line_home"]
    )


def test_line_and_spread_errors_follow_the_repo_sign_convention():
    games = _team_games(
        [
            dict(
                id="g1",
                date="2025-11-01",
                home=1,
                away=2,
                pts_h=110,
                pts_a=102,
                fta_h=20,
                fta_a=20,
                pf_h=20,
                pf_a=20,
            ),
            dict(
                id="g2",
                date="2025-11-01",
                home=3,
                away=4,
                pts_h=95,
                pts_a=105,
                fta_h=20,
                fta_a=20,
                pf_h=20,
                pf_a=20,
            ),
        ]
    )
    # g1: home handicap -5 (home favoured by 5), wins by 8 -> covers by 3.
    # g2: home handicap +4 (away favoured by 4), loses by 10 -> away covers by 6.
    odds = _odds([("g1", 210.5, -5.0), ("g2", 200.0, 4.0)])
    hist = build_referee_game_history(
        games, odds, total_line_book=BOOK, spread_book=BOOK
    ).set_index("GAME_ID")

    assert hist.loc["g1", "LINE_ERROR"] == pytest.approx(212 - 210.5)
    assert hist.loc["g1", "SPREAD_ERROR"] == pytest.approx(3.0)
    assert hist.loc["g1", "FAV_SPREAD_ERROR"] == pytest.approx(3.0)
    assert hist.loc["g2", "SPREAD_ERROR"] == pytest.approx(-6.0)
    assert hist.loc["g2", "FAV_SPREAD_ERROR"] == pytest.approx(6.0)


def test_quantities_are_centred_on_earlier_dates_of_the_season_only():
    games = _team_games(
        [
            dict(
                id="g1",
                date="2025-11-01",
                home=1,
                away=2,
                pts_h=110,
                pts_a=100,
                fta_h=20,
                fta_a=20,
                pf_h=20,
                pf_a=20,
            ),
            dict(
                id="g2",
                date="2025-11-02",
                home=1,
                away=2,
                pts_h=110,
                pts_a=100,
                fta_h=20,
                fta_a=20,
                pf_h=20,
                pf_a=20,
            ),
        ]
    )
    odds = _odds([("g1", 200.0, -1.0), ("g2", 200.0, -1.0)])
    hist = build_referee_game_history(
        games, odds, total_line_book=BOOK, spread_book=BOOK
    ).set_index("GAME_ID")
    assert hist.loc["g1", "LINE_ERROR"] == pytest.approx(
        10.0
    )  # no earlier date: centre 0
    assert hist.loc["g2", "LINE_ERROR"] == pytest.approx(0.0)  # centred on g1's +10


def test_overtime_whistle_volume_is_scaled_to_regulation():
    base = dict(
        date="2025-11-02", home=1, away=2, pts_h=110, pts_a=100, pf_h=20, pf_a=20
    )
    games = _team_games(
        [
            dict(
                id="g0",
                date="2025-11-01",
                home=1,
                away=2,
                pts_h=100,
                pts_a=100,
                fta_h=20,
                fta_a=20,
                pf_h=20,
                pf_a=20,
            ),
            dict(id="reg", fta_h=24, fta_a=24, min=240, **base),
        ]
    )
    games_ot = _team_games(
        [
            dict(
                id="g0",
                date="2025-11-01",
                home=1,
                away=2,
                pts_h=100,
                pts_a=100,
                fta_h=20,
                fta_a=20,
                pf_h=20,
                pf_a=20,
            ),
            dict(id="reg", fta_h=26.5, fta_a=26.5, min=265, **base),
        ]
    )
    reg = build_referee_game_history(
        games, None, total_line_book=BOOK, spread_book=BOOK
    ).set_index("GAME_ID")
    ot = build_referee_game_history(
        games_ot, None, total_line_book=BOOK, spread_book=BOOK
    ).set_index("GAME_ID")
    assert ot.loc["reg", "FTA_RESID"] == pytest.approx(26.5 * 2 * 240 / 265 - 40)
    assert reg.loc["reg", "FTA_RESID"] == pytest.approx(8.0)


def _swap_home_away(team_games, odds):
    swapped = team_games.copy()
    swapped["HOME"] = ~swapped["HOME"]
    swapped_odds = odds.copy()
    swapped_odds[f"spread_{BOOK}_line_home"] = -swapped_odds[f"spread_{BOOK}_line_home"]
    return swapped, swapped_odds


def test_swapping_home_and_away_flips_only_the_home_signed_features():
    rng = np.random.default_rng(0)
    game_rows, odds_rows, ref_rows = [], [], []
    officials = ["A", "B", "C", "D", "E"]
    for i, day in enumerate(pd.date_range("2025-10-20", periods=40, freq="D")):
        gid = f"g{i:02d}"
        game_rows.append(
            dict(
                id=gid,
                date=str(day.date()),
                home=int(rng.integers(1, 4)) * 2,
                away=int(rng.integers(1, 4)) * 2 + 1,
                pts_h=int(rng.integers(95, 125)),
                pts_a=int(rng.integers(95, 125)),
                fta_h=float(rng.integers(10, 35)),
                fta_a=float(rng.integers(10, 35)),
                pf_h=float(rng.integers(15, 28)),
                pf_a=float(rng.integers(15, 28)),
                poss=float(rng.integers(92, 106)),
            )
        )
        odds_rows.append(
            (gid, float(rng.integers(200, 240)) + 0.5, float(rng.integers(-9, 9)) + 0.5)
        )
        for off in rng.choice(officials, 3, replace=False):
            ref_rows.append((gid, off, off, "x", str(day.date())))

    team_games, odds, refs = _team_games(game_rows), _odds(odds_rows), _refs(ref_rows)
    merged = team_games[team_games.HOME][["GAME_ID", "GAME_DATE"]]

    def features(tg, od):
        out = add_referee_tendency_features(
            merged,
            df_team_games=tg,
            df_refs=refs,
            df_odds=od,
            total_line_book=BOOK,
            spread_book=BOOK,
        )
        return out.set_index("GAME_ID")

    original = features(team_games, odds)
    swapped = features(*_swap_home_away(team_games, odds))
    for spec in DEFAULT_REFEREE_TENDENCY_SPECS:
        a, b = original[spec.feature].to_numpy(), swapped[spec.feature].to_numpy()
        assert np.abs(a).sum() > 0, spec.feature
        if spec.home_signed:
            np.testing.assert_allclose(a, -b, atol=1e-9, err_msg=spec.feature)
        else:
            np.testing.assert_allclose(a, b, atol=1e-9, err_msg=spec.feature)


def test_training_and_prediction_views_of_a_game_agree():
    """A scheduled game gets the same features as when it is later a stored game."""
    rows, ref_rows = [], []
    for i, day in enumerate(pd.date_range("2025-10-20", periods=12, freq="D")):
        gid = f"g{i:02d}"
        rows.append(
            dict(
                id=gid,
                date=str(day.date()),
                home=2,
                away=3,
                pts_h=100 + i,
                pts_a=100,
                fta_h=20.0 + i,
                fta_a=20.0,
                pf_h=20.0,
                pf_a=20.0,
            )
        )
        for off, name in (("1", "Ann One"), ("2", "Bob Two"), ("3", "Cy Three")):
            ref_rows.append(
                (gid, off, name.split()[0], name.split()[1], str(day.date()))
            )
    team_games, refs = _team_games(rows), _refs(ref_rows)
    odds = _odds([(r["id"], 200.5, -2.5) for r in rows])
    merged = team_games[team_games.HOME][["GAME_ID", "GAME_DATE"]]

    training = (
        add_referee_tendency_features(
            merged,
            df_team_games=team_games,
            df_refs=refs,
            df_odds=odds,
            total_line_book=BOOK,
            spread_book=BOOK,
        )
        .set_index("GAME_ID")
        .loc["g11"]
    )

    history_only = team_games[team_games.GAME_ID != "g11"]
    scheduled = pd.DataFrame(
        {
            "GAME_ID": ["g11"],
            "REF_1": ["Cy Three"],
            "REF_2": ["Ann One"],
            "REF_3": ["Bob Two"],
        }
    )
    prediction = (
        add_referee_tendency_features(
            merged,
            df_team_games=history_only,
            df_refs=refs[refs.GAME_ID != "g11"],
            df_odds=odds,
            total_line_book=BOOK,
            spread_book=BOOK,
            df_referees_scheduled=scheduled,
        )
        .set_index("GAME_ID")
        .loc["g11"]
    )

    columns = referee_tendency_feature_columns()
    pd.testing.assert_series_equal(
        training[columns].astype(float), prediction[columns].astype(float)
    )


# ---------------------------------------------------------------- output contract


def test_feature_names_are_leakage_safe_and_not_odds_shaped():
    columns = referee_tendency_feature_columns(include_same_season_variants=True) + [
        FTA_X_EXPECTED_TOTAL_FTA_FEATURE,
        FTA_X_ABS_SPREAD_FEATURE,
        POSS_X_ABS_SPREAD_FEATURE,
        FTA_X_HOME_FTA_RATE_EDGE_FEATURE,
    ]
    assert len(set(columns)) == len(columns)
    for column in columns:
        assert column.endswith("_BEFORE"), column
        assert not is_odds_shaped_column(column), column


def test_interaction_features_use_absolute_spread_and_skip_missing_inputs():
    df = pd.DataFrame(
        {
            "REF_CREW_FTA_TENDENCY_BEFORE": [2.0],
            "REF_CREW_POSS_TENDENCY_BEFORE": [-1.0],
            "STYLE_EXPECTED_TOTAL_FTA_BEFORE": [40.0],
            spread_col(BOOK): [-6.5],
            "STYLE_EXPECTED_FTA_RATE_HOME_BEFORE": [0.30],
            "STYLE_EXPECTED_FTA_RATE_AWAY_BEFORE": [0.25],
        }
    )
    out = add_referee_interaction_features(df, spread_book=BOOK).iloc[0]
    assert out[FTA_X_EXPECTED_TOTAL_FTA_FEATURE] == pytest.approx(80.0)
    assert out[FTA_X_ABS_SPREAD_FEATURE] == pytest.approx(13.0)
    assert out[POSS_X_ABS_SPREAD_FEATURE] == pytest.approx(-6.5)
    assert out[FTA_X_HOME_FTA_RATE_EDGE_FEATURE] == pytest.approx(0.1)

    partial = add_referee_interaction_features(
        df[["REF_CREW_FTA_TENDENCY_BEFORE"]], spread_book=BOOK
    )
    assert list(partial.columns) == ["REF_CREW_FTA_TENDENCY_BEFORE"]


# ---------------------------------------------------------------- review fixes


def test_same_date_rows_for_a_team_do_not_enter_each_others_expectation():
    """Team expectations use strictly earlier dates, not the previous row."""

    def fta_resid_g2(fta_in_g1):
        games = _team_games(
            [
                dict(
                    id="g0",
                    date="2025-11-01",
                    home=1,
                    away=2,
                    pts_h=100,
                    pts_a=100,
                    fta_h=20,
                    fta_a=20,
                    pf_h=20,
                    pf_a=20,
                ),
                dict(
                    id="g1",
                    date="2025-11-02",
                    home=1,
                    away=3,
                    pts_h=100,
                    pts_a=100,
                    fta_h=fta_in_g1,
                    fta_a=20,
                    pf_h=20,
                    pf_a=20,
                ),
                dict(
                    id="g2",
                    date="2025-11-02",
                    home=4,
                    away=1,
                    pts_h=100,
                    pts_a=100,
                    fta_h=20,
                    fta_a=20,
                    pf_h=20,
                    pf_a=20,
                ),
            ]
        )
        hist = build_referee_game_history(
            games, None, total_line_book=BOOK, spread_book=BOOK
        ).set_index("GAME_ID")
        return hist.loc["g2", "FTA_RESID"]

    assert fta_resid_g2(20.0) == pytest.approx(fta_resid_g2(60.0))


def _single_official_crew(n_officials):
    est = pd.DataFrame(
        {
            "GAME_ID": ["g1"] * n_officials,
            SPEC.feature: [1.0] * n_officials,
            "PRIOR_GAMES": [40] * n_officials,
        }
    )
    return aggregate_crew_features(est, specs=(SPEC,)).iloc[0]


@pytest.mark.parametrize("n_officials", [1, 2])
def test_missing_crew_slots_are_unknown_officials_not_extrapolated(n_officials):
    crew = _single_official_crew(n_officials)
    assert crew[SPEC.feature] == pytest.approx(float(n_officials))
    assert crew[UNKNOWN_COUNT_FEATURE] == 3 - n_officials
    assert crew[PRIOR_GAMES_FEATURE] == 0


def test_complete_crew_is_a_plain_sum_with_no_unknowns():
    crew = _single_official_crew(3)
    assert crew[SPEC.feature] == pytest.approx(3.0)
    assert crew[UNKNOWN_COUNT_FEATURE] == 0
    assert crew[PRIOR_GAMES_FEATURE] == 40


def _small_league(n_days=6):
    rows, ref_rows = [], []
    for i, day in enumerate(pd.date_range("2025-10-20", periods=n_days, freq="D")):
        gid = f"g{i:02d}"
        rows.append(
            dict(
                id=gid,
                date=str(day.date()),
                home=2,
                away=3,
                pts_h=100 + i,
                pts_a=100,
                fta_h=20.0 + i,
                fta_a=20.0,
                pf_h=20.0,
                pf_a=20.0,
            )
        )
        for off in ("1", "2", "3"):
            ref_rows.append((gid, off, f"F{off}", f"L{off}", str(day.date())))
    team_games = _team_games(rows)
    merged = team_games[team_games.HOME][["GAME_ID", "GAME_DATE"]]
    odds = _odds([(r["id"], 200.5, -2.5) for r in rows])
    return team_games, _refs(ref_rows), odds, merged


def _attach(team_games, refs, odds, merged, **kwargs):
    return add_referee_tendency_features(
        merged,
        df_team_games=team_games,
        df_refs=refs,
        df_odds=odds,
        total_line_book=BOOK,
        spread_book=BOOK,
        **kwargs,
    )


def test_empty_referee_source_raises_a_clear_error():
    team_games, _, odds, merged = _small_league()
    with pytest.raises(ValueError, match="No referee assignments"):
        _attach(team_games, pd.DataFrame(), odds, merged)


def test_completed_games_without_a_crew_above_the_threshold_raise():
    team_games, refs, odds, merged = _small_league()
    lagging = refs[refs.GAME_ID != "g05"]  # 1 of 6 completed games has no crew
    with pytest.raises(ValueError, match="completed games"):
        _attach(team_games, lagging, odds, merged)
    out = _attach(team_games, lagging, odds, merged, max_missing_crew_share=0.2)
    # Below a looser threshold the build proceeds and the gap stays visible.
    assert pd.isna(out.set_index("GAME_ID").loc["g05", SPEC_FEATURE_FTA])


def test_scheduled_game_without_an_assignment_only_warns():
    team_games, refs, odds, merged = _small_league()
    scheduled = pd.DataFrame(
        {"GAME_ID": ["g99"], "GAME_DATE": pd.to_datetime(["2025-11-30"])}
    )
    with pytest.warns(UserWarning, match="without a referee assignment"):
        out = _attach(team_games, refs, odds, pd.concat([merged, scheduled]))
    assert out.set_index("GAME_ID")[SPEC_FEATURE_FTA].isna().sum() == 1


SPEC_FEATURE_FTA = "REF_CREW_FTA_TENDENCY_BEFORE"
