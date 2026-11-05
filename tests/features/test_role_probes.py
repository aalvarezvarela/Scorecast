import numpy as np
import pandas as pd
import pytest

from lab.text_to_features import role_probes as rp


def swap(text: str, a: str, b: str) -> str:
    return text.replace(a, "\0").replace(b, a).replace("\0", b)


def test_swapped_is_the_anchor_with_the_labels_exchanged():
    items = rp.triplets(target_label="AAA", opponent_label="BBB")
    assert len(items) == len(rp.TRIPLET_TEMPLATES) * len(rp.TRIPLET_VALUES)
    for item in items:
        assert item.swapped == swap(item.anchor, "AAA", "BBB")
        assert item.same != item.anchor


def test_in_context_wraps_all_three_texts_in_the_same_paragraph():
    short = rp.triplets()
    long = rp.triplets(in_context=True)
    for s, item in zip(short, long, strict=True):
        for plain, wrapped in zip(
            (s.anchor, s.same, s.swapped),
            (item.anchor, item.same, item.swapped),
            strict=True,
        ):
            prefix, _, suffix = wrapped.partition(plain)
            assert prefix.startswith("In this text, Target Team is the home team")
            assert suffix == rp.CONTEXT_AFTER


def test_triplet_scores_prefer_the_closer_text():
    items = rp.triplets()[:2]
    e = np.eye(3)
    near = (e[0] + 0.1 * e[1]) / np.linalg.norm(e[0] + 0.1 * e[1])
    vectors = np.stack([e[0], near, e[2], e[0], e[2], near])
    scores = rp.triplet_scores(items, vectors)
    assert scores["correct"].tolist() == [True, False]
    assert scores["margin"].iloc[0] == pytest.approx(-scores["margin"].iloc[1])


@pytest.mark.parametrize(
    ("value", "minutes"),
    [("25:30", 25.5), ("22.000000:10", 22 + 10 / 60), (None, 0.0), ("", 0.0)],
)
def test_parse_minutes(value, minutes):
    assert rp.parse_minutes(value) == pytest.approx(minutes)


def team_season(n_games: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_id": "1",
            "game_id": [f"00224{i:05d}" for i in range(n_games)],
            "season_year": 2024,
            "game_index": range(n_games),
        }
    )


def test_regulars_out_counts_listed_regulars_without_minutes():
    order = team_season(13)
    games = order["game_id"].tolist()
    rows = []
    for i, game in enumerate(games):
        if i < 10:
            rows.append((game, "1", "star", 32.0))
        elif i == 10:
            rows.append((game, "1", "star", 0.0))  # DNP in the box score
        rows.append((game, "1", "bench", 9.0))
    minutes = pd.DataFrame(rows, columns=["game_id", "team_id", "player_id", "minutes"])
    # Game 11: inactive list; game 12: not listed at all (traded).
    listed = pd.concat(
        [
            minutes[["game_id", "team_id", "player_id"]],
            pd.DataFrame({"game_id": [games[11]], "team_id": ["1"], "player_id": ["star"]}),
        ]
    )
    out = rp.regulars_out(order, minutes, listed).set_index("game_id")["regulars_out"]
    assert out[games[:10]].eq(0).all()
    assert out[games[10]] == 1
    assert out[games[11]] == 1
    assert out[games[12]] == 0


def test_regulars_out_needs_enough_games_played():
    order = team_season(5)
    games = order["game_id"].tolist()
    minutes = pd.DataFrame(
        [(g, "1", "star", 35.0) for g in games[:4]] + [(games[4], "1", "star", 0.0)],
        columns=["game_id", "team_id", "player_id", "minutes"],
    )
    listed = minutes[["game_id", "team_id", "player_id"]]
    out = rp.regulars_out(order, minutes, listed)
    assert out["regulars_out"].eq(0).all()  # 4 games < MIN_PLAYED


def labels_for(home_pts: int, away_pts: int) -> pd.DataFrame:
    row = {"game_id": "g1", "home_prior_games": 20, "away_prior_games": 20}
    for side, pts in (("home", home_pts), ("away", away_pts)):
        row.update(
            {
                f"{side}_pts": pts,
                f"{side}_fg_pct": 0.5 if side == "home" else 0.4,
                f"{side}_fg3m": 12 if side == "home" else 9,
                f"{side}_reb": 44,
                f"{side}_tov": 13,
                f"{side}_pace": 99.0,
                f"{side}_regulars_out": 0 if side == "home" else 2,
            }
        )
    return pd.DataFrame([row])


def test_view_labels_flip_roles_and_keep_shared_labels():
    documents = pd.DataFrame({"doc_index": [0], "game_id": ["g1"], "source": ["report"]})
    frame = rp.view_labels(documents, labels_for(110, 100))
    home, away = frame.iloc[0], frame.iloc[1]
    assert (home["view"], away["view"]) == rp.ROLE_VIEWS
    assert (home["target_won"], away["target_won"]) == (1.0, 0.0)
    assert home["target_margin"] == -away["target_margin"] == 10
    assert home["target_points"] == away["opponent_points"] == 110
    assert home["fg3m_diff"] == -away["fg3m_diff"] == 3
    assert (home["target_regulars_out"], away["target_regulars_out"]) == (0, 2)
    assert home["total_points"] == away["total_points"] == 210


def test_view_labels_raise_for_a_game_without_labels():
    documents = pd.DataFrame({"doc_index": [0], "game_id": ["g2"], "source": ["report"]})
    with pytest.raises(ValueError, match="g2"):
        rp.view_labels(documents, labels_for(110, 100))


def test_group_folds_keep_each_game_in_one_fold():
    groups = np.repeat(np.arange(50), 2)
    folds = rp.group_folds(groups, 5)
    assert (folds.reshape(-1, 2)[:, 0] == folds.reshape(-1, 2)[:, 1]).all()
    assert set(np.bincount(folds)) == {20}


def test_ridge_oof_recovers_a_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 40))
    y = X[:, 0] - 2 * X[:, 1] + 0.1 * rng.normal(size=200)
    noise = rng.normal(size=200)
    pred, alphas = rp.ridge_oof(X, np.column_stack([y, noise]), np.arange(200))
    assert np.corrcoef(pred[:, 0], y)[0, 1] > 0.95
    assert abs(np.corrcoef(pred[:, 1], noise)[0, 1]) < 0.3
    assert alphas.shape == (5, 2)


def test_pair_correct_counts_ties_as_half():
    frame = pd.DataFrame(
        {
            "doc_index": [0, 0, 1, 1, 2, 2, 3, 3],
            "target_won": [1, 0, 0, 1, 1, 0, 1, 1],
        }
    )
    pred = np.array([0.9, 0.1, 0.8, 0.2, 0.5, 0.5, 0.3, 0.6])
    correct = rp.pair_correct(frame, "target_won", pred)
    assert correct.to_dict() == {0: 1.0, 1: 0.0, 2: 0.5}  # doc 3: equal labels


def test_auc_matches_hand_count():
    assert rp.auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])) == 0.75


def test_paired_bootstrap_of_identical_outcomes_is_zero():
    a = pd.Series([1.0, 0.0, 1.0, 1.0])
    assert rp.paired_bootstrap(a, a) == (0.0, 0.0, 0.0)


def test_raw_side_features_flip_with_the_side():
    frame = pd.DataFrame({"doc_index": [0, 0], "target_is_home": [True, False]})
    raw = np.array([[1.0, 2.0]])
    X = rp.raw_side_features(frame, raw)
    assert X.tolist() == [[1, 2, 1, 2, 1], [1, 2, -1, -2, -1]]


def test_sample_keys_only_draws_eligible_games():
    keys = {
        "report": [f"r/season=2024-25/reports/{g}.json.gz" for g in ("a", "b", "c")],
        "transcript": [f"t/transcripts/{g}.json.gz" for g in ("a", "c")],
    }
    picked = rp.sample_keys(keys, {"a", "c"}, 2, seed=1)
    assert sorted(rp.key_game_id(k) for _, k in picked) == ["a", "a", "c", "c"]
    with pytest.raises(ValueError, match="report"):
        rp.sample_keys(keys, {"a"}, 2, seed=1)


def test_ridge_oof_applied_to_the_same_matrix_is_the_plain_probe():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 8))
    y = X[:, 0] + 0.1 * rng.normal(size=60)
    groups = np.repeat(np.arange(30), 2)
    plain, _ = rp.ridge_oof(X, y, groups)
    applied, _ = rp.ridge_oof(X, y, groups, apply=X.copy())
    assert np.allclose(plain, applied)
    flipped, _ = rp.ridge_oof(X, y, groups, apply=-X)
    assert np.corrcoef(flipped[:, 0], y)[0, 1] < -0.9


def test_role_direction_cosine_sees_a_swap():
    rng = np.random.default_rng(2)
    home, away = rng.normal(size=(4, 6)), rng.normal(size=(4, 6))
    reference = {"home_target": home, "away_target": away}
    swapped = {"home_target": away, "away_target": home}
    assert np.allclose(rp.role_direction_cosine(reference, reference), 1.0)
    assert np.allclose(rp.role_direction_cosine(reference, swapped), -1.0)
