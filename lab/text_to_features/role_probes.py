"""Do embeddings of the team-perspective rewrites know which team did what?

Two checks, cheapest first. Both compare task instructions (``INSTRUCTIONS``)
against no instruction and against a neutral control instruction of similar
length, so that the effect of a shared prefix is not read as role content.

1. **Minimal pairs** (``triplets``). Synthetic sentences: an anchor ("Target
   Team beat Opponent Team 112-104."), a paraphrase with the same roles
   ("Opponent Team lost to Target Team 112-104.") and the anchor with the roles
   swapped ("Opponent Team beat Target Team 112-104."). The swap shares more
   words with the anchor than the paraphrase does, so an embedding that only
   sees which words occur puts the swap closer. Success: the paraphrase is
   closer. That is strict; ``swap_ratio`` grades it: how far the swap moves
   the anchor per unit of the paraphrase's move. ``in_context`` wraps each
   sentence in the same neutral paragraph.
2. **Linear probes** (``view_labels``, ``ridge_oof``, ``probe_metrics``). Real
   documents, both rewrites; labels from the box score, seen from the Target
   Team (won, margin, points, shooting, regulars out). A ridge probe is fitted
   out of fold, folds grouped by game. The main metric is the **pair
   accuracy**: within a game, does the home rewrite get the higher prediction
   exactly when the home team has the higher label? An embedding that ignores
   the roles gives both rewrites the same prediction and scores 50%.

The probes only say whether the information can be read from the vector, not
whether it helps predict anything; that is a walk-forward ablation.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from nba_ou.data_processing.game_texts.embedding_inputs import (
    TARGET_VIEW_INSTRUCTION,
)
from nba_ou.data_processing.game_texts.team_perspective import (
    OPPONENT_LABEL,
    TARGET_LABEL,
)

CURRENT_INSTRUCTION = (
    "Represent this NBA game text while distinguishing Target Team from Opponent "
    "Team, preserving which team's scoring, offense, defense, pace, player "
    "availability and game outcome each fact describes, following any explicit "
    "team-role mapping, treating the two roles as distinct rather than "
    "interchangeable, and not inventing roles when no mapping is provided."
)

# "none" and "neutral" are the controls; the rest are candidates.
INSTRUCTIONS: dict[str, str | None] = {
    "none": None,
    "neutral": (
        "Given an NBA game text written by a sports journalist, represent it so "
        "that it can be stored and searched in a basketball news archive."
    ),
    "current": CURRENT_INSTRUCTION,
    "target_focus": (
        "Given an NBA game recap, represent how the Target Team played against "
        "the Opponent Team: result, scoring, defense, pace and missing players."
    ),
    "target_view": TARGET_VIEW_INSTRUCTION,
    # Mirror of target_view: if the roles matter, it should flip them.
    "opponent_view": (
        "Represent this NBA game recap from the Opponent Team's point of view."
    ),
    "next_game": (
        "Given an NBA game recap, represent what it reveals about the Target Team "
        "that would help predict its points scored and allowed in its next game."
    ),
}
CONTROLS = ("none", "neutral")
ROLE_VIEWS = ("home_target", "away_target")

# ---------------------------------------------------------------------------
# 1. Minimal pairs
# ---------------------------------------------------------------------------

# (facet, anchor, same roles reworded, anchor with the roles swapped)
TRIPLET_TEMPLATES = (
    ("result", "{T} beat {O} {w}-{l}.", "{O} lost to {T} {w}-{l}.", "{O} beat {T} {w}-{l}."),
    (
        "result",
        "{T} won {w}-{l} over {O}.",
        "{O} fell {l}-{w} to {T}.",
        "{O} won {w}-{l} over {T}.",
    ),
    (
        "comeback",
        "{T} rallied from {n} points down to beat {O}.",
        "{O} blew a {n}-point lead and lost to {T}.",
        "{O} rallied from {n} points down to beat {T}.",
    ),
    (
        "scoring",
        "{T} scored {p} points on {pct}% shooting against {O}.",
        "{T} shot {pct}% from the field and finished with {p} points against {O}.",
        "{O} scored {p} points on {pct}% shooting against {T}.",
    ),
    (
        "scoring",
        "{T} made {k} 3-pointers against {O}.",
        "{T} hit {k} shots from beyond the arc against {O}.",
        "{O} made {k} 3-pointers against {T}.",
    ),
    (
        "defense",
        "{T}'s defense held {O} to {pct2}% shooting.",
        "{O} shot only {pct2}% against {T}'s defense.",
        "{O}'s defense held {T} to {pct2}% shooting.",
    ),
    (
        "defense",
        "{T} forced {k2} turnovers by {O}.",
        "{O} committed {k2} turnovers against {T}'s pressure.",
        "{O} forced {k2} turnovers by {T}.",
    ),
    (
        "rebounding",
        "{T} outrebounded {O} {r1}-{r2}.",
        "{O} lost the battle on the boards {r2}-{r1} to {T}.",
        "{O} outrebounded {T} {r1}-{r2}.",
    ),
    (
        "availability",
        "{T} played without its leading scorer, who was out with a {inj} injury.",
        "{T}'s leading scorer missed the game with a {inj} injury.",
        "{O} played without its leading scorer, who was out with a {inj} injury.",
    ),
    (
        "availability",
        "{P} ({T} player) sat out against {O} with a {inj} injury.",
        "{T} were missing {P} against {O} because of a {inj} injury.",
        "{P} ({O} player) sat out against {T} with a {inj} injury.",
    ),
    (
        "player",
        "{P} ({T} player) scored {p2} points against {O}.",
        "{T} got {p2} points from {P} against {O}.",
        "{P} ({O} player) scored {p2} points against {T}.",
    ),
)
TRIPLET_VALUES = (
    {"w": 112, "l": 104, "n": 15, "p": 128, "pct": 54, "k": 18, "pct2": 39,
     "k2": 21, "r1": 52, "r2": 38, "inj": "sprained ankle", "P": "Brunson", "p2": 38},
    {"w": 99, "l": 97, "n": 22, "p": 141, "pct": 58, "k": 23, "pct2": 36,
     "k2": 19, "r1": 49, "r2": 41, "inj": "hamstring", "P": "Edwards", "p2": 41},
    {"w": 130, "l": 118, "n": 11, "p": 109, "pct": 49, "k": 16, "pct2": 41,
     "k2": 17, "r1": 55, "r2": 40, "inj": "knee", "P": "Haliburton", "p2": 29},
    {"w": 105, "l": 88, "n": 18, "p": 120, "pct": 51, "k": 20, "pct2": 38,
     "k2": 24, "r1": 47, "r2": 35, "inj": "back", "P": "Tatum", "p2": 33},
)  # fmt: skip
CONTEXT_BEFORE = (
    "In this text, {T} is the home team and {O} is the visiting team.\n\n"
    "The game was played in front of a sellout crowd on a Friday night. Both "
    "teams were coming off a day of rest and had met twice earlier in the "
    "season. The first quarter was played at a quick pace, with several lead "
    "changes and few stoppages. "
)
CONTEXT_AFTER = (
    " Both coaches used their full rotations in the second half. The teams meet "
    "again next month, and each plays its next game on Sunday."
)


@dataclass(frozen=True)
class Triplet:
    facet: str
    anchor: str
    same: str  # the anchor reworded, same roles
    swapped: str  # the anchor with the two labels exchanged


def triplets(
    *,
    in_context: bool = False,
    target_label: str = TARGET_LABEL,
    opponent_label: str = OPPONENT_LABEL,
) -> list[Triplet]:
    """Every template filled with every value set."""
    labels = {"T": target_label, "O": opponent_label}
    out = []
    for values in TRIPLET_VALUES:
        for facet, *texts in TRIPLET_TEMPLATES:
            filled = [text.format(**labels, **values) for text in texts]
            if in_context:
                before = CONTEXT_BEFORE.format(**labels)
                filled = [before + text + CONTEXT_AFTER for text in filled]
            out.append(Triplet(facet, *filled))
    return out


def triplet_texts(items: Sequence[Triplet]) -> list[str]:
    """Anchor, same, swapped for each triplet, in order (3 texts per triplet)."""
    return [text for t in items for text in (t.anchor, t.same, t.swapped)]


def triplet_scores(items: Sequence[Triplet], vectors: np.ndarray) -> pd.DataFrame:
    """One row per triplet; ``vectors`` are L2-normalized, as ``triplet_texts``."""
    vectors = np.asarray(vectors).reshape(len(items), 3, -1)
    same = np.einsum("nd,nd->n", vectors[:, 0], vectors[:, 1])
    swapped = np.einsum("nd,nd->n", vectors[:, 0], vectors[:, 2])
    return pd.DataFrame(
        {
            "facet": [t.facet for t in items],
            "cos_same": same,
            "cos_swapped": swapped,
            "margin": same - swapped,
            "correct": same > swapped,
            # How far the swap moves the anchor, per unit of the paraphrase's
            # move: 0 = the swap is invisible. Unchanged when an instruction
            # compresses every distance by the same factor.
            "swap_ratio": (1 - swapped) / (1 - same),
        }
    )


# ---------------------------------------------------------------------------
# 2. Labels from the box score
# ---------------------------------------------------------------------------

GAME_TYPES = ("002", "004", "005")  # regular season, playoffs, play-in
WINDOW = 10  # team games before the game that define a regular
MIN_PLAYED = 5  # of those, games the player must have played
MIN_MINUTES = 24.0  # average minutes in the games he played
TEAM_STATS = ("pts", "fg_pct", "fg3m", "reb", "tov", "pace")

# Labels seen from the Target Team (``view_labels``). Role labels change when
# the roles swap; shared labels are the same for both rewrites (controls).
ROLE_LABELS = (
    "target_won",
    "target_margin",
    "target_points",
    "opponent_points",
    "target_fg_pct",
    "opponent_fg_pct",
    "fg3m_diff",
    "reb_diff",
    "tov_diff",
    "target_regulars_out",
    "opponent_regulars_out",
)
SHARED_LABELS = ("total_points", "pace")
BINARY_LABELS = ("target_won",)


def parse_minutes(value) -> float:
    """Box-score minutes as a number: "25:30", "22.000000:10", None -> 0."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    minutes, _, seconds = text.partition(":")
    return float(minutes) + (float(seconds) / 60 if seconds else 0.0)


def team_game_order(team_games: pd.DataFrame) -> pd.DataFrame:
    """``team_id, game_id, season_year, game_index``: the team's games of the
    season in date order, counted from 0 (only ``GAME_TYPES``)."""
    games = team_games[team_games["game_id"].str[:3].isin(GAME_TYPES)]
    games = games[["team_id", "game_id", "season_year", "game_date"]].copy()
    games["team_id"] = games["team_id"].astype(str)
    games = games.sort_values(["team_id", "season_year", "game_date", "game_id"])
    games["game_index"] = games.groupby(["team_id", "season_year"]).cumcount()
    return games.drop(columns="game_date").reset_index(drop=True)


def regulars_out(
    order: pd.DataFrame,
    minutes: pd.DataFrame,
    listed: pd.DataFrame,
    *,
    window: int = WINDOW,
    min_played: int = MIN_PLAYED,
    min_minutes: float = MIN_MINUTES,
) -> pd.DataFrame:
    """Regulars of each team who did not play: ``team_id, game_id, regulars_out``.

    A regular, at a game, played at least ``min_played`` of the team's previous
    ``window`` games of the season, averaging ``min_minutes`` in them. He is out
    when the game lists him for the team (box-score row or inactive list,
    ``listed``: ``game_id, team_id, player_id``) and he logged no minutes
    (``minutes``: ``game_id, team_id, player_id, minutes``). A player the game
    does not list (traded, waived) is not counted.
    """
    keys = ["team_id", "game_id"]
    played = minutes.assign(team_id=minutes["team_id"].astype(str))
    played = played.merge(order, on=keys)
    listed = listed.assign(team_id=listed["team_id"].astype(str))
    listed = listed[keys + ["player_id"]].drop_duplicates()

    # Every player the team ever listed in the season, at every team game.
    roster = played[["team_id", "season_year", "player_id"]].drop_duplicates()
    grid = roster.merge(order, on=["team_id", "season_year"])
    grid = grid.merge(
        played[keys + ["player_id", "minutes"]], on=keys + ["player_id"], how="left"
    )
    grid["minutes"] = grid["minutes"].fillna(0.0)
    grid = grid.sort_values(["team_id", "season_year", "player_id", "game_index"])
    groups = grid.groupby(["team_id", "season_year", "player_id"])["minutes"]
    prior = lambda s: s.shift(1).rolling(window, min_periods=1)  # noqa: E731
    grid["prior_played"] = groups.transform(lambda s: prior(s.gt(0).astype(float)).sum())
    grid["prior_minutes"] = groups.transform(lambda s: prior(s).sum())
    regular = (grid["prior_played"] >= min_played) & (
        grid["prior_minutes"] >= min_minutes * grid["prior_played"]
    )
    out = grid[regular & grid["minutes"].eq(0)]
    out = out.merge(listed, on=keys + ["player_id"])
    counts = out.groupby(keys).size().rename("regulars_out")
    result = order[keys].merge(counts, on=keys, how="left")
    result["regulars_out"] = result["regulars_out"].fillna(0).astype(int)
    return result


def game_labels(
    team_games: pd.DataFrame, out: pd.DataFrame, order: pd.DataFrame
) -> pd.DataFrame:
    """One row per game: ``home_*`` / ``away_*`` box-score stats, regulars out
    and the number of earlier season games of each team (``*_prior_games``)."""
    games = team_games[team_games["game_id"].str[:3].isin(GAME_TYPES)].copy()
    games["team_id"] = games["team_id"].astype(str)
    games = games.merge(out, on=["team_id", "game_id"], how="left")
    games = games.merge(order[["team_id", "game_id", "game_index"]], on=["team_id", "game_id"])
    columns = list(TEAM_STATS) + ["regulars_out", "game_index"]
    sides = {}
    for side, is_home in (("home", True), ("away", False)):
        part = games[games["home"].astype(bool) == is_home].set_index("game_id")
        part = part[columns].rename(columns={"game_index": "prior_games"})
        sides[side] = part.add_prefix(f"{side}_")
    labels = sides["home"].join(sides["away"], how="inner")
    return labels.reset_index()


def view_labels(documents: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Two rows per document, ``home_target`` and ``away_target``: ``doc_index``,
    ``game_id``, ``source``, ``view``, ``target_is_home`` and every label seen
    from the Target Team (``ROLE_LABELS`` + ``SHARED_LABELS``)."""
    frame = documents[["doc_index", "game_id", "source"]].merge(
        labels, on="game_id", how="left", validate="many_to_one"
    )
    if frame["home_pts"].isna().any():
        missing = frame.loc[frame["home_pts"].isna(), "game_id"].tolist()
        raise ValueError(f"No box-score labels for games {missing[:5]}")
    rows = []
    for view, t, o in (("home_target", "home", "away"), ("away_target", "away", "home")):
        part = frame[["doc_index", "game_id", "source"]].copy()
        part["view"] = view
        part["target_is_home"] = view == "home_target"
        margin = frame[f"{t}_pts"] - frame[f"{o}_pts"]
        part["target_won"] = (margin > 0).astype(float)
        part["target_margin"] = margin
        part["target_points"] = frame[f"{t}_pts"]
        part["opponent_points"] = frame[f"{o}_pts"]
        part["target_fg_pct"] = frame[f"{t}_fg_pct"]
        part["opponent_fg_pct"] = frame[f"{o}_fg_pct"]
        part["fg3m_diff"] = frame[f"{t}_fg3m"] - frame[f"{o}_fg3m"]
        part["reb_diff"] = frame[f"{t}_reb"] - frame[f"{o}_reb"]
        part["tov_diff"] = frame[f"{t}_tov"] - frame[f"{o}_tov"]
        part["target_regulars_out"] = frame[f"{t}_regulars_out"]
        part["opponent_regulars_out"] = frame[f"{o}_regulars_out"]
        part["total_points"] = frame["home_pts"] + frame["away_pts"]
        part["pace"] = frame["home_pace"]
        rows.append(part)
    frame = pd.concat(rows, ignore_index=True)
    frame["view_order"] = frame["view"].map(ROLE_VIEWS.index)
    frame = frame.sort_values(["doc_index", "view_order"], kind="stable")
    return frame.drop(columns="view_order").reset_index(drop=True)


def eligible_games(labels: pd.DataFrame, min_prior_games: int = WINDOW) -> set[str]:
    """Games where both teams have ``min_prior_games`` earlier season games, so
    that "regular" is defined on a full window."""
    keep = (labels["home_prior_games"] >= min_prior_games) & (
        labels["away_prior_games"] >= min_prior_games
    )
    return set(labels.loc[keep, "game_id"])


def key_game_id(key: str) -> str:
    """``.../0022400374.json.gz`` -> ``0022400374``."""
    return key.rsplit("/", 1)[-1].split(".", 1)[0]


def sample_keys(
    keys_by_source: dict[str, list[str]],
    games: set[str],
    per_source: int,
    seed: int,
) -> list[tuple[str, str]]:
    """``(source, key)`` pairs: ``per_source`` random keys of each source whose
    game is in ``games``."""
    rng = random.Random(seed)
    picked = []
    for source, keys in keys_by_source.items():
        pool = sorted(key for key in keys if key_game_id(key) in games)
        if len(pool) < per_source:
            raise ValueError(f"{source}: {len(pool)} eligible documents < {per_source}")
        picked += [(source, key) for key in rng.sample(pool, per_source)]
    return picked


# ---------------------------------------------------------------------------
# 3. Probes
# ---------------------------------------------------------------------------


def group_folds(groups: Sequence, n_splits: int, seed: int = 0) -> np.ndarray:
    """Fold number per row; every row of a group (game) in the same fold."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    order = np.random.default_rng(seed).permutation(len(unique))
    fold_of = {group: i % n_splits for i, group in zip(order, unique, strict=True)}
    return np.array([fold_of[group] for group in groups])


def _kernel_ridge(
    X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, alphas: np.ndarray
) -> np.ndarray:
    """Predictions ``(len(alphas), n_test, n_labels)``; ridge with intercept on
    features standardized with the training rows, solved in the kernel form."""
    mean, std = X_train.mean(0), X_train.std(0)
    std[std == 0] = 1.0
    A, B = (X_train - mean) / std, (X_test - mean) / std
    y_mean = Y_train.mean(0)
    values, vectors = np.linalg.eigh(A @ A.T)
    projected = vectors.T @ (Y_train - y_mean)
    cross = B @ A.T @ vectors
    return np.stack(
        [cross @ (projected / (values + alpha)[:, None]) + y_mean for alpha in alphas]
    )


def ridge_oof(
    X: np.ndarray,
    Y: np.ndarray,
    groups: Sequence,
    *,
    n_splits: int = 5,
    alphas: Sequence[float] | None = None,
    seed: int = 0,
    apply: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Out-of-fold predictions of every label (columns of ``Y``) and the alpha
    chosen per outer fold and label. Alpha is chosen per label on inner folds
    of the training rows, also grouped; grid relative to the feature count.

    ``apply`` (same rows as ``X``, another embedding of them): the probe fitted
    on ``X`` predicts the held-out rows of ``apply`` instead (transfer)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    target = X if apply is None else np.asarray(apply, dtype=np.float64)
    if target.shape != X.shape:
        raise ValueError(f"apply has shape {target.shape}, X has {X.shape}")
    if Y.ndim == 1:
        Y = Y[:, None]
    if alphas is None:
        alphas = X.shape[1] * np.logspace(-3, 2, 11)
    alphas = np.asarray(alphas, dtype=np.float64)
    groups = np.asarray(groups)
    folds = group_folds(groups, n_splits, seed)
    predictions = np.full(Y.shape, np.nan)
    chosen = []
    for fold in range(n_splits):
        train, test = folds != fold, folds == fold
        inner = group_folds(groups[train], n_splits, seed + 1 + fold)
        errors = np.zeros((len(alphas), Y.shape[1]))
        X_train, Y_train = X[train], Y[train]
        for inner_fold in range(n_splits):
            fit, held = inner != inner_fold, inner == inner_fold
            pred = _kernel_ridge(X_train[fit], Y_train[fit], X_train[held], alphas)
            errors += ((pred - Y_train[held]) ** 2).sum(1)
        best = errors.argmin(0)
        pred = _kernel_ridge(X_train, Y_train, target[test], alphas)
        predictions[test] = pred[best, :, np.arange(Y.shape[1])].T
        chosen.append(alphas[best])
    return predictions, pd.DataFrame(chosen, columns=range(Y.shape[1]))


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve (ties count half)."""
    y = np.asarray(y).astype(bool)
    ranks = pd.Series(score).rank().to_numpy()
    positives, negatives = y.sum(), (~y).sum()
    if not positives or not negatives:
        return float("nan")
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def pair_correct(frame: pd.DataFrame, label: str, prediction: np.ndarray) -> pd.Series:
    """Per document, 1 when the home rewrite's prediction is above the away
    rewrite's exactly when the home label is, 0 when the other way round and
    0.5 for equal predictions; documents whose two labels are equal are left
    out. ``frame`` is in ``view_labels`` order (home, away per document)."""
    values = frame[label].to_numpy().reshape(-1, 2)
    pred = np.asarray(prediction).reshape(-1, 2)
    label_sign = np.sign(values[:, 0] - values[:, 1])
    pred_sign = np.sign(pred[:, 0] - pred[:, 1])
    keep = label_sign != 0
    doc_index = frame["doc_index"].to_numpy().reshape(-1, 2)[:, 0]
    correct = np.where(pred_sign == 0, 0.5, (pred_sign == label_sign).astype(float))
    return pd.Series(correct[keep], index=doc_index[keep])


def wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def probe_metrics(
    frame: pd.DataFrame, predictions: np.ndarray, labels: Sequence[str]
) -> pd.DataFrame:
    """Per label: out-of-fold Pearson r (AUC for binary labels) and, for role
    labels, pair accuracy with its 95% Wilson interval."""
    rows = []
    for index, label in enumerate(labels):
        y, pred = frame[label].to_numpy(dtype=float), predictions[:, index]
        row = {"label": label, "kind": "role" if label in ROLE_LABELS else "shared"}
        if label in BINARY_LABELS:
            row["auc"] = auc(y, pred)
        varies = y.std() > 0 and pred.std() > 0
        row["r"] = float(np.corrcoef(y, pred)[0, 1]) if varies else float("nan")
        if label in ROLE_LABELS:
            correct = pair_correct(frame, label, pred)
            row["pair_accuracy"] = float(correct.mean())
            row["pair_low"], row["pair_high"] = wilson(correct.sum(), len(correct))
            row["pairs"] = len(correct)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(
    a: pd.Series, b: pd.Series, *, n: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of ``a - b`` (per-document ``pair_correct`` outcomes on the same
    documents) and its 95% bootstrap interval."""
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    diff = (joined["a"].astype(float) - joined["b"].astype(float)).to_numpy()
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(1)
    return float(diff.mean()), float(np.quantile(means, 0.025)), float(
        np.quantile(means, 0.975)
    )


def side_features(frame: pd.DataFrame) -> np.ndarray:
    """Baseline: only which side the Target Team is (+1 home, -1 away)."""
    return np.where(frame["target_is_home"], 1.0, -1.0)[:, None]


def raw_side_features(frame: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    """Baseline: the raw text's embedding and its product with the side, so a
    linear probe can read "home team's X" and flip it for the away view.

    ``raw`` is one row per document, indexed by ``doc_index``."""
    vectors = np.asarray(raw)[frame["doc_index"].to_numpy()]
    side = side_features(frame)
    return np.hstack([vectors, side * vectors, side])


# ---------------------------------------------------------------------------
# 4. Reading a run (``run_role_probes``)
# ---------------------------------------------------------------------------


def triplet_summary(model_dir, conditions: Sequence[str]) -> pd.DataFrame:
    """One row per condition, setting and facet (plus ``all``): triplets,
    accuracy, mean margin and median swap ratio. Conditions without files are
    skipped."""
    rows = []
    for condition in conditions:
        for setting, in_context in (("short", False), ("context", True)):
            path = model_dir / f"triplets_{setting}_{condition}.npy"
            if not path.exists():
                continue
            scores = triplet_scores(triplets(in_context=in_context), np.load(path))
            for facet, part in [("all", scores), *scores.groupby("facet", sort=False)]:
                rows.append(
                    {
                        "condition": condition,
                        "setting": setting,
                        "facet": facet,
                        "triplets": len(part),
                        "accuracy": part["correct"].mean(),
                        "margin": part["margin"].mean(),
                        "swap_ratio": part["swap_ratio"].median(),
                    }
                )
    return pd.DataFrame(rows)


def document_vectors(model_dir, condition: str, n_docs: int) -> dict[str, np.ndarray]:
    """``{view: (n_docs, dim)}``; "none" also holds the raw text."""
    views = ("raw", *ROLE_VIEWS) if condition == "none" else ROLE_VIEWS
    flat = np.load(model_dir / f"documents_{condition}.npy")
    cube = flat.reshape(n_docs, len(views), -1)
    return {view: cube[:, i] for i, view in enumerate(views)}


def view_matrix(frame: pd.DataFrame, vectors: dict[str, np.ndarray]) -> np.ndarray:
    """The rewrite's vector for every row of ``frame`` (``view_labels``)."""
    stacked = np.stack([vectors[view] for view in ROLE_VIEWS], axis=1)
    position = frame["view"].map(ROLE_VIEWS.index).to_numpy()
    return stacked[frame["doc_index"].to_numpy(), position]


def run_probes(
    frame: pd.DataFrame,
    features: dict[str, np.ndarray],
    labels: Sequence[str] = ROLE_LABELS + SHARED_LABELS,
    *,
    n_splits: int = 5,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.Series]]]:
    """``probe_metrics`` of every feature set (same folds for all), and the
    per-document ``pair_correct`` outcomes by feature set and role label."""
    Y = frame[list(labels)].to_numpy(dtype=float)
    metrics, outcomes = [], {}
    for name, X in features.items():
        pred, _ = ridge_oof(X, Y, frame["game_id"], n_splits=n_splits, seed=seed)
        table = probe_metrics(frame, pred, labels)
        table.insert(0, "features", name)
        metrics.append(table)
        outcomes[name] = {
            label: pair_correct(frame, label, pred[:, i])
            for i, label in enumerate(labels)
            if label in ROLE_LABELS
        }
    return pd.concat(metrics, ignore_index=True), outcomes


def run_transfer(
    frame: pd.DataFrame,
    fit: np.ndarray,
    apply: dict[str, np.ndarray],
    labels: Sequence[str] = ROLE_LABELS,
    *,
    n_splits: int = 5,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.Series]]]:
    """Fit the probes on one condition's vectors and read other conditions with
    them (same folds as ``run_probes``). A condition with the same role
    reading keeps the pair accuracy high; one that swaps the roles drives it
    below 50%. Returns ``probe_metrics`` per applied condition and the
    per-document ``pair_correct`` outcomes."""
    Y = frame[list(labels)].to_numpy(dtype=float)
    metrics, outcomes = [], {}
    for name, X in apply.items():
        pred, _ = ridge_oof(
            fit, Y, frame["game_id"], n_splits=n_splits, seed=seed, apply=X
        )
        table = probe_metrics(frame, pred, labels)
        table.insert(0, "applied_to", name)
        metrics.append(table)
        outcomes[name] = {
            label: pair_correct(frame, label, pred[:, i])
            for i, label in enumerate(labels)
            if label in ROLE_LABELS
        }
    return pd.concat(metrics, ignore_index=True), outcomes


def role_direction_cosine(
    reference: dict[str, np.ndarray], other: dict[str, np.ndarray]
) -> np.ndarray:
    """Per document, cosine between the home-minus-away rewrite difference of
    two conditions. Both differences share the label swap in the text, so
    they agree (cosine > 0) unless the instruction turns the roles around."""
    a = reference["home_target"] - reference["away_target"]
    b = other["home_target"] - other["away_target"]
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
