"""Is there repeatable content in the presser vectors beyond what is known?

"Known" is everything already available as numbers: the transcript's own
properties (length, punctuation, season), the box score, the regulars out,
the team's record and the closing line. After removing it (and, at the last
level, who the teams and the coach are), what is left of a vector is judged
without any label:

- **halves**: two halves of the same presser (odd and even chunks) should
  have closer residuals than halves of different pressers. An upper bound:
  it also holds for a coach's mood or wording that says nothing of the game.
- **game**: the two coaches of the same game are independent observers of
  it. If their residuals are closer than those of coaches of different
  games, the residual holds something about the game that the numbers do
  not. This is the measure that matters.

Both are reported as an AUC: the probability that a matched pair is more
similar (cosine) than an unmatched one; 0.5 is nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from lab.coach_pressers.embedding_analysis import analysis as A
from lab.text_to_features import role_probes as rp

LEVELS = ("nothing", "transcript", "known", "identity")


def normalize(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def chunk_means(
    chunks: pd.DataFrame, vectors: np.ndarray, n_pressers: int
) -> dict[str, np.ndarray]:
    """``all``: mean of every chunk of a presser; ``odd`` / ``even``: of every
    other chunk (NaN rows for pressers with one chunk). Rows are L2-normalized."""
    presser = chunks["presser"].to_numpy()
    parity = chunks["chunk"].to_numpy() % 2
    vectors = np.asarray(vectors, dtype=np.float32)
    out = {}
    for name, keep in (
        ("all", parity >= 0),
        ("even", parity == 0),
        ("odd", parity == 1),
    ):
        total = np.zeros((n_pressers, vectors.shape[1]), np.float64)
        np.add.at(total, presser[keep], vectors[keep])
        count = np.bincount(presser[keep], minlength=n_pressers)
        total[count == 0] = np.nan
        out[name] = normalize(total)
    return out


def controls(frame: pd.DataFrame, level: str) -> np.ndarray | None:
    """The variables removed at each level of ``LEVELS`` (cumulative)."""
    if level == "nothing":
        return None
    blocks = [A.design(frame, A.NUISANCE), A.one_hot(frame["season"])]
    if level in ("known", "identity"):
        blocks.append(A.design(frame, A.label_names()))
    if level == "identity":
        blocks += [A.one_hot(frame[name]) for name in ("team", "opponent", "coach")]
    return np.hstack(blocks)


def residual(X: np.ndarray, C: np.ndarray | None) -> np.ndarray:
    """``X`` centered, minus its least-squares fit on ``C``."""
    X = X - X.mean(0)
    if C is None:
        return X
    C = C - C.mean(0)
    C = C[:, C.std(0) > 0]
    coefficients, *_ = np.linalg.lstsq(C, X, rcond=None)
    return X - C @ coefficients


def matched_rates(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row i of ``a`` and row i of ``b`` are a pair: per pair, the share of
    the other rows of ``b`` that are less similar (cosine) to ``a[i]``."""
    similarity = normalize(a) @ normalize(b).T
    own = np.diag(similarity)[:, None]
    wins = (own > similarity).sum(1) + 0.5 * ((own == similarity).sum(1) - 1)
    return wins / (len(a) - 1)


def matched_auc(a: np.ndarray, b: np.ndarray) -> float:
    """Mean of ``matched_rates``: 0.5 when a pair is no closer than chance."""
    return float(matched_rates(a, b).mean())


def interval(rates: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval of the mean of per-pair rates."""
    rng = np.random.default_rng(seed)
    means = rates[rng.integers(0, len(rates), (n, len(rates)))].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def game_pairs(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Row positions of the home and the away coach of games with both."""
    position = pd.Series(np.arange(len(frame)), index=frame.index)
    home = frame[frame["is_home"]].set_index("game_id")
    away = frame[~frame["is_home"]].set_index("game_id")
    games = home.index.intersection(away.index)
    return (
        position[frame.index[frame["is_home"]]].set_axis(home.index)[games].to_numpy(),
        position[frame.index[~frame["is_home"]]].set_axis(away.index)[games].to_numpy(),
    )


def repeatability(
    frame: pd.DataFrame,
    whole: np.ndarray,
    halves: tuple[np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Per level of controls removed: the ``game`` AUC (two coaches, same
    game) and, with ``halves``, the ``halves`` AUC (same presser)."""
    home, away = game_pairs(frame)
    rows = []
    for level in LEVELS:
        C = controls(frame, level)
        row = {"removed": level}
        if halves is not None:
            keep = ~np.isnan(halves[0]).any(1) & ~np.isnan(halves[1]).any(1)
            Ck = None if C is None else C[keep]
            row["halves"] = matched_auc(
                residual(halves[0][keep], Ck), residual(halves[1][keep], Ck)
            )
        left = residual(whole, C)
        rates = matched_rates(left[home], left[away])
        row["game"] = float(rates.mean())
        row["game_low"], row["game_high"] = interval(rates)
        rows.append(row)
    return pd.DataFrame(rows)


def composition(frame: pd.DataFrame, X: np.ndarray, n_components: int = 100) -> dict:
    """What a representation is made of: out-of-fold share of its variance
    explained by each block (cumulative), and how well its length and the
    regulars out can be read from it."""
    reduced = A.pca(X)
    groups = frame["game_id"].to_numpy()
    blocks = {
        "transcript": np.hstack(
            [A.design(frame, A.NUISANCE), A.one_hot(frame["season"])]
        ),
        "known": A.design(frame, A.label_names()),
        "identity": np.hstack(
            [A.one_hot(frame[n]) for n in ("team", "opponent", "coach")]
        ),
    }
    shares = A.explained_by(reduced, blocks, groups, n_components=n_components)
    cumulative = shares["variance_explained"].to_numpy()
    probes = A.probe(
        X, frame, ["log_words", "target_won", "target_regulars_out", "pace"]
    ).set_index("label")["r"]
    return {
        "PC1": float(reduced.ratio[0]),
        "effective_dimension": reduced.effective_dimension,
        "var_transcript": float(cumulative[0]),
        "var_known": float(cumulative[1] - cumulative[0]),
        "var_identity": float(cumulative[2] - cumulative[1]),
        "read_length_r": float(probes["log_words"]),
        "read_won_r": float(probes["target_won"]),
        "read_regulars_out_r": float(probes["target_regulars_out"]),
        "read_pace_r": float(probes["pace"]),
    }


# ---------------------------------------------------------------------------
# Concept anchors
# ---------------------------------------------------------------------------


def axis_directions(
    anchors: pd.DataFrame, vectors: np.ndarray
) -> dict[str, np.ndarray]:
    """Unit vector from the ``low`` pole to the ``high`` pole of each axis."""
    directions = {}
    for axis, rows in anchors.groupby("axis", sort=False):
        high = vectors[rows.index[rows["pole"] == "high"]].mean(0)
        low = vectors[rows.index[rows["pole"] == "low"]].mean(0)
        directions[axis] = (high - low) / np.linalg.norm(high - low)
    return directions


def axis_scores(
    chunks: pd.DataFrame, vectors: np.ndarray, directions: dict[str, np.ndarray]
) -> dict[str, pd.DataFrame]:
    """Per presser, the mean projection of its chunks on each axis: over all
    chunks, the even ones and the odd ones."""
    projected = pd.DataFrame(
        np.asarray(vectors, np.float32) @ np.stack(list(directions.values())).T,
        columns=list(directions),
    )
    presser = chunks["presser"].to_numpy()
    even = chunks["chunk"].to_numpy() % 2 == 0
    n = int(presser.max()) + 1
    index = pd.RangeIndex(n)
    return {
        "all": projected.groupby(presser).mean().reindex(index),
        "even": projected[even].groupby(presser[even]).mean().reindex(index),
        "odd": projected[~even].groupby(presser[~even]).mean().reindex(index),
    }


def axis_report(frame: pd.DataFrame, scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per axis: split-half r, r with the result and with length, out-of-fold
    R2 from the transcript and from everything known, and what is left after
    the known part: the r between the two coaches of a game and the share a
    coach's identity explains (a trait rather than a game)."""
    groups = frame["game_id"].to_numpy()
    home, away = game_pairs(frame)
    transcript, known = controls(frame, "transcript"), controls(frame, "known")
    coach = A.one_hot(frame["coach"])
    rows = []
    for axis in scores["all"].columns:
        y = scores["all"][axis].to_numpy()
        row = {"axis": axis}
        row["split_half_r"] = scores["even"][axis].corr(scores["odd"][axis])
        row["r_won"] = float(np.corrcoef(y, frame["target_won"])[0, 1])
        row["r_margin_vs_spread"] = frame["target_margin_vs_spread"].corr(pd.Series(y))
        row["r_length"] = float(np.corrcoef(y, frame["log_words"])[0, 1])
        for name, C in (("transcript", transcript), ("known", known)):
            prediction, _ = rp.ridge_oof(C, y, groups)
            row[f"r2_{name}"] = float(
                1 - ((y - prediction[:, 0]) ** 2).mean() / y.var()
            )
        left = residual(y[:, None], known)[:, 0]
        row["coaches_r_raw"] = float(np.corrcoef(y[home], y[away])[0, 1])
        row["coaches_r_residual"] = float(np.corrcoef(left[home], left[away])[0, 1])
        prediction, _ = rp.ridge_oof(coach, left, groups)
        row["coach_trait_r2"] = float(
            1 - ((left - prediction[:, 0]) ** 2).mean() / left.var()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def load_variants(
    out: Path = A.DEFAULT_OUT, names: Sequence[str] | None = None
) -> dict:
    """Chunk tables, chunk vectors (as stored, float16), whole vectors, the
    anchors and the report vectors written by ``run_variants``."""
    vectors = {
        path.stem: np.load(path, mmap_mode="r")
        for path in sorted((out / "vectors").glob("*.npy"))
        if names is None or path.stem in names
    }
    return {
        "chunks": {
            name: pd.read_parquet(out / f"chunks_{name}.parquet")
            for name in ("raw", "masked")
        },
        "anchors": pd.read_parquet(out / "anchors.parquet"),
        "vectors": vectors,
    }
