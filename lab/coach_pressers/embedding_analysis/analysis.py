"""PCA of the presser vectors against the same-game statistics.

Functions for the notebook; every fit that is scored is out of fold, with
both pressers of a game in the same fold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lab.coach_pressers.embedding_analysis.labels import LABEL_GROUPS
from lab.coach_pressers.embedding_analysis.presser_inputs import VIEWS
from lab.text_to_features import role_probes as rp

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "artifacts" / "coach_pressers" / "embedding_analysis"
#: Properties of the transcript, not of the game.
NUISANCE = ("log_words", "punct_per_100_words", "combined_video")
BINARY = ("target_won", "target_is_home")


@dataclass(frozen=True)
class Run:
    frame: pd.DataFrame  # sample + text counts + labels, one row per presser
    texts: pd.DataFrame
    vectors: dict[str, np.ndarray]
    info: dict


def load_run(out: Path = DEFAULT_OUT) -> Run:
    import json

    sample = pd.read_parquet(out / "sample.parquet")
    texts = pd.read_parquet(out / "texts.parquet")
    labels = pd.read_parquet(out / "labels.parquet")
    if not (sample["presser_id"].equals(texts["presser_id"])) or not (
        sample["presser_id"].equals(labels["presser_id"])
    ):
        raise ValueError("sample, texts and labels are not in the same order")
    counts = texts.drop(columns=["presser_id", "raw", "target_view"])
    frame = pd.concat(
        [sample, counts, labels.drop(columns=["presser_id", "game_id", "team"])],
        axis=1,
    )
    frame["log_words"] = np.log(frame["n_words"])
    frame["combined_video"] = (frame["kind"] == "combined").astype(float)
    vectors = {}
    for view in VIEWS:  # the other files of the folder belong to run_variants
        path = out / "vectors" / f"{view}.npy"
        if not path.exists():
            continue
        vectors[path.stem] = np.load(path)
        if len(vectors[path.stem]) != len(frame):
            raise ValueError(f"{path.name} does not match the sample")
    info_file = out / "run.json"
    info = json.loads(info_file.read_text()) if info_file.exists() else {}
    return Run(frame, texts, vectors, info)


def label_names(groups: Sequence[str] | None = None) -> list[str]:
    return [name for group in groups or LABEL_GROUPS for name in LABEL_GROUPS[group]]


@dataclass(frozen=True)
class PCA:
    scores: np.ndarray  # (n, k), column j has variance ratio[j] * total
    ratio: np.ndarray  # share of the total variance per component
    components: np.ndarray  # (k, d)
    mean: np.ndarray

    @property
    def effective_dimension(self) -> float:
        """Participation ratio: 1 / sum of squared variance shares."""
        return float(1.0 / np.sum(self.ratio**2))


def pca(X: np.ndarray) -> PCA:
    """Every component of the centered vectors (at most n - 1)."""
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    # Fix the sign so that a component does not flip between runs.
    sign = np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(1)])
    variance = S**2
    return PCA(U * S * sign, variance / variance.sum(), Vt * sign[:, None], mean)


def correlations(
    scores: np.ndarray, frame: pd.DataFrame, n_components: int = 12
) -> pd.DataFrame:
    """Pearson r of each of the first components (rows) with each column."""
    names = [f"PC{i + 1}" for i in range(n_components)]
    parts = pd.DataFrame(scores[:, :n_components], columns=names, index=frame.index)
    table = pd.concat([parts, frame.astype(float)], axis=1).corr()
    return table.loc[names, frame.columns]


def design(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Numeric columns with missing values set to the column mean."""
    block = frame[list(columns)].astype(float)
    return block.fillna(block.mean()).to_numpy()


def one_hot(values: pd.Series) -> np.ndarray:
    return pd.get_dummies(values).to_numpy(dtype=float)


def explained_by(
    reduced: PCA,
    blocks: dict[str, np.ndarray],
    groups: Sequence,
    *,
    n_components: int = 100,
) -> pd.DataFrame:
    """How much of the vectors' variance each block of features explains.

    The blocks are added one after another (cumulative). Each row gives the
    out-of-fold share of the total variance explained, counting only the
    first `n_components` components (the rest count as unexplained), and the
    out-of-fold R2 of PC1..PC5.
    """
    Y = reduced.scores[:, :n_components]
    total = Y.var(0)
    rows, features = [], []
    for name, block in blocks.items():
        features.append(block)
        predictions, _ = rp.ridge_oof(np.hstack(features), Y, groups)
        r2 = 1 - ((Y - predictions) ** 2).mean(0) / total
        row = {"features": "+ " + name if rows else name}
        row["variance_explained"] = float((r2 * reduced.ratio[:n_components]).sum())
        row.update({f"PC{i + 1}_r2": float(r2[i]) for i in range(5)})
        rows.append(row)
    return pd.DataFrame(rows)


def component_r2(
    reduced: PCA, features: np.ndarray, groups: Sequence, n_components: int = 12
) -> np.ndarray:
    """Out-of-fold R2 of each of the first components from `features`."""
    Y = reduced.scores[:, :n_components]
    predictions, _ = rp.ridge_oof(features, Y, groups)
    return 1 - ((Y - predictions) ** 2).mean(0) / Y.var(0)


def probe(
    X: np.ndarray,
    frame: pd.DataFrame,
    labels: Sequence[str],
    groups_column: str = "game_id",
) -> pd.DataFrame:
    """Ridge probe of each label from `X`, out of fold: Pearson r with its
    95% interval (Fisher), AUC for binary labels. Rows with a missing label
    are left out of every label."""
    keep = frame[list(labels)].notna().all(1).to_numpy()
    Y = frame.loc[keep, list(labels)].to_numpy(dtype=float)
    predictions, _ = rp.ridge_oof(X[keep], Y, frame.loc[keep, groups_column].to_numpy())
    n = int(keep.sum())
    rows = []
    for index, label in enumerate(labels):
        y, pred = Y[:, index], predictions[:, index]
        r = float(np.corrcoef(y, pred)[0, 1]) if pred.std() > 0 else float("nan")
        half = 1.96 / np.sqrt(n - 3)
        row = {
            "label": label,
            "r": r,
            "r_low": float(np.tanh(np.arctanh(r) - half)),
            "r_high": float(np.tanh(np.arctanh(r) + half)),
            "n": n,
        }
        if label in BINARY:
            row["auc"] = rp.auc(y, pred)
        rows.append(row)
    return pd.DataFrame(rows)


def view_cosines(vectors: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per presser, the cosine between each pair of views."""
    names = list(vectors)
    columns = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            columns[f"{a} / {b}"] = (vectors[a] * vectors[b]).sum(1)
    return pd.DataFrame(columns)


def extremes(
    scores: np.ndarray, frame: pd.DataFrame, component: int, n: int = 5
) -> pd.DataFrame:
    """The pressers at both ends of one component (1-based)."""
    order = np.argsort(scores[:, component - 1])
    picked = np.concatenate([order[:n], order[-n:][::-1]])
    columns = [
        "season",
        "team",
        "opponent",
        "coach",
        "target_margin",
        "n_words",
        "title",
    ]
    table = frame.iloc[picked][columns].copy()
    table.insert(0, "score", scores[picked, component - 1].round(3))
    table.insert(0, "end", ["low"] * n + ["high"] * n)
    return table.reset_index(drop=True)
