"""Build the frozen phase 3 rating book for phase 5 (and check it).

``rating_book.walk_forward_rating_book`` with the frozen configuration
(``rating_book.FROZEN``: offense and defense toward the profile prior at
3,000, pace toward zero at 10,000, half-life 180 days) and phase 3's prior
provider (``frozen_prior_provider``), on the stints from 2016-17, for every
game date from ``--first-date`` to the end of ``--last-season``: the stints'
dates and the closing 2_6 file's, so every game graph finds its own date.

Writes ``data/player_graph/rating_book/phase3_profile_prior.parquet`` and its
``.metadata.json`` (the format ``lineups.features.load_rating_book`` reads).

``--verify`` scores phase 3's 2019-25 evaluation rows with the book (the same
stint-sides, built as the evaluation built them) and compares them with the
stored predictions of that walk-forward
(``rating_diagnostics/evaluation/profile_prior_{efficiency,pace}_2019_2025``):
offense/defense with the profile prior, pace with the zero prior.

    python scripts/player_graph/build_rating_book.py
    python scripts/player_graph/build_rating_book.py --verify
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.data_processing.lineups.stint_store import read_stints
from nba_ou.data_processing.player_graph.as_of import FIRST_SEASON
from nba_ou.data_processing.player_graph.profile_prior import LAMBDA_OFFDEF, LAMBDA_PACE
from nba_ou.data_processing.player_graph.rating_book import (
    BOOK_PATH,
    FROZEN,
    book_metadata,
    frozen_prior_provider,
    walk_forward_rating_book,
)
from nba_ou.data_processing.player_graph.rating_diagnostics import (
    decayed_exposure,
    decayed_league_means,
    scored_rows,
)

CLOSING_2_6 = Path("data/train_data/closing_line_data_2_6_20261003.parquet")
STORED_2_6 = Path("data/lineup_ratings/player_ratings.parquet")
EVALUATION = Path("player_graph") / "rating_diagnostics" / "evaluation"
EVALUATION_SEASONS = range(2019, 2026)


def target_dates(
    history: pd.DataFrame, first: pd.Timestamp, last_season: int
) -> list[pd.Timestamp]:
    stint_dates = pd.to_datetime(history["game_date"]).dt.normalize()
    closing = pd.read_parquet(CLOSING_2_6, columns=["GAME_DATE", "SEASON_YEAR"])
    closing = closing.loc[
        pd.to_numeric(closing["SEASON_YEAR"], errors="coerce").le(last_season)
    ]
    dates = set(stint_dates) | set(pd.to_datetime(closing["GAME_DATE"]).dt.normalize())
    return sorted(d for d in dates if d >= first)


def build(args) -> None:
    started = time.time()
    history = read_stints(
        list(range(FIRST_SEASON, args.last_season + 1)), local_root=args.local_root
    )
    provider = frozen_prior_provider(history, local_root=args.local_root)
    dates = target_dates(history, pd.Timestamp(args.first_date), args.last_season)
    book = walk_forward_rating_book(history, dates, spec=FROZEN, prior=provider)
    path = args.local_root / BOOK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    book.to_parquet(path, index=False)
    frozen = json.loads(
        (args.local_root / "player_graph" / "profile_prior" / "alphas.json").read_text()
    )
    metadata = book_metadata(
        FROZEN,
        first_season=FIRST_SEASON,
        last_season=args.last_season,
        as_of_from=str(book["as_of_date"].min().date()),
        as_of_to=str(book["as_of_date"].max().date()),
        rating_dates=int(book["as_of_date"].nunique()),
        rows=len(book),
        fit_max_game_date=str(pd.Timestamp(book["fit_max_game_date"].max()).date()),
        f_alphas=frozen["alphas"],
        f_selected_on=frozen["selected_on"],
        f_pairs="profile_prior/pairs_evaluation.parquet",
        f_first_checkpoint="2016-12-01",
    )
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    sources = pd.Series(
        [s for day in provider.sources.values() for s in day.values()]
    ).value_counts()
    print(
        f"{len(book):,} rows on {book['as_of_date'].nunique()} dates "
        f"({metadata['as_of_from']} to {metadata['as_of_to']}); prior sources over "
        f"solver players and dates: {sources.to_dict()}; {time.time() - started:.0f}s "
        f"-> {path}"
    )


def verify(args) -> None:
    started = time.time()
    root = args.local_root
    history = read_stints(
        list(range(FIRST_SEASON, max(EVALUATION_SEASONS) + 1)), local_root=root
    )
    stints = history.loc[history["season_year"].isin(list(EVALUATION_SEASONS))]
    dates = sorted(pd.to_datetime(stints["game_date"]).dt.normalize().unique())
    # The evaluation's rows: stints on the dates of 2_6's stored cache.
    stored = pd.read_parquet(STORED_2_6)
    stored_dates = set(
        pd.to_datetime(stored.loc[stored["as_of_date"].isin(dates), "as_of_date"])
    )
    book = pd.read_parquet(root / BOOK_PATH)
    book = book.loc[pd.to_datetime(book["as_of_date"]).isin(stored_dates)]
    exposure = decayed_exposure(history, dates)
    efficiency, pace = scored_rows(
        stints,
        book,
        exposure,
        lambda_offdef=LAMBDA_OFFDEF,
        lambda_pace=LAMBDA_PACE,
        league_means=decayed_league_means(history, dates),
    )
    reference = {
        "efficiency": (
            pd.read_parquet(
                root / EVALUATION / "profile_prior_efficiency_2019_2025.parquet"
            ),
            efficiency,
            "profile prior",
        ),
        "pace": (
            pd.read_parquet(root / EVALUATION / "profile_prior_pace_2019_2025.parquet"),
            pace,
            "zero prior",
        ),
    }
    for name, (stored_rows, rows, column) in reference.items():
        same = (
            len(stored_rows) == len(rows)
            and stored_rows["game_id"].to_numpy().tolist()
            == rows["game_id"].to_numpy().tolist()
            and np.allclose(stored_rows["weight"], rows["weight"])
            and np.allclose(stored_rows["actual"], rows["actual"])
        )
        diff = (
            (rows["predicted"].to_numpy() - stored_rows[column].to_numpy())
            if same
            else None
        )
        print(
            f"{name}: {len(rows):,} rows, same rows as the evaluation: {same}; "
            + (
                f"|book - stored {column!r}| max {np.abs(diff).max():.2e}, "
                f"mean {np.abs(diff).mean():.2e}"
                if same
                else "rows differ"
            )
        )
    print(f"{time.time() - started:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--first-date", default="2018-10-01")
    parser.add_argument("--last-season", type=int, default=2025)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
