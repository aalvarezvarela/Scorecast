"""The season partitions: one file per season in S3, a full build's frame when
read back, and a daily update that touches only the games it must.

The builders are stood in for: what is under test is the storage contract --
rows, order and dtypes as a full build has them -- the update window and its
overlap check, and that a refit on the partitions trains on the spec's window
of games, with its seed, exactly as on the full build.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from nba_ou.create_training_data import season_partitions as sp
from nba_ou.create_training_data.season_partitions import (
    PartitionError,
    SeasonPartitions,
    dataset_checksum,
    split_into_partitions,
    update_partitions,
)
from nba_ou.create_training_data.train_data_store import (
    parse_season_partition_key,
    season_partition_key,
)
from nba_ou.modeling.refit import refit_from_spec, resolve_training_frame
from nba_ou.modeling.registry_models import (
    SpecCleaning,
    SpecFeatures,
    SpecIdentity,
    SpecTraining,
    finalize_spec,
)
from nba_ou.modeling.registry_paths import ModelSlot, ModelTarget

from tests.support.fake_s3 import FakeS3Client

BUCKET = "b"
LINE = "ODDS_TOTAL_LINE_bet365"
FEATURES = ["FEAT_A", "FEAT_B", "FEAT_C"]
HORIZONS = (60, 360)


def _games(n_per_season=30, seasons=(2023, 2024, 2025)) -> pd.DataFrame:
    """A closing frame: two games a day, seasons in their own date ranges."""
    rng = np.random.default_rng(0)
    parts = []
    for season in seasons:
        days = pd.date_range(f"{season}-10-22", periods=n_per_season // 2)
        dates = np.repeat(days, 2)
        n = len(dates)
        line = rng.normal(225, 5, size=n)
        total = line + rng.normal(0, 12, size=n)
        parts.append(
            pd.DataFrame(
                {
                    "GAME_ID": [f"002{season % 100:02d}{i:05d}" for i in range(n)],
                    "GAME_DATE": dates,
                    "SEASON_YEAR": season,
                    "TOTAL_POINTS": total,
                    LINE: line,
                    "FEAT_A": rng.normal(size=n),
                    "FEAT_B": rng.normal(size=n),
                    "FEAT_C": rng.normal(size=n),
                    "TEAM_NAME": ["x"] * n,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _snapshots(closing: pd.DataFrame) -> pd.DataFrame:
    """An intermediate frame: each game at every horizon, in a full build's
    order (date, game, horizon), with the residual against each snapshot's
    line."""
    parts = []
    for step, horizon in enumerate(HORIZONS):
        part = closing.assign(TIME_TO_MATCH_MIN=horizon)
        part[LINE] = part[LINE] + step
        part["LINE_ERROR"] = part["TOTAL_POINTS"] - part[LINE]
        part["FEAT_B"] = part["FEAT_B"] + step
        parts.append(part)
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["GAME_DATE", "GAME_ID", "TIME_TO_MATCH_MIN"], kind="mergesort")
        .reset_index(drop=True)
    )


def _partitions(s3, dataset_type="closing_line", tmp_path=None, **kwargs):
    return SeasonPartitions(
        s3_client=s3,
        bucket=BUCKET,
        dataset_type=dataset_type,
        schema_version="2_5",
        local_dir=tmp_path,
        use_cache=tmp_path is not None,
        **kwargs,
    )


# --- storage -------------------------------------------------------------------


def test_a_season_is_named_like_the_other_datasets():
    manifest = season_partition_key(
        dataset_type="intermediate_line", schema_version="2_5", season=2025, ext="json"
    )
    assert manifest == "train_data/2_5/seasons/intermediate_line/season=2025.json"
    data = season_partition_key(
        dataset_type="intermediate_line",
        schema_version="2_5",
        season=2025,
        digest="0123456789abcdef",
    )
    assert data.endswith("/season=2025.0123456789abcdef.parquet")
    assert parse_season_partition_key(manifest) == (2025, "json")
    assert parse_season_partition_key(data) == (2025, "parquet")
    # The first layout's data file, so an update deletes it.
    assert parse_season_partition_key("p/season=2025.parquet") == (2025, "parquet")
    assert parse_season_partition_key("train_data/2_5/seasons/x/other.parquet") is None


def test_the_seasons_read_back_as_the_full_build():
    s3 = FakeS3Client()
    full = _games()
    partitions = _partitions(s3)
    written = split_into_partitions(
        full, partitions, limit_date="2026-10-16", source="t"
    )

    assert sorted(written) == [2023, 2024, 2025]
    assert written[2024].n_games == 30
    assert written[2025].limit_date == date(2026, 10, 16)
    stored = [k for k in s3.store if k.startswith(partitions.prefix)]
    assert len(stored) == 6  # one file and one manifest per season, nothing per day

    loaded = partitions.load()
    pd.testing.assert_frame_equal(loaded.df, full, check_exact=True)
    assert loaded.build_id == "closing_line_data_2_5_20261016"
    assert loaded.limit_date == date(2026, 10, 16)


def test_an_intermediate_horizon_reads_alone_in_build_order():
    s3 = FakeS3Client()
    full = _snapshots(_games())
    partitions = _partitions(s3, "intermediate_line")
    split_into_partitions(full, partitions, limit_date="2026-10-16", source="t")

    one = partitions.load(horizon=360)
    expected = full[full["TIME_TO_MATCH_MIN"] == 360].reset_index(drop=True)
    pd.testing.assert_frame_equal(one.df, expected, check_exact=True)
    pd.testing.assert_frame_equal(partitions.load().df, full, check_exact=True)
    with pytest.raises(ValueError, match="horizons"):
        _partitions(s3).read(2025, horizon=360)


def _count_downloads(monkeypatch) -> list[str]:
    reads: list[str] = []
    original = sp.read_s3_object_bytes
    monkeypatch.setattr(
        sp,
        "read_s3_object_bytes",
        lambda **kw: reads.append(kw["key"]) or original(**kw),
    )
    return reads


def test_a_local_copy_is_used_only_when_it_is_the_manifests(tmp_path, monkeypatch):
    s3 = FakeS3Client()
    partitions = _partitions(s3, tmp_path=tmp_path)
    split_into_partitions(_games(), partitions, limit_date="2026-10-16", source="t")
    manifest = partitions.manifests()[2025]
    local = partitions.local_dir / manifest.file
    assert local.exists()

    reads = _count_downloads(monkeypatch)
    fresh = _partitions(s3, tmp_path=tmp_path)
    fresh.read(2025)
    assert not [k for k in reads if k.endswith(".parquet")]

    local.write_bytes(b"stale")
    rows = _partitions(s3, tmp_path=tmp_path).read(2025)
    assert [k for k in reads if k.endswith(".parquet")] == [
        partitions.prefix + manifest.file
    ]
    assert len(rows) == 30
    assert dataset_checksum(local.read_bytes()) == manifest.checksum


@pytest.mark.parametrize("cached", [True, False])
def test_each_file_is_checked_once_however_many_horizons_are_read(
    tmp_path, monkeypatch, cached
):
    """16 horizons must not download or hash every season 16 times."""
    s3 = FakeS3Client()
    full = _snapshots(_games())
    split_into_partitions(
        full,
        _partitions(s3, "intermediate_line", tmp_path=tmp_path if cached else None),
        limit_date="2026-10-16",
        source="t",
    )
    partitions = _partitions(
        s3, "intermediate_line", tmp_path=tmp_path if cached else None
    )
    reads = _count_downloads(monkeypatch)
    hashed = []
    original = sp.dataset_checksum
    monkeypatch.setattr(
        sp, "dataset_checksum", lambda data: hashed.append(1) or original(data)
    )

    for horizon in HORIZONS:
        partitions.load(horizon=horizon)

    downloads = [k for k in reads if k.endswith(".parquet")]
    assert len(hashed) == 3  # one per season
    assert len(downloads) == (0 if cached else 3)


def test_a_write_that_stops_before_its_manifest_changes_nothing(monkeypatch):
    s3 = FakeS3Client()
    full = _games()
    partitions = _partitions(s3)
    split_into_partitions(full, partitions, limit_date="2026-10-16", source="t")
    before = partitions.manifests()[2025]
    original = sp.upload_bytes_to_s3

    def manifest_upload_fails(**kw):
        if kw["key"].endswith(".json"):
            raise ConnectionError("network down")
        return original(**kw)

    monkeypatch.setattr(sp, "upload_bytes_to_s3", manifest_upload_fails)
    changed = full[full["SEASON_YEAR"] == 2025].assign(FEAT_A=0.0)
    with pytest.raises(ConnectionError):
        partitions.write(2025, changed, limit_date="2026-10-17", source="t")
    monkeypatch.setattr(sp, "upload_bytes_to_s3", original)

    # Another runner, with no local copy: the season is still the old one.
    elsewhere = _partitions(s3)
    assert elsewhere.manifests()[2025] == before
    pd.testing.assert_frame_equal(elsewhere.load().df, full, check_exact=True)

    # The next write succeeds and leaves one data file for the season.
    partitions.write(2025, changed, limit_date="2026-10-17", source="t")
    files = [
        k
        for k in s3.store
        if k.startswith(partitions.prefix + "season=2025") and k.endswith(".parquet")
    ]
    assert files == [partitions.prefix + partitions.manifests()[2025].file]


def test_a_data_file_changed_after_it_was_written_is_refused():
    s3 = FakeS3Client()
    partitions = _partitions(s3)
    split_into_partitions(_games(), partitions, limit_date="2026-10-16", source="t")
    manifest = partitions.manifests()[2025]
    s3.put_object(Bucket=BUCKET, Key=partitions.prefix + manifest.file, Body=b"x")
    with pytest.raises(PartitionError, match="changed after it was written"):
        _partitions(s3).read(2025)


def test_seasons_with_different_columns_are_refused():
    s3 = FakeS3Client()
    partitions = _partitions(s3)
    full = _games()
    split_into_partitions(full, partitions, limit_date="2026-10-16", source="t")
    partitions.write(
        2023,
        full[full["SEASON_YEAR"] == 2023].drop(columns="FEAT_C"),
        limit_date="2026-10-16",
        source="t",
    )
    with pytest.raises(PartitionError, match="columns differ"):
        partitions.load()


# --- the daily update --------------------------------------------------------------


class _Database:
    """The games and the builder, as of a full build of ``full``."""

    def __init__(self, full: pd.DataFrame):
        self.full = full
        self.built: list[list[str]] = []
        #: Games whose line history is not stored.
        self.no_history: set[str] = set()

    def game_ids_between(self, start, end):
        dates = pd.to_datetime(self.full["GAME_DATE"]).dt.date
        return self.full.loc[(dates >= start) & (dates <= end), "GAME_ID"].tolist()

    def build_rows(self, game_ids, limit):
        self.built.append(list(game_ids))
        rows = self.full[self.full["GAME_ID"].isin(game_ids)].copy()
        # A target build infers dtypes over its own rows only.
        rows["FEAT_C"] = rows["FEAT_C"].astype(object)
        return rows

    def update(self, partitions, limit_date):
        return update_partitions(
            partitions,
            limit_date=limit_date,
            game_ids_between=self.game_ids_between,
            build_rows=self.build_rows,
            ready_game_ids=lambda ids: [g for g in ids if g not in self.no_history],
        )


def test_an_update_builds_only_the_window_and_matches_the_full_build():
    s3 = FakeS3Client()
    full = _games(seasons=(2024, 2025))
    stored = full[pd.to_datetime(full["GAME_DATE"]) <= "2025-10-30"]
    partitions = _partitions(s3)
    split_into_partitions(stored, partitions, limit_date="2025-10-30", source="t")
    database = _Database(full)

    report = database.update(partitions, "2025-11-05")

    # The window: three days before the last cutoff, through the new one.
    assert report.window_start == date(2025, 10, 27)
    built_dates = full.set_index("GAME_ID").loc[database.built[0], "GAME_DATE"]
    assert built_dates.min() == pd.Timestamp("2025-10-27")
    assert built_dates.max() == pd.Timestamp("2025-11-05")
    assert sorted(report.written) == [2025]  # 2024 untouched
    assert not report.drifted
    assert report.overlap[0].n_games == 8
    loaded = partitions.load()
    pd.testing.assert_frame_equal(loaded.df, full, check_exact=True)
    assert loaded.limit_date == date(2025, 11, 5)
    assert partitions.manifests()[2024].limit_date == date(2025, 10, 30)


def test_a_new_season_gets_its_own_partition():
    s3 = FakeS3Client()
    full = _games(seasons=(2024, 2025))
    partitions = _partitions(s3)
    split_into_partitions(
        full[full["SEASON_YEAR"] == 2024],
        partitions,
        limit_date="2025-10-22",
        source="t",
    )
    report = _Database(full).update(partitions, "2025-11-06")
    # From the day after 2024's last game: the finished season is not rebuilt.
    assert report.window_start == date(2024, 11, 6)
    assert sorted(report.written) == [2025]
    assert report.overlap == []
    pd.testing.assert_frame_equal(partitions.load().df, full, check_exact=True)


def test_a_rebuilt_row_that_changed_is_reported():
    s3 = FakeS3Client()
    full = _games(seasons=(2025,))
    partitions = _partitions(s3)
    split_into_partitions(full, partitions, limit_date="2025-11-05", source="t")
    corrected = full.copy()
    late = pd.to_datetime(corrected["GAME_DATE"]) == "2025-11-04"
    corrected.loc[late, "TOTAL_POINTS"] += 2  # a late box-score correction

    report = _Database(corrected).update(partitions, "2025-11-06")

    assert report.drifted
    assert report.overlap[0].differing == {"TOTAL_POINTS": pytest.approx(2 / 8)}
    assert "TOTAL_POINTS: 25.0% of rows" in report.describe()
    pd.testing.assert_frame_equal(partitions.load().df, corrected, check_exact=True)


def test_a_stored_game_the_rebuild_misses_is_kept_and_reported():
    s3 = FakeS3Client()
    full = _games(seasons=(2025,))
    partitions = _partitions(s3)
    split_into_partitions(full, partitions, limit_date="2025-11-05", source="t")
    database = _Database(full)
    gone = full.loc[pd.to_datetime(full["GAME_DATE"]) == "2025-11-03", "GAME_ID"].iloc[
        0
    ]
    database.full = full[full["GAME_ID"] != gone]

    report = database.update(partitions, "2025-11-06")

    assert report.overlap[0].missing_games == [gone]
    assert report.drifted
    pd.testing.assert_frame_equal(partitions.load().df, full, check_exact=True)


def test_an_off_season_update_rebuilds_nothing():
    """A finished season is not rebuilt day after day: the window starts the
    day after its last game, so only games still unbuilt would be tried."""
    s3 = FakeS3Client()
    full = _games(seasons=(2025,))
    partitions = _partitions(s3)
    split_into_partitions(full, partitions, limit_date="2026-07-01", source="t")
    before = partitions.manifests()[2025]
    database = _Database(full)

    report = database.update(partitions, "2026-07-02")

    assert report.window_start == date(2025, 11, 6)
    assert report.n_target_games == 0
    assert database.built == []
    after = partitions.manifests()[2025]
    assert after.checksum == before.checksum
    assert after.limit_date == date(2026, 7, 2)


def test_a_game_that_cannot_be_built_yet_is_retried():
    s3 = FakeS3Client()
    full = _games(seasons=(2025,))
    partitions = _partitions(s3)
    stored = full[pd.to_datetime(full["GAME_DATE"]) <= "2025-10-30"]
    split_into_partitions(stored, partitions, limit_date="2025-10-30", source="t")
    database = _Database(full)
    later = pd.to_datetime(full["GAME_DATE"]) > "2025-10-30"
    database.full = full[~later]  # their line history is not ingested yet
    database.game_ids_between = lambda start, end: _Database(full).game_ids_between(
        start, end
    )

    report = database.update(partitions, "2025-11-03")
    due = pd.to_datetime(full["GAME_DATE"]) <= "2025-11-03"
    assert report.not_built == sorted(full.loc[later & due, "GAME_ID"])
    assert "not built" in report.describe()

    for day in range(4, 10):  # daily updates while the ingestion lags
        database.update(partitions, f"2025-11-{day:02d}")
    database.full = full  # ingested a week late
    report = database.update(partitions, "2025-11-10")
    assert report.not_built == []
    pd.testing.assert_frame_equal(partitions.load().df, full, check_exact=True)


def _frozen_game(full: pd.DataFrame, day: str) -> str:
    return full.loc[pd.to_datetime(full["GAME_DATE"]) == day, "GAME_ID"].iloc[0]


def test_a_game_without_line_history_waits_behind_later_games():
    """An SBR feed freeze: one game's line history is not stored, so its close
    would be the frozen day-page number. It is not built, and stays in the
    window after later games are stored, until its history arrives."""
    s3 = FakeS3Client()
    full = _games(seasons=(2025,))
    partitions = _partitions(s3)
    stored = full[pd.to_datetime(full["GAME_DATE"]) <= "2025-10-30"]
    split_into_partitions(stored, partitions, limit_date="2025-10-30", source="t")
    database = _Database(full)
    frozen = _frozen_game(full, "2025-10-31")
    database.no_history = {frozen}

    for day in range(1, 6):
        report = database.update(partitions, f"2025-11-{day:02d}")
        assert report.waiting == [frozen]
        assert frozen not in set(partitions.load().df["GAME_ID"])
    # The plain window (from 2025-11-01) has moved past it; it reaches back.
    assert report.window_start == date(2025, 10, 29)
    assert "wait for their line history" in report.describe()

    database.no_history = set()  # SBR recovered
    report = database.update(partitions, "2025-11-06")
    assert report.waiting == [] and report.overdue == []
    pd.testing.assert_frame_equal(partitions.load().df, full, check_exact=True)


def test_a_game_that_never_gets_line_history_is_built_after_waiting():
    s3 = FakeS3Client()
    full = _games(seasons=(2025,))
    partitions = _partitions(s3)
    stored = full[pd.to_datetime(full["GAME_DATE"]) <= "2025-10-24"]
    split_into_partitions(stored, partitions, limit_date="2025-10-24", source="t")
    database = _Database(full)
    frozen = _frozen_game(full, "2025-10-25")
    database.no_history = {frozen}

    for day in range(25, 32):
        report = database.update(partitions, f"2025-10-{day:02d}")
        assert report.waiting == [frozen] and report.overdue == []
    # MAX_WAIT_DAYS after its date: built with what there is, and reported.
    report = database.update(partitions, "2025-11-01")
    assert report.waiting == []
    assert report.overdue == [frozen]
    assert "built without line history" in report.describe()
    due = full[pd.to_datetime(full["GAME_DATE"]) <= "2025-11-01"]
    pd.testing.assert_frame_equal(partitions.load().df, due, check_exact=True)
    # Once built it is not waited for again.
    report = database.update(partitions, "2025-11-02")
    assert report.waiting == [] and report.overdue == []


def test_without_any_game_to_build_only_the_cutoff_moves():
    s3 = FakeS3Client()
    full = _games(seasons=(2025,))
    partitions = _partitions(s3)
    split_into_partitions(full, partitions, limit_date="2026-07-01", source="t")
    before = partitions.manifests()[2025]

    report = _Database(full.iloc[:0]).update(partitions, "2026-07-02")

    after = partitions.manifests()[2025]
    assert report.n_target_games == 0
    assert after.checksum == before.checksum
    assert after.limit_date == date(2026, 7, 2)


def test_an_update_never_moves_the_cutoff_back():
    s3 = FakeS3Client()
    partitions = _partitions(s3)
    split_into_partitions(_games(), partitions, limit_date="2026-10-16", source="t")
    with pytest.raises(PartitionError, match="refusing to move them back"):
        _Database(_games()).update(partitions, "2026-10-01")


def test_new_rows_with_other_columns_are_refused():
    s3 = FakeS3Client()
    full = _games(seasons=(2025,))
    partitions = _partitions(s3)
    split_into_partitions(full, partitions, limit_date="2025-11-01", source="t")
    database = _Database(full.assign(NEW_FEATURE=1.0))
    with pytest.raises(PartitionError, match="builder changed"):
        database.update(partitions, "2025-11-06")


# --- what the refit trains on ---------------------------------------------------------


def _spec(dataset_type: str, horizon: int, train_games: int):
    return finalize_spec(
        identity=SpecIdentity(
            schema_version="2_5",
            target=ModelTarget.LINE_ERROR,
            prediction_strategy="line_error_regressor",
            horizon_minutes=horizon,
            dataset_type=dataset_type,
        ),
        features=SpecFeatures(feature_names=FEATURES, n_features=len(FEATURES)),
        cleaning=SpecCleaning(nan_threshold=5.0, max_na_per_row=0),
        training=SpecTraining(
            params={"max_depth": 3, "subsample": 0.6, "colsample_bytree": 0.7},
            n_estimators=20,
            train_games=train_games,
            random_state=16,
            device="cpu",
        ),
    )


@pytest.mark.parametrize(
    ("dataset_type", "horizon"), [("closing_line", 0), ("intermediate_line", 360)]
)
def test_the_refit_trains_on_the_specs_window_as_on_the_full_build(
    dataset_type, horizon
):
    """Window in games (not rows), and the same model as a fit on the full
    build's frame: same rows, same order, same seed."""
    s3 = FakeS3Client()
    full = _games(n_per_season=40)
    if dataset_type == "intermediate_line":
        full = _snapshots(full)
    partitions = _partitions(s3, dataset_type)
    split_into_partitions(full, partitions, limit_date="2026-10-16", source="t")
    spec = _spec(dataset_type, horizon, train_games=50)
    slot = ModelSlot(schema_version="2_5", target="line_error", horizon_minutes=horizon)

    frame = resolve_training_frame(spec, partitions={dataset_type: partitions})
    result = refit_from_spec(spec, slot=slot, frame=frame)

    assert result.fit.n_train_games == 50
    assert result.fit.dataset_build_id == f"{dataset_type}_data_2_5_20261016"
    assert result.fit.dataset_checksum == frame.checksum
    # The last 50 games: all 40 of 2025 and the last 10 of 2024 (two a day).
    assert result.fit.train_date_min == pd.Timestamp("2024-11-06").to_pydatetime()

    direct = full
    if dataset_type == "intermediate_line":
        direct = full[full["TIME_TO_MATCH_MIN"] == horizon].reset_index(drop=True)
    reference = refit_from_spec(
        spec,
        slot=slot,
        frame=type(frame)(df=sp_with_targets(direct), build_id=None, checksum=None),
    )
    X = direct[FEATURES].astype(float)
    np.testing.assert_array_equal(result.model.predict(X), reference.model.predict(X))


def sp_with_targets(df: pd.DataFrame) -> pd.DataFrame:
    from nba_ou.modeling.refit import with_residual_targets

    return with_residual_targets(df)
