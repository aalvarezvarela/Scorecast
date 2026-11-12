"""Export the current S3 season partitions to a local intermediate dataset file.

The scoring file (closing bet365 total per game x snapshot) is taken from the
T-0 snapshot row of each game, which is the closing line.
"""
import sys
from pathlib import Path

import pandas as pd

from nba_ou.config.settings import SETTINGS
from nba_ou.create_training_data.season_partitions import SeasonPartitions
from nba_ou.utils.s3_models import make_s3_client

out = Path(sys.argv[1])
store = SeasonPartitions(
    s3_client=make_s3_client(
        profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region
    ),
    bucket=SETTINGS.s3_bucket,
    dataset_type="intermediate_line",
    schema_version="2_5",
)
for season, m in store.manifests().items():
    print(season, m.limit_date, m.n_games, m.n_rows, m.git_commit, m.written_at, m.source)
frame = store.load()
print(frame.build_id, frame.checksum, frame.df.shape)
frame.df.to_parquet(out / f"{frame.build_id}.parquet")
