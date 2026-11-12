"""Prepare cleaned one-horizon matrices exactly as the production recipe does."""
import os, sys, json, numpy as np, pandas as pd
from pathlib import Path
from training_pipeline.cli import load_config
from training_pipeline.data import prepare_dataset

OUT = Path(sys.argv[1]); horizons = [int(h) for h in sys.argv[2:]]
# The study ran on the 20261105 build; STUDY_FILE points a rerun at another one.
F = os.environ.get("STUDY_FILE", "data/train_data/intermediate_line_data_2_5_20261105.parquet")
for h in horizons:
    cfg = load_config(f"experiments/retrain_line_error_2_5_2026_10/intermediate_line_error_t{h}.yaml")
    data = cfg.data.model_copy(update=dict(csv_path=F, expected_checksum=None,
        scoring_csv_path=F.replace(".parquet", "_scoring.parquet")))
    cfg = cfg.model_copy(update=dict(data=data))
    p = prepare_dataset(cfg)
    df = p.df_full
    print(h, p.X.shape, len(df), p.target_line_col, p.scoring_columns[:3])
    keep = [c for c in ["GAME_ID", "GAME_DATE", "SEASON_YEAR", "TOTAL_POINTS", "HOME_MARGIN",
            "LINE_ERROR", "IS_OVERTIME", "TEAM_ID_TEAM_HOME", "TEAM_ID_TEAM_AWAY",
            p.target_line_col, "ODDS_CLOSING_TOTAL_LINE_bet365"] if c in df.columns]
    print("meta", keep)
    assert (df.index == p.X.index).all()
    meta = df[keep].copy(); meta["y"] = p.y.values
    meta.to_parquet(OUT / f"meta_t{h}.parquet")
    p.X.astype("float32").to_parquet(OUT / f"X_t{h}.parquet")
