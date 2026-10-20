import pandas as pd


def compute_total_points_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total points and related features.

    - TOTAL_POINTS: sum of PTS per GAME_ID
    - For each totals line column:
        * DIFF_FROM_<linecol>: TOTAL_POINTS - line
        * IS_OVER_<linecol>: indicator(TOTAL_POINTS > line)
    """
    # Total points per game
    df["TOTAL_POINTS"] = df.groupby("GAME_ID")["PTS"].transform("sum")

    # Identify all totals line columns
    total_line_cols: list[str] = []

    total_line_cols.extend([c for c in df.columns if c.startswith("ODDS_TOTAL_LINE_")])

    # De-dup while preserving order
    seen = set()
    total_line_cols = [c for c in total_line_cols if not (c in seen or seen.add(c))]

    # Compute features for each line column
    for line_col in total_line_cols:
        # Make a stable suffix for new column names
        suffix = line_col.replace("TOTAL_", "")  # e.g. OVER_UNDER_LINE or LINE_betmgm

        diff_col = f"DIFF_FROM_{suffix}"

        line_vals = pd.to_numeric(df[line_col], errors="coerce")

        df[diff_col] = df["TOTAL_POINTS"] - line_vals

    # Dates
    df["GAME_DATE"] = pd.to_datetime(
        df["GAME_DATE"], format="%Y-%m-%d", errors="coerce"
    )

    return df
