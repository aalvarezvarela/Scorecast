from nba_ou.config.market_columns import (
    HOME_MARGIN_COL,
    PTS_AWAY_COL,
    PTS_HOME_COL,
)
from nba_ou.config.odds_columns import (
    is_odds_column,
    moneyline_col,
    spread_col,
    total_line_col,
)
from nba_ou.data_processing.odds.canonical_markets import add_canonical_market_columns

MAIN_TOTAL_LINE_COL = total_line_col()
MAIN_SPREAD_COL = spread_col()
MAIN_MONEYLINE_COL = moneyline_col()

# Column name constants for training data selection
TEAM_INFO_COLUMNS = [
    "TEAM_ID",
    "TEAM_CITY",
    "TEAM_ABBREVIATION",
    "TEAM_NAME",
    "MATCHUP",
    "GAME_NUMBER",
]

STATIC_COLUMNS = [
    "SEASON_ID",
    "IS_OVERTIME",
    "GAME_ID",
    "GAME_DATE",
    "SEASON_TYPE",
    "IS_PLAYOFF_GAME_BEFORE",
    "IS_PLAYOFF_GAME",
    "PLAYOFF_GAMES_LAST_SEASON_TEAM_AWAY",
    "PLAYOFF_GAMES_LAST_SEASON_TEAM_HOME",
    "SEASON_YEAR",
]

ODDS_COLUMNS = [
    # Configured main sportsbook columns
    MAIN_TOTAL_LINE_COL,
    MAIN_SPREAD_COL,
    f"{MAIN_MONEYLINE_COL}_TEAM_HOME",
    f"{MAIN_MONEYLINE_COL}_TEAM_AWAY",
    # Main spread gets _TEAM_HOME/_TEAM_AWAY suffix after merge_home_away_data
    f"{MAIN_SPREAD_COL}_TEAM_HOME",
    f"{MAIN_SPREAD_COL}_TEAM_AWAY",
    # Yahoo odds columns
    "spread_home",
    "spread_away",
    "moneyline_home",
    "moneyline_away",
    # "total_line", # This column is dropped after filling BetMGM missing values, so we won't have it in the final merged df
    # Yahoo percentage columns (betting behavior data)
    "total_pct_bets_over",
    "total_pct_bets_under",
    "total_pct_money_over",
    "total_pct_money_under",
    "spread_pct_bets_away",
    "spread_pct_bets_home",
    "spread_pct_money_away",
    "spread_pct_money_home",
    "moneyline_pct_bets_away",
    "moneyline_pct_bets_home",
    "moneyline_pct_money_away",
    "moneyline_pct_money_home",
    # Consensus data from sportsbooks
    "total_consensus_pct_over",
    "total_consensus_pct_under",
    "spread_consensus_pct_away",
    "spread_consensus_pct_home",
    "spread_consensus_opener_line_away",
    "spread_consensus_opener_line_home",
    "spread_consensus_opener_price_away",
    "spread_consensus_opener_price_home",
    "total_consensus_opener_line_over",
    "total_consensus_opener_price_over",
    "total_consensus_opener_line_under",
    "total_consensus_opener_price_under",
    "ml_consensus_opener_price_away",
    "ml_consensus_opener_price_home",
]

# Add all sportsbook-specific columns dynamically
# Common sportsbooks in the data
SPORTSBOOKS = [
    "pinnacle",
    "betmgm",
    "bet365",
    "caesars",
    "fanduel",
    "fanatics",
    "draftkings",
    "mybookie",
    "bovada",
    "betonline",
    "intertops",
    "heritage",
    "bookmaker",
    "lowvig",
    "betcris",
    "justbet",
    "sportsbetting",
    "gtbets",
    "consensus",
    "fanatics_sportsbook",
    "betrivers",
]

# Add total line and price columns for each book
for book in SPORTSBOOKS:
    if book != "consensus":  # consensus is handled separately above
        ODDS_COLUMNS.extend(
            [
                f"total_{book}_line_over",
                f"total_{book}_line_under",
                f"total_{book}_price_over",
                f"total_{book}_price_under",
            ]
        )

# Add spread columns for each book
for book in SPORTSBOOKS:
    if book not in [
        "consensus",
        "consensus_opener",
    ]:  # consensus variants handled separately
        ODDS_COLUMNS.extend(
            [
                f"spread_{book}_line_away",
                f"spread_{book}_line_home",
                f"spread_{book}_price_away",
                f"spread_{book}_price_home",
            ]
        )

# Add moneyline columns for each book
for book in SPORTSBOOKS:
    ODDS_COLUMNS.extend(
        [
            f"ml_{book}_price_away",
            f"ml_{book}_price_home",
        ]
    )

TARGET_COLUMN = "TOTAL_POINTS"

#: Every outcome column that must survive selection.
#:
#: ``TARGET_COLUMN`` stays as the totals target's name so nothing that imports it
#: changes behaviour. The three additions are what the spread market needs:
#: PTS_TEAM_HOME/PTS_TEAM_AWAY are the raw per-team finals (kept so HOME_MARGIN
#: stays reproducible and bets can be settled), and HOME_MARGIN is derived in
#: merge_home_away.
#:
#: These are OUTCOME facts, not features. Keeping them in the CSV is what makes a
#: spread target possible at all; keeping them out of X is enforced separately
#: and unconditionally by training_pipeline (LEAKING_TARGET_COLUMNS +
#: assert_no_leaking_features), for every strategy including the totals ones.
TARGET_COLUMNS: tuple[str, ...] = (
    TARGET_COLUMN,
    HOME_MARGIN_COL,
    PTS_HOME_COL,
    PTS_AWAY_COL,
)

FORBIDDEN_COLUMNS = [
    "DIFFERENCE_FROM_LINE",
    "DIFF_FROM_LINE",
    "TOTAL_PF",
    "IS_OVER_LINE",
    # PTS_PER_40 is a completed-game outcome. Only its shifted _BEFORE
    # derivatives may be used as features; the raw home/away versions are
    # target leakage.
    "PTS_PER_40_TEAM_HOME",
    "PTS_PER_40_TEAM_AWAY",
]


def select_training_columns(
    df_merged, original_columns, debug=False, keep_game_time=False
):
    """
    Select and organize columns for training dataset.

    Args:
        df_merged (pd.DataFrame): Merged home/away DataFrame with all features
        original_columns (list): List of original column names to check against
        debug (bool): If True, print information about deleted columns
        keep_game_time (bool): If True, keep GAME_TIME column for prediction purposes. Default: False

    Returns:
        pd.DataFrame: DataFrame with selected columns for training

    Raises:
        ValueError: If any disallowed original columns are present in the final training data
    """
    # Generate new list with _HOME and _AWAY appended
    columns_info_before = [f"{col}_TEAM_HOME" for col in TEAM_INFO_COLUMNS] + [
        f"{col}_TEAM_AWAY" for col in TEAM_INFO_COLUMNS
    ]

    columns_info_before.extend(STATIC_COLUMNS)

    # Insert columns that have BEFORE in the name
    columns_info_before.extend([col for col in df_merged.columns if "BEFORE" in col])

    # Add odds columns that exist in the dataframe (permit but don't require)
    odds_cols_present = [col for col in ODDS_COLUMNS if col in df_merged.columns]
    columns_info_before.extend(odds_cols_present)

    # Add any columns already carrying the unified odds-derived prefix
    columns_info_before.extend(
        [col for col in df_merged.columns if is_odds_column(col)]
    )

    # Add any columns that start with "ODDS_TOTAL_LINE_" prefix
    columns_info_before.extend(
        [col for col in df_merged.columns if col.startswith("ODDS_TOTAL_LINE_")]
    )

    # Add GAME_TIME if requested (for prediction purposes)
    if keep_game_time and "GAME_TIME" in df_merged.columns:
        columns_info_before.append("GAME_TIME")

    # Filter to only include columns that actually exist in df_merged
    columns_to_select = [col for col in columns_info_before if col in df_merged.columns]
    # Deduplicate while preserving order (some columns can be added by multiple rules)
    columns_to_select = list(dict.fromkeys(columns_to_select))

    # Add target/outcome columns if they exist
    columns_to_select.extend(
        column for column in TARGET_COLUMNS if column in df_merged.columns
    )
    columns_to_select = list(dict.fromkeys(columns_to_select))

    if debug:
        excluded_columns = [
            col for col in df_merged.columns if col not in columns_to_select
        ]
        if excluded_columns:
            print(f"Debug: {len(excluded_columns)} columns not selected for training:")
            for col in excluded_columns:
                print(f"  - {col}")

    df_training = df_merged[columns_to_select].copy()
    # Drop any forbidden columns if they exist
    columns_to_drop = [col for col in FORBIDDEN_COLUMNS if col in df_training.columns]
    # add to columns to discard anything that contains "DIFF_FROM" (unless it also contains "BEFORE")
    columns_to_drop.extend(
        [
            col
            for col in df_training.columns
            if "DIFF_FROM" in col and "_BEFORE" not in col
        ]
    )

    if columns_to_drop:
        if debug:
            print(f"Debug: Dropping {len(columns_to_drop)} forbidden columns:")
            for col in columns_to_drop:
                print(f"  - {col}")
        df_training = df_training.drop(columns=columns_to_drop)

    # Safety check: Ensure no disallowed original columns are present
    # Allowed columns include: static columns, target, odds, and team info with suffixes
    allowed_columns = set(STATIC_COLUMNS + list(TARGET_COLUMNS) + ODDS_COLUMNS)

    # Add team info columns with HOME/AWAY suffixes
    for col in TEAM_INFO_COLUMNS:
        allowed_columns.add(f"{col}_TEAM_HOME")
        allowed_columns.add(f"{col}_TEAM_AWAY")

    disallowed_columns = []

    for col in df_training.columns:
        # Check if this column is in original columns but not in allowed list
        # Skip columns with BEFORE (they are temporal features)
        # Skip columns that carry the unified "ODDS_" prefix (odds features)
        if (
            "_BEFORE" not in col
            and not is_odds_column(col)
            and col not in allowed_columns
            and col in original_columns
        ):
            disallowed_columns.append(col)

    if disallowed_columns:
        raise ValueError(
            f"Disallowed original columns found in training data: {disallowed_columns}. "
            "These columns should have '_BEFORE' suffix or be excluded."
        )

    # NOTE: the ODDS_ unification rename deliberately does NOT happen here.
    # Selection is not the last stage: engineer_odds_features() runs afterwards
    # and resolves its inputs by their raw names (total_<book>_price_over,
    # spread_consensus_opener_line_home, ...), so renaming at this point hides
    # them and silently drops every vig / no-vig / price-dispersion feature.
    # The unification is applied once at the end of each pipeline entry point
    # via nba_ou.config.odds_columns.apply_odds_prefix, and enforced there by
    # assert_odds_columns_prefixed.
    #
    # Canonical market normalisation runs HERE, at the one gate both datasets
    # pass through (create_df_to_predict for the closing data,
    # create_base_game_features for the intermediate data's base). Deriving the
    # spread's home orientation once, in one place, is the whole defence against
    # the two datasets disagreeing about a sign -- which they demonstrably would,
    # since their raw spread columns are stored with OPPOSITE conventions.
    df_training = add_canonical_market_columns(df_training)

    return df_training
