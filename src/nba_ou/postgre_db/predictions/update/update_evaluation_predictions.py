import numpy as np
import pandas as pd
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_predictions,
)
from psycopg import sql


def get_available_training_code_tags() -> list[str]:
    """
    Return distinct non-empty training_code_tag values stored in the predictions table.
    """
    schema = get_schema_name_predictions()
    table = schema

    conn = connect_nba_db()
    try:
        query_obj = sql.SQL("""
            SELECT DISTINCT training_code_tag
            FROM {}.{}
            WHERE training_code_tag IS NOT NULL
              AND btrim(training_code_tag) <> ''
        """).format(sql.Identifier(schema), sql.Identifier(table))
        query = query_obj.as_string(conn)
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    if df.empty or "training_code_tag" not in df.columns:
        return []

    tags = (
        df["training_code_tag"]
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    return sorted(tags)


def get_games_with_total_scored_points(
    date: str = None,
    start_date: str = None,
    end_date: str = None,
    only_null: bool = False,
) -> pd.DataFrame:
    """
    Returns predictions from the configured predictions table
    (``SCHEMA_NAME_PREDICTIONS``, currently ``nba_ou_predictions``).
    By default, returns only games where total_scored_points IS NOT NULL.

    Args:
        date (str, optional): Filter for a specific date (YYYY-MM-DD).
        start_date (str, optional): Start date for range (inclusive, YYYY-MM-DD).
        end_date (str, optional): End date for range (inclusive, YYYY-MM-DD).
        only_null (bool, optional): If True, returns games where total_scored_points IS NULL.
                                       If False (default), returns games where total_scored_points IS NOT NULL.

    Returns:
        pd.DataFrame: All columns of the predictions table for rows matching the filter.
    """
    schema = get_schema_name_predictions()
    table = schema  # Your convention: table name = schema name

    conn = connect_nba_db()
    try:
        # Set the total_scored_points filter based on only_null parameter
        if only_null:
            where_clauses = ["total_scored_points IS NULL"]
        else:
            where_clauses = ["total_scored_points IS NOT NULL"]

        params = []
        if date:
            where_clauses.append("game_date = %s")
            params.append(date)
        if start_date:
            where_clauses.append("game_date >= %s")
            params.append(start_date)
        if end_date:
            where_clauses.append("game_date <= %s")
            params.append(end_date)
        where_sql = " AND ".join(where_clauses)

        query_obj = sql.SQL("""
            SELECT *
            FROM {}.{}
            WHERE {}
        """).format(sql.Identifier(schema), sql.Identifier(table), sql.SQL(where_sql))
        query = query_obj.as_string(conn)

        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

    return df


def add_ou_betting_metrics(
    df: pd.DataFrame,
    *,
    line_col: str = "total_over_under_line",
    actual_total_col: str = "total_scored_points",
    regressor_pred_total_col: str = "predicted_total_score",
    classifier_pred_side_col: str = "classifier_prediction_model2",
    over_odds_col: str = "average_total_over_money",
    under_odds_col: str = "average_total_under_money",
) -> pd.DataFrame:
    """
    Adds over/under correctness and $1-bet profit metrics for:
      - regressor (using predicted total vs line)
      - classifier (using predicted side vs line)
      - both (only if both models pick the same side; else profit=0)

    Output columns added:
      - regressor_side, classifier_side, actual_side
      - regressor_correct, classifier_correct, both_correct
      - profit_regressor, profit_classifier, profit_both_agree

    Odds handling:
      - Accepts either American odds (e.g., -110, +120) OR decimal odds (e.g., 1.91, 2.20).
      - If abs(odds) >= 100, assumes American and converts to decimal first.
      - Profit on a $1 stake when winning = (decimal_odds - 1); loss = -1; push/unknown = 0.
      - Push (actual_total == line) => profit = 0, correct = False.
      - Missing data => profit = 0, correct = False (conservative).
    """
    required = [
        line_col,
        actual_total_col,
        regressor_pred_total_col,
        classifier_pred_side_col,
        over_odds_col,
        under_odds_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    def american_to_decimal(a: pd.Series) -> pd.Series:
        a_num = pd.to_numeric(a, errors="coerce")
        dec = pd.Series(np.nan, index=a.index, dtype=float)
        pos = a_num > 0
        neg = a_num < 0
        dec[pos] = 1.0 + (a_num[pos] / 100.0)
        dec[neg] = 1.0 + (100.0 / a_num[neg].abs())
        return dec

    def to_decimal_odds(odds: pd.Series) -> pd.Series:
        """
        If abs(odds) >= 100 -> treat as American, convert.
        Else treat as decimal already (expects values like 1.01+).
        """
        o = pd.to_numeric(odds, errors="coerce")
        dec = o.astype(float).copy()

        american_mask = o.notna() & (o.abs() >= 100)
        if american_mask.any():
            dec.loc[american_mask] = american_to_decimal(o.loc[american_mask]).values

        # Guardrails: typical decimal odds should be > 1.0; otherwise set NaN
        dec = dec.where(dec > 1.0, np.nan)
        return dec

    def normalize_side(series: pd.Series) -> pd.Series:
        s = series.copy()
        if s.dtype == object:
            s_str = s.astype(str).str.strip().str.lower()
            over_mask = s_str.isin(["over", "o", "1", "true", "yes"])
            under_mask = s_str.isin(["under", "u", "0", "false", "no"])

            s_num = pd.to_numeric(s, errors="coerce")
            num_mask = s_num.notna() & ~(over_mask | under_mask)

            side = pd.Series(np.nan, index=s.index, dtype=object)
            side[over_mask] = "over"
            side[under_mask] = "under"
            side[num_mask & (s_num > 0)] = "over"
            side[num_mask & (s_num <= 0)] = "under"
            return side

        s_num = pd.to_numeric(s, errors="coerce")
        return pd.Series(
            np.where(s_num > 0, "over", np.where(s_num <= 0, "under", np.nan)),
            index=s.index,
            dtype=object,
        )

    # --- sides ---
    line = pd.to_numeric(df[line_col], errors="coerce")
    actual_total = pd.to_numeric(df[actual_total_col], errors="coerce")
    reg_pred_total = pd.to_numeric(df[regressor_pred_total_col], errors="coerce")

    df["actual_side"] = np.where(
        actual_total > line, "over", np.where(actual_total < line, "under", np.nan)
    )
    df["regressor_side"] = np.where(
        reg_pred_total > line, "over", np.where(reg_pred_total < line, "under", np.nan)
    )
    df["classifier_side"] = normalize_side(df[classifier_pred_side_col])
    df["both_agree"] = df["regressor_side"] == df["classifier_side"]
    df["both_agree_side"] = np.where(df["both_agree"], df["regressor_side"], np.nan)

    # --- correctness (push or missing => False) ---
    df["regressor_correct"] = (df["regressor_side"] == df["actual_side"]) & df[
        "actual_side"
    ].notna()
    df["classifier_correct"] = (df["classifier_side"] == df["actual_side"]) & df[
        "actual_side"
    ].notna()
    df["both_agree_correct"] = (df["both_agree_side"] == df["actual_side"]) & df[
        "actual_side"
    ].notna()

    # --- profit per $1 stake (using decimal odds) ---
    df[over_odds_col] = to_decimal_odds(df[over_odds_col])
    df[under_odds_col] = to_decimal_odds(df[under_odds_col])

    def profit_from_pick(prediction, is_correct, odds_over, odds_under) -> float:
        """
        Profit for a $1 (or 1 euro) bet following `prediction` side ('over'/'under').

        Returns:
            - If win: (decimal_odds * 1) - 1 = profit from winning bet
            - If loss: -1 (lost the 1 euro stake)
            - If no prediction/push/unknown: 0

        Example:
            - Bet 1 euro on over at odds 1.91, win: profit = 1.91 - 1 = 0.91
            - Bet 1 euro on under at odds 2.10, lose: profit = -1.00
        """
        # No bet placed or missing prediction
        if not prediction or pd.isna(prediction):
            return 0.0

        # Missing odds data - can't calculate profit
        if pd.isna(odds_over) or pd.isna(odds_under):
            return 0.0

        # Calculate profit based on outcome
        if is_correct:
            # Win: return stake * odds - stake = stake * (odds - 1)
            # For 1 euro stake: 1 * (odds - 1) = odds - 1
            if prediction == "over":
                return float(odds_over - 1)
            elif prediction == "under":
                return float(odds_under - 1)
            else:
                return 0.0
        else:
            # Loss: lose the 1 euro stake
            return -1.0

    df["profit_regressor"] = df.apply(
        lambda row: profit_from_pick(
            row["regressor_side"],
            row["regressor_correct"],
            row[over_odds_col],
            row[under_odds_col],
        ),
        axis=1,
    )
    df["profit_classifier"] = df.apply(
        lambda row: profit_from_pick(
            row["classifier_side"],
            row["classifier_correct"],
            row[over_odds_col],
            row[under_odds_col],
        ),
        axis=1,
    )

    # both agree profit
    df["profit_both_agree"] = df.apply(
        lambda row: (
            profit_from_pick(
                row["both_agree_side"],
                row["both_agree_correct"],
                row[over_odds_col],
                row[under_odds_col],
            )
            if pd.notna(row["both_agree_side"])
            else 0.0
        ),
        axis=1,
    )

    # Ensure bool dtype
    for c in ["regressor_correct", "classifier_correct", "both_agree_correct"]:
        df[c] = df[c].fillna(False).astype(bool)

    return df


def compute_ou_betting_statistics(
    df: pd.DataFrame,
    *,
    actual_side_col: str = "actual_side",
    reg_correct_col: str = "regressor_correct",
    clf_correct_col: str = "classifier_correct",
    both_correct_col: str = "both_agree_correct",
    reg_profit_col: str = "profit_regressor",
    clf_profit_col: str = "profit_classifier",
    both_profit_col: str = "profit_both_agree",
    both_pick_col: str = "both_agree_side",
    game_date_col: str = "game_date",
    print_report: bool = True,
    title: str = "Over/Under Betting Performance Summary",
    float_decimals: int = 4,
) -> pd.DataFrame:
    """
    Builds summary statistics for:
      - regressor
      - classifier
      - both-agree strategy (only bets when both agree)

    Returns a single-row DataFrame with:
      - n_games_total
      - n_resolved (non-push rows, where actual_side is known)
      - per-method: n_bets, accuracy, total_profit, avg_profit_per_bet, avg_profit_per_day

    If print_report=True, prints a formatted report.
    """

    required = [
        actual_side_col,
        reg_correct_col,
        clf_correct_col,
        both_correct_col,
        reg_profit_col,
        clf_profit_col,
        both_profit_col,
        both_pick_col,
        game_date_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns (run add_ou_betting_metrics first): {missing}"
        )

    n_games_total = int(len(df))

    # "Resolved" games are those that are not pushes and have known actual side
    resolved_mask = df[actual_side_col].notna()
    n_resolved = int(resolved_mask.sum())

    # Calculate number of unique days
    n_days = df[game_date_col].nunique() if game_date_col in df.columns else 0

    # Calculate mean prediction error (predicted - actual)
    # Only for resolved games where both predicted_total_score and total_scored_points exist
    pred_total_col = "predicted_total_score"
    actual_total_col = "total_scored_points"

    if pred_total_col in df.columns and actual_total_col in df.columns:
        pred_total = pd.to_numeric(df[pred_total_col], errors="coerce")
        actual_total = pd.to_numeric(df[actual_total_col], errors="coerce")
        valid_mask = pred_total.notna() & actual_total.notna()

        prediction_errors = pred_total - actual_total
        mean_prediction_error = (
            float(prediction_errors[valid_mask].mean())
            if valid_mask.sum() > 0
            else float("nan")
        )
        mean_abs_prediction_error = (
            float(prediction_errors[valid_mask].abs().mean())
            if valid_mask.sum() > 0
            else float("nan")
        )
    else:
        mean_prediction_error = float("nan")
        mean_abs_prediction_error = float("nan")

    def _accuracy(mask: pd.Series, correct_col: str) -> float:
        denom = int(mask.sum())
        if denom == 0:
            return float("nan")
        return float(df.loc[mask, correct_col].mean())

    def _profit_stats(
        mask: pd.Series, profit_col: str
    ) -> tuple[int, float, float, float, float]:
        """
        Returns: (n_bets, total_profit, avg_profit_per_bet, avg_profit_per_day, avg_stake_per_day)
        avg_profit_per_bet excludes zeros (0 means no-bet / push / unknown by your convention).
        avg_profit_per_day is the average daily profit across all unique game dates.
        avg_stake_per_day is the average amount of money bet per day (number of bets per day * 1 euro).
        """
        profits = pd.to_numeric(df.loc[mask, profit_col], errors="coerce").fillna(0.0)

        total_profit = float(profits.sum())

        bet_mask = profits != 0
        n_bets = int(bet_mask.sum())
        avg_profit_per_bet = (
            float(profits.loc[bet_mask].mean()) if n_bets > 0 else float("nan")
        )

        # Calculate average profit per day and average stake per day
        if n_days > 0 and game_date_col in df.columns:
            # Group by game_date, sum profits per day, then average across days
            daily_profits = df.loc[mask].groupby(game_date_col)[profit_col].sum()
            avg_profit_per_day = float(daily_profits.mean())

            # Count bets per day (non-zero profits) and average across days
            # Each bet is 1 euro, so avg_stake_per_day = avg number of bets per day * 1
            daily_bets = (
                df.loc[mask]
                .groupby(game_date_col)[profit_col]
                .apply(lambda x: (x != 0).sum())
            )
            avg_stake_per_day = float(daily_bets.mean())
        else:
            avg_profit_per_day = float("nan")
            avg_stake_per_day = float("nan")

        return (
            n_bets,
            total_profit,
            avg_profit_per_bet,
            avg_profit_per_day,
            avg_stake_per_day,
        )

    # Regressor and classifier: bet whenever game is resolved (non-push)
    reg_mask = resolved_mask
    clf_mask = resolved_mask

    # Both-agree: bet only when both agree AND game resolved
    both_mask = resolved_mask & df[both_pick_col].notna()

    (
        reg_n_bets,
        reg_total_profit,
        reg_avg_profit,
        reg_avg_profit_per_day,
        reg_avg_stake_per_day,
    ) = _profit_stats(reg_mask, reg_profit_col)
    (
        clf_n_bets,
        clf_total_profit,
        clf_avg_profit,
        clf_avg_profit_per_day,
        clf_avg_stake_per_day,
    ) = _profit_stats(clf_mask, clf_profit_col)
    (
        both_n_bets,
        both_total_profit,
        both_avg_profit,
        both_avg_profit_per_day,
        both_avg_stake_per_day,
    ) = _profit_stats(both_mask, both_profit_col)

    out = {
        "n_games_total": n_games_total,
        "n_resolved": n_resolved,
        "n_days": n_days,
        "mean_prediction_error": mean_prediction_error,
        "mean_abs_prediction_error": mean_abs_prediction_error,
        "regressor_n_bets": reg_n_bets,
        "regressor_accuracy": _accuracy(reg_mask, reg_correct_col),
        "regressor_total_profit": reg_total_profit,
        "regressor_avg_profit_per_bet": reg_avg_profit,
        "regressor_avg_profit_per_day": reg_avg_profit_per_day,
        "regressor_avg_stake_per_day": reg_avg_stake_per_day,
        "classifier_n_bets": clf_n_bets,
        "classifier_accuracy": _accuracy(clf_mask, clf_correct_col),
        "classifier_total_profit": clf_total_profit,
        "classifier_avg_profit_per_bet": clf_avg_profit,
        "classifier_avg_profit_per_day": clf_avg_profit_per_day,
        "classifier_avg_stake_per_day": clf_avg_stake_per_day,
        "both_agree_n_bets": both_n_bets,
        "both_agree_accuracy": _accuracy(both_mask, both_correct_col),
        "both_agree_total_profit": both_total_profit,
        "both_agree_avg_profit_per_bet": both_avg_profit,
        "both_agree_avg_profit_per_day": both_avg_profit_per_day,
        "both_agree_avg_stake_per_day": both_avg_stake_per_day,
    }

    stats_df = pd.DataFrame([out])

    if print_report:
        _print_ou_stats_report(
            stats_df=stats_df,
            title=title,
            float_decimals=float_decimals,
        )

    return stats_df


def compute_daily_prediction_errors(
    df: pd.DataFrame,
    *,
    game_date_col: str = "game_date",
    pred_total_col: str = "predicted_total_score",
    actual_total_col: str = "total_scored_points",
) -> pd.DataFrame:
    """
    Computes daily prediction errors (predicted - actual).

    Returns a DataFrame with columns:
        - game_date
        - n_games (number of games per day)
        - mean_error (average prediction error per day)
        - mean_abs_error (average absolute prediction error per day)
    """
    # Filter to games with valid predicted and actual totals
    pred_total = pd.to_numeric(df[pred_total_col], errors="coerce")
    actual_total = pd.to_numeric(df[actual_total_col], errors="coerce")
    valid_mask = pred_total.notna() & actual_total.notna()

    valid_df = df[valid_mask].copy()
    valid_df["prediction_error"] = pred_total[valid_mask] - actual_total[valid_mask]
    valid_df["abs_prediction_error"] = valid_df["prediction_error"].abs()

    # Group by date and calculate errors
    daily_errors = []

    for date, group in valid_df.groupby(game_date_col):
        n_games = len(group)
        mean_error = group["prediction_error"].mean()
        mean_abs_error = group["abs_prediction_error"].mean()

        daily_errors.append(
            {
                "game_date": date,
                "n_games": n_games,
                "mean_error": mean_error,
                "mean_abs_error": mean_abs_error,
            }
        )

    return pd.DataFrame(daily_errors).sort_values("game_date")


def compute_daily_accuracy(
    df: pd.DataFrame,
    *,
    game_date_col: str = "game_date",
    reg_correct_col: str = "regressor_correct",
    clf_correct_col: str = "classifier_correct",
    both_correct_col: str = "both_agree_correct",
    both_pick_col: str = "both_agree_side",
    actual_side_col: str = "actual_side",
) -> pd.DataFrame:
    """
    Computes daily accuracy for each strategy.

    Returns a DataFrame with columns:
        - game_date
        - regressor_accuracy
        - classifier_accuracy
        - both_agree_accuracy
        - n_games (number of resolved games per day)
    """
    # Filter to resolved games only
    resolved_df = df[df[actual_side_col].notna()].copy()

    # Group by date and calculate accuracy
    daily_stats = []

    for date, group in resolved_df.groupby(game_date_col):
        n_games = len(group)

        # Regressor accuracy
        reg_acc = group[reg_correct_col].mean() if n_games > 0 else np.nan

        # Classifier accuracy
        clf_acc = group[clf_correct_col].mean() if n_games > 0 else np.nan

        # Both agree accuracy (only count games where both models agree)
        both_games = group[group[both_pick_col].notna()]
        both_acc = (
            both_games[both_correct_col].mean() if len(both_games) > 0 else np.nan
        )

        daily_stats.append(
            {
                "game_date": date,
                "n_games": n_games,
                "regressor_accuracy": reg_acc,
                "classifier_accuracy": clf_acc,
                "both_agree_accuracy": both_acc,
            }
        )

    return pd.DataFrame(daily_stats).sort_values("game_date")


def plot_daily_accuracy(
    daily_df: pd.DataFrame,
    *,
    show_plot: bool = True,
) -> None:
    """
    Plots daily accuracy for each strategy over time.

    Args:
        daily_df: DataFrame from compute_daily_accuracy
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    # Convert game_date to datetime if it's not already
    dates = pd.to_datetime(daily_df["game_date"])

    # Plot each strategy
    ax.plot(
        dates,
        daily_df["regressor_accuracy"] * 100,
        marker="o",
        label="Regressor",
        linewidth=2,
        markersize=6,
    )
    ax.plot(
        dates,
        daily_df["classifier_accuracy"] * 100,
        marker="s",
        label="Classifier",
        linewidth=2,
        markersize=6,
    )
    ax.plot(
        dates,
        daily_df["both_agree_accuracy"] * 100,
        marker="^",
        label="Both Agree",
        linewidth=2,
        markersize=6,
    )

    # Add 50% reference line
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% (Break-even)")

    # Formatting
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        "Daily Prediction Accuracy by Strategy", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45, ha="right")

    # Set y-axis range
    ax.set_ylim(0, 100)

    plt.tight_layout()

    if show_plot:
        plt.show()
    else:
        plt.close()


def _print_ou_stats_report(
    *,
    stats_df: pd.DataFrame,
    title: str,
    float_decimals: int = 2,
) -> None:
    """Pretty-prints the single-row stats_df produced by compute_ou_betting_statistics."""
    if stats_df.empty:
        print(f"{title}\n(no data)")
        return

    row = stats_df.iloc[0].to_dict()

    n_games_total = int(row["n_games_total"])
    n_resolved = int(row["n_resolved"])
    n_days = int(row.get("n_days", 0))

    def fmt_pct(x: float) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/a"
        return f"{100.0 * float(x):.{float_decimals}f}%"

    def fmt_num(x: float) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/a"
        return f"{float(x):.{float_decimals}f}"

    def fmt_int(x: int) -> str:
        return f"{int(x):d}"

    methods = [
        ("Regressor", "regressor"),
        ("Classifier", "classifier"),
        ("Both Agree", "both_agree"),
    ]

    # Header
    print()
    print(f"{title} | Games: {n_games_total} | Resolved: {n_resolved} | Days: {n_days}")
    print("=" * 110)

    # Compact table
    print(
        f"{'Method':<12} {'Bets':>5} {'Acc%':>6} {'Tot€':>7} {'€/Bet':>7} {'€/Day':>7} {'Stake/Day':>10}"
    )
    print("-" * 110)

    for label, prefix in methods:
        n_bets = row[f"{prefix}_n_bets"]
        acc = row[f"{prefix}_accuracy"]
        total_profit = row[f"{prefix}_total_profit"]
        avg_profit = row[f"{prefix}_avg_profit_per_bet"]
        avg_profit_per_day = row[f"{prefix}_avg_profit_per_day"]
        avg_stake_per_day = row[f"{prefix}_avg_stake_per_day"]

        acc_str = fmt_pct(acc) if not np.isnan(acc) else "n/a"

        line = (
            f"{label:<12} "
            f"{fmt_int(n_bets):>5} "
            f"{acc_str:>6} "
            f"{fmt_num(total_profit):>7} "
            f"{fmt_num(avg_profit):>7} "
            f"{fmt_num(avg_profit_per_day):>7} "
            f"{fmt_num(avg_stake_per_day):>10}"
        )
        print(line)

    print()
    print("Note: Accuracy = correct predictions / resolved games (excl. pushes)")
    print()


if __name__ == "__main__":
    df = get_games_with_total_scored_points()
    print(df)
    df_with_metrics = add_ou_betting_metrics(df)
    print(df_with_metrics)
    # Save to Excel
    # df_with_metrics.to_excel("/home/adrian_alvarez/Projects/NBA_over_under_predictor/Predictions/temp.xlsx", index=False)
    stats = compute_ou_betting_statistics(df_with_metrics, print_report=True)
    # print(stats)
