"""Compact result tables and inspectable historical prediction details."""

from html import escape

import pandas as pd

from nba_ou.dashboard.config import SIGNAL_COLORS
from nba_ou.dashboard.domain.dates import madrid_clock, madrid_time
from nba_ou.dashboard.domain.picks import bet_text
from nba_ou.dashboard.ui.formatting import number

RESULT_COLORS = {"WIN": "green", "LOSS": "red"}


def checkpoint_summary_table(summary: pd.DataFrame, total_games: int) -> pd.DataFrame:
    """Use the same compact checkpoint breakdown for every model family."""
    display = pd.DataFrame(index=summary.index)
    display["Snapshot"] = summary["Checkpoint"].map(
        {
            "Closing · T−6h": "Closing at 6h",
            "Closing · T−4h": "Closing at 4h",
            "Closing · Final pre-game": "Closing at closing",
            "Intermediate · T−8h": "Intermediate at 8h",
            "Intermediate · T−6h": "Intermediate at 6h",
            "Intermediate · T−4h": "Intermediate at 4h",
        }
    )
    display["Games"] = summary["Games"].map(lambda n: f"{n}/{total_games}")
    display["Wins"] = [
        f"{wins}/{wins + losses}"
        for wins, losses in zip(summary["Wins"], summary["Losses"], strict=True)
    ]
    display["Win rate"] = summary["Win rate"].map(
        lambda n: f"{n:.1f}%" if pd.notna(n) else "—"
    )
    display["MAE"] = summary["MAE (pts)"].map(
        lambda n: f"{n:.2f} pts" if pd.notna(n) else "—"
    )
    return display.set_index("Snapshot")


def game_results_table(games: pd.DataFrame) -> pd.DataFrame:
    """Keep scores visible in Table view even when there are no model rows."""
    result = pd.DataFrame(index=games.index)
    result["Matchup"] = games["away_team"] + " @ " + games["home_team"]
    result["Tip-off (Madrid)"] = games["tipoff_at"].map(madrid_time)
    final = (
        games["status"].eq(1)
        & games["home_points"].notna()
        & games["away_points"].notna()
    )
    result["Away points"] = games["away_points"].where(final)
    result["Home points"] = games["home_points"].where(final)
    result["Status"] = [
        "Final" if done else "Void" if status == 2 else "Pending"
        for done, status in zip(final, games["status"], strict=True)
    ]
    return result


def _badge(text: str, color: str) -> str:
    return f'<span class="pick pick-{color}">{escape(text)}</span>'


def _pick(row) -> str:
    if row["signal"] not in ("OVER", "UNDER", "HOME", "AWAY"):
        return "NO BET" if row["signal"] == "NO EDGE" else "NO LINE"
    return bet_text(
        row["signal"],
        row["reference_line"],
        home=row["home_team"],
        away=row["away_team"],
    )


def history_table_html(rows: pd.DataFrame, *, spread: bool) -> str:
    body = []
    for _, row in rows.iterrows():
        available = row["available"]
        checkpoint = escape(row["checkpoint"])
        prediction = "—"
        pick = "—"
        if available:
            published = row["available_at"]
            checkpoint += (
                f'<small title="{escape(madrid_time(published))}">'
                f"Available {madrid_clock(published)} Madrid</small>"
            )
            value = row["predicted_error"] if spread else row["predicted_total"]
            prediction = f"<strong>{number(value, signed=spread)}</strong>"
            if not spread:
                prediction += f"<small>{number(row['comparison_edge'], signed=True)} vs line</small>"
            pick = _badge(_pick(row), SIGNAL_COLORS[row["signal"]])
        result = _badge(row["result"], RESULT_COLORS.get(row["result"], "gray"))
        body.append(
            f'<tr><th scope="row">{escape(row["model_label"])}</th>'
            f'<th scope="row">{checkpoint}</th><td class="prediction">{prediction}</td>'
            f"<td>{pick}</td><td>{result}</td></tr>"
        )
    label = "Spread error" if spread else "Predicted total"
    return (
        '<div class="prediction-comparison history-comparison" tabindex="0" '
        'role="region" aria-label="Historical predictions and game results">'
        '<table><thead><tr><th scope="col">Model</th><th scope="col">Checkpoint</th>'
        f'<th scope="col">{label}</th><th scope="col">Pick / Line</th>'
        '<th scope="col">Result</th></tr></thead><tbody>'
        + "".join(body)
        + "</tbody></table></div>"
    )


def history_details(rows: pd.DataFrame, *, spread: bool) -> pd.DataFrame:
    result = pd.DataFrame(index=rows.index)
    result["Matchup"] = rows["away_team"] + " @ " + rows["home_team"]
    result["Final score (away–home)"] = [
        f"{int(row['away_points'])}–{int(row['home_points'])}"
        if row["status"] == 1
        and pd.notna(row["away_points"])
        and pd.notna(row["home_points"])
        else "Void"
        if row["status"] == 2
        else "Pending"
        for _, row in rows.iterrows()
    ]
    result["Model"] = rows["model_label"]
    result["Target"] = rows["target"].map(
        {1: "Line error", 2: "Total points", 3: "Spread"}
    )
    result["Checkpoint"] = rows["checkpoint"]
    result["Source"] = rows["source"]
    result["Prediction available"] = rows["available"]
    result["Predicted total" if not spread else "Spread error"] = rows[
        "predicted_total" if not spread else "predicted_error"
    ]
    result["Stored line"] = rows["reference_line"]
    result["Edge vs stored line"] = rows["comparison_edge"]
    result["Pick"] = rows["signal"]
    result["Result"] = rows["result"]
    result["Actual total" if not spread else "Actual home margin"] = rows[
        "actual_value"
    ]
    result["Prediction error (pts)"] = rows["prediction_error"]
    for column, label in (
        ("predicted_at", "Predicted at"),
        ("available_at", "Available at"),
        ("input_at", "Input at"),
        ("checkpoint_at", "Checkpoint at"),
    ):
        result[f"{label} (Madrid)"] = (
            rows[column].map(madrid_time) if column in rows else "—"
        )
    for column, label in (
        ("model_horizon_minutes", "Model horizon (min)"),
        ("snapshot_horizon_minutes", "Input horizon (min)"),
        ("prediction_value", "Original output"),
        ("model_line", "Original model line"),
        ("anchor_book", "Bookmaker"),
        ("input_mode", "Input mode"),
        ("workflow", "Workflow"),
        ("run_id", "Run"),
        ("fit_id", "Fit"),
    ):
        result[label] = rows[column] if column in rows else None
    return result
