"""The line and the compact table of a bet card: one row per model's prediction."""

from dataclasses import dataclass
from html import escape


@dataclass(frozen=True)
class BetTableRow:
    model: str
    detail: str
    prediction: str
    edge: str
    pick: str
    color: str
    #: The line this prediction was made against, and whether it is another
    #: than the one the card offers (shown only when the card's lines differ).
    line: str = "—"
    line_differs: bool = False


def bet_line_html(label: str, value: str | None) -> str:
    """The card's line of a market, large: ``Totals · Bet365 line`` over
    ``224.5``. ``value`` None is a market without a stored line."""
    missing = value is None
    return (
        f'<div class="bet-line{" bet-line-missing" if missing else ""}" role="group" '
        f'aria-label="{escape(label)}">'
        f'<span class="bet-line-label">{escape(label)}</span>'
        f'<span class="bet-line-value">{escape("No line" if missing else value)}</span>'
        "</div>"
    )


def bet_predictions_html(rows: list[BetTableRow], *, show_line: bool = False) -> str:
    """The model and its target on the left, the numbers aligned on the right.

    ``show_line`` adds each prediction's own line, for a card whose
    predictions were made against different lines.
    """

    def line_cell(row: BetTableRow) -> str:
        if not show_line:
            return ""
        differs = ' class="line-differs"' if row.line_differs else ""
        return f"<td{differs}>{escape(row.line)}</td>"

    body = "".join(
        f'<tr><th scope="row">{escape(row.model)}<small>{escape(row.detail)}</small>'
        f"</th>{line_cell(row)}"
        f'<td class="prediction">{escape(row.prediction)}</td>'
        f"<td>{escape(row.edge)}</td>"
        f'<td><span class="pick pick-{escape(row.color)}">{escape(row.pick)}</span>'
        "</td></tr>"
        for row in rows
    )
    line_head = '<th scope="col">Line</th>' if show_line else ""
    return (
        '<div class="prediction-comparison bet-predictions" tabindex="0" '
        'role="region" aria-label="Model predictions for this game">'
        f'<table><thead><tr><th scope="col">Model</th>{line_head}'
        '<th scope="col">Prediction</th>'
        '<th scope="col">Edge</th><th scope="col">Pick</th></tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )
