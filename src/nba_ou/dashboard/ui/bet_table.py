"""The compact table of a bet card: one line per model's prediction."""

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


def bet_predictions_html(rows: list[BetTableRow]) -> str:
    """The model and its target on the left, the numbers aligned on the right."""
    body = "".join(
        f'<tr><th scope="row">{escape(row.model)}<small>{escape(row.detail)}</small>'
        f'</th><td class="prediction">{escape(row.prediction)}</td>'
        f"<td>{escape(row.edge)}</td>"
        f'<td><span class="pick pick-{escape(row.color)}">{escape(row.pick)}</span>'
        "</td></tr>"
        for row in rows
    )
    return (
        '<div class="prediction-comparison bet-predictions" tabindex="0" '
        'role="region" aria-label="Model predictions for this game">'
        '<table><thead><tr><th scope="col">Model</th><th scope="col">Prediction</th>'
        '<th scope="col">Edge</th><th scope="col">Pick</th></tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )
