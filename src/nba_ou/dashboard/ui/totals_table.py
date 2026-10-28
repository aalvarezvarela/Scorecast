"""A semantic comparison table with real, spanning column-group headers."""

import re
from html import escape

import pandas as pd


def _cell(value: str) -> str:
    """Render only the bold numbers and signal badges emitted by our formatter."""
    text = escape(str(value))
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return re.sub(
        r":(red|blue|gray)-badge\[(OVER|UNDER|NO EDGE|UNAVAILABLE)\]",
        r'<span class="pick pick-\1">\2</span>',
        text,
    )


def totals_table_html(table: pd.DataFrame) -> str:
    """Preserve every formatted model/snapshot row and its four data columns."""
    body = []
    for (model, snapshot), row in table.iterrows():
        cells = "".join(
            f'<td class="{column[1].lower()}">{_cell(value)}</td>'
            for column, value in row.items()
        )
        body.append(
            f'<tr><th scope="row">{escape(str(model))}</th>'
            f'<th scope="row">{escape(str(snapshot))}</th>{cells}</tr>'
        )
    return (
        '<div class="prediction-comparison" tabindex="0" '
        'role="region" aria-label="Model predictions and market margins">'
        '<table><thead><tr><th rowspan="2" scope="col">Model</th>'
        '<th rowspan="2" scope="col">Snapshot</th>'
        '<th colspan="2" scope="colgroup">Total points</th>'
        '<th colspan="2" scope="colgroup">Line error</th></tr><tr>'
        '<th scope="col">Prediction</th><th scope="col">Margin</th>'
        '<th scope="col">Prediction</th><th scope="col">Margin</th>'
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
    )
