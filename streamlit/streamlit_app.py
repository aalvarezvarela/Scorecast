"""Unified Aiven dashboard. Run from the repository root."""

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
NBA_MARK = APP_DIR / "assets/nba-mark.svg"
sys.path.insert(0, str(APP_DIR.parent / "src"))

st.set_page_config(
    page_title="NBA · Predictions",
    page_icon=str(NBA_MARK),
    layout="wide",
)
st.logo(str(NBA_MARK), size="large", icon_image=str(NBA_MARK))
st.html(APP_DIR.parent / "src/nba_ou/dashboard/ui/dashboard.css")

for key, value in {
    "today_target": 1,
    "today_view": "Cards",
    "today_reference": "Bet365",
    "today_charts": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

from nba_ou.dashboard.domain.access import Access, can_use_app  # noqa: E402
from nba_ou.dashboard.ui.auth import resolve_access, sign_out_button  # noqa: E402


def _page(name: str, title: str, icon: str, **kwargs) -> st.Page:
    return st.Page(
        str(APP_DIR / f"app_pages/{name}.py"), title=title, icon=icon, **kwargs
    )


# The gate: signed out, only the welcome page exists; signed in but not
# approved, only the pending page. Nothing else is built for them.
access, user = resolve_access()
st.session_state["current_access"] = access
st.session_state["current_user"] = user
if access == Access.ANONYMOUS:
    pages = [_page("welcome", "Welcome", ":material/waving_hand:", default=True)]
elif not can_use_app(access):
    pages = [
        _page("pending", "Access pending", ":material/hourglass_top:", default=True)
    ]
else:
    pages = [
        _page(
            "today",
            "Today's predictions",
            ":material/sports_basketball:",
            default=True,
        ),
        _page("bets", "Bets", ":material/payments:"),
        _page("bet_history", "Bet history", ":material/receipt_long:"),
        _page("yesterday", "Yesterday's predictions", ":material/history:"),
        _page("historical", "Historical analysis", ":material/analytics:"),
    ]
    if access == Access.ADMIN:
        pages.append(_page("admin", "Accounts", ":material/manage_accounts:"))
    with st.sidebar:
        st.caption(f"Signed in as {user.label}")
        sign_out_button()

page = st.navigation(pages, position="top")
page.run()
