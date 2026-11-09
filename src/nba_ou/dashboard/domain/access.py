"""Who may see what: the dashboard's sign-in gate, without Streamlit.

Sign-in is Google (``st.login``); the dashboard's own ``app.users`` row says
whether the signed-in account is approved. A first sign-in creates the row as
pending, unless the email is one of the configured admins (the bootstrap):
those start active and admin.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from nba_ou.postgre_db.app_aiven.schema import USER_ACTIVE, USER_DISABLED


class Access(StrEnum):
    ANONYMOUS = "anonymous"
    PENDING = "pending"
    DISABLED = "disabled"
    ACTIVE = "active"
    ADMIN = "admin"


@dataclass(frozen=True)
class AppUser:
    user_id: int
    email: str
    display_name: str | None
    status: int
    is_admin: bool

    @property
    def label(self) -> str:
        return self.display_name or self.email


def normalize_email(email: str | None) -> str:
    return (email or "").strip().lower()


def admin_emails(raw: str | Iterable[str] | None) -> frozenset[str]:
    """``ADMIN_EMAILS`` from the secrets: a list, or one comma-separated string."""
    if raw is None:
        return frozenset()
    items = raw.split(",") if isinstance(raw, str) else raw
    return frozenset(e for e in (normalize_email(item) for item in items) if e)


def access_of(user: AppUser | None) -> Access:
    if user is None:
        return Access.ANONYMOUS
    if user.status == USER_DISABLED:
        return Access.DISABLED
    if user.status != USER_ACTIVE:
        return Access.PENDING
    return Access.ADMIN if user.is_admin else Access.ACTIVE


def can_use_app(access: Access) -> bool:
    return access in (Access.ACTIVE, Access.ADMIN)
