"""
Centralized database configuration module.
Reads database credentials from config.ini file.
"""

import configparser
import os
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent.parent

_CONFIG_INI = _PROJECT_ROOT / "config.ini"
_SECRETS_INI = _PROJECT_ROOT / "config.secrets.ini"
# A second secrets file lives at the repo root. Both are gitignored and in use,
# so read both; the package-local one is read last and therefore wins on any
# key the two files share.
_REPO_ROOT_SECRETS_INI = _PROJECT_ROOT.parent.parent / "config.secrets.ini"


def get_config() -> configparser.ConfigParser:
    """Load configuration from config.ini and overlay config.secrets.ini if present."""
    if not _CONFIG_INI.exists():
        raise FileNotFoundError(f"Config file not found at: {_CONFIG_INI}")

    config = configparser.ConfigParser()
    config.read(_CONFIG_INI)

    # Overlay optional local secrets files (gitignored)
    if _REPO_ROOT_SECRETS_INI.exists():
        config.read(_REPO_ROOT_SECRETS_INI)
    if _SECRETS_INI.exists():
        config.read(_SECRETS_INI)

    return config


def _get_env_or_config(
    config: configparser.ConfigParser,
    section: str,
    key: str,
    env_var: str,
    *,
    required: bool = True,
    fallback: str | None = None,
) -> str:
    """Resolve value with priority: env var -> config -> fallback -> error."""
    v = os.getenv(env_var)
    if v is not None and v != "":
        return v

    if config.has_option(section, key):
        v = config.get(section, key).strip()
        if v != "":
            return v

    if fallback is not None:
        return fallback

    if required:
        raise ValueError(
            f"Missing required setting [{section}] {key}. "
            f"Set env var {env_var} or define it in config.secrets.ini."
        )

    return ""


def get_db_env() -> str:
    """Return current default DB environment: local|supabase|aiven."""
    config = get_config()
    return (
        os.getenv("DB_ENV", config.get("Database", "DB_ENV", fallback="local"))
        .strip()
        .lower()
    )


# Environments that are configured by a full connection URI rather than by
# discrete host/user/port settings, as (config section, key, env var).
_DSN_SOURCES: dict[str, tuple[str, str, str]] = {
    "supabase": ("DatabaseSupabase", "SUPABASE_DB_URL", "SUPABASE_DB_URL"),
    "aiven": ("Aiven", "AIVEN_DB_URL", "AIVEN_DB_URL"),
}

_CREDENTIAL_SECTIONS: dict[str, str] = {
    "supabase": "DatabaseSupabase",
    "aiven": "DatabaseAiven",
    "local": "DatabaseLocal",
}


def get_db_dsn(env: str | None = None) -> str:
    """Return the connection URI for ``env``, or "" if it is not URI-configured."""
    resolved = (env or get_db_env()).strip().lower()
    spec = _DSN_SOURCES.get(resolved)
    if spec is None:
        return ""

    section, key, env_var = spec
    return _get_env_or_config(
        get_config(), section, key, env_var, required=False, fallback=""
    )


def get_db_credentials(env: str | None = None) -> dict[str, Any]:
    """
    Get database credentials with env var overrides and secrets support.
    """
    config = get_config()
    db_env = (env or get_db_env()).strip().lower()
    section = _CREDENTIAL_SECTIONS.get(db_env, "DatabaseLocal")

    # Non-secret settings (config.ini)
    user = config.get(section, "DB_USER")
    host = config.get(section, "DB_HOST")
    port = config.get(section, "DB_PORT")
    sslmode = config.get(section, "DB_SSLMODE", fallback="").strip() or None

    # DB name can be per-environment; fallback to [Database].DB_NAME
    dbname = config.get(section, "DB_NAME", fallback=config.get("Database", "DB_NAME"))
    dbname = dbname.strip().strip('"').strip("'")

    # Secret password: env var preferred, else secrets file
    password_env = {
        "supabase": "SUPABASE_DB_PASSWORD",
        "aiven": "AIVEN_DB_PASSWORD",
    }.get(db_env, "LOCAL_DB_PASSWORD")
    password = _get_env_or_config(
        config, section, "DB_PASSWORD", password_env, required=True
    )

    return {
        "env": db_env,
        "user": user,
        "password": password,
        "host": host,
        "port": port,
        "dbname": dbname,
        "sslmode": sslmode,
    }


def get_schema_name_games() -> str:
    return get_config().get("Database", "SCHEMA_NAME_GAMES")


def get_schema_name_game_time_index() -> str:
    return get_config().get(
        "Database",
        "SCHEMA_NAME_GAME_TIME_INDEX",
        fallback="nba_game_time_index",
    )


def get_schema_name_players() -> str:
    return get_config().get("Database", "SCHEMA_NAME_PLAYERS")


def get_schema_name_all_star_voting() -> str:
    return get_config().get(
        "Database",
        "SCHEMA_NAME_ALL_STAR_VOTING",
        fallback="nba_all_star_voting",
    )


def get_schema_name_odds_sportsbook() -> str:
    return get_config().get("Database", "SCHEMA_NAME_ODDS_SPORTSBOOK")


def get_schema_name_odds_yahoo() -> str:
    return get_config().get("Database", "SCHEMA_NAME_ODDS_YAHOO")


def get_schema_name_predictions() -> str:
    return get_config().get("Database", "SCHEMA_NAME_PREDICTIONS")


def get_schema_name_injuries() -> str:
    return get_config().get("Database", "SCHEMA_NAME_INJURIES")


def get_schema_name_refs() -> str:
    return get_config().get("Database", "SCHEMA_NAME_REFS")


def connect_postgres_db() -> psycopg.Connection:
    """
    Connect to the 'postgres' database for admin tasks.

    Note: On Supabase you typically do not need admin DB creation; still usable for
    maintenance queries if your role permits.
    """
    c = get_db_credentials()

    kwargs: dict[str, Any] = dict(
        dbname="postgres",
        user=c["user"],
        password=c["password"],
        host=c["host"],
        port=c["port"],
        autocommit=True,
    )
    if c["sslmode"]:
        kwargs["sslmode"] = c["sslmode"]

    return psycopg.connect(**kwargs)


def connect_nba_db(env: str | None = None) -> psycopg.Connection:
    """Connect to a configured database (local, Supabase or Aiven).

    ``env`` overrides the process-wide ``DB_ENV`` for this connection only, so a
    single process can read one environment and write another -- which the
    line-history load needs (read games from Supabase, write to Aiven).
    """
    resolved_env = (env or get_db_env()).strip().lower()

    # Prefer DSN where one is configured (Supabase pooler/session mode, Aiven
    # service URI). This also fixes IPv6 issues in GitHub runners.
    dsn = get_db_dsn(resolved_env)
    if dsn:
        return psycopg.connect(dsn)

    c = get_db_credentials(env=resolved_env)
    kwargs: dict[str, Any] = dict(
        dbname=c["dbname"],
        user=c["user"],
        password=c["password"],
        host=c["host"],
        port=c["port"],
    )
    if c.get("sslmode"):
        kwargs["sslmode"] = c["sslmode"]

    return psycopg.connect(**kwargs)


def connect_line_history_db() -> psycopg.Connection:
    """Connect to the Aiven instance that hosts the line-history store."""
    return connect_nba_db(env="aiven")


def connect_schema_db(schema: str) -> psycopg.Connection:
    """Connect and set search_path to the given schema."""
    conn = connect_nba_db()
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SET search_path TO {}, public;").format(sql.Identifier(schema))
        )
    conn.commit()
    return conn


def connect_games_db() -> psycopg.Connection:
    return connect_schema_db(get_schema_name_games())


def connect_players_db() -> psycopg.Connection:
    return connect_schema_db(get_schema_name_players())


def connect_all_star_voting_db() -> psycopg.Connection:
    return connect_schema_db(get_schema_name_all_star_voting())


def connect_odds_sportsbook_db() -> psycopg.Connection:
    return connect_schema_db(get_schema_name_odds_sportsbook())


def connect_odds_yahoo_db() -> psycopg.Connection:
    return connect_schema_db(get_schema_name_odds_yahoo())


def connect_predictions_db() -> psycopg.Connection:
    return connect_schema_db(get_schema_name_predictions())


def connect_injuries_db() -> psycopg.Connection:
    return connect_schema_db(get_schema_name_injuries())


def connect_refs_db() -> psycopg.Connection:
    return connect_schema_db(get_schema_name_refs())
