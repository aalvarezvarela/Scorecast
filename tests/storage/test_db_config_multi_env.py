import configparser

import pytest
from nba_ou.postgre_db.config import db_config


def _config(**sections: dict[str, str]) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read_dict(
        {
            "Database": {"DB_ENV": "supabase", "DB_NAME": "nba"},
            **sections,
        }
    )
    return config


@pytest.fixture
def fake_config(monkeypatch):
    def _install(**sections):
        config = _config(**sections)
        monkeypatch.setattr(db_config, "get_config", lambda: config)
        return config

    return _install


def test_aiven_dsn_is_read_from_the_aiven_section(fake_config, monkeypatch):
    monkeypatch.delenv("AIVEN_DB_URL", raising=False)
    fake_config(Aiven={"AIVEN_DB_URL": "postgres://u:p@aiven.example:1/defaultdb"})

    assert db_config.get_db_dsn("aiven") == "postgres://u:p@aiven.example:1/defaultdb"


def test_supabase_dsn_is_unaffected_by_the_aiven_section(fake_config, monkeypatch):
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    fake_config(
        DatabaseSupabase={
            "SUPABASE_DB_URL": "postgres://u:p@supabase.example:2/postgres"
        },
        Aiven={"AIVEN_DB_URL": "postgres://u:p@aiven.example:1/defaultdb"},
    )

    assert "supabase.example" in db_config.get_db_dsn("supabase")


def test_env_var_overrides_the_configured_dsn(fake_config, monkeypatch):
    fake_config(Aiven={"AIVEN_DB_URL": "postgres://from-file"})
    monkeypatch.setenv("AIVEN_DB_URL", "postgres://from-env")

    assert db_config.get_db_dsn("aiven") == "postgres://from-env"


def test_local_env_has_no_dsn(fake_config):
    fake_config(Aiven={"AIVEN_DB_URL": "postgres://u:p@aiven.example:1/defaultdb"})

    assert db_config.get_db_dsn("local") == ""


def test_missing_dsn_returns_empty_rather_than_raising(fake_config, monkeypatch):
    monkeypatch.delenv("AIVEN_DB_URL", raising=False)
    fake_config()

    assert db_config.get_db_dsn("aiven") == ""


def test_credentials_select_the_section_for_the_requested_env(fake_config, monkeypatch):
    monkeypatch.delenv("LOCAL_DB_PASSWORD", raising=False)
    monkeypatch.delenv("SUPABASE_DB_PASSWORD", raising=False)
    fake_config(
        DatabaseLocal={
            "DB_USER": "local_user",
            "DB_HOST": "127.0.0.1",
            "DB_PORT": "5432",
            "DB_PASSWORD": "local_pw",
        },
        DatabaseSupabase={
            "DB_USER": "supabase_user",
            "DB_HOST": "db.supabase.example",
            "DB_PORT": "5432",
            "DB_PASSWORD": "supabase_pw",
        },
    )

    # DB_ENV says supabase, but an explicit env argument must win.
    assert db_config.get_db_credentials(env="local")["user"] == "local_user"
    assert db_config.get_db_credentials()["user"] == "supabase_user"


def test_connect_nba_db_prefers_the_dsn_for_the_requested_env(fake_config, monkeypatch):
    monkeypatch.delenv("AIVEN_DB_URL", raising=False)
    fake_config(
        DatabaseSupabase={"SUPABASE_DB_URL": "postgres://supabase"},
        Aiven={"AIVEN_DB_URL": "postgres://aiven"},
    )

    seen: list[str] = []
    monkeypatch.setattr(db_config.psycopg, "connect", lambda dsn: seen.append(dsn))

    db_config.connect_nba_db()
    db_config.connect_nba_db(env="aiven")
    db_config.connect_line_history_db()

    # Default keeps the existing DB_ENV behaviour; the override targets Aiven.
    assert seen == ["postgres://supabase", "postgres://aiven", "postgres://aiven"]


def test_repo_root_secrets_file_is_overlaid(tmp_path, monkeypatch):
    """[Aiven] lives in the repo-root secrets file, not the package-local one."""
    package_root = tmp_path / "src" / "nba_ou"
    package_root.mkdir(parents=True)
    (package_root / "config.ini").write_text(
        "[Database]\nDB_ENV = local\nDB_NAME = nba\n"
    )
    (package_root / "config.secrets.ini").write_text(
        "[DatabaseSupabase]\nDB_PASSWORD = from_package\n"
    )
    (tmp_path / "config.secrets.ini").write_text(
        "[Aiven]\nAIVEN_DB_URL = postgres://root\n"
        "[DatabaseSupabase]\nDB_PASSWORD = from_root\n"
    )

    monkeypatch.setattr(db_config, "_CONFIG_INI", package_root / "config.ini")
    monkeypatch.setattr(db_config, "_SECRETS_INI", package_root / "config.secrets.ini")
    monkeypatch.setattr(
        db_config, "_REPO_ROOT_SECRETS_INI", tmp_path / "config.secrets.ini"
    )

    config = db_config.get_config()

    # Root-only keys become visible...
    assert config.get("Aiven", "AIVEN_DB_URL") == "postgres://root"
    # ...but the package-local file still wins on shared keys.
    assert config.get("DatabaseSupabase", "DB_PASSWORD") == "from_package"
