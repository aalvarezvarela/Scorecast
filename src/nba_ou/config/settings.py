"""
NBA Over/Under Predictor - Settings Module

This module handles application settings and configuration loading from config.ini.
It provides a centralized way to access all configuration parameters.

ADDING NEW SETTINGS:
--------------------
To add a new setting, simply add it to the CONFIG_SCHEMA dictionary:

    "property_name": ("Section", "CONFIG_KEY", type, default_value)

Where:
    - property_name: The Python property name (snake_case)
    - Section: The INI file section name
    - CONFIG_KEY: The key name in the INI file
    - type: str, bool, int, or "secret" (for sensitive data)
    - default_value: Fallback value (None if required)

Example:
    "my_new_setting": ("MySection", "MY_SETTING", str, "default_value")

The setting will automatically be available as SETTINGS.my_new_setting

For settings with custom logic (like s3_aws_profile), create a @property method.
"""

import configparser
import os
from pathlib import Path
from typing import Any

# Locate the config.ini file (assumes it's in the project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_FILE = PROJECT_ROOT / "nba_ou/config.ini"
SECRETS_FILE = PROJECT_ROOT / "nba_ou/config.secrets.ini"


# Configuration schema: maps property names to (section, key, type, default)
# type can be: str, bool, int, secret
CONFIG_SCHEMA = {
    # Scraping / Browser
    "headless": ("Scraping", "HEADLESS", bool, True),
    # Database
    "db_env": ("Database", "DB_ENV", str, None),
    "db_name": ("Database", "DB_NAME", str, None),
    "schema_name_games": ("Database", "SCHEMA_NAME_GAMES", str, None),
    "schema_name_game_time_index": (
        "Database",
        "SCHEMA_NAME_GAME_TIME_INDEX",
        str,
        "nba_game_time_index",
    ),
    "schema_name_players": ("Database", "SCHEMA_NAME_PLAYERS", str, None),
    "schema_name_odds": ("Database", "SCHEMA_NAME_ODDS", str, None),
    "schema_name_odds_mgm": ("Database", "SCHEMA_NAME_ODDS_MGM", str, None),
    "schema_name_odds_sportsbook": (
        "Database",
        "SCHEMA_NAME_ODDS_SPORTSBOOK",
        str,
        None,
    ),
    "schema_name_odds_yahoo": ("Database", "SCHEMA_NAME_ODDS_YAHOO", str, None),
    "schema_name_predictions": ("Database", "SCHEMA_NAME_PREDICTIONS", str, None),
    "schema_name_injuries": ("Database", "SCHEMA_NAME_INJURIES", str, None),
    "schema_name_refs": ("Database", "SCHEMA_NAME_REFS", str, None),
    # Database Local
    "db_user_local": ("DatabaseLocal", "DB_USER", str, None),
    "db_host_local": ("DatabaseLocal", "DB_HOST", str, None),
    "db_port_local": ("DatabaseLocal", "DB_PORT", int, None),
    "db_sslmode_local": ("DatabaseLocal", "DB_SSLMODE", str, None),
    # Database Supabase
    "db_user_supabase": ("DatabaseSupabase", "DB_USER", str, None),
    "db_port_supabase": ("DatabaseSupabase", "DB_PORT", int, None),
    "db_sslmode_supabase": ("DatabaseSupabase", "DB_SSLMODE", str, None),
    "db_name_supabase": ("DatabaseSupabase", "DB_NAME", str, None),
    "db_host_supabase": ("DatabaseSupabase", "DB_HOST", str, None),
    # Paths
    "report_path": ("Paths", "REPORT_PATH", str, None),
    # Injuries
    "nba_injury_reports_url": ("Injuries", "NBA_INJURY_REPORTS_URL", str, None),
    # Refs
    "nba_official_assignments_url": ("Refs", "NBA_OFFICIAL_ASSIGNMENTS_URL", str, None),
    # Odds
    "odds_data_folder": ("Odds", "ODDS_DATA_FOLDER", str, None),
    "odds_base_url": ("Odds", "BASE_URL", str, None),
    "odds_save_pickle": ("Odds", "SAVE_ODDS_PICKLE", bool, None),
    "odds_pickle_path": ("Odds", "ODDS_PICKLE_PATH", str, None),
    "main_sportsbook": ("Odds", "MAIN_SPORTSBOOK", str, "consensus_opener"),
    # Secrets (not in config.ini, must be in secrets file or env vars)
    "odds_api_key": ("Odds", "ODDS_API_KEY", "secret", None),
    "db_password": ("DatabaseSupabase", "DB_PASSWORD", "secret", None),
    # S3 (Note: s3_aws_profile has custom @property logic below)
    "s3_aws_region": ("S3", "AWS_REGION", str, None),
    "s3_bucket": ("S3", "BUCKET", str, None),
    "s3_models_prefix": ("S3", "MODELS_PREFIX", str, None),
    "s3_local_model_cache_dir": ("S3", "LOCAL_MODEL_CACHE_DIR", str, None),
}


class Settings:
    def __init__(self, config_path: str | None = None, secrets_path: str | None = None):
        self.config = configparser.ConfigParser()

        main_path = Path(config_path) if config_path else CONFIG_FILE
        if not main_path.exists():
            raise FileNotFoundError(
                f"Config file not found at {main_path}. "
                "Please ensure config.ini exists in the project root."
            )

        # 1) Load committed config
        self.config.read(main_path)

        # 2) Overlay optional secrets config (local)
        secrets_path = Path(secrets_path) if secrets_path else SECRETS_FILE
        if secrets_path.exists():
            self.config.read(secrets_path)

        # Cache for resolved values
        self._cache: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        """
        Dynamically resolve configuration values based on the schema.

        This method is called when an attribute is not found through normal lookup.
        It checks the CONFIG_SCHEMA and resolves the value from the config file.
        """
        # Avoid infinite recursion for special attributes
        if name.startswith("_") or name in ("config", "_cache"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # Check cache first
        if name in self._cache:
            return self._cache[name]

        # Check if this property is in the schema
        if name not in CONFIG_SCHEMA:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        section, key, value_type, default = CONFIG_SCHEMA[name]

        # Handle secrets
        if value_type == "secret":
            env_var = key  # Environment variable name
            value = self._get_secret(
                section, key, env_var, required=default is None, fallback=default
            )
        # Handle booleans
        elif value_type is bool:
            value = self.config.getboolean(section, key, fallback=default)
        # Handle integers
        elif value_type is int:
            value = self.config.getint(section, key, fallback=default)
        # Handle strings
        else:
            value = self.config.get(section, key, fallback=default)

        # Cache the result
        self._cache[name] = value
        return value

    # -------------------------------------------------------------------------
    # Secret resolver
    # -------------------------------------------------------------------------
    def _get_secret(
        self,
        section: str,
        key: str,
        env_var: str,
        *,
        required: bool = True,
        fallback: str | None = None,
    ) -> str:
        """
        Resolve secrets with priority:
          1) Environment variable
          2) config.ini/config.secrets.ini (overlay)
          3) fallback (if provided)
        """
        v = os.getenv(env_var)
        if v is not None and v != "":
            return v

        if self.config.has_option(section, key):
            v = self.config.get(section, key).strip()
            if v != "":
                return v

        if fallback is not None:
            return fallback

        if required:
            raise ValueError(
                f"Missing required secret [{section}] {key}. "
                f"Set env var {env_var} or define it in config.secrets.ini."
            )

        return ""

    def _get_multiline_list(
        self,
        section: str,
        key: str,
        *,
        fallback: list[str] | None = None,
    ) -> list[str]:
        raw_value = self.config.get(section, key, fallback="")
        items = [
            line.strip()
            for line in raw_value.replace(",", "\n").splitlines()
            if line.strip()
        ]
        return items or list(fallback or [])

    # -------------------------------------------------------------------------
    # Special properties with custom logic
    # -------------------------------------------------------------------------
    @property
    def s3_aws_profile(self) -> str | None:
        """
        Select AWS profile for S3 access.

        Resolution order:
          1) `S3_AWS_PROFILE` environment variable (empty disables profile)
          2) If running in GitHub Actions (`GITHUB_ACTIONS=true`), use no profile
          3) `S3.AWS_PROFILE` from config.ini
        """
        env_profile = os.getenv("S3_AWS_PROFILE")
        if env_profile is not None:
            env_profile = env_profile.strip()
            return env_profile or None

        if os.getenv("GITHUB_ACTIONS", "").lower() == "true":
            return None

        profile = self.config.get("S3", "AWS_PROFILE", fallback=None)
        if profile is None:
            return None

        profile = profile.strip()
        return profile or None

    @property
    def default_schema_version(self) -> str:
        """Training-data schema version used by CLI tools when none is given.

        A default for typing convenience only. It is deliberately NOT an
        implicit part of an enabled slot's identity -- every row of
        ``ENABLED_MODELS`` states its own version, so a 2_6 slot can be
        promoted while the 2_5 slots keep serving, with no flag day.
        """
        from nba_ou.config.dataset_versions import TRAINING_DATA_SCHEMA_VERSION

        configured = self.config.get(
            "PredictionModels", "DEFAULT_SCHEMA_VERSION", fallback=""
        ).strip()
        return configured or TRAINING_DATA_SCHEMA_VERSION

    @property
    def prediction_model_slots(self) -> list:
        """Model slots enabled for prediction and retraining runs.

        Each row of ``[PredictionModels] ENABLED_MODELS`` is::

            schema_version | target | horizon_minutes | variant

        ``variant`` may be omitted and defaults to ``main``;
        ``horizon_minutes`` may be ``tpool`` for a pooled model. An EMPTY table
        is a valid state meaning "nothing promoted yet", and every caller must
        treat it as nothing to do rather than an error -- the system has to be
        runnable before the first promotion.

        Returns ``list[nba_ou.modeling.registry_paths.ModelSlot]``; the import
        is deferred so that settings stays importable without the modeling
        package.
        """
        if "prediction_model_slots" in self._cache:
            return self._cache["prediction_model_slots"]

        from nba_ou.modeling.registry_paths import ModelSlot, parse_horizon

        rows = self._get_multiline_list("PredictionModels", "ENABLED_MODELS")
        slots = []
        for row in rows:
            if row.startswith((";", "#")):
                continue
            fields = [field.strip() for field in row.split("|")]
            if len(fields) == 3:
                fields.append("main")
            if len(fields) != 4:
                raise ValueError(
                    "Each [PredictionModels] ENABLED_MODELS row must be "
                    "'schema_version | target | horizon_minutes | variant' "
                    f"(variant optional). Got: {row!r}"
                )
            schema_version, target, horizon, variant = fields
            horizon_minutes = (
                parse_horizon(horizon)
                if not horizon.lstrip("-").isdigit()
                else int(horizon)
            )
            slots.append(
                ModelSlot(
                    schema_version=schema_version,
                    target=target,
                    horizon_minutes=horizon_minutes,
                    variant=variant or "main",
                    root=self.s3_models_prefix or "models/",
                )
            )

        duplicates = [
            slot.describe()
            for slot in slots
            if [other.describe() for other in slots].count(slot.describe()) > 1
        ]
        if duplicates:
            raise ValueError(
                "Duplicate slots in [PredictionModels] ENABLED_MODELS: "
                f"{sorted(set(duplicates))}"
            )

        self._cache["prediction_model_slots"] = slots
        return slots

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def get_absolute_path(self, relative_path: str) -> Path:
        """
        Convert a relative path from config to an absolute path.

        Args:
            relative_path: Path relative to project root

        Returns:
            Absolute Path object
        """
        if os.path.isabs(relative_path):
            return Path(relative_path)
        return PROJECT_ROOT / relative_path

    def ensure_directories_exist(self):
        """
        Create all necessary directories if they don't exist.
        """
        directories = [
            self.report_path,
            self.odds_data_folder,
        ]

        for directory in directories:
            abs_path = self.get_absolute_path(directory)
            abs_path.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        """String representation of settings."""
        return (
            f"Settings(\n"
            f"  db_env='{self.db_env}',\n"
            f"  s3_bucket='{self.s3_bucket}',\n"
            f"  report_path='{self.report_path}'\n"
            f")"
        )


# Create a default settings instance for easy importing
SETTINGS = Settings()
