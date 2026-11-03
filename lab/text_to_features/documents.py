"""Read archived game texts (NBA.com written reports, recap transcripts) from S3.

Both archives store one gzip JSON per game with ``text``, ``home`` and
``away`` (schedule nicknames); see ``lab/game_reports/README.md`` and
``docs/pipeline/recap_transcripts.md``.
"""

from __future__ import annotations

import gzip
import json
import random
from collections.abc import Iterable

from nba_ou.config.settings import SETTINGS

from lab.text_to_features.players import Player

SOURCES = {
    "report": ("train_data/game_reports/v1", "reports"),
    "transcript": (
        "train_data/game_recap_transcripts/v1/profile=wishper-v1",
        "transcripts",
    ),
}


def s3_client():
    from nba_ou.utils.s3_models import make_s3_client

    return make_s3_client(
        profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region
    )


def list_keys(client, source: str, season: str | None = None) -> list[str]:
    """Every document key of one source, optionally one season ("2025-26")."""
    prefix, folder = SOURCES[source]
    if season:
        prefix = f"{prefix}/season={season}"
    keys = []
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=SETTINGS.s3_bucket, Prefix=f"{prefix}/"
    ):
        keys += [
            item["Key"]
            for item in page.get("Contents", [])
            if f"/{folder}/" in item["Key"] and item["Key"].endswith(".json.gz")
        ]
    return sorted(keys)


def load_document(client, key: str, source: str) -> dict:
    body = client.get_object(Bucket=SETTINGS.s3_bucket, Key=key)["Body"].read()
    document = json.loads(gzip.decompress(body))
    document["source"] = source
    document["s3_key"] = key
    return document


def sample_documents(
    client,
    per_source: int = 2,
    sources: tuple[str, ...] = ("report", "transcript"),
    season: str | None = None,
    seed: int | None = None,
) -> list[dict]:
    """``per_source`` random documents from each source."""
    rng = random.Random(seed)
    documents = []
    for source in sources:
        keys = list_keys(client, source, season)
        for key in rng.sample(keys, min(per_source, len(keys))):
            documents.append(load_document(client, key, source))
    return documents


def load_game_players(game_ids: Iterable[str]) -> dict[str, list[Player]]:
    """Players listed for each game: box score (played, DNP, inactive) plus the
    NBA inactive list, one entry per player. Teams come from ``team_id``."""
    from nba_api.stats.static import players as static_players
    from nba_api.stats.static import teams
    from nba_ou.postgre_db.injuries_refs.fetch_injury_db.get_injury_data_from_db import (  # noqa: E501
        get_injury_data_from_db,
    )
    from nba_ou.postgre_db.players.fetch_players_data_from_db import (
        load_players_from_db,
    )

    ids = sorted({str(game_id) for game_id in game_ids})
    nicknames = {str(team["id"]): team["nickname"] for team in teams.get_teams()}
    # 2024-25 box scores carry only "S. Castle"; earlier ones only a full name.
    static = {
        str(p["id"]): (p["first_name"], p["last_name"])
        for p in static_players.get_players()
    }
    box = load_players_from_db(extra_game_ids=ids)
    inactive = get_injury_data_from_db(extra_game_ids=ids)
    if box is None or inactive is None:
        raise RuntimeError("Could not load the games' players from the database")

    rows = {}
    for row in box.itertuples(index=False):
        names = (row.firstname, row.familyname)
        if not all(names):
            names = static.get(str(row.player_id)) or (
                str(row.player_name).partition(" ")[::2]
            )
        rows[(row.game_id, str(row.player_id))] = (row.team_id, *names)
    for row in inactive.itertuples(index=False):
        rows.setdefault(
            (row.GAME_ID, str(row.PLAYER_ID)),
            (row.TEAM_ID, row.FIRST_NAME, row.LAST_NAME),
        )
    players: dict[str, list[Player]] = {game_id: [] for game_id in ids}
    for (game_id, _), (team_id, first, last) in rows.items():
        if game_id in players and str(team_id) in nicknames:
            players[game_id].append(
                Player(first or "", last or "", nicknames[str(team_id)])
            )
    return players
