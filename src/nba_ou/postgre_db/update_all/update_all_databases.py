import time

from nba_ou.config.settings import SETTINGS
from nba_ou.fetch_data.live_games.live_games import (
    get_game_ids_for_date,
    get_live_game_ids,
)
from nba_ou.postgre_db.games.update_games.update_database import (
    update_team_players_database,
)
from nba_ou.postgre_db.injuries_refs.update_ref_injuries_database.update_refs_injuries_database import (
    update_refs_injuries_database,
)
from nba_ou.postgre_db.line_history_aiven.update import resolve_season_years
from nba_ou.postgre_db.odds_sportsbook.update_sportsbook.update_sportsbook_database import (
    update_odds_sportsbook_database,
)
from nba_ou.postgre_db.odds_yahoo.update_yahoo.update_yahoo_database import (
    update_odds_yahoo_database,
)


def update_all_databases(
    start_season_year: int | None = None,
    end_season_year: int | None = None,
    only_new_games: bool = True,
    headless: bool | None = None,
    sleep_seconds_between_seasons: float = 2.0,
    exclude_game_date: str | None = None,
) -> None:
    """
    Update from `start_season_year`-YY through `end_season_year`-(YY+1).

    Args:
        start_season_year: First season start year (2005 -> 2005-06). Defaults
            to the current season.
        end_season_year: Last season start year (2024 -> 2024-25). Defaults to
            `start_season_year`.
        sleep_seconds_between_seasons: Small pause to be polite with the NBA API.
    """
    if headless is None:
        headless = SETTINGS.headless

    seasons = resolve_season_years(start_season_year, end_season_year)
    print(f"Updating season(s): {seasons}")

    games_id_to_exclude = set(get_live_game_ids())
    if exclude_game_date:
        games_id_to_exclude.update(get_game_ids_for_date(exclude_game_date))
        print(
            f"Excluding {len(games_id_to_exclude)} game IDs from upload "
            f"(live games + games on {exclude_game_date})."
        )

    games_id_to_exclude = list(games_id_to_exclude)

    for y in seasons:
        season_year = str(y)
        print(f"\n=== Backfilling season starting {season_year} season) ===")

        limit_reached, exclude_game_ids, new_data_found = update_team_players_database(
            season_year=season_year,
            games_id_to_exclude=games_id_to_exclude,
        )
        if not new_data_found and only_new_games:
            print(
                f"No new games found for season {season_year}. Skipping odds and refs/injuries updates."
            )
            continue

        print(f"\n--- Updating sportsbook odds for season {y} ---")

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                sportsbook_results = update_odds_sportsbook_database(
                    season_year=season_year,
                    headless=headless,
                )
                print(f"Sportsbook update results: {sportsbook_results}")
                break
            except Exception as e:
                if attempt < max_retries:
                    print(
                        f"Sportsbook update failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                    )
                    print("Retrying...")
                    time.sleep(2.0)
                else:
                    print(
                        f"Sportsbook update failed after {max_retries + 1} attempts: {e}"
                    )
                    raise RuntimeError("Sportsbook update failed during backfill.") from e

        print(f"\n--- Updating Yahoo odds for season {y} ---")

        for attempt in range(max_retries + 1):
            try:
                yahoo_results = update_odds_yahoo_database(season_year=season_year)
                print(f"Yahoo update results: {yahoo_results}")
                break
            except Exception as e:
                if attempt < max_retries:
                    print(
                        f"Yahoo update failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                    )
                    print("Retrying...")
                    time.sleep(2.0)
                else:
                    print(f"Yahoo update failed after {max_retries + 1} attempts: {e}")
                    raise RuntimeError("Yahoo update failed during backfill.") from e

        print(f"\n--- Updating refs/injuries for season {y} ---")
        limit_reached = update_refs_injuries_database(
            season_year=season_year,
            exclude_game_ids=exclude_game_ids,
        )

        if limit_reached:
            print(
                "\n⚠️ API limit reached or throttling detected while updating refs/injuries. "
                "Stopping backfill now. Re-run this script later to continue."
            )
            raise RuntimeError("API limit reached during refs/injuries backfill.")

        time.sleep(sleep_seconds_between_seasons)

    print("\n✅ Backfill completed for all requested seasons.")
