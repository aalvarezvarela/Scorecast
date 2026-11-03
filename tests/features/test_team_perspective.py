import pytest
from nba_ou.data_processing.game_texts.team_perspective import (
    TEAMS,
    document_perspectives,
    perspective_texts,
    replace_teams,
    strip_boilerplate,
)


def rewrite(text: str, target: str, opponent: str) -> str:
    return replace_teams(text, target, opponent).text


def test_full_names_city_and_short_forms_become_role_labels():
    text = (
        "The Los Angeles Lakers beat the Minnesota Timberwolves. "
        "The Wolves trailed before Minnesota and the T-Wolves rallied; L.A. held on."
    )
    assert rewrite(text, "Lakers", "Timberwolves") == (
        "The Target Team beat the Opponent Team. "
        "The Opponent Team trailed before Opponent Team and the Opponent Team "
        "rallied; Target Team held on."
    )


def test_other_teams_stay_whole():
    text = "The Lakers host the LA Clippers on Friday; Minnesota visits Denver."
    assert rewrite(text, "Lakers", "Timberwolves") == (
        "The Target Team host the LA Clippers on Friday; Opponent Team visits Denver."
    )


def test_shared_city_names_neither_team():
    text = "The Lakers beat the Clippers in Los Angeles; L.A. Clippers fans left early."
    assert rewrite(text, "Lakers", "Clippers") == (
        "The Target Team beat the Opponent Team in Los Angeles; "
        "Opponent Team fans left early."
    )


def test_common_words_match_only_capitalized():
    text = "He went into heat check mode as the Heat ran; the magic faded, Magic lost."
    assert rewrite(text, "Heat", "Magic") == (
        "He went into heat check mode as the Target Team ran; "
        "the magic faded, Opponent Team lost."
    )


def test_city_of_a_team_that_did_not_play_is_kept():
    text = "Dallas lost in Washington, then beat Boston."
    assert rewrite(text, "Mavericks", "Celtics") == (
        "Target Team lost in Washington, then beat Opponent Team."
    )


def test_names_containing_an_alias_are_replaced_too():
    # Accepted error: no phrase is protected, so a player named like a team
    # loses that part of his name.
    text = "P.J. Washington scored 20 against Washington."
    assert rewrite(text, "Mavericks", "Wizards") == (
        "P.J. Opponent Team scored 20 against Opponent Team."
    )


def test_distinctive_names_match_in_any_case_and_possessive():
    text = "the lakers lead. Indiana’s bench and the PACERS' run."
    assert rewrite(text, "Pacers", "Lakers") == (
        "the Opponent Team lead. Target Team’s bench and the Target Team' run."
    )


def test_mentions_record_roles():
    result = replace_teams("Lakers beat Pacers; Denver next.", "Lakers", "Pacers")
    assert [(m.surface, m.role) for m in result.mentions] == [
        ("Lakers", "target"),
        ("Pacers", "opponent"),
        ("Denver", "other"),
    ]
    assert result.count("target") == result.count("opponent") == 1


def test_strip_boilerplate_drops_dateline_and_footer():
    text = (
        "SACRAMENTO, Calif. (AP) — De’Aaron Fox had 29 points.\n\n"
        "Kings: Host Phoenix on Friday.\n\n"
        "---\n\n"
        "More AP NBA: https://apnews.com/hub/nba\n\n"
        "See AP’s full NBA coverage here"
    )
    assert strip_boilerplate(text) == (
        "De’Aaron Fox had 29 points.\n\nKings: Host Phoenix on Friday."
    )
    assert strip_boilerplate("- Luka Doncic and the LA Lakers.") == (
        "Luka Doncic and the LA Lakers."
    )


def test_perspective_texts_mirror_each_other():
    texts = perspective_texts(
        "TORONTO (AP) Raptors beat Nets.", "Raptors", "Nets", intro=False
    )
    assert texts == {
        "raw": "Raptors beat Nets.",
        "home_target": "Target Team beat Opponent Team.",
        "away_target": "Opponent Team beat Target Team.",
    }


def test_title_and_intro_sentence():
    texts = document_perspectives(
        {
            "title": "Lakers beat Timberwolves 118-110",
            "text": "LOS ANGELES (AP) — Minnesota lost.",
            "home": "Lakers",
            "away": "Timberwolves",
        }
    )
    assert texts["raw"] == ("Lakers beat Timberwolves 118-110\n\nMinnesota lost.")
    assert texts["away_target"] == (
        "In this text, Target Team is the Minnesota Timberwolves (visiting team) "
        "and Opponent Team is the Los Angeles Lakers (home team).\n\n"
        "Opponent Team beat Target Team 118-110\n\n"
        "Target Team lost."
    )


def test_intro_sentence_without_team_names():
    texts = perspective_texts("x", "Lakers", "Clippers", name_teams=False)
    assert texts["home_target"] == (
        "In this text, Target Team is the home team and Opponent Team is the "
        "visiting team.\n\nx"
    )


def test_unknown_or_same_team_is_rejected():
    with pytest.raises(KeyError):
        replace_teams("text", "SuperSonics", "Lakers")
    with pytest.raises(ValueError):
        replace_teams("text", "Lakers", "Lakers")


def test_every_franchise_is_listed():
    assert len(TEAMS) == 30
    assert TEAMS["Clippers"].full_name == "Los Angeles Clippers"
    assert TEAMS["Trail Blazers"].full_name == "Portland Trail Blazers"
