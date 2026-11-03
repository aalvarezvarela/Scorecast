import pytest
from nba_ou.data_processing.game_texts.players import (
    Player,
    sentence_spans,
    tag_players,
)
from nba_ou.data_processing.game_texts.team_perspective import perspective_texts

HEAT = [Player("Tyler", "Herro", "Heat"), Player("Bam", "Adebayo", "Heat")]
KNICKS = [Player("Jalen", "Brunson", "Knicks"), Player("OG", "Anunoby", "Knicks")]


def tag(text: str, players=HEAT + KNICKS, rule: str = "any") -> str:
    return tag_players(
        text,
        players,
        "Heat",
        "Knicks",
        target_label="Target Team",
        opponent_label="Opponent Team",
        label_rule=rule,
    ).text


def test_label_rule_any_skips_sentences_with_either_label():
    text = (
        "The Opponent Team played without Tyler Herro (toe). "
        "Brunson scored 30 and Adebayo had 12."
    )
    assert tag(text) == (
        "The Opponent Team played without Tyler Herro (toe). "
        "Brunson (Opponent Team player) scored 30 and "
        "Adebayo (Target Team player) had 12."
    )


def test_label_rule_own_skips_only_the_players_own_label():
    text = "The Opponent Team played without Tyler Herro (toe)."
    assert tag(text, rule="own") == (
        "The Opponent Team played without Tyler Herro (Target Team player) (toe)."
    )
    text = "The Target Team played without Tyler Herro (toe)."
    assert tag(text, rule="own") == text


def test_label_rule_never_notes_every_sentence():
    text = "The Target Team played without Tyler Herro (toe)."
    assert tag(text, rule="never") == (
        "The Target Team played without Tyler Herro (Target Team player) (toe)."
    )


def test_first_mention_per_sentence_and_after_the_possessive():
    text = "Anunoby's defense held Herro, and Anunoby scored. Anunoby rested."
    assert tag(text) == (
        "Anunoby's (Opponent Team player) defense held Herro (Target Team player), "
        "and Anunoby scored. Anunoby (Opponent Team player) rested."
    )


def test_accents_initials_and_suffixes():
    players = [
        Player("Luka", "Dončić", "Heat"),
        Player("OG", "Anunoby", "Knicks"),
        Player("Derrick", "Jones Jr.", "Knicks"),
    ]
    text = "Doncic watched. O.G. Anunoby scored. Jones Jr. dunked."
    assert tag(text, players) == (
        "Doncic (Target Team player) watched. "
        "O.G. Anunoby (Opponent Team player) scored. "
        "Jones Jr. (Opponent Team player) dunked."
    )


def test_lone_name_next_to_another_name_is_somebody_else():
    players = [Player("Kobe", "Bufkin", "Heat"), Player("Kevin", "Love", "Heat")]
    text = "Before the game, fans saluted Kobe Bryant. Coach Kevin Durant waved."
    result = tag_players(
        text,
        players,
        "Heat",
        "Knicks",
        target_label="Target Team",
        opponent_label="Opponent Team",
    )
    assert result.text == text
    assert {m.outcome for m in result.mentions} == {"other_person"}


def test_misspelt_other_half_is_the_same_player():
    players = [Player("Austin", "Reaves", "Heat")]
    assert tag("Then came Austin Reeves on the break.", players) == (
        "Then came Austin Reeves (Target Team player) on the break."
    )


def test_form_shared_by_both_teams_is_not_used():
    players = [
        Player("Jalen", "Williams", "Heat"),
        Player("Jalen", "Brunson", "Knicks"),
    ]
    assert tag("Jalen scored. Brunson passed.", players) == (
        "Jalen scored. Brunson (Opponent Team player) passed."
    )


def test_sentences_ignore_abbreviations_and_initials():
    text = "Jaren Jackson Jr. scored at 7 p.m. Friday. P.J. Tucker sat. Then it ended."
    assert [text[s:e] for s, e in sentence_spans(text)] == [
        "Jaren Jackson Jr. scored at 7 p.m. Friday.",
        "P.J. Tucker sat.",
        "Then it ended.",
    ]


def test_unknown_label_rule_is_rejected():
    with pytest.raises(ValueError):
        tag("x", rule="sometimes")


def test_perspective_texts_note_players_from_each_side():
    texts = perspective_texts(
        "Herro beat Brunson.", "Heat", "Knicks", players=HEAT + KNICKS, intro=False
    )
    assert texts["home_target"] == (
        "Herro (Target Team player) beat Brunson (Opponent Team player)."
    )
    assert texts["away_target"] == (
        "Herro (Opponent Team player) beat Brunson (Target Team player)."
    )
