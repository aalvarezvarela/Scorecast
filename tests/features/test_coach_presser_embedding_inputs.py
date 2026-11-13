"""Caption text and the three embedding views of a coach presser."""

from nba_ou.data_processing.game_texts.players import Player

from lab.coach_pressers.embedding_analysis.captions import caption_text, parse_vtt
from lab.coach_pressers.embedding_analysis.presser_inputs import (
    NOTE_GAP_WORDS,
    VIEWS,
    rewrite_presser,
)

VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.480 --> 00:00:02.030 align:start position:0%
 
we<00:00:00.900><c> played</c><00:00:01.200><c> hard</c>

00:00:02.030 --> 00:00:02.040 align:start position:0%
we played hard
 

00:00:02.040 --> 00:00:04.870 align:start position:0%
we played hard
&gt;&gt; [laughter]<00:00:02.880><c> and</c><00:00:02.960><c> Brunson</c>

00:00:10.000 --> 00:00:12.000 align:start position:0%
and Brunson
was<00:00:10.500><c> great</c>
"""


def test_caption_text_drops_repeats_sounds_and_marks_speaker_changes():
    assert [cue.start for cue in parse_vtt(VTT)] == [0.48, 2.04, 10.0]
    assert caption_text(VTT) == "we played hard\nand Brunson was great"


def test_caption_text_keeps_the_phrases_that_start_in_the_coach_part():
    assert caption_text(VTT, 2.0, 10.0) == "and Brunson"


def test_rewrite_labels_teams_and_notes_players_once_per_gap():
    players = [Player("Jalen", "Brunson", "Knicks"), Player("Tyler", "Herro", "Heat")]
    filler = " ".join(["uh"] * NOTE_GAP_WORDS)
    text = f"Brunson was great and Brunson's shot fell {filler} Brunson and Herro"
    rewrite = rewrite_presser(
        text + " against the Heat", "Knicks", "Heat", False, players
    )
    assert rewrite.text.startswith(
        "In this text, the head coach of the Target Team (visiting team) answers"
    )
    assert "Brunson (Target Team player) was great and Brunson's shot" in rewrite.text
    assert rewrite.text.endswith(
        "Brunson (Target Team player) and Herro (Opponent Team player) "
        "against the Opponent Team"
    )
    assert (rewrite.player_mentions, rewrite.player_notes) == (4, 3)
    assert (rewrite.target_mentions, rewrite.opponent_mentions) == (0, 1)


def test_views_share_one_instruction_wording():
    assert VIEWS["raw"] == ("raw", None)
    assert VIEWS["instructed"][0] == "raw"
    assert "Target Team" in VIEWS["target_view"][1]
    assert "Target Team" not in VIEWS["instructed"][1]
