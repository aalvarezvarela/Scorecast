"""Contract for the injury-report change-log store.

Every test here corresponds to a way the store could silently produce a
look-ahead error or lose a player. All are pure -- no network, no database.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from nba_ou.postgre_db.injury_report_aiven.parse import category_code, split_reason
from nba_ou.postgre_db.injury_report_aiven.resolve import initial_key, normalise_name
from nba_ou.postgre_db.injury_report_aiven.schema import STATUS_BY_LABEL, STATUSES
from nba_ou.postgre_db.injury_report_aiven.spans import build_spans

UTC = UTC
TIP = datetime(2026, 2, 12, 0, 30, tzinfo=UTC)


def _obs(minutes_before: list[int], statuses: list[int], reasons=None):
    """Observations for one (game, player), at N minutes before tipoff."""
    reasons = reasons or [10] * len(statuses)
    return pd.DataFrame(
        {
            "game_id": ["0022500001"] * len(statuses),
            "player_id": [201939] * len(statuses),
            "team_id": [1] * len(statuses),
            "season_year": [2025] * len(statuses),
            "observed_at": [TIP - timedelta(minutes=m) for m in minutes_before],
            "tipoff_utc": [TIP] * len(statuses),
            "status_id": statuses,
            "reason_id": reasons,
            "report_id": list(range(1, len(statuses) + 1)),
        }
    )


class TestStatusVocabulary:
    def test_exactly_five_statuses(self):
        """Verified closed over 34,505 rows; a sixth means the NBA changed."""
        assert len(STATUSES) == 5
        assert len(STATUS_BY_LABEL) == 5

    def test_status_ids_are_ordinal_by_severity(self):
        codes = [code for _, code in sorted(STATUSES)]
        assert codes == ["available", "probable", "questionable", "doubtful", "out"]

    def test_labels_map_onto_the_dimension(self):
        assert set(STATUS_BY_LABEL.values()) == {sid for sid, _ in STATUSES}


class TestReasonSplitting:
    def test_splits_on_the_first_separator_only(self):
        """Splitting on the last turns wrapped detail into a bogus category."""
        assert split_reason("Not with Team - Return to Competition Reconditioning") == (
            "Not with Team",
            "Return to Competition Reconditioning",
        )

    def test_keeps_semicolon_detail_intact(self):
        assert split_reason("Injury/Illness - Right Knee; Soreness") == (
            "Injury/Illness",
            "Right Knee; Soreness",
        )

    @pytest.mark.parametrize("raw", ["-", "", None, "nan"])
    def test_empty_reasons_collapse_to_none(self, raw):
        assert split_reason(raw) == ("(none)", "")

    def test_category_without_detail(self):
        assert split_reason("Ineligible To Play") == ("Ineligible To Play", "")

    def test_case_variants_are_one_category(self):
        """The NBA prints both spellings; they must not become two rows."""
        assert category_code("Not With Team") == category_code("Not with Team")


class TestNameNormalisation:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Gilgeous-Alexander, Shai", "gilgeousalexander|shai"),
            ("Smith Jr., Dennis", "smith|dennis"),
            ("Williams III, Robert", "williams|robert"),
            ("Aminu, Al-Farouq", "aminu|alfarouq"),
        ],
    )
    def test_suffixes_and_hyphens_are_stripped(self, raw, expected):
        assert normalise_name(raw) == expected

    def test_family_and_given_stay_separate(self):
        """Otherwise "Smith, Jaden" and "Jaden, Smith" would collide."""
        assert normalise_name("Smith, Jaden") != normalise_name("Jaden, Smith")

    def test_initial_key_bridges_the_abbreviated_box_score_form(self):
        """nba_players stores "B. Adebayo" for every season before 2025."""
        assert initial_key(normalise_name("Adebayo, Bam")) == initial_key(
            normalise_name("Adebayo, B.")
        )


class TestSpans:
    def test_unchanged_observations_collapse_to_one_span(self):
        spans = build_spans(_obs([180, 120, 60], [5, 5, 5]))
        assert len(spans) == 1
        assert spans.loc[0, "status_id"] == 5

    def test_a_change_opens_a_new_span(self):
        spans = build_spans(_obs([180, 120, 60], [3, 3, 5]))
        assert list(spans["status_id"]) == [3, 5]

    def test_a_span_ends_where_the_next_begins(self):
        spans = build_spans(_obs([180, 60], [3, 5])).sort_values("valid_from")
        assert spans.iloc[0]["valid_to"] == spans.iloc[1]["valid_from"]

    def test_the_final_span_is_clamped_at_tipoff(self):
        """The whole point: no span may outlive its own game."""
        spans = build_spans(_obs([180, 60], [3, 5]))
        assert spans["valid_to"].max() == pd.Timestamp(TIP)

    def test_no_span_outlives_tipoff_even_with_a_late_report(self):
        spans = build_spans(_obs([60, -30], [4, 4]))
        assert spans["valid_to"].max() <= pd.Timestamp(TIP)

    def test_observations_after_tipoff_cannot_open_a_span(self):
        """A report published after tip is not information about the game."""
        spans = build_spans(_obs([-30, -60], [4, 5]))
        assert spans.empty

    def test_reason_change_alone_opens_a_span(self):
        """Same status, different reason, is a different fact."""
        spans = build_spans(_obs([180, 60], [5, 5], reasons=[10, 20]))
        assert len(spans) == 2

    def test_mins_to_tip_is_measured_from_the_span_start(self):
        spans = build_spans(_obs([180, 60], [3, 5])).sort_values("valid_from")
        assert list(spans["mins_to_tip"]) == [180, 60]

    def test_spans_are_always_ordered(self):
        spans = build_spans(_obs([300, 180, 60], [3, 4, 5]))
        assert (spans["valid_to"] > spans["valid_from"]).all()

    def test_empty_input_yields_empty_spans(self):
        assert build_spans(pd.DataFrame()).empty

    def test_two_players_do_not_share_a_span(self):
        other = _obs([120, 60], [5, 5])
        other["player_id"] = 1628983
        frame = pd.concat([_obs([120, 60], [3, 5]), other], ignore_index=True)
        spans = build_spans(frame)
        assert set(spans["player_id"]) == {201939, 1628983}


# --- derived observations: removal, filing, tipoff fallback ---------------------

from nba_ou.postgre_db.injury_report_aiven.ingest import (  # noqa: E402
    build_filing_observations,
    build_removal_observations,
)
from nba_ou.postgre_db.injury_report_aiven.parse import decode_game_time  # noqa: E402


class TestGameTimeDecoding:
    """The report prints a 12-hour clock with no AM/PM."""

    @pytest.mark.parametrize(
        "raw,expected_et_hour",
        [
            ("08:00 (ET)", 20),
            ("10:30 (ET)", 22),
            ("01:00 (ET)", 13),
            ("12:00 (ET)", 12),
            ("11:00 (ET)", 11),
        ],
    )
    def test_nba_tip_window_resolves_the_meridiem(self, raw, expected_et_hour):
        tip = decode_game_time(pd.Timestamp("2026-01-15").date(), raw)
        assert tip.tz_convert("America/New_York").hour == expected_et_hour

    def test_utc_offset_follows_dst(self):
        winter = decode_game_time(pd.Timestamp("2026-01-15").date(), "07:00 (ET)")
        summer = decode_game_time(pd.Timestamp("2026-05-21").date(), "08:00 (ET)")
        assert winter == pd.Timestamp("2026-01-16 00:00", tz="UTC")
        assert summer == pd.Timestamp("2026-05-22 00:00", tz="UTC")

    @pytest.mark.parametrize("raw", ["", None, "TBD", "8:00 PM"])
    def test_unparseable_times_are_none(self, raw):
        assert decode_game_time(pd.Timestamp("2026-01-15").date(), raw) is None


def _t(minutes_before: int) -> pd.Timestamp:
    return pd.Timestamp(TIP) - pd.Timedelta(minutes=minutes_before)


def _player_obs(minutes_before: list[int], player_id=201939, raw_team="Chicago Bulls"):
    n = len(minutes_before)
    return pd.DataFrame(
        {
            "game_id": ["0022500001"] * n,
            "player_id": [player_id] * n,
            "team_id": [1] * n,
            "nba_team_id": ["1610612741"] * n,
            "season_year": [2025] * n,
            "tipoff_utc": [pd.Timestamp(TIP)] * n,
            "raw_name": ["X, Y"] * n,
            "raw_team": [raw_team] * n,
            "observed_at": [_t(m) for m in minutes_before],
            "status_id": [4] * n,
            "reason_category": ["Injury/Illness"] * n,
            "reason_detail": ["Knee"] * n,
        }
    )


def _listed(minutes_before: list[int]):
    return pd.DataFrame(
        {
            "game_id": ["0022500001"] * len(minutes_before),
            "observed_at": [_t(m) for m in minutes_before],
            "tipoff_utc": [pd.Timestamp(TIP)] * len(minutes_before),
            "season_year": [2025] * len(minutes_before),
            "team_home": ["DET"] * len(minutes_before),
            "team_away": ["CHI"] * len(minutes_before),
        }
    )


class TestRemovals:
    """A player dropping off a filed team's list must stop reading "Doubtful"."""

    def test_dropping_off_the_list_emits_a_null_status(self):
        obs = _player_obs([180, 120])
        listed = _listed([180, 120, 60])
        removed = build_removal_observations(
            obs, listed, pd.DataFrame(), listed["observed_at"]
        )
        assert list(removed["observed_at"]) == [_t(60)]
        assert removed["status_id"].isna().all()

    def test_absence_while_team_has_not_filed_is_not_a_removal(self):
        """Missing because NOT YET SUBMITTED is unknown, not removed."""
        obs = _player_obs([180])
        listed = _listed([180, 120])
        nys = pd.DataFrame(
            {
                "game_id": ["0022500001"],
                "raw_team": ["Chicago Bulls"],
                "observed_at": [_t(120)],
            }
        )
        removed = build_removal_observations(obs, listed, nys, listed["observed_at"])
        assert removed.empty

    def test_nothing_before_first_sighting_counts(self):
        obs = _player_obs([60])
        listed = _listed([180, 120, 60])
        assert build_removal_observations(
            obs, listed, pd.DataFrame(), listed["observed_at"]
        ).empty

    def test_a_removal_span_ends_the_doubtful_span_early(self):
        """End to end: the stale-Doubtful case the tipoff clamp alone missed."""
        obs = _player_obs([180, 120])
        listed = _listed([180, 120, 60])
        removed = build_removal_observations(
            obs, listed, pd.DataFrame(), listed["observed_at"]
        )
        both = pd.concat([obs, removed], ignore_index=True)
        both["reason_id"] = both["reason_category"] + "|" + both["reason_detail"]
        both["report_id"] = 1
        spans = build_spans(both).sort_values("valid_from")
        assert list(spans["status_id"].astype("object")) == [4, pd.NA]
        assert spans.iloc[0]["valid_to"] == _t(60)

    def test_repeated_absence_collapses_into_one_null_span(self):
        obs = _player_obs([240, 180])
        listed = _listed([240, 180, 120, 60, 30])
        removed = build_removal_observations(
            obs, listed, pd.DataFrame(), listed["observed_at"]
        )
        both = pd.concat([obs, removed], ignore_index=True)
        both["reason_id"] = both["reason_category"] + "|" + both["reason_detail"]
        both["report_id"] = 1
        assert len(build_spans(both)) == 2


class TestFilings:
    TEAMS = {"DET": 10, "CHI": 20}

    def test_a_listed_team_not_marked_nys_has_filed(self):
        listed = _listed([120])
        filings = build_filing_observations(
            listed, pd.DataFrame(), listed["observed_at"], self.TEAMS
        )
        assert filings["submitted"].all() and len(filings) == 2

    def test_nys_team_is_not_submitted_and_flips_when_it_files(self):
        listed = _listed([180, 60])
        nys = pd.DataFrame(
            {"game_id": ["0022500001"], "team_id": [20], "observed_at": [_t(180)]}
        )
        filings = build_filing_observations(
            listed, nys, listed["observed_at"], self.TEAMS
        )
        chi = filings.loc[filings["team_id"] == 20].sort_values("observed_at")
        assert list(chi["submitted"]) == [False, True]

    def test_a_game_that_vanishes_before_tip_means_both_teams_filed(self):
        """Seen in the 2026 East Finals: both lists filed empty, game dropped."""
        listed = _listed([300])
        nys = pd.DataFrame(
            {"game_id": ["0022500001"], "team_id": [20], "observed_at": [_t(300)]}
        )
        report_times = pd.Series([_t(300), _t(240), _t(180)])
        filings = build_filing_observations(listed, nys, report_times, self.TEAMS)
        late = filings.loc[filings["observed_at"] == _t(240)]
        assert set(late["team_id"]) == {10, 20} and late["submitted"].all()


# --------------------------------------------------------------------------- #
# Publication time and season replacement (2025-12 restamp recovery)
# --------------------------------------------------------------------------- #

from nba_ou.postgre_db.injury_report_aiven import ingest as ingest_module  # noqa: E402
from nba_ou.postgre_db.injury_report_aiven import load as load_module  # noqa: E402
from nba_ou.postgre_db.injury_report_aiven.parse import ParsedReport  # noqa: E402


class _Storage:
    def __init__(self, keys):
        self.keys = set(keys)

    def get(self, key):
        return b"%PDF" if key in self.keys else None


def _manifest_rows(rows):
    return pd.DataFrame(
        [
            {
                "report_key": key,
                "season_year": 2025,
                "source_era": "hourly_24",
                "nba_available": "true",
                "s3_key": key,
                "download_status": status,
                "report_datetime_utc": pd.Timestamp(derived),
                "report_published_utc": (
                    pd.Timestamp(published) if published else pd.NaT
                ),
            }
            for key, derived, published, status in rows
        ]
    )


def _capture_parse(monkeypatch):
    seen = []

    def fake_parse(data, observed_at):
        seen.append(observed_at)
        empty = pd.DataFrame()
        return ParsedReport(observed_at, empty, empty)

    monkeypatch.setattr(ingest_module, "parse_report", fake_parse)
    return seen


def test_reports_are_read_at_their_publication_time(monkeypatch):
    """A report stamped 16:45 must not be observed at the filename's 16:30."""
    seen = _capture_parse(monkeypatch)
    manifest = _manifest_rows(
        [
            ("a", "2025-12-19T21:30Z", "2025-12-19T21:45Z", "stored"),
            ("b", "2025-12-19T20:30Z", None, "stored"),
        ]
    )
    parsed, _, _ = ingest_module.read_reports(
        manifest, _Storage({"a", "b"}), quiet=True
    )
    assert [p.report.observed_at for p in parsed] == seen
    assert [pd.Timestamp(t) for t in seen] == [
        pd.Timestamp("2025-12-19T20:30Z"),
        pd.Timestamp("2025-12-19T21:45Z"),
    ]


def test_two_files_at_one_publication_instant_keep_the_stored_later_one(monkeypatch):
    seen = _capture_parse(monkeypatch)
    manifest = _manifest_rows(
        [
            ("early_label", "2025-12-19T21:30Z", "2025-12-19T22:00Z", "stored"),
            ("late_label", "2025-12-19T22:00Z", None, "stored"),
            ("never_downloaded", "2025-12-19T22:15Z", "2025-12-19T22:00Z", "invalid"),
        ]
    )
    parsed, _, _ = ingest_module.read_reports(
        manifest, _Storage({"early_label", "late_label"}), quiet=True
    )
    assert len(parsed) == 1
    assert pd.Timestamp(seen[0]) == pd.Timestamp("2025-12-19T22:00Z")


class _Cursor:
    def __init__(self, log, fail_on=None):
        self.log, self.fail_on, self.rowcount = log, fail_on, 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, statement, params=None):
        text = (
            statement.as_string(None)
            if hasattr(statement, "as_string")
            else str(statement)
        )
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("boom")
        self.log.append((text, params))
        self.rowcount = 7


class _Conn:
    def __init__(self, fail_on=None):
        self.log, self.fail_on = [], fail_on
        self.committed = self.rolled_back = False

    def cursor(self):
        return _Cursor(self.log, self.fail_on)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def test_clear_season_deletes_facts_before_reports_for_one_season():
    conn = _Conn()
    deleted = load_module.clear_season(conn, 2025)
    tables = [t for t, _ in load_module.CLEAR_SEASON_STATEMENTS]
    assert list(deleted) == tables
    assert tables.index("ir_status_span") < tables.index("ir_report")
    assert all(params == (2025,) for _, params in conn.log)
    assert "ir_player_alias" not in " ".join(text for text, _ in conn.log)
    assert conn.committed and not conn.rolled_back


def test_clear_season_rolls_back_everything_on_failure():
    conn = _Conn(fail_on="ir_report")
    with pytest.raises(RuntimeError):
        load_module.clear_season(conn, 2025)
    assert conn.rolled_back and not conn.committed


# --------------------------------------------------------------------------------
# Curated player aliases
# --------------------------------------------------------------------------------

from nba_ou.postgre_db.injury_report_aiven.resolve import (  # noqa: E402
    CURATED_PLAYER_ALIASES,
    METHOD_CURATED_ALIAS,
    ResolutionReport,
    resolve_players,
)

_HOU = 1610612745


def _roster(rows):
    frame = pd.DataFrame(
        rows, columns=["game_id", "nba_team_id", "player_id", "season_year", "name"]
    )
    frame["norm_name"] = frame["name"].map(normalise_name)
    frame["norm_initial"] = frame["norm_name"].map(initial_key)
    return frame.drop(columns=["name"])


def _listing(raw_name, team=_HOU, season=2024, game="0022400001"):
    return pd.DataFrame(
        {
            "raw_name": [raw_name],
            "raw_team": ["Houston Rockets"],
            "nba_team_id": [team],
            "season_year": [season],
            "game_id": [game],
        }
    )


def test_curated_alias_resolves_an_initial_collision_on_the_team():
    # Jalen and Jeff Green are both "J. Green" in HOU 2024-25 box scores.
    rosters = _roster(
        [
            ("0022400001", _HOU, 1630224, 2024, "Green, J"),
            ("0022400001", _HOU, 201145, 2024, "Green, J"),
        ]
    )
    report = ResolutionReport()
    out = resolve_players(_listing("Green, Jalen"), rosters, report)
    assert out["player_id"].tolist() == [CURATED_PLAYER_ALIASES["green|jalen"]]
    assert out["method"].tolist() == [METHOD_CURATED_ALIAS]
    assert not report.unresolved


def test_curated_alias_requires_the_player_on_that_team_season():
    # Same alias, but 1630224 is not on this team's roster that season.
    rosters = _roster([("0022400001", _HOU, 201935, 2024, "Harden, James")])
    report = ResolutionReport()
    out = resolve_players(_listing("Green, Jalen"), rosters, report)
    assert out.empty
    assert len(report.unresolved) == 1


def test_a_roster_match_wins_over_a_curated_alias():
    rosters = _roster(
        [
            ("0022400001", _HOU, 1630224, 2024, "Green, Jalen"),
            ("0022400001", _HOU, 999, 2024, "Jones, David"),
        ]
    )
    out = resolve_players(_listing("Jones, David"), rosters, ResolutionReport())
    assert out["player_id"].tolist() == [999]
    assert out["method"].tolist() != [METHOD_CURATED_ALIAS]
