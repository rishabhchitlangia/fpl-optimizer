"""Tests for parsing FPL player news into availability.

The point of this module is that ``status`` and ``chance_of_playing_next_round``
describe only the *next* gameweek, so a player due back in three days and one
due back in three months look identical. These tests pin down the behaviour that
distinguishes them.
"""

from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta, timezone

from fpl import data, news, scoring

GW1 = datetime(2026, 8, 21, 19, 0, tzinfo=timezone.utc)
GW2 = datetime(2026, 8, 29, 14, 0, tzinfo=timezone.utc)
LATER = datetime(2026, 11, 1, 14, 0, tzinfo=timezone.utc)


def flagged(text: str, added: str = "2026-08-18T10:00:00Z") -> dict:
    return {"news": text, "news_added": added, "status": "i",
            "chance_of_playing_next_round": 0}


class TestParsing(unittest.TestCase):
    """Turning free text into structure."""

    def test_no_news_is_not_flagged(self):
        flag = news.parse_news({"news": "", "news_added": None})
        self.assertFalse(flag.is_flagged)
        self.assertIsNone(flag.return_date)

    def test_expected_back_date_is_extracted(self):
        flag = news.parse_news(flagged("Ankle injury - Expected back 23 Aug"))
        self.assertEqual(flag.return_date, date(2026, 8, 23))
        self.assertEqual(flag.kind, "injury")
        self.assertFalse(flag.indefinite)

    def test_suspension_end_date_is_extracted(self):
        flag = news.parse_news(flagged("Suspended until 6 Sep"))
        self.assertEqual(flag.return_date, date(2026, 9, 6))
        self.assertEqual(flag.kind, "suspension")

    def test_unknown_return_date_is_marked_indefinite(self):
        flag = news.parse_news(flagged("Groin injury - Unknown return date"))
        self.assertTrue(flag.indefinite)
        self.assertIsNone(flag.return_date)

    def test_departures_are_identified(self):
        for text in ("Has joined Getafe permanently",
                     "Has joined Rangers on loan for the rest of the season",
                     "has returned to Getafe CF"):
            self.assertEqual(news.parse_news(flagged(text)).kind, "departed")

    def test_year_rolls_over_using_the_publication_date(self):
        """A December item pointing at January must resolve to the next year."""
        flag = news.parse_news(
            flagged("Knee injury - Expected back 12 Jan", "2026-12-20T10:00:00Z"))
        self.assertEqual(flag.return_date, date(2027, 1, 12))

    def test_unparseable_news_degrades_quietly(self):
        flag = news.parse_news(flagged("Something the parser has never seen"))
        self.assertTrue(flag.is_flagged)
        self.assertIsNone(flag.return_date)

    def test_malformed_timestamp_is_survivable(self):
        flag = news.parse_news(
            {"news": "Ankle injury - Expected back 23 Aug", "news_added": "nonsense"})
        self.assertIsNotNone(flag.return_date)


class TestAvailabilityOverTime(unittest.TestCase):
    """The behaviour the structured fields cannot express."""

    def test_player_is_unavailable_before_their_return_date(self):
        flag = news.parse_news(flagged("Ankle injury - Expected back 29 Aug"))
        self.assertEqual(news.availability_multiplier(flag, GW1), 0.0)

    def test_player_becomes_available_after_their_return_date(self):
        flag = news.parse_news(flagged("Ankle injury - Expected back 23 Aug"))
        self.assertGreater(news.availability_multiplier(flag, GW2), 0.0)

    def test_returning_players_ramp_rather_than_snap_to_full(self):
        """Players back from a lay-off are eased in, or miss the date entirely."""
        flag = news.parse_news(flagged("Ankle injury - Expected back 23 Aug"))
        day_one = news.availability_multiplier(
            flag, datetime(2026, 8, 23, 14, tzinfo=timezone.utc))
        self.assertGreater(day_one, 0.0)
        self.assertLess(day_one, 1.0)
        self.assertEqual(news.availability_multiplier(flag, LATER), 1.0)

    def test_ramp_is_monotonic(self):
        flag = news.parse_news(flagged("Ankle injury - Expected back 23 Aug"))
        values = [
            news.availability_multiplier(
                flag, datetime(2026, 8, 23, tzinfo=timezone.utc) + timedelta(days=d))
            for d in range(0, 20, 2)
        ]
        self.assertEqual(values, sorted(values))
        self.assertLessEqual(max(values), 1.0)

    def test_a_long_term_absence_stays_zero_across_the_horizon(self):
        flag = news.parse_news(flagged("Leg injury - Expected back 28 Nov"))
        for when in (GW1, GW2, datetime(2026, 10, 1, tzinfo=timezone.utc)):
            self.assertEqual(news.availability_multiplier(flag, when), 0.0)

    def test_departed_players_are_zero_regardless_of_date(self):
        flag = news.parse_news(flagged("Has joined Getafe permanently"))
        self.assertEqual(news.availability_multiplier(flag, LATER), 0.0)

    def test_no_date_information_defers_to_the_caller(self):
        """Returning None means 'no opinion', never 'available'."""
        flag = news.parse_news(flagged("Groin injury - Unknown return date"))
        self.assertIsNone(news.availability_multiplier(flag, GW1))
        self.assertIsNone(news.availability_multiplier(flag, None))


class TestScoringIntegration(unittest.TestCase):
    """The end the optimizer actually sees."""

    def test_indefinite_absence_still_yields_zero_availability(self):
        player = flagged("Groin injury - Unknown return date")
        player["status"] = "i"
        self.assertEqual(scoring.availability(player, GW1), 0.0)

    def test_return_date_beats_the_structured_fields(self):
        """status='i' and chance=0 must not veto a stated later return."""
        player = flagged("Ankle injury - Expected back 23 Aug")
        self.assertEqual(scoring.availability(player, GW1), 0.0)
        self.assertGreater(scoring.availability(player, LATER), 0.0)

    def test_stale_percentages_are_not_trusted_precisely(self):
        fresh = {"news": "Knock - 25% chance of playing",
                 "news_added": datetime.now(timezone.utc).isoformat().replace(
                     "+00:00", "Z"),
                 "status": "d", "chance_of_playing_next_round": 25}
        stale = dict(fresh, news_added="2026-01-01T10:00:00Z")
        self.assertAlmostEqual(scoring.availability(fresh), 0.25)
        self.assertGreater(scoring.availability(stale), 0.25)

    def test_a_returning_player_is_worth_more_later_in_the_horizon(self):
        """The whole point: a mid-horizon return must show up as rising points."""
        bootstrap = data.get_bootstrap()
        fixtures = data.get_fixtures()
        summaries = data.get_all_element_summaries(
            [e["id"] for e in bootstrap["elements"]])
        scorer = scoring.BayesianRateScorer(bootstrap, fixtures, summaries)

        returning = [
            p for p in bootstrap["elements"]
            if (flag := news.parse_news(p)).return_date
            and flag.kind != "departed"
            and date(2026, 8, 22) <= flag.return_date <= date(2026, 9, 15)
        ]
        if not returning:
            self.skipTest("no mid-horizon returns flagged right now")

        projections = scorer.project(list(range(1, 8)))
        improved = 0
        for player in returning:
            per_gw = projections[player["id"]].per_gameweek
            if per_gw.get(7, 0.0) > per_gw.get(1, 0.0):
                improved += 1
        self.assertGreater(improved, 0,
                           "no returning player gained points later in the horizon")

    def test_fixtures_carry_kickoff_times(self):
        """Per-fixture availability depends on this being populated."""
        fixtures = data.get_fixtures()
        context = scoring.FixtureContext(fixtures)
        bootstrap = data.get_bootstrap()
        team = bootstrap["teams"][0]["id"]
        upcoming = context.fixtures_for(team, data.target_gameweek(bootstrap))
        self.assertTrue(upcoming)
        self.assertIsNotNone(upcoming[0].kickoff)


if __name__ == "__main__":
    unittest.main(verbosity=2)
