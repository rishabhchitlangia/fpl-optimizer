"""Tests for the point-in-time evaluation harness.

The harness exists to stop the projection model being tuned on faith, so its
own correctness matters more than most: a leaky rewind would make every model
look good and the numbers it produces would be worse than having none.
"""

from __future__ import annotations

import unittest

from fpl import backtest, data


class TestRewind(unittest.TestCase):
    """Rewinding must remove everything the model could not have known."""

    def setUp(self):
        self.summaries = {
            1: {"history": [{"round": r, "minutes": 90, "total_points": r}
                            for r in range(1, 10)],
                "history_past": [{"minutes": 3000, "total_points": 150}]},
        }
        self.fixtures = [
            {"event": 3, "finished": True, "team_h_score": 2, "team_a_score": 1},
            {"event": 8, "finished": True, "team_h_score": 0, "team_a_score": 3},
            {"event": None, "finished": False},
        ]

    def test_history_after_the_cutoff_is_dropped(self):
        summaries, _ = backtest.rewind(self.summaries, self.fixtures, 5)
        rounds = [m["round"] for m in summaries[1]["history"]]
        self.assertEqual(rounds, [1, 2, 3, 4, 5])

    def test_past_seasons_survive(self):
        summaries, _ = backtest.rewind(self.summaries, self.fixtures, 5)
        self.assertEqual(len(summaries[1]["history_past"]), 1)

    def test_later_results_are_hidden(self):
        _, fixtures = backtest.rewind(self.summaries, self.fixtures, 5)
        later = fixtures[1]
        self.assertFalse(later["finished"])
        self.assertIsNone(later["team_h_score"])
        self.assertIsNone(later["team_a_score"])

    def test_earlier_results_are_kept(self):
        _, fixtures = backtest.rewind(self.summaries, self.fixtures, 5)
        self.assertTrue(fixtures[0]["finished"])
        self.assertEqual(fixtures[0]["team_h_score"], 2)

    def test_the_originals_are_not_mutated(self):
        backtest.rewind(self.summaries, self.fixtures, 5)
        self.assertEqual(len(self.summaries[1]["history"]), 9)
        self.assertTrue(self.fixtures[1]["finished"])

    def test_a_fixture_with_no_gameweek_is_treated_as_unplayed(self):
        _, fixtures = backtest.rewind(self.summaries, self.fixtures, 5)
        self.assertFalse(fixtures[2]["finished"])


class TestEligibility(unittest.TestCase):
    """The pool is players who were actually pickable at the time."""

    def _summary(self, minutes_per_round):
        return {"history": [{"round": i + 1, "minutes": m, "total_points": 0}
                            for i, m in enumerate(minutes_per_round)]}

    def test_a_regular_starter_qualifies(self):
        self.assertTrue(backtest.eligible(self._summary([90] * 6), 6))

    def test_a_never_used_squad_player_does_not(self):
        self.assertFalse(backtest.eligible(self._summary([0] * 6), 6))

    def test_minutes_after_the_cutoff_do_not_count(self):
        """Otherwise the pool itself would be chosen with hindsight."""
        summary = self._summary([0, 0, 0, 90, 90, 90])
        self.assertFalse(backtest.eligible(summary, 3))

    def test_before_any_gameweek_nobody_qualifies(self):
        self.assertFalse(backtest.eligible(self._summary([]), 0))


class TestMetrics(unittest.TestCase):
    """Correlation, error and the top-N edge."""

    def _pool(self, pairs):
        """Build summaries and predictions from (predicted, actual) pairs."""
        summaries, predictions = {}, {}
        for i, (p, a) in enumerate(pairs):
            summaries[i] = {"history": [{"round": 1, "minutes": 90, "total_points": 0},
                                        {"round": 2, "minutes": 90, "total_points": a}]}
            predictions[i] = p
        return predictions, summaries

    def test_a_perfect_ranking_scores_one(self):
        pairs = [(float(i), float(i)) for i in range(40)]
        predictions, summaries = self._pool(pairs)
        result = backtest.evaluate(predictions, summaries, 1, [2], top_n=5)
        self.assertAlmostEqual(result.rank_correlation, 1.0)
        self.assertAlmostEqual(result.correlation, 1.0)

    def test_a_reversed_ranking_scores_minus_one(self):
        pairs = [(float(i), float(39 - i)) for i in range(40)]
        predictions, summaries = self._pool(pairs)
        result = backtest.evaluate(predictions, summaries, 1, [2], top_n=5)
        self.assertAlmostEqual(result.rank_correlation, -1.0)

    def test_edge_is_positive_for_a_good_ranking(self):
        pairs = [(float(i), float(i)) for i in range(40)]
        predictions, summaries = self._pool(pairs)
        result = backtest.evaluate(predictions, summaries, 1, [2], top_n=5)
        self.assertGreater(result.edge, 0.0)
        self.assertAlmostEqual(result.top_n - result.pool_mean, result.edge)

    def test_edge_is_negative_for_a_bad_one(self):
        pairs = [(float(i), float(39 - i)) for i in range(40)]
        predictions, summaries = self._pool(pairs)
        result = backtest.evaluate(predictions, summaries, 1, [2], top_n=5)
        self.assertLess(result.edge, 0.0)

    def test_too_small_a_pool_returns_nothing(self):
        predictions, summaries = self._pool([(1.0, 1.0)] * 5)
        self.assertIsNone(backtest.evaluate(predictions, summaries, 1, [2]))


class TestAgainstLiveData(unittest.TestCase):
    """The model must beat the pool on real data, or it is not worth running."""

    @classmethod
    def setUpClass(cls):
        cls.bootstrap = data.get_bootstrap()
        cls.fixtures = data.get_fixtures()
        cls.summaries = data.get_all_element_summaries(
            [e["id"] for e in cls.bootstrap["elements"]])
        played = max((m.get("round") or 0) for s in cls.summaries.values()
                     for m in (s.get("history") or [])) if cls.summaries else 0
        if played < 5:
            raise unittest.SkipTest("not enough of the season played")
        cls.cutoff = played - 2

    def test_the_model_ranks_better_than_chance(self):
        results = backtest.walk_forward(
            self.bootstrap, self.fixtures, self.summaries,
            backtest.model_predictor("bayes"), [self.cutoff], horizon=2)
        self.assertTrue(results)
        self.assertGreater(results[0].rank_correlation, 0.15)

    def test_the_model_beats_the_average_player(self):
        results = backtest.walk_forward(
            self.bootstrap, self.fixtures, self.summaries,
            backtest.model_predictor("bayes"), [self.cutoff], horizon=2)
        self.assertGreater(results[0].edge, 0.0)

    def test_baselines_run_and_are_beatable(self):
        """A baseline that errored, or that won, would both be worth knowing."""
        model = backtest.walk_forward(
            self.bootstrap, self.fixtures, self.summaries,
            backtest.model_predictor("bayes"), [self.cutoff], horizon=2)[0]
        for kind in ("total", "form", "minutes"):
            base = backtest.walk_forward(
                self.bootstrap, self.fixtures, self.summaries,
                backtest.baseline_predictor(kind), [self.cutoff], horizon=2)[0]
            self.assertGreaterEqual(
                model.rank_correlation, base.rank_correlation,
                msg=f"the {kind!r} baseline out-ranks the model")


class TestRewindBootstrap(unittest.TestCase):
    """The player list must be rewound too: price, ownership, availability."""

    def setUp(self):
        self.bootstrap = {
            "total_players": 1000,
            "elements": [
                {"id": 1, "now_cost": 80, "selected_by_percent": "40.0",
                 "status": "i", "news": "Knee injury", "ep_next": "9.0",
                 "chance_of_playing_next_round": 0},
                {"id": 2, "now_cost": 45, "selected_by_percent": "1.0",
                 "status": "a", "news": "", "ep_next": "2.0",
                 "chance_of_playing_next_round": None},
            ],
        }
        self.summaries = {
            1: {"history": [
                {"round": 3, "value": 70, "selected": 50, "minutes": 90},
                {"round": 6, "value": 75, "selected": 200, "minutes": 90},
                {"round": 9, "value": 80, "selected": 400, "minutes": 90}]},
            2: {"history": []},
        }

    def _player(self, cutoff, pid=1):
        rewound = backtest.rewind_bootstrap(self.bootstrap, self.summaries, cutoff)
        return next(e for e in rewound["elements"] if e["id"] == pid)

    def test_price_is_the_price_at_the_cutoff(self):
        self.assertEqual(self._player(6)["now_cost"], 75)
        self.assertEqual(self._player(4)["now_cost"], 70)

    def test_ownership_is_the_ownership_at_the_cutoff(self):
        self.assertEqual(self._player(6)["selected_by_percent"], "20.0")

    def test_a_later_injury_is_not_applied_to_the_past(self):
        player = self._player(6)
        self.assertEqual(player["status"], "a")
        self.assertEqual(player["news"], "")
        self.assertIsNone(player["chance_of_playing_next_round"])

    def test_ep_next_is_blanked(self):
        self.assertEqual(float(self._player(6)["ep_next"]), 0.0)

    def test_a_player_with_no_matches_keeps_his_price(self):
        self.assertEqual(self._player(6, pid=2)["now_cost"], 45)

    def test_the_original_is_not_mutated(self):
        backtest.rewind_bootstrap(self.bootstrap, self.summaries, 6)
        self.assertEqual(self.bootstrap["elements"][0]["now_cost"], 80)
        self.assertEqual(self.bootstrap["elements"][0]["status"], "i")


class TestCalibrationMetrics(unittest.TestCase):
    """Slope and top-decile excess describe over-confidence."""

    def _pool(self, pairs):
        summaries, predictions = {}, {}
        for i, (p, a) in enumerate(pairs):
            summaries[i] = {"history": [{"round": 1, "minutes": 90, "total_points": 0},
                                        {"round": 2, "minutes": 90, "total_points": a}]}
            predictions[i] = p
        return predictions, summaries

    def test_an_honest_predictor_has_slope_one_and_no_excess(self):
        predictions, summaries = self._pool([(float(i), float(i)) for i in range(40)])
        result = backtest.evaluate(predictions, summaries, 1, [2], top_n=5)
        self.assertAlmostEqual(result.slope, 1.0)
        self.assertAlmostEqual(result.top_decile_excess, 0.0)

    def test_an_exaggerating_predictor_has_slope_below_one(self):
        """Right order, double the spread: every gap is half what it claims."""
        predictions, summaries = self._pool([(2.0 * i, float(i)) for i in range(40)])
        result = backtest.evaluate(predictions, summaries, 1, [2], top_n=5)
        self.assertAlmostEqual(result.slope, 0.5)
        self.assertGreater(result.top_decile_excess, 0.0)


class TestWhatCannotBeBacktested(unittest.TestCase):
    """ep_next has no recoverable history, so evaluating it must fail loudly."""

    def test_ep_next_is_refused(self):
        with self.assertRaises(ValueError):
            backtest.model_predictor("ep_next")

    def test_blended_is_refused(self):
        with self.assertRaises(ValueError):
            backtest.model_predictor("blended")

    def test_the_ep_next_baseline_is_gone(self):
        predictor = backtest.baseline_predictor("ep_next")
        with self.assertRaises(ValueError):
            predictor({"elements": []}, [], {1: {"history": []}}, [2])
