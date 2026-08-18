"""Tests for the Dixon-Coles goals model.

The model cannot be fitted to real data pre-season — there are no results — so
correctness is established against **simulated seasons with known parameters**.
That is the only way to check a fitted model actually recovers what it claims
to, and it catches things live data cannot: here it caught the regularisation
crushing every club toward average, and rho pinning itself to its bound.
"""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from fpl import data, dixon_coles, scoring

SEASON_START = datetime(2026, 8, 21, tzinfo=timezone.utc)


def simulate_season(n_teams: int = 20, seed: int = 7,
                    attack_sd: float = 0.35, defence_sd: float = 0.30,
                    home_advantage: float = 0.26, rounds: int = 1):
    """Return simulated fixtures plus the true parameters that generated them."""
    rng = np.random.default_rng(seed)
    ids = list(range(1, n_teams + 1))
    attack = rng.normal(0, attack_sd, n_teams)
    attack -= attack.mean()
    defence = rng.normal(0, defence_sd, n_teams)

    fixtures = []
    counter = 0
    for _ in range(rounds):
        for h in range(n_teams):
            for a in range(n_teams):
                if h == a:
                    continue
                lambda_home = math.exp(attack[h] + defence[a] + home_advantage)
                lambda_away = math.exp(attack[a] + defence[h])
                kickoff = SEASON_START + timedelta(days=counter // 10)
                fixtures.append({
                    "team_h": ids[h], "team_a": ids[a],
                    "team_h_score": int(rng.poisson(lambda_home)),
                    "team_a_score": int(rng.poisson(lambda_away)),
                    "finished": True,
                    "kickoff_time": kickoff.isoformat().replace("+00:00", "Z"),
                })
                counter += 1
    return fixtures, ids, attack, defence, home_advantage


class TestFittingRefusal(unittest.TestCase):
    """The model must decline to fit rather than invent parameters."""

    def test_no_results_means_no_fit(self):
        self.assertIsNone(dixon_coles.fit([], [1, 2, 3]))

    def test_too_few_results_means_no_fit(self):
        fixtures, ids, *_ = simulate_season()
        self.assertIsNone(dixon_coles.fit(fixtures[:5], ids))

    def test_live_preseason_data_declines_to_fit(self):
        """Against the real fixture list today, there is nothing to fit."""
        bootstrap = data.get_bootstrap()
        fixtures = data.get_fixtures()
        ids = [t["id"] for t in bootstrap["teams"]]
        if dixon_coles.finished_matches(fixtures):
            self.skipTest("season has started; this test only covers pre-season")
        self.assertIsNone(dixon_coles.fit(fixtures, ids))

    def test_unfinished_fixtures_are_ignored(self):
        fixtures, ids, *_ = simulate_season()
        for fixture in fixtures:
            fixture["finished"] = False
        self.assertEqual(dixon_coles.finished_matches(fixtures), [])


class TestParameterRecovery(unittest.TestCase):
    """A fit on simulated data must recover the parameters that generated it."""

    @classmethod
    def setUpClass(cls):
        fixtures, ids, attack, defence, home = simulate_season()
        cls.fit = dixon_coles.fit(fixtures, ids,
                                  now=SEASON_START + timedelta(days=280))
        cls.ids, cls.attack, cls.defence, cls.home = ids, attack, defence, home

    def test_attack_strengths_are_recovered(self):
        estimated = np.array([self.fit.strengths[i].attack for i in self.ids])
        self.assertGreater(np.corrcoef(estimated, self.attack)[0, 1], 0.85)

    def test_defence_strengths_are_recovered(self):
        estimated = np.array([self.fit.strengths[i].defence for i in self.ids])
        self.assertGreater(np.corrcoef(estimated, self.defence)[0, 1], 0.65)

    def test_home_advantage_is_recovered(self):
        self.assertAlmostEqual(self.fit.home_advantage, self.home, delta=0.06)

    def test_rho_stays_near_zero_on_independent_poisson_data(self):
        """Regression: rho previously pinned itself to its bound on noise."""
        self.assertLess(abs(self.fit.rho), 0.05)
        self.assertNotAlmostEqual(abs(self.fit.rho), dixon_coles.RHO_BOUNDS[1],
                                  places=3)

    def test_strengths_are_not_crushed_toward_average(self):
        """Regression: over-strong regularisation flattened the whole league."""
        estimated = np.array([self.fit.strengths[i].attack for i in self.ids])
        self.assertGreater(estimated.std(), 0.5 * self.attack.std())

    def test_more_data_improves_the_fit(self):
        fixtures, ids, attack, _, _ = simulate_season()
        short = dixon_coles.fit(fixtures[:40], ids,
                                now=SEASON_START + timedelta(days=30))
        long = dixon_coles.fit(fixtures, ids,
                               now=SEASON_START + timedelta(days=280))
        short_corr = np.corrcoef(
            [short.strengths[i].attack for i in ids], attack)[0, 1]
        long_corr = np.corrcoef(
            [long.strengths[i].attack for i in ids], attack)[0, 1]
        self.assertGreater(long_corr, short_corr)


class TestProbabilities(unittest.TestCase):
    """Score matrices and derived probabilities must be coherent."""

    @classmethod
    def setUpClass(cls):
        fixtures, ids, *_ = simulate_season()
        cls.fit = dixon_coles.fit(fixtures, ids,
                                  now=SEASON_START + timedelta(days=280))
        cls.ids = ids

    def test_score_matrix_is_a_probability_distribution(self):
        matrix = self.fit.score_matrix(self.ids[0], self.ids[1])
        self.assertAlmostEqual(float(matrix.sum()), 1.0, places=9)
        self.assertTrue((matrix >= 0).all())

    def test_clean_sheet_probability_is_a_probability(self):
        for home in self.ids[:5]:
            for away in self.ids[:5]:
                if home == away:
                    continue
                p = self.fit.project(home, away, True).clean_sheet_probability
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)

    def test_facing_a_stronger_attack_lowers_clean_sheet_odds(self):
        best = max(self.ids, key=lambda i: self.fit.strengths[i].attack)
        worst = min(self.ids, key=lambda i: self.fit.strengths[i].attack)
        defender = self.ids[0]
        vs_best = self.fit.project(defender, best, True).clean_sheet_probability
        vs_worst = self.fit.project(defender, worst, True).clean_sheet_probability
        self.assertLess(vs_best, vs_worst)

    def test_home_advantage_raises_expected_goals(self):
        home_goals, _ = self.fit.expected_goals(self.ids[0], self.ids[1])
        away_goals = self.fit.project(self.ids[0], self.ids[1], False).expected_goals_for
        self.assertGreater(home_goals, away_goals)

    def test_goals_conceded_penalty_is_negative_and_bounded(self):
        projection = self.fit.project(self.ids[0], self.ids[1], True)
        self.assertLessEqual(projection.expected_goals_conceded_points, 0.0)
        self.assertGreater(projection.expected_goals_conceded_points, -3.0)

    def test_tau_only_touches_the_four_lowest_scorelines(self):
        for home_goals in range(4):
            for away_goals in range(4):
                tau = dixon_coles._tau(home_goals, away_goals, 1.4, 1.1, -0.05)
                if home_goals <= 1 and away_goals <= 1:
                    self.assertNotAlmostEqual(tau, 1.0)
                else:
                    self.assertEqual(tau, 1.0)

    def test_unknown_teams_fall_back_to_league_average(self):
        goals = self.fit.expected_goals(9999, 8888)
        self.assertEqual(len(goals), 2)
        self.assertTrue(all(g > 0 for g in goals))


class TestBlendWeight(unittest.TestCase):
    """The handover from fixture difficulty to the results model."""

    def test_no_matches_means_no_trust(self):
        self.assertEqual(dixon_coles.blend_weight(0), 0.0)

    def test_weight_is_monotonic_in_evidence(self):
        weights = [dixon_coles.blend_weight(n) for n in range(0, 40, 4)]
        self.assertEqual(weights, sorted(weights))

    def test_weight_never_reaches_certainty(self):
        self.assertLess(dixon_coles.blend_weight(38), 1.0)

    def test_half_weight_at_the_documented_half_life(self):
        self.assertAlmostEqual(
            dixon_coles.blend_weight(dixon_coles.BLEND_HALF_LIFE_MATCHES), 0.5)


class TestFixtureModelIntegration(unittest.TestCase):
    """How the fit feeds the scorer, and the double-counting guard."""

    @classmethod
    def setUpClass(cls):
        cls.bootstrap = data.get_bootstrap()
        cls.fixtures = data.get_fixtures()
        cls.context = scoring.FixtureContext(cls.fixtures)

    def test_without_a_fit_it_falls_back_to_fixture_difficulty(self):
        model = scoring.FixtureModel(self.context, None)
        self.assertFalse(model.has_results_model)
        fixture = scoring.TeamFixture(1, 2, True, 3)
        self.assertAlmostEqual(model.multiplier(2, 1, fixture), 1.0)

    def test_no_fit_means_no_clean_sheet_or_goals_output(self):
        model = scoring.FixtureModel(self.context, None)
        fixture = scoring.TeamFixture(1, 2, True, 3)
        self.assertIsNone(model.clean_sheet_probability(1, fixture))
        self.assertIsNone(model.expected_goals(1, fixture))

    def test_multiplier_is_normalised_to_one_at_league_average(self):
        """The guard against double-counting.

        The projection scales an *observed* points-per-90 that already contains
        the clean sheets a player historically earned. So the fixture model must
        express "better or worse than a typical fixture", averaging to 1.0 —
        never absolute points.
        """
        fixtures, ids, *_ = simulate_season()
        fit = dixon_coles.fit(fixtures, ids,
                              now=SEASON_START + timedelta(days=280))
        model = scoring.FixtureModel(self.context, fit)

        for element_type in (1, 2, 3, 4):
            multipliers = [
                model.multiplier(element_type, team,
                                 scoring.TeamFixture(1, opponent, is_home, 3))
                for team in ids for opponent in ids if team != opponent
                for is_home in (True, False)
            ]
            self.assertAlmostEqual(float(np.mean(multipliers)), 1.0, delta=0.12,
                                   msg=f"position {element_type} is biased")

    def test_forwards_ignore_clean_sheets(self):
        """Forwards score nothing for a clean sheet (RULES.md 2.1)."""
        self.assertEqual(scoring.POSITION_DEFENSIVE_SHARE[4], 0.0)

    def test_defenders_are_more_clean_sheet_driven_than_midfielders(self):
        self.assertGreater(scoring.POSITION_DEFENSIVE_SHARE[2],
                           scoring.POSITION_DEFENSIVE_SHARE[3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
