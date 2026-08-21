"""Tests for multi-gameweek transfer planning.

The behaviour that matters is the behaviour a greedy weekly optimizer cannot
produce: banking a transfer in a quiet week to spend two in a good one, and
declining a transfer that does not pay for its own hit.
"""

from __future__ import annotations

import unittest

from fpl import data, optimizer, planner, scoring


class PlannerTestCase(unittest.TestCase):
    """Shared fixture: real projections over a real horizon."""

    HORIZON = list(range(1, 6))

    @classmethod
    def setUpClass(cls):
        cls.bootstrap = data.get_bootstrap()
        cls.fixtures = data.get_fixtures()
        cls.players = data.player_lookup(cls.bootstrap)
        summaries = data.get_all_element_summaries(list(cls.players))
        scorer = scoring.build_scorer(cls.bootstrap, cls.fixtures, summaries, "bayes")
        cls.projections = scorer.project(cls.HORIZON)
        # A deliberately mediocre starting squad, so there is something to improve.
        weak = optimizer.optimize_squad(cls.bootstrap, cls.projections, budget=900)
        cls.squad = list(weak.squad_ids)
        cls.bank = 1000 - sum(cls.players[p]["now_cost"] for p in cls.squad)


class TestPathValidity(PlannerTestCase):
    """Every gameweek in a plan must be a legal FPL squad."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.path = planner.plan_transfers(
            cls.bootstrap, cls.projections, cls.squad, cls.HORIZON,
            free_transfers=1, bank=cls.bank)

    def test_one_plan_per_gameweek(self):
        self.assertEqual(len(self.path.gameweeks), len(self.HORIZON))
        self.assertEqual([p.gameweek for p in self.path.gameweeks], self.HORIZON)

    def test_every_squad_is_legal(self):
        for plan in self.path.gameweeks:
            self.assertEqual(len(plan.squad_ids), 15, f"GW{plan.gameweek}")
            counts: dict[int, int] = {}
            clubs: dict[int, int] = {}
            for pid in plan.squad_ids:
                player = self.players[pid]
                counts[player["element_type"]] = counts.get(player["element_type"], 0) + 1
                clubs[player["team"]] = clubs.get(player["team"], 0) + 1
            self.assertEqual(counts, {1: 2, 2: 5, 3: 5, 4: 3}, f"GW{plan.gameweek}")
            self.assertLessEqual(max(clubs.values()), 3, f"GW{plan.gameweek}")

    def test_every_starting_xi_is_legal(self):
        for plan in self.path.gameweeks:
            self.assertEqual(len(plan.starting_ids), 11)
            counts = {1: 0, 2: 0, 3: 0, 4: 0}
            for pid in plan.starting_ids:
                counts[self.players[pid]["element_type"]] += 1
            self.assertEqual(counts[1], 1)
            self.assertGreaterEqual(counts[2], 3)
            self.assertGreaterEqual(counts[3], 2)
            self.assertGreaterEqual(counts[4], 1)
            self.assertIn(plan.captain_id, plan.starting_ids)

    def test_budget_is_respected_every_week(self):
        limit = self.bank + sum(self.players[p]["now_cost"] for p in self.squad)
        for plan in self.path.gameweeks:
            self.assertLessEqual(plan.squad_cost, limit, f"GW{plan.gameweek}")

    def test_squads_are_continuous_across_gameweeks(self):
        """Transfers must exactly explain the difference between weeks."""
        previous = set(self.squad)
        for plan in self.path.gameweeks:
            current = set(plan.squad_ids)
            self.assertEqual(set(plan.transfers_out), previous - current,
                             f"GW{plan.gameweek} outgoing mismatch")
            self.assertEqual(set(plan.transfers_in), current - previous,
                             f"GW{plan.gameweek} incoming mismatch")
            self.assertEqual(len(plan.transfers_in), len(plan.transfers_out))
            previous = current


class TestFreeTransferAccounting(PlannerTestCase):
    """The rules from RULES.md §3, as implemented in the recursion."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.path = planner.plan_transfers(
            cls.bootstrap, cls.projections, cls.squad, cls.HORIZON,
            free_transfers=1, bank=cls.bank)

    def test_free_transfers_never_exceed_the_cap(self):
        for plan in self.path.gameweeks:
            self.assertLessEqual(plan.free_transfers_available,
                                 planner.MAX_FREE_TRANSFERS)
            self.assertGreaterEqual(plan.free_transfers_available, 1)

    def test_banking_a_transfer_increases_next_week_s_allowance(self):
        """The behaviour a weekly optimizer cannot find."""
        plans = self.path.gameweeks
        for earlier, later in zip(plans, plans[1:]):
            used = min(len(earlier.transfers_in), earlier.free_transfers_available)
            expected = min(planner.MAX_FREE_TRANSFERS,
                           earlier.free_transfers_available - used + 1)
            self.assertLessEqual(later.free_transfers_available, expected,
                                 f"GW{later.gameweek} claims too many free transfers")

    def test_hits_match_transfers_beyond_the_free_allowance(self):
        for plan in self.path.gameweeks:
            beyond = max(0, len(plan.transfers_in) - plan.free_transfers_available)
            self.assertAlmostEqual(plan.hits, 4.0 * beyond, places=4,
                                   msg=f"GW{plan.gameweek}")

    def test_starting_allowance_is_honoured(self):
        path = planner.plan_transfers(
            self.bootstrap, self.projections, self.squad, self.HORIZON,
            free_transfers=3, bank=self.bank)
        self.assertEqual(path.gameweeks[0].free_transfers_available, 3)

    def test_totals_are_consistent(self):
        self.assertAlmostEqual(
            self.path.total_hits, sum(p.hits for p in self.path.gameweeks))
        self.assertAlmostEqual(
            self.path.net_points, self.path.total_points - self.path.total_hits)


class TestPlanQuality(PlannerTestCase):
    """A plan must beat doing nothing, and beat week-by-week greed."""

    def test_planning_beats_holding_the_squad(self):
        path = planner.plan_transfers(
            self.bootstrap, self.projections, self.squad, self.HORIZON,
            free_transfers=1, bank=self.bank)
        positions = data.position_lookup(self.bootstrap)
        held = optimizer.pick_starting_xi(
            self.squad, self.players, self.projections, positions)
        # pick_starting_xi scores the whole horizon at once, so compare totals.
        self.assertGreater(path.net_points, held.predicted_points)

    def test_a_strong_squad_is_left_largely_alone(self):
        """Nothing to fix means few transfers — not churn for its own sake."""
        best = optimizer.optimize_squad(self.bootstrap, self.projections)
        path = planner.plan_transfers(
            self.bootstrap, self.projections, list(best.squad_ids), self.HORIZON,
            free_transfers=1, bank=0)
        self.assertEqual(path.total_hits, 0.0)

    def test_locked_players_are_held_all_horizon(self):
        target = self.squad[0]
        path = planner.plan_transfers(
            self.bootstrap, self.projections, self.squad, self.HORIZON,
            free_transfers=1, bank=self.bank, locked=[target])
        for plan in path.gameweeks:
            self.assertIn(target, plan.squad_ids, f"GW{plan.gameweek}")

    def test_banned_players_never_appear(self):
        target = max(self.players, key=lambda p: self.players[p]["now_cost"])
        path = planner.plan_transfers(
            self.bootstrap, self.projections, self.squad, self.HORIZON,
            free_transfers=1, bank=self.bank, banned=[target])
        for plan in path.gameweeks:
            self.assertNotIn(target, plan.squad_ids)

    def test_short_horizons_use_the_whole_player_pool(self):
        """Pruning a five-gameweek plan trades accuracy for a few seconds.

        Measured: the full pool solves five gameweeks to proven optimality in
        about six seconds, and returns the same plan as a pruned pool. Pruning
        here was an unmeasured assumption that turned out to be wrong.
        """
        path = planner.plan_transfers(
            self.bootstrap, self.projections, self.squad, self.HORIZON,
            free_transfers=1, bank=self.bank)
        self.assertFalse(path.pruned)
        self.assertGreater(path.pool_size, 300)

    def test_pruning_is_reported_when_it_happens(self):
        """The approximation must never be silent."""
        pool, pruned = planner._build_pool(
            self.bootstrap, self.projections, list(range(1, 12)),
            self.squad, set())
        self.assertTrue(pruned)
        self.assertLess(len(pool), len(self.players))

    def test_a_truncated_solve_is_flagged_rather_than_called_optimal(self):
        """CBC labels its incumbent "Optimal" when it stops on the clock.

        Regression: a truncated solve on a larger pool once returned a *worse*
        plan than a pruned one while reporting optimality, which is how this
        was found. Wall time is the only reliable signal.
        """
        path = planner.plan_transfers(
            self.bootstrap, self.projections, self.squad,
            list(range(1, 9)), free_transfers=1, bank=self.bank,
            time_limit=2)
        self.assertTrue(path.hit_time_limit)
        self.assertEqual(path.status, "Time limit")

    def test_an_untruncated_solve_is_not_flagged(self):
        path = planner.plan_transfers(
            self.bootstrap, self.projections, self.squad, self.HORIZON,
            free_transfers=1, bank=self.bank)
        self.assertFalse(path.hit_time_limit)
        self.assertEqual(path.status, "Optimal")
        self.assertGreater(path.solve_seconds, 0.0)


class TestEdgeCases(PlannerTestCase):
    """Degenerate inputs."""

    def test_empty_horizon_is_rejected(self):
        with self.assertRaises(optimizer.OptimizerError):
            planner.plan_transfers(
                self.bootstrap, self.projections, self.squad, [])

    def test_single_gameweek_horizon_works(self):
        path = planner.plan_transfers(
            self.bootstrap, self.projections, self.squad, [1],
            free_transfers=1, bank=self.bank)
        self.assertEqual(len(path.gameweeks), 1)

    def test_describe_path_covers_every_gameweek(self):
        path = planner.plan_transfers(
            self.bootstrap, self.projections, self.squad, self.HORIZON,
            free_transfers=1, bank=self.bank)
        rows = planner.describe_path(path, self.bootstrap)
        self.assertEqual(len(rows), len(self.HORIZON))
        for row in rows:
            self.assertIn("captain", row)
            self.assertIsInstance(row["moves"], str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
