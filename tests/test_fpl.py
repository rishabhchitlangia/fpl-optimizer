"""Tests for the FPL optimizer.

Uses live cached API data rather than fixtures, because the point of most of
these assertions is that the real payloads satisfy the rules in RULES.md. Run
``python main.py --refresh`` first if the cache is cold.
"""

from __future__ import annotations

import unittest

from fpl import chips, data, optimizer, scoring, transfers


class TestSellingPrice(unittest.TestCase):
    """The 50%-of-profit, rounded-down sell-on fee. RULES.md §3."""

    def test_profit_is_halved_and_rounded_down(self):
        self.assertEqual(transfers.selling_price(74, 70), 72)   # +0.4 -> keep 0.2
        self.assertEqual(transfers.selling_price(73, 70), 71)   # +0.3 -> keep 0.1
        self.assertEqual(transfers.selling_price(71, 70), 70)   # +0.1 -> keep 0.0

    def test_losses_are_borne_in_full(self):
        self.assertEqual(transfers.selling_price(66, 70), 66)

    def test_unknown_purchase_price_falls_back_to_now_cost(self):
        self.assertEqual(transfers.selling_price(74, None), 74)


class TestFixtureMultiplier(unittest.TestCase):
    """Fixture difficulty scaling."""

    def test_neutral_difficulty_is_identity(self):
        for position in (1, 2, 3, 4):
            self.assertAlmostEqual(scoring.fixture_multiplier(3, position), 1.0)

    def test_easier_fixtures_scale_up_and_harder_down(self):
        self.assertGreater(scoring.fixture_multiplier(1, 2), 1.0)
        self.assertLess(scoring.fixture_multiplier(5, 2), 1.0)

    def test_defenders_are_more_fixture_sensitive_than_forwards(self):
        self.assertGreater(scoring.fixture_multiplier(1, 2),
                           scoring.fixture_multiplier(1, 4))


class TestAvailability(unittest.TestCase):
    """Injured and suspended players must be unselectable."""

    def test_unavailable_statuses_score_zero(self):
        for status in ("i", "s", "u", "n"):
            self.assertEqual(scoring.availability({"status": status}), 0.0)

    def test_stated_chance_is_respected(self):
        self.assertAlmostEqual(
            scoring.availability({"status": "d", "chance_of_playing_next_round": 25}),
            0.25)

    def test_fit_player_is_fully_available(self):
        self.assertEqual(scoring.availability({"status": "a"}), 1.0)


class TestLiveData(unittest.TestCase):
    """Assertions against the real API payloads."""

    @classmethod
    def setUpClass(cls):
        cls.bootstrap = data.get_bootstrap()
        cls.fixtures = data.get_fixtures()
        cls.players = data.player_lookup(cls.bootstrap)
        summaries = data.get_all_element_summaries(list(cls.players))
        cls.scorer = scoring.build_scorer(cls.bootstrap, cls.fixtures, summaries, "bayes")
        cls.gw = data.target_gameweek(cls.bootstrap)
        cls.projections = cls.scorer.project([cls.gw])

    def test_game_rules_match_rules_md(self):
        rules = self.bootstrap["game_config"]["rules"]
        self.assertEqual(rules["squad_squadsize"], 15)
        self.assertEqual(rules["squad_squadplay"], 11)
        self.assertEqual(rules["squad_team_limit"], 3)
        self.assertEqual(rules["squad_total_spend"], 1000)
        self.assertEqual(rules["max_extra_free_transfers"], 4)

    def test_assistant_manager_chip_is_gone(self):
        names = {c["name"] for c in self.bootstrap["chips"]}
        self.assertNotIn("manager", names)
        scoring_cfg = self.bootstrap["game_config"]["scoring"]
        self.assertEqual(scoring_cfg["mng_loss"], 0)

    def test_eight_chips_in_two_sets(self):
        self.assertEqual(len(self.bootstrap["chips"]), 8)
        first_half = [c for c in self.bootstrap["chips"] if c["stop_event"] == 19]
        self.assertEqual(len(first_half), 4)

    def test_unavailable_players_project_zero(self):
        for player in self.bootstrap["elements"]:
            if player["status"] in ("i", "s", "u", "n"):
                self.assertEqual(self.projections[player["id"]].expected_points, 0.0,
                                 f"{player['web_name']} is {player['status']}")

    def test_model_discriminates_unlike_ep_next(self):
        """The whole reason the Bayes model exists: ep_next saturates."""
        ep_values = [float(p["ep_next"] or 0) for p in self.bootstrap["elements"]]
        model_values = [p.expected_points for p in self.projections.values()]
        # ep_next has far fewer distinct values at the top end than the model.
        self.assertGreater(len(set(round(v, 2) for v in model_values)),
                           len(set(ep_values)))


class TestOptimizer(unittest.TestCase):
    """The optimizer must produce squads that satisfy every FPL constraint."""

    @classmethod
    def setUpClass(cls):
        cls.bootstrap = data.get_bootstrap()
        cls.fixtures = data.get_fixtures()
        cls.players = data.player_lookup(cls.bootstrap)
        cls.positions = data.position_lookup(cls.bootstrap)
        summaries = data.get_all_element_summaries(list(cls.players))
        scorer = scoring.build_scorer(cls.bootstrap, cls.fixtures, summaries, "bayes")
        cls.gw = data.target_gameweek(cls.bootstrap)
        cls.projections = scorer.project([cls.gw])
        cls.selection = optimizer.optimize_squad(cls.bootstrap, cls.projections)

    def _assert_legal(self, selection):
        squad = selection.squad_ids
        self.assertEqual(len(squad), 15)
        self.assertEqual(len(set(squad)), 15, "duplicate players in squad")

        counts = {}
        for pid in squad:
            counts[self.players[pid]["element_type"]] = \
                counts.get(self.players[pid]["element_type"], 0) + 1
        self.assertEqual(counts, {1: 2, 2: 5, 3: 5, 4: 3})

        clubs = {}
        for pid in squad:
            clubs[self.players[pid]["team"]] = clubs.get(self.players[pid]["team"], 0) + 1
        self.assertLessEqual(max(clubs.values()), 3, "more than 3 from one club")

        cost = sum(self.players[pid]["now_cost"] for pid in squad)
        self.assertLessEqual(cost, 1000, "over budget")

    def test_optimal_squad_is_legal(self):
        self._assert_legal(self.selection)

    def test_starting_xi_is_a_legal_formation(self):
        xi = self.selection.starting_ids
        self.assertEqual(len(xi), 11)
        counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for pid in xi:
            counts[self.players[pid]["element_type"]] += 1
        self.assertEqual(counts[1], 1, "must start exactly one goalkeeper")
        self.assertGreaterEqual(counts[2], 3)
        self.assertGreaterEqual(counts[3], 2)
        self.assertGreaterEqual(counts[4], 1)

    def test_bench_is_the_remaining_four(self):
        self.assertEqual(len(self.selection.bench_ids), 4)
        self.assertEqual(set(self.selection.starting_ids) | set(self.selection.bench_ids),
                         set(self.selection.squad_ids))

    def test_captain_starts_and_is_top_projected(self):
        self.assertIn(self.selection.captain_id, self.selection.starting_ids)
        best = max(self.selection.starting_ids,
                   key=lambda p: self.projections[p].expected_points)
        self.assertEqual(self.selection.captain_id, best)
        self.assertNotEqual(self.selection.captain_id, self.selection.vice_captain_id)

    def test_locked_player_is_selected(self):
        haaland = next(p["id"] for p in self.bootstrap["elements"]
                       if p["web_name"] == "Haaland")
        selection = optimizer.optimize_squad(self.bootstrap, self.projections,
                                             locked=[haaland])
        self.assertIn(haaland, selection.squad_ids)
        self._assert_legal(selection)

    def test_banned_player_is_excluded(self):
        banned = self.selection.squad_ids[0]
        selection = optimizer.optimize_squad(self.bootstrap, self.projections,
                                             banned=[banned])
        self.assertNotIn(banned, selection.squad_ids)

    def test_lower_budget_yields_cheaper_and_weaker_squad(self):
        cheap = optimizer.optimize_squad(self.bootstrap, self.projections, budget=850)
        self.assertLessEqual(sum(self.players[p]["now_cost"] for p in cheap.squad_ids), 850)
        self.assertLess(cheap.predicted_points, self.selection.predicted_points)

    def test_eight_valid_formations_derived_from_rules(self):
        formations = optimizer._valid_formations(self.positions)
        self.assertEqual(len(formations), 8, "RULES.md §1 says 8 legal formations")
        self.assertIn((5, 2, 3), formations)   # the commonly forgotten one
        self.assertNotIn((3, 3, 4), formations)


class TestTransfers(unittest.TestCase):
    """Transfer planning against a deliberately sub-optimal starting squad."""

    @classmethod
    def setUpClass(cls):
        cls.bootstrap = data.get_bootstrap()
        cls.fixtures = data.get_fixtures()
        cls.players = data.player_lookup(cls.bootstrap)
        summaries = data.get_all_element_summaries(list(cls.players))
        scorer = scoring.build_scorer(cls.bootstrap, cls.fixtures, summaries, "bayes")
        cls.gw = data.target_gameweek(cls.bootstrap)
        cls.projections = scorer.project([cls.gw])

        # Build a legal but weak squad by starving the optimizer of budget,
        # then hand back the difference as bank so upgrades are affordable.
        weak = optimizer.optimize_squad(cls.bootstrap, cls.projections, budget=830)
        spent = sum(cls.players[p]["now_cost"] for p in weak.squad_ids)
        cls.squad = transfers.CurrentSquad(
            player_ids=list(weak.squad_ids),
            bank=1000 - spent,
            squad_value=spent,
            gameweek=cls.gw,
        )

    def test_plans_include_a_no_transfer_baseline(self):
        plans = transfers.evaluate_transfer_plans(
            self.bootstrap, self.projections, self.squad, free_transfers=1)
        self.assertTrue(any(p.n_transfers == 0 for p in plans))

    def test_plans_are_ranked_by_net_gain(self):
        plans = transfers.evaluate_transfer_plans(
            self.bootstrap, self.projections, self.squad, free_transfers=1)
        gains = [p.net_gain for p in plans]
        self.assertEqual(gains, sorted(gains, reverse=True))

    def test_hits_are_charged_beyond_free_transfers(self):
        plans = transfers.evaluate_transfer_plans(
            self.bootstrap, self.projections, self.squad,
            free_transfers=1, max_transfers=3)
        for plan in plans:
            expected = optimizer.TRANSFER_HIT_COST * max(0, plan.n_transfers - 1)
            self.assertAlmostEqual(plan.hit_cost, expected,
                                   msg=f"{plan.n_transfers} transfers")
            self.assertAlmostEqual(plan.net_gain, plan.gross_gain - plan.hit_cost)

    def test_more_free_transfers_means_no_hits(self):
        plans = transfers.evaluate_transfer_plans(
            self.bootstrap, self.projections, self.squad,
            free_transfers=5, max_transfers=4)
        self.assertTrue(all(p.hit_cost == 0 for p in plans))

    def test_a_weak_squad_has_a_worthwhile_upgrade(self):
        plans = transfers.evaluate_transfer_plans(
            self.bootstrap, self.projections, self.squad, free_transfers=1)
        best = transfers.best_plan(plans)
        self.assertGreater(best.net_gain, 0.0)
        self.assertGreater(best.n_transfers, 0)

    def test_resulting_squad_stays_legal(self):
        plans = transfers.evaluate_transfer_plans(
            self.bootstrap, self.projections, self.squad, free_transfers=2)
        for plan in plans:
            squad = plan.selection.squad_ids
            self.assertEqual(len(squad), 15)
            clubs = {}
            for pid in squad:
                clubs[self.players[pid]["team"]] = clubs.get(self.players[pid]["team"], 0) + 1
            self.assertLessEqual(max(clubs.values()), 3)

    def test_moves_match_transfer_count(self):
        plans = transfers.evaluate_transfer_plans(
            self.bootstrap, self.projections, self.squad, free_transfers=1)
        for plan in plans:
            self.assertEqual(len(plan.moves), plan.n_transfers)


class TestChips(unittest.TestCase):
    """Chip availability windows and blank/double detection."""

    @classmethod
    def setUpClass(cls):
        cls.bootstrap = data.get_bootstrap()
        cls.fixtures = data.get_fixtures()

    def test_wildcard_and_free_hit_locked_in_gameweek_1(self):
        playable = chips.available_chips(self.bootstrap, 1)
        self.assertNotIn("wildcard", playable)
        self.assertNotIn("freehit", playable)
        self.assertIn("bboost", playable)
        self.assertIn("3xc", playable)

    def test_all_four_chips_available_mid_first_half(self):
        playable = chips.available_chips(self.bootstrap, 10)
        self.assertEqual(playable, {"wildcard", "freehit", "bboost", "3xc"})

    def test_used_chips_are_excluded(self):
        playable = chips.available_chips(self.bootstrap, 10, chips_used=["bboost"])
        self.assertNotIn("bboost", playable)

    def test_second_set_available_after_gameweek_19(self):
        self.assertEqual(chips.available_chips(self.bootstrap, 25),
                         {"wildcard", "freehit", "bboost", "3xc"})

    def test_gameweek_shape_detects_normal_week(self):
        shape = chips.gameweek_shape(self.bootstrap, self.fixtures, 1)
        self.assertEqual(shape["label"], "normal")
        self.assertEqual(shape["fixtures"], 10)
        self.assertEqual(shape["blank_teams"], [])

    def test_blank_and_double_detection_is_symmetric(self):
        """Every team-gameweek must be accounted for as blank, single or double."""
        context = scoring.FixtureContext(self.fixtures)
        teams = [t["id"] for t in self.bootstrap["teams"]]
        for gw in range(1, 39):
            total = sum(context.fixture_count(t, gw) for t in teams)
            self.assertEqual(total % 2, 0,
                             f"GW{gw} has an odd number of team-fixtures")


if __name__ == "__main__":
    unittest.main(verbosity=2)
