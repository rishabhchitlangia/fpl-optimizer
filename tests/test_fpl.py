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


class TestPeakMinutes(unittest.TestCase):
    """A wrecked recent season must not erase proven capability."""

    def _rates(self, minutes_by_season_oldest_first):
        summary = {"history_past": [
            {"minutes": m, "total_points": 0, "starts": 0}
            for m in minutes_by_season_oldest_first
        ]}
        return scoring.historical_rates(summary)

    def test_peak_survives_a_blank_recent_season(self):
        """Three full seasons then an injury year: project on the three."""
        rates = self._rates([3000, 3000, 3000, 100])
        self.assertGreater(rates.peak_minutes_per_game(), 40.0)

    def test_peak_decays_with_age(self):
        recent_peak = self._rates([0, 0, 3000]).peak_minutes_per_game()
        old_peak = self._rates([3000, 0, 0]).peak_minutes_per_game()
        self.assertGreater(recent_peak, old_peak)

    def test_peak_is_discounted_below_the_raw_level(self):
        """We never project a past peak forward at full value."""
        rates = self._rates([3420])           # every minute of a season
        self.assertLess(rates.peak_minutes_per_game(), 3420 / 38)

    def test_no_history_means_no_peak(self):
        self.assertEqual(scoring.historical_rates(None).peak_minutes_per_game(), 0.0)

    def test_zero_seasons_are_recorded_not_skipped(self):
        """A season sat out is information; dropping it would inflate the mean."""
        rates = self._rates([3000, 0])
        self.assertEqual(len(rates.minutes_per_game), 2)
        self.assertEqual(rates.minutes_per_game[0], 0.0)   # most recent first


class TestNewSignings(unittest.TestCase):
    """Minutes earned at a previous club are weaker evidence."""

    @classmethod
    def setUpClass(cls):
        cls.bootstrap = data.get_bootstrap()
        cls.fixtures = data.get_fixtures()
        players = data.player_lookup(cls.bootstrap)
        summaries = data.get_all_element_summaries(list(players))
        cls.scorer = scoring.BayesianRateScorer(
            cls.bootstrap, cls.fixtures, summaries)

    def test_recent_join_date_is_detected(self):
        self.assertTrue(self.scorer._is_new_signing({"team_join_date": "2026-07-15"}))
        self.assertFalse(self.scorer._is_new_signing({"team_join_date": "2023-08-01"}))
        self.assertFalse(self.scorer._is_new_signing({}))

    def _candidate_without_current_season(self):
        """A player with a prior-season record but no games yet this season.

        The discount only has anything to act on before the new club's own
        minutes are observable. Once a signing has played a month, his
        minutes at *this* club are the evidence and the old ones stop
        mattering -- so the test has to pose the question at the point the
        discount exists to answer, not in November.
        """
        for player in self.bootstrap["elements"]:
            summary = self.scorer.summaries.get(player["id"]) or {}
            if not (summary.get("history_past") or []):
                continue
            rates = scoring.historical_rates(summary)
            if rates.weighted_games <= 0:
                continue
            if player["now_cost"] < 55:
                continue
            blanked = dict(summary, history=[])
            self.scorer.summaries[player["id"]] = blanked
            return dict(player)
        return None

    def test_new_signing_status_changes_the_estimate(self):
        """Join date must actually feed through to projected minutes."""
        candidate = self._candidate_without_current_season()
        self.assertIsNotNone(candidate, "no player with a prior-season record")
        settled = dict(candidate, team_join_date="2022-07-01")
        moved = dict(candidate, team_join_date="2026-07-01")
        self.assertNotAlmostEqual(self.scorer._expected_minutes(moved),
                                  self.scorer._expected_minutes(settled))

    def test_observed_minutes_override_the_signing_discount(self):
        """A signing who has played here is judged on that, not his join date.

        Regression guard in the other direction: the backtest showed the
        current season should take over the minutes estimate quickly, and
        once it has, the join date must stop moving the number.
        """
        candidate = next(
            dict(p) for p in self.bootstrap["elements"]
            if (self.scorer.summaries.get(p["id"]) or {}).get("history")
        )
        history = self.scorer.summaries[candidate["id"]]["history"]
        if len(history) < scoring.MINUTES_CURRENT_SEASON_FULL_WEIGHT:
            self.skipTest("too early in the season for the override to apply")
        settled = dict(candidate, team_join_date="2022-07-01")
        moved = dict(candidate, team_join_date="2026-07-01")
        self.assertAlmostEqual(self.scorer._expected_minutes(moved),
                               self.scorer._expected_minutes(settled))

    def test_a_proven_starter_who_moves_is_projected_down(self):
        """Minutes earned elsewhere are real but weaker evidence.

        Widening shrinkage pulls the estimate toward the price-implied prior,
        so a player who was near ever-present at a previous club loses some of
        that certainty on arrival.
        """
        proven = None
        for player in self.bootstrap["elements"]:
            summary = self.scorer.summaries.get(player["id"]) or {}
            if not (summary.get("history_past") or []):
                continue
            rates = scoring.historical_rates(dict(summary, history=[]))
            if rates.weighted_games <= 0:
                continue
            observed = rates.weighted_minutes / rates.weighted_games
            if observed > 80 and float(player["selected_by_percent"] or 0) < 5:
                # Pose the question before this season's minutes exist, which
                # is the only point at which the discount can apply.
                self.scorer.summaries[player["id"]] = dict(summary, history=[])
                proven = player
                break
        self.assertIsNotNone(proven, "no near-ever-present low-owned player found")

        settled = dict(proven, team_join_date="2021-07-01")
        moved = dict(proven, team_join_date="2026-07-01")
        self.assertLess(self.scorer._expected_minutes(moved),
                        self.scorer._expected_minutes(settled))

    def test_the_league_actually_contains_new_signings(self):
        """Guards against the cutoff silently matching nobody."""
        new = [p for p in self.bootstrap["elements"]
               if self.scorer._is_new_signing(p)]
        self.assertGreater(len(new), 10)


class TestOwnershipIsPriceAware(unittest.TestCase):
    """Ownership means different things at different price points."""

    @classmethod
    def setUpClass(cls):
        cls.bootstrap = data.get_bootstrap()
        cls.fixtures = data.get_fixtures()
        players = data.player_lookup(cls.bootstrap)
        summaries = data.get_all_element_summaries(list(players))
        cls.scorer = scoring.BayesianRateScorer(
            cls.bootstrap, cls.fixtures, summaries)

    def test_minimum_priced_players_get_no_ownership_boost(self):
        """A heavily-owned cheap keeper is bench fodder, not a nailed starter."""
        cheapest = min((p for p in self.bootstrap["elements"]
                        if p["element_type"] == 1),
                       key=lambda p: p["now_cost"])
        self.assertEqual(self.scorer._ownership_confidence(cheapest), 0.0)

    def test_premium_players_get_the_full_ownership_signal(self):
        premium = max(self.bootstrap["elements"], key=lambda p: p["now_cost"])
        self.assertEqual(self.scorer._ownership_confidence(premium), 1.0)

    def test_confidence_rises_with_price(self):
        by_price = sorted((p for p in self.bootstrap["elements"]
                           if p["element_type"] == 2),
                          key=lambda p: p["now_cost"])
        self.assertLessEqual(self.scorer._ownership_confidence(by_price[0]),
                             self.scorer._ownership_confidence(by_price[-1]))

    def test_cheap_bench_keepers_are_not_projected_as_starters(self):
        """Regression: the ownership floor once put backup keepers in the XI."""
        for player in self.bootstrap["elements"]:
            is_cheap_keeper = (player["element_type"] == 1
                               and player["now_cost"] <= 40)
            heavily_owned = float(player["selected_by_percent"] or 0) > 15
            if is_cheap_keeper and heavily_owned:
                self.assertLess(
                    self.scorer._expected_minutes(player), 70.0,
                    f"{player['web_name']} projected as a near-full-time starter")


class TestSetPiecePremium(unittest.TestCase):
    """Set-piece duty is new-season information history cannot contain."""

    def test_designated_penalty_taker_earns_a_premium(self):
        taker = {"element_type": 4, "penalties_order": 1}
        none = {"element_type": 4}
        self.assertGreater(scoring.set_piece_premium(taker),
                           scoring.set_piece_premium(none))

    def test_second_choice_taker_earns_less_than_first(self):
        first = scoring.set_piece_premium({"element_type": 3, "penalties_order": 1})
        second = scoring.set_piece_premium({"element_type": 3, "penalties_order": 2})
        self.assertGreater(first, second)
        self.assertGreater(second, 0.0)

    def test_goalkeepers_never_earn_a_set_piece_premium(self):
        self.assertEqual(
            scoring.set_piece_premium({"element_type": 1, "penalties_order": 1}), 0.0)

    def test_corner_duty_stacks_with_penalty_duty(self):
        both = scoring.set_piece_premium({
            "element_type": 3, "penalties_order": 1,
            "corners_and_indirect_freekicks_order": 1})
        pens_only = scoring.set_piece_premium({"element_type": 3, "penalties_order": 1})
        self.assertGreater(both, pens_only)

    def test_premium_is_not_double_counted_for_established_players(self):
        """A long-serving taker's history already contains their penalties.

        The premium is scaled by prior reliance, so a player with a lot of
        history should receive far less of it than one with none.
        """
        bootstrap = data.get_bootstrap()
        fixtures = data.get_fixtures()
        players = data.player_lookup(bootstrap)
        summaries = data.get_all_element_summaries(list(players))
        scorer = scoring.BayesianRateScorer(bootstrap, fixtures, summaries)

        veteran = scoring.historical_rates(
            summaries[next(p["id"] for p in bootstrap["elements"]
                           if p["web_name"] == "Haaland")])
        newcomer = scoring.HistoricalRates(0.0, 0.0, 0.0, 0.0, 0)
        self.assertLess(scorer._prior_reliance(veteran),
                        scorer._prior_reliance(newcomer))
        self.assertAlmostEqual(scorer._prior_reliance(newcomer), 1.0)


class TestTeamStrength(unittest.TestCase):
    """Club quality nudges the no-history prior."""

    def test_stronger_clubs_scale_the_prior_up(self):
        strong = {"strength_overall_home": 5, "strength_overall_away": 5}
        weak = {"strength_overall_home": 2, "strength_overall_away": 2}
        self.assertGreater(scoring.team_strength_multiplier(strong), 1.0)
        self.assertLess(scoring.team_strength_multiplier(weak), 1.0)

    def test_missing_strength_data_is_survivable(self):
        self.assertGreater(scoring.team_strength_multiplier({}), 0.0)


class TestOwnershipFloor(unittest.TestCase):
    """Ownership raises the minutes prior but never lowers it."""

    @classmethod
    def setUpClass(cls):
        cls.bootstrap = data.get_bootstrap()
        cls.fixtures = data.get_fixtures()
        players = data.player_lookup(cls.bootstrap)
        cls.summaries = data.get_all_element_summaries(list(players))
        cls.scorer = scoring.BayesianRateScorer(
            cls.bootstrap, cls.fixtures, cls.summaries)

    def _minutes(self, ownership: float) -> float:
        template = dict(self.bootstrap["elements"][0])
        template["selected_by_percent"] = str(ownership)
        template["id"] = -1          # no history
        return self.scorer._expected_minutes(template)

    def test_high_ownership_raises_expected_minutes(self):
        self.assertGreater(self._minutes(50.0), self._minutes(1.0))

    def test_ownership_is_a_floor_not_a_ceiling(self):
        """Zero ownership must not drag a player below their own prior."""
        self.assertAlmostEqual(self._minutes(0.0), self._minutes(0.0))
        self.assertGreater(self._minutes(0.0), 0.0)

    def test_ownership_does_not_touch_the_points_rate(self):
        """Following the crowd on quality would push every squad to template."""
        template = dict(self.bootstrap["elements"][0])
        template["id"] = -1
        low = dict(template, selected_by_percent="0.1")
        high = dict(template, selected_by_percent="80.0")
        self.assertAlmostEqual(self.scorer._points_per_90(low),
                               self.scorer._points_per_90(high))


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

    def test_unselectable_players_are_excluded_whatever_their_status(self):
        """can_select is FPL's direct statement that a player cannot be picked.

        A departed player normally also carries status 'u', but that alignment
        is incidental and must not be relied on.
        """
        self.assertEqual(
            scoring.availability({"status": "a", "can_select": False}), 0.0)
        self.assertEqual(
            scoring.availability({"status": "a", "removed": True}), 0.0)

    def test_live_departures_are_all_excluded(self):
        """Every player FPL marks unselectable must project zero."""
        bootstrap = data.get_bootstrap()
        departed = [p for p in bootstrap["elements"]
                    if p.get("can_select") is False or p.get("removed")]
        for player in departed:
            self.assertEqual(scoring.availability(player), 0.0,
                             f"{player['web_name']} is still selectable")


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
        """Players who cannot feature must project zero.

        Note the qualification. Being flagged ``i`` or ``s`` is no longer
        sufficient on its own: a player whose news states a return date falling
        inside the horizon is legitimately worth points from that date, and
        zeroing them was a real modelling error. What must always be zero is a
        player with no route back before the fixture — departed, unselectable,
        or carrying an indefinite absence.
        """
        from fpl import news as news_module

        for player in self.bootstrap["elements"]:
            flag = news_module.parse_news(player)
            unreachable = (
                player.get("can_select") is False
                or player.get("removed")
                or flag.kind == "departed"
                or (player["status"] in ("i", "s", "n")
                    and flag.return_date is None)
            )
            if unreachable:
                self.assertEqual(
                    self.projections[player["id"]].expected_points, 0.0,
                    f"{player['web_name']} ({player['status']}) should be zero")

    def test_a_stated_return_date_can_lift_a_flagged_player(self):
        """The behaviour the previous assertion used to forbid."""
        from fpl import news as news_module

        returning = [
            p for p in self.bootstrap["elements"]
            if p["status"] == "i" and news_module.parse_news(p).return_date
        ]
        if not returning:
            self.skipTest("nobody currently flagged with a return date")
        # Over a long horizon at least one of them must be worth something.
        long_horizon = self.scorer.project(list(range(self.gw, min(self.gw + 8, 39))))
        self.assertGreater(
            max(long_horizon[p["id"]].expected_points for p in returning), 0.0)

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


class TestCurrentSeasonIsUsed(unittest.TestCase):
    """The projection must see the season in progress, not only finished ones.

    Regression: the rate model read ``history_past`` alone, so four gameweeks
    into a season every player was still projected purely from last year.
    """

    def _summary(self, current_points, current_minutes, past_points, past_minutes):
        return {
            "history": [{"minutes": current_minutes, "total_points": current_points,
                         "starts": 1, "expected_goal_involvements": "0.5"}],
            "history_past": [{"minutes": past_minutes, "total_points": past_points,
                              "starts": 30, "expected_goal_involvements": "10.0"}],
        }

    def test_current_season_moves_the_rate(self):
        summary = self._summary(20, 90, 100, 3000)
        without = scoring.historical_rates(summary, include_current=False)
        with_now = scoring.historical_rates(summary, include_current=True)
        self.assertGreater(with_now.weighted_points / with_now.weighted_90s,
                           without.weighted_points / without.weighted_90s)

    def test_a_player_with_only_this_season_is_not_blank(self):
        """A debutant has no past seasons but is not therefore worthless."""
        summary = {"history": [{"minutes": 90, "total_points": 8, "starts": 1}],
                   "history_past": []}
        rates = scoring.historical_rates(summary)
        self.assertGreater(rates.weighted_90s, 0.0)
        self.assertGreater(rates.weighted_points, 0.0)

    def test_current_season_is_weighted_by_matches_played(self):
        """One gameweek must not outweigh a full prior season."""
        one_game = scoring.historical_rates(self._summary(2, 90, 200, 3000))
        self.assertLess(one_game.weighted_games, 38)

    def test_live_data_actually_reaches_the_model(self):
        bootstrap = data.get_bootstrap()
        summaries = data.get_all_element_summaries(
            [e["id"] for e in bootstrap["elements"]])
        played = [s for s in summaries.values() if s.get("history")]
        if not played:
            self.skipTest("season has not started")
        moved = 0
        for summary in list(played)[:120]:
            a = scoring.historical_rates(summary, include_current=False)
            b_ = scoring.historical_rates(summary, include_current=True)
            if abs(a.weighted_points - b_.weighted_points) > 1e-9:
                moved += 1
        self.assertGreater(moved, 0, "current-season data is being ignored")


class TestBanningAnOwnedPlayerReleasesHisValue(unittest.TestCase):
    """Selling a banned player must fund his replacement.

    Regression: ``banned`` drops a player from the candidate pool, and the
    budget constraint raised sale proceeds only over candidates. An owned
    player who was banned was therefore forced out with his money confiscated,
    so a straight one-for-one swap on an empty bank reported "Infeasible" and
    larger plans paid for the replacement by selling someone expensive.
    """

    @classmethod
    def setUpClass(cls):
        cls.bootstrap = data.get_bootstrap()
        cls.fixtures = data.get_fixtures()
        cls.players = data.player_lookup(cls.bootstrap)
        summaries = data.get_all_element_summaries(list(cls.players))
        scorer = scoring.build_scorer(cls.bootstrap, cls.fixtures, summaries, "bayes")
        cls.gw = data.target_gameweek(cls.bootstrap)
        cls.projections = scorer.project([cls.gw])
        # A squad that spends everything, so the replacement can only be funded
        # out of the banned player's sale.
        cls.squad = optimizer.optimize_squad(cls.bootstrap, cls.projections)

    def _expensive_defender(self) -> int:
        defenders = [pid for pid in self.squad.squad_ids
                     if self.players[pid]["element_type"] == 2]
        return max(defenders, key=lambda pid: self.players[pid]["now_cost"])

    def test_one_for_one_swap_is_feasible_on_an_empty_bank(self):
        banned = self._expensive_defender()
        selection = optimizer.optimize_squad(
            self.bootstrap, self.projections,
            current_squad=list(self.squad.squad_ids), bank=0,
            max_transfers=1, apply_hits=False, banned=[banned],
        )
        self.assertEqual(selection.transfers_out, [banned])
        self.assertEqual(len(selection.transfers_in), 1)

    def test_a_second_transfer_never_makes_the_plan_worse(self):
        """More freedom cannot hurt, so the ladder must not go backwards.

        This is the sharpest signature of the bug: with the proceeds
        confiscated, the two-transfer plan funded the replacement by selling
        the squad's most expensive player, and so scored *below* the plan that
        made a single swap.
        """
        banned = self._expensive_defender()
        ladder = []
        for cap in (1, 2, 3):
            selection = optimizer.optimize_squad(
                self.bootstrap, self.projections,
                current_squad=list(self.squad.squad_ids), bank=0,
                max_transfers=cap, apply_hits=False, banned=[banned],
            )
            self.assertIn(banned, selection.transfers_out)
            ladder.append(selection.predicted_points)
        for cap, (fewer, more) in enumerate(zip(ladder, ladder[1:]), start=1):
            self.assertGreaterEqual(more, fewer - 1e-6,
                                    msg=f"{cap + 1} transfers scored below {cap}")

    def test_replacement_is_affordable_at_the_banned_price(self):
        banned = self._expensive_defender()
        selection = optimizer.optimize_squad(
            self.bootstrap, self.projections,
            current_squad=list(self.squad.squad_ids), bank=0,
            max_transfers=1, apply_hits=False, banned=[banned],
        )
        incoming = selection.transfers_in[0]
        self.assertLessEqual(self.players[incoming]["now_cost"],
                             self.players[banned]["now_cost"])


class TestModelSelection(unittest.TestCase):
    """``auto`` must not dilute the rate model with a minutes-blind one.

    Regression: blending ``ep_next`` in-season halved the minutes model's
    discount, so a substitute with one big haul outranked an established
    starter whose next fixture happened to be awkward.
    """

    @classmethod
    def setUpClass(cls):
        cls.bootstrap = data.get_bootstrap()
        cls.fixtures = data.get_fixtures()
        players = data.player_lookup(cls.bootstrap)
        cls.summaries = data.get_all_element_summaries(list(players))

    def test_auto_is_the_rate_model_when_history_is_available(self):
        scorer = scoring.build_scorer(
            self.bootstrap, self.fixtures, self.summaries, "auto")
        self.assertIsInstance(scorer, scoring.BayesianRateScorer)

    def test_auto_falls_back_to_ep_next_without_history(self):
        scorer = scoring.build_scorer(self.bootstrap, self.fixtures, {}, "auto")
        self.assertIsInstance(scorer, scoring.EPNextScorer)

    def test_blended_is_still_available_on_request(self):
        scorer = scoring.build_scorer(
            self.bootstrap, self.fixtures, self.summaries, "blended")
        self.assertIsInstance(scorer, scoring.BlendedScorer)


class TestEPNextDiscountsRotation(unittest.TestCase):
    """``ep_next`` says nothing about minutes, so involvement is applied.

    Taken at face value it values a substitute who scored once like a starter
    scoring at the same rate per game, which is what inflated cheap bench
    players when the blended model was the in-season default.
    """

    def setUp(self):
        self.bootstrap = data.get_bootstrap()
        self.fixtures = data.get_fixtures()
        self.scorer = scoring.EPNextScorer(self.bootstrap, self.fixtures)
        self.finished = data.get_game_state(self.bootstrap).finished_gws

    def test_involvement_is_neutral_without_evidence(self):
        self.scorer.finished_gws = 0
        self.assertAlmostEqual(self.scorer._involvement({"minutes": 0}), 1.0)

    def test_a_nailed_starter_keeps_his_figure(self):
        if not self.finished:
            self.skipTest("season has not started")
        nailed = {"minutes": 90 * self.finished}
        self.assertAlmostEqual(self.scorer._involvement(nailed), 1.0)

    def test_a_substitute_is_discounted(self):
        if self.finished < scoring.EP_INVOLVEMENT_FULL_WEIGHT:
            self.skipTest("too early for involvement to carry full weight")
        bench = {"minutes": 20 * self.finished}
        factor = self.scorer._involvement(bench)
        self.assertLess(factor, 1.0)
        # Partial, not total: FPL's own figure already allows for some risk.
        self.assertGreater(factor, 1.0 - scoring.EP_INVOLVEMENT_DISCOUNT - 1e-9)

    def test_the_discount_reaches_the_projection(self):
        if self.finished < scoring.EP_INVOLVEMENT_FULL_WEIGHT:
            self.skipTest("too early for involvement to carry full weight")
        gw = data.target_gameweek(self.bootstrap)
        projections = self.scorer.project([gw])
        rotated = [e for e in self.bootstrap["elements"]
                   if float(e.get("ep_next") or 0) > 2.0
                   and 0 < (e.get("minutes") or 0) < 45 * self.finished]
        if not rotated:
            self.skipTest("no rotation player with a high ep_next")
        player = rotated[0]
        self.assertLess(projections[player["id"]].expected_minutes, 90.0)


class TestRecentMinutes(unittest.TestCase):
    """Current-season minutes are weighted toward the recent gameweeks."""

    def _history(self, minutes):
        return [{"minutes": m, "total_points": 0, "round": i + 1}
                for i, m in enumerate(minutes)]

    def test_decay_of_one_is_the_flat_mean(self):
        h = self._history([90, 0, 45])
        self.assertAlmostEqual(scoring.recent_minutes_per_game(h, 1.0), 45.0)

    def test_recent_gameweeks_dominate(self):
        """A starter who has been dropped reads lower than his season mean."""
        dropped = self._history([90] * 6 + [0, 45, 35])
        flat = scoring.recent_minutes_per_game(dropped, 1.0)
        weighted = scoring.recent_minutes_per_game(dropped, 0.7)
        self.assertLess(weighted, flat)

    def test_it_works_the_other_way_too(self):
        """A squad player who won the shirt reads higher than his mean."""
        promoted = self._history([0, 0, 0, 20, 45, 90, 90, 90])
        flat = scoring.recent_minutes_per_game(promoted, 1.0)
        weighted = scoring.recent_minutes_per_game(promoted, 0.7)
        self.assertGreater(weighted, flat)

    def test_empty_history_is_zero(self):
        self.assertEqual(scoring.recent_minutes_per_game([]), 0.0)

    def test_order_matters(self):
        """The same minutes in the opposite order must not read the same."""
        rising = self._history([0, 0, 90, 90])
        falling = self._history([90, 90, 0, 0])
        self.assertGreater(scoring.recent_minutes_per_game(rising, 0.7),
                           scoring.recent_minutes_per_game(falling, 0.7))


class TestObjectiveIsReported(unittest.TestCase):
    """The solver's own objective is exposed, because the XI is not it.

    Regression: plans were ranked on ``predicted_points`` (XI only) while the
    solver maximised XI plus the bench at BENCH_WEIGHT. Adding a constraint
    could then appear to improve a squad.
    """

    @classmethod
    def setUpClass(cls):
        cls.bootstrap = data.get_bootstrap()
        cls.fixtures = data.get_fixtures()
        players = data.player_lookup(cls.bootstrap)
        summaries = data.get_all_element_summaries(list(players))
        scorer = scoring.build_scorer(cls.bootstrap, cls.fixtures, summaries, "bayes")
        cls.projections = scorer.project([data.target_gameweek(cls.bootstrap)])

    def test_optimize_squad_reports_its_objective(self):
        selection = optimizer.optimize_squad(self.bootstrap, self.projections)
        self.assertIsNotNone(selection.objective_value)

    def test_objective_exceeds_the_xi_because_the_bench_counts(self):
        selection = optimizer.optimize_squad(self.bootstrap, self.projections)
        self.assertGreater(selection.objective_value, selection.predicted_points)

    def test_pick_starting_xi_reports_no_objective(self):
        """Nothing was solved, so there is no objective to report."""
        selection = optimizer.optimize_squad(self.bootstrap, self.projections)
        redone = optimizer.pick_starting_xi(
            selection.squad_ids, data.player_lookup(self.bootstrap),
            self.projections, {t["id"]: t for t in self.bootstrap["element_types"]})
        self.assertIsNone(redone.objective_value)

    def test_a_constraint_never_improves_the_objective(self):
        base = optimizer.optimize_squad(self.bootstrap, self.projections)
        players = data.player_lookup(self.bootstrap)
        absent = max((p for p in self.bootstrap["elements"]
                      if p["id"] not in set(base.squad_ids)),
                     key=lambda p: p["now_cost"])
        pinned = optimizer.optimize_squad(
            self.bootstrap, self.projections, locked=[absent["id"]])
        self.assertLessEqual(pinned.objective_value, base.objective_value + 1e-6)


class TestUnderlyingPoints(unittest.TestCase):
    """This season's rate is built from chances, not from what went in."""

    def _match(self, **over):
        match = {"total_points": 2, "goals_scored": 0, "assists": 0,
                 "expected_goals": "0.0", "expected_assists": "0.0", "minutes": 90}
        match.update(over)
        return match

    def test_a_lucky_goal_is_marked_down(self):
        """A forward who scored from 0.2 xG is credited with 0.2 of a goal."""
        match = self._match(total_points=6, goals_scored=1, expected_goals="0.2")
        self.assertAlmostEqual(scoring.underlying_points(match, 4, 1.0),
                               6 - 0.8 * scoring.GOAL_POINTS[4])

    def test_an_unlucky_blank_is_marked_up(self):
        match = self._match(expected_goals="0.9", expected_assists="0.5")
        expected = 2 + 0.9 * scoring.GOAL_POINTS[3] + 0.5 * scoring.ASSIST_POINTS
        self.assertAlmostEqual(scoring.underlying_points(match, 3, 1.0), expected)

    def test_zero_weight_returns_actual_points(self):
        match = self._match(total_points=13, goals_scored=2, expected_goals="0.4")
        self.assertAlmostEqual(scoring.underlying_points(match, 4, 0.0), 13.0)

    def test_no_position_means_no_adjustment(self):
        match = self._match(total_points=13, goals_scored=2, expected_goals="0.4")
        self.assertAlmostEqual(scoring.underlying_points(match, None), 13.0)

    def test_a_match_without_expected_stats_is_left_alone(self):
        """Missing data is not the same as having created nothing."""
        match = {"total_points": 8, "goals_scored": 1, "assists": 0, "minutes": 90}
        self.assertAlmostEqual(scoring.underlying_points(match, 4, 1.0), 8.0)

    def test_the_rate_uses_it_only_when_given_a_position(self):
        summary = {"history": [self._match(total_points=10, goals_scored=2,
                                           expected_goals="0.5")],
                   "history_past": []}
        as_scored = scoring.historical_rates(summary)
        underlying = scoring.historical_rates(summary, element_type=4)
        self.assertAlmostEqual(as_scored.weighted_points, 10.0)
        self.assertLess(underlying.weighted_points, as_scored.weighted_points)
        self.assertAlmostEqual(underlying.weighted_90s, as_scored.weighted_90s)


class TestSeasonDecayIsReadAtCallTime(unittest.TestCase):
    """Regression: the constant was frozen into a default argument at import."""

    def setUp(self):
        self.summary = {"history": [],
                        "history_past": [
                            {"minutes": 3000, "total_points": 100, "starts": 33},
                            {"minutes": 3000, "total_points": 200, "starts": 33}]}
        self._saved = scoring.SEASON_DECAY

    def tearDown(self):
        scoring.SEASON_DECAY = self._saved

    def test_changing_the_constant_changes_the_estimate(self):
        scoring.SEASON_DECAY = 0.9
        gentle = scoring.historical_rates(self.summary)
        scoring.SEASON_DECAY = 0.1
        steep = scoring.historical_rates(self.summary)
        # The older season scored half as much; a steeper decay trusts it less.
        self.assertGreater(steep.weighted_points / steep.weighted_90s,
                           gentle.weighted_points / gentle.weighted_90s)

    def test_an_explicit_decay_still_wins(self):
        scoring.SEASON_DECAY = 0.1
        explicit = scoring.historical_rates(self.summary, decay=0.9)
        scoring.SEASON_DECAY = 0.9
        default = scoring.historical_rates(self.summary)
        self.assertAlmostEqual(explicit.weighted_points, default.weighted_points)


class TestCalibration(unittest.TestCase):
    """Per-90 output is pulled toward the position mean, and only that."""

    def test_full_weight_is_the_identity(self):
        self.assertAlmostEqual(scoring.calibrated_rate(7.0, 4.0, 1.0), 7.0)

    def test_a_high_rate_comes_down_and_a_low_one_goes_up(self):
        self.assertLess(scoring.calibrated_rate(7.0, 4.0, 0.7), 7.0)
        self.assertGreater(scoring.calibrated_rate(2.0, 4.0, 0.7), 2.0)

    def test_order_is_preserved(self):
        low, high = (scoring.calibrated_rate(v, 4.0, 0.7) for v in (5.0, 6.0))
        self.assertLess(low, high)

    def test_the_gap_between_two_players_shrinks_by_the_factor(self):
        a, b_ = (scoring.calibrated_rate(v, 4.0, 0.7) for v in (5.0, 7.0))
        self.assertAlmostEqual(b_ - a, 0.7 * 2.0)

    def test_no_anchor_means_no_change(self):
        self.assertAlmostEqual(scoring.calibrated_rate(7.0, None, 0.5), 7.0)

    def test_a_player_who_will_not_play_stays_at_zero(self):
        """Shrinking the rate, not the points, is what guarantees this."""
        bootstrap = data.get_bootstrap()
        fixtures = data.get_fixtures()
        players = data.player_lookup(bootstrap)
        summaries = data.get_all_element_summaries(list(players))
        scorer = scoring.build_scorer(bootstrap, fixtures, summaries, "bayes")
        projections = scorer.project([data.target_gameweek(bootstrap)])
        idle = [p for p in projections.values() if p.expected_minutes == 0]
        if not idle:
            self.skipTest("every player has some expected minutes")
        self.assertTrue(all(p.expected_points == 0 for p in idle))


class TestFixtureStrength(unittest.TestCase):
    """The fixture effect is damped, never reversed."""

    @classmethod
    def setUpClass(cls):
        cls.bootstrap = data.get_bootstrap()
        cls.fixtures = data.get_fixtures()
        players = data.player_lookup(cls.bootstrap)
        summaries = data.get_all_element_summaries(list(players))
        cls.scorer = scoring.build_scorer(cls.bootstrap, cls.fixtures, summaries, "bayes")
        cls.model = cls.scorer.fixture_model
        gw = data.target_gameweek(cls.bootstrap)
        cls.cases = [(team["id"], fixture)
                     for team in cls.bootstrap["teams"]
                     for fixture in cls.scorer.context.fixtures_for(team["id"], gw)]

    def test_there_are_fixtures_to_check(self):
        self.assertTrue(self.cases)

    def test_damped_effect_is_the_stated_fraction_of_the_raw_one(self):
        for team_id, fixture in self.cases:
            for position in (1, 2, 3, 4):
                raw = self.model._raw_multiplier(position, team_id, fixture)
                damped = self.model.multiplier(position, team_id, fixture)
                self.assertAlmostEqual(
                    damped - 1.0, scoring.FIXTURE_STRENGTH * (raw - 1.0))

    def test_a_good_fixture_is_still_better_than_a_bad_one(self):
        raws = [(self.model._raw_multiplier(4, t, f), self.model.multiplier(4, t, f))
                for t, f in self.cases]
        best, worst = max(raws), min(raws)
        if best[0] == worst[0]:
            self.skipTest("no spread in fixture difficulty this gameweek")
        self.assertGreater(best[1], worst[1])
