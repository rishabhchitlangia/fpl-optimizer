"""Tests for the interactive squad editor.

Covers :class:`fpl.server.SquadEditor`, which holds the session state and calls
the real optimizer. The HTTP layer around it is a thin adapter and is exercised
by a live-server smoke test rather than mocked.
"""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer

from fpl import data, optimizer, scoring, server, visualize


class EditorTestCase(unittest.TestCase):
    """Shared fixture: a real squad from real projections."""

    @classmethod
    def setUpClass(cls):
        cls.bootstrap = data.get_bootstrap()
        cls.fixtures = data.get_fixtures()
        cls.players = data.player_lookup(cls.bootstrap)
        summaries = data.get_all_element_summaries(list(cls.players))
        scorer = scoring.build_scorer(cls.bootstrap, cls.fixtures, summaries, "bayes")
        cls.gameweek = data.target_gameweek(cls.bootstrap)
        cls.projections = scorer.project([cls.gameweek])
        cls.baseline = optimizer.optimize_squad(cls.bootstrap, cls.projections)

    def make_editor(self) -> server.SquadEditor:
        return server.SquadEditor(
            self.bootstrap, self.projections, self.baseline,
            visualize.PitchMeta(gameweek=self.gameweek),
        )


class TestReplacement(EditorTestCase):
    """Replacing players must change those players and nobody else."""

    def test_one_replacement_changes_exactly_one_player(self):
        editor = self.make_editor()
        target = self.baseline.captain_id
        keep = [p for p in self.baseline.squad_ids if p != target]
        result = editor.replace([target], keep)

        self.assertNotIn(target, result.squad_ids)
        self.assertEqual(set(keep) - set(result.squad_ids), set(),
                         "a player the user kept was dropped")
        self.assertEqual(len(set(self.baseline.squad_ids) - set(result.squad_ids)), 1)

    def test_replacing_several_keeps_the_rest(self):
        editor = self.make_editor()
        targets = self.baseline.squad_ids[:3]
        keep = [p for p in self.baseline.squad_ids if p not in targets]
        result = editor.replace(targets, keep)

        for target in targets:
            self.assertNotIn(target, result.squad_ids)
        for kept in keep:
            self.assertIn(kept, result.squad_ids)

    def test_result_is_still_a_legal_squad(self):
        editor = self.make_editor()
        targets = self.baseline.squad_ids[:4]
        keep = [p for p in self.baseline.squad_ids if p not in targets]
        result = editor.replace(targets, keep)

        self.assertEqual(len(result.squad_ids), 15)
        counts: dict[int, int] = {}
        clubs: dict[int, int] = {}
        for pid in result.squad_ids:
            player = self.players[pid]
            counts[player["element_type"]] = counts.get(player["element_type"], 0) + 1
            clubs[player["team"]] = clubs.get(player["team"], 0) + 1
        self.assertEqual(counts, {1: 2, 2: 5, 3: 5, 4: 3})
        self.assertLessEqual(max(clubs.values()), 3)
        self.assertLessEqual(
            sum(self.players[p]["now_cost"] for p in result.squad_ids), 1000)

    def test_replacement_never_costs_more_than_it_has_to(self):
        """Removing a player cannot improve the squad — it only constrains it."""
        editor = self.make_editor()
        target = self.baseline.captain_id
        keep = [p for p in self.baseline.squad_ids if p != target]
        result = editor.replace([target], keep)
        self.assertLessEqual(result.predicted_points,
                             self.baseline.predicted_points + 1e-6)


class TestBanAccumulation(EditorTestCase):
    """A rejected player must stay rejected until the user resets."""

    def test_bans_accumulate_across_requests(self):
        editor = self.make_editor()
        first = self.baseline.squad_ids[0]
        editor.replace([first], [p for p in self.baseline.squad_ids if p != first])

        second = editor.current.squad_ids[1]
        result = editor.replace([second],
                                [p for p in editor.current.squad_ids if p != second])
        self.assertNotIn(first, result.squad_ids,
                         "a previously rejected player was re-signed")
        self.assertNotIn(second, result.squad_ids)
        self.assertEqual(editor.banned, {first, second})

    def test_reset_clears_bans_and_restores_the_baseline(self):
        editor = self.make_editor()
        target = self.baseline.captain_id
        editor.replace([target], [p for p in self.baseline.squad_ids if p != target])
        restored = editor.reset()

        self.assertEqual(editor.banned, set())
        self.assertEqual(set(restored.squad_ids), set(self.baseline.squad_ids))
        self.assertAlmostEqual(restored.predicted_points,
                               self.baseline.predicted_points)

    def test_impossible_request_raises_rather_than_returning_nonsense(self):
        """Banning every goalkeeper in the league leaves no legal squad."""
        editor = self.make_editor()
        keepers = [p["id"] for p in self.bootstrap["elements"]
                   if p["element_type"] == 1]
        with self.assertRaises(optimizer.OptimizerError):
            editor.replace(keepers, [])


class TestRequiredPlayers(EditorTestCase):
    """Pinning a player rebuilds the best squad that contains them."""

    def _find(self, name: str) -> int:
        return next(p["id"] for p in self.bootstrap["elements"]
                    if p["web_name"] == name)

    def test_pinned_player_appears_in_the_squad(self):
        editor = self.make_editor()
        haaland = self._find("Haaland")
        result = editor.require([haaland])
        self.assertIn(haaland, result.squad_ids)
        self.assertIn(haaland, editor.required)

    def test_pinning_an_expensive_player_restructures_the_squad(self):
        """The rest must be free to change — that is what affording him means."""
        editor = self.make_editor()
        haaland = self._find("Haaland")
        result = editor.require([haaland])
        changed = set(self.baseline.squad_ids) - set(result.squad_ids)
        self.assertGreater(len(changed), 0)
        self.assertLessEqual(
            sum(self.players[p]["now_cost"] for p in result.squad_ids), 1000)

    def test_pinning_costs_points_but_stays_legal(self):
        editor = self.make_editor()
        result = editor.require([self._find("Haaland")])
        self.assertLessEqual(result.predicted_points,
                             self.baseline.predicted_points + 1e-6)
        counts: dict[int, int] = {}
        for pid in result.squad_ids:
            position = self.players[pid]["element_type"]
            counts[position] = counts.get(position, 0) + 1
        self.assertEqual(counts, {1: 2, 2: 5, 3: 5, 4: 3})

    def test_pins_accumulate(self):
        editor = self.make_editor()
        first, second = self._find("Haaland"), self._find("Raya")
        editor.require([first])
        result = editor.require([second])
        self.assertIn(first, result.squad_ids)
        self.assertIn(second, result.squad_ids)

    def test_unpinning_restores_the_free_optimum(self):
        editor = self.make_editor()
        haaland = self._find("Haaland")
        editor.require([haaland])
        result = editor.unrequire([haaland])
        self.assertEqual(editor.required, set())
        self.assertAlmostEqual(result.predicted_points,
                               self.baseline.predicted_points, places=4)

    def test_rejecting_a_pinned_player_overrides_the_pin(self):
        """The newer intent wins, rather than deadlocking."""
        editor = self.make_editor()
        haaland = self._find("Haaland")
        editor.require([haaland])
        keep = [p for p in editor.current.squad_ids if p != haaland]
        result = editor.replace([haaland], keep)
        self.assertNotIn(haaland, result.squad_ids)
        self.assertNotIn(haaland, editor.required)

    def test_pinning_a_rejected_player_overrides_the_rejection(self):
        editor = self.make_editor()
        target = self.baseline.captain_id
        editor.replace([target], [p for p in self.baseline.squad_ids if p != target])
        result = editor.require([target])
        self.assertIn(target, result.squad_ids)
        self.assertNotIn(target, editor.banned)

    def test_an_unaffordable_set_of_pins_raises(self):
        """Four premium forwards cannot coexist in one squad."""
        editor = self.make_editor()
        priciest = sorted(self.bootstrap["elements"],
                          key=lambda p: -p["now_cost"])[:9]
        with self.assertRaises(optimizer.OptimizerError):
            editor.require([p["id"] for p in priciest])

    def test_reset_clears_pins_as_well_as_rejections(self):
        editor = self.make_editor()
        editor.require([self._find("Haaland")])
        editor.reset()
        self.assertEqual(editor.required, set())
        self.assertEqual(editor.banned, set())


class TestNewsSection(EditorTestCase):
    """The team-news panel on the page."""

    def test_news_section_always_renders(self):
        markup = self.make_editor().render()
        self.assertIn("Team news", markup)

    def test_squad_news_is_reported_or_explicitly_cleared(self):
        markup = self.make_editor().render()
        has_items = "news-item" in markup
        says_clear = "No flagged players in this squad" in markup
        self.assertTrue(has_items or says_clear)

    def test_flagged_players_elsewhere_are_listed(self):
        markup = self.make_editor().render()
        self.assertIn("Flagged players elsewhere", markup)

    def test_departed_players_are_not_offered_for_pinning(self):
        """No point searching for someone who has left the league."""
        markup = self.make_editor().render()
        departed = [p for p in self.bootstrap["elements"]
                    if p.get("can_select") is False]
        if not departed:
            self.skipTest("nobody currently marked unselectable")
        index_start = markup.index('id="player-index"')
        index_block = markup[index_start:index_start + 200000]
        for player in departed[:5]:
            self.assertNotIn(f'"id": {player["id"]},', index_block)

    def test_player_index_is_present_for_search(self):
        markup = self.make_editor().render()
        self.assertIn('id="player-index"', markup)
        self.assertIn('id="pin-search"', markup)


class TestChangeList(EditorTestCase):
    """The change list is what the user reads to understand what happened."""

    def test_no_changes_before_anything_is_replaced(self):
        self.assertEqual(self.make_editor().changes(), [])

    def test_changes_are_paired_by_position(self):
        editor = self.make_editor()
        keeper = next(p for p in self.baseline.squad_ids
                      if self.players[p]["element_type"] == 1)
        forward = next(p for p in self.baseline.squad_ids
                       if self.players[p]["element_type"] == 4)
        keep = [p for p in self.baseline.squad_ids if p not in (keeper, forward)]
        editor.replace([keeper, forward], keep)

        names = {self.players[p]["web_name"] for p in self.baseline.squad_ids}
        for change in editor.changes():
            self.assertIn(change["out"], names)
            self.assertIsInstance(change["delta"], float)

        outgoing = {c["out"] for c in editor.changes()}
        self.assertIn(self.players[keeper]["web_name"], outgoing)
        self.assertIn(self.players[forward]["web_name"], outgoing)

    def test_change_count_matches_squad_difference(self):
        editor = self.make_editor()
        targets = self.baseline.squad_ids[:2]
        keep = [p for p in self.baseline.squad_ids if p not in targets]
        editor.replace(targets, keep)
        difference = set(self.baseline.squad_ids) - set(editor.current.squad_ids)
        self.assertEqual(len(editor.changes()), len(difference))


class TestRendering(EditorTestCase):
    """Markup the browser depends on."""

    def test_interactive_render_makes_players_clickable(self):
        markup = self.make_editor().render()
        # The attribute itself appears once per card; the bare selector also
        # appears in the script, so match on the attribute assignment.
        self.assertEqual(markup.count('data-player-id="'), 15)
        self.assertIn('aria-pressed="false"', markup)
        self.assertIn('id="reoptimise"', markup)

    def test_static_render_has_no_controls(self):
        markup = visualize.render_squad_html(
            self.baseline, self.bootstrap, self.projections,
            visualize.PitchMeta(gameweek=self.gameweek))
        self.assertNotIn("data-player-id", markup)
        self.assertNotIn("reoptimise", markup)

    def test_body_only_returns_a_single_wrap_element(self):
        body = self.make_editor().body_only()
        self.assertTrue(body.startswith('<div class="wrap">'))
        self.assertTrue(body.rstrip().endswith("</div>"))
        self.assertNotIn("<!doctype", body.lower())

    def test_every_player_name_survives_rendering(self):
        markup = self.make_editor().render()
        for pid in self.baseline.squad_ids:
            name = self.players[pid]["web_name"]
            # Names carrying markup-sensitive characters are escaped, so only
            # assert on the plain ones.
            if all(c not in name for c in "<>&\"'"):
                self.assertIn(name, markup, f"{name} missing from the page")


class TestHttpLayer(EditorTestCase):
    """Smoke test against a real server on an ephemeral port."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.editor = server.SquadEditor(
            cls.bootstrap, cls.projections, cls.baseline,
            visualize.PitchMeta(gameweek=cls.gameweek),
        )
        cls.httpd = HTTPServer(("127.0.0.1", 0), server._handler_factory(cls.editor))
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _post(self, payload: dict):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/optimise",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=30)

    def test_index_serves_a_full_page(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=30) as r:
            body = r.read().decode()
        self.assertEqual(r.status, 200)
        self.assertIn("<!doctype html>", body.lower())
        self.assertIn("data-player-id", body)

    def test_unknown_path_is_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/nope", timeout=30)
        self.assertEqual(ctx.exception.code, 404)

    def test_favicon_is_answered_quietly(self):
        with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/favicon.ico", timeout=30) as r:
            self.assertEqual(r.status, 204)

    def test_replacing_returns_rendered_html(self):
        target = self.baseline.captain_id
        keep = [p for p in self.baseline.squad_ids if p != target]
        with self._post({"replace": [target], "keep": keep}) as response:
            payload = json.loads(response.read())
        self.assertIn("html", payload)
        self.assertIn('<div class="wrap">', payload["html"])
        self.editor.reset()

    def test_empty_selection_is_rejected_with_a_readable_message(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post({"replace": [], "keep": []})
        self.assertEqual(ctx.exception.code, 400)
        self.assertIn("at least one", json.loads(ctx.exception.read())["error"])

    def test_malformed_body_is_rejected(self):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/optimise",
            data=b"not json",
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(request, timeout=30)
        self.assertEqual(ctx.exception.code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
