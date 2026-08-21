"""A small local web server for editing a squad interactively.

Serves the pitch view with players clickable. Selecting players and pressing
*Replace* posts the selection back here, where the **real optimizer** re-solves
with those players banned and the rest locked in place, and the freshly
rendered squad is returned.

That round-trip is the point. A static page cannot run a mixed-integer solver,
so a purely client-side version could only ever offer precomputed swaps. Going
through the server means every suggestion is genuinely optimal under the same
constraints as the command line — one implementation of the logic, not two.

Bans accumulate for the lifetime of the session: once you have said you do not
want a player, they stay out until you press *Reset squad*. That matches what
selecting them meant, and stops the optimizer immediately re-signing someone you
just rejected.

The server binds to localhost only and is intended for local use while you plan
a gameweek. It is not hardened for exposure to a network.
"""

from __future__ import annotations

import json
import logging
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Sequence

from fpl import visualize
from fpl.optimizer import OptimizerError, SquadSelection, optimize_squad
from fpl.scoring import Projection

log = logging.getLogger(__name__)

#: Refuse absurd payloads outright rather than letting the solver chew on them.
MAX_SELECTION = 15


class SquadEditor:
    """Holds the session's squad state and re-solves on request.

    Attributes:
        banned: Players rejected so far this session. Accumulates across
            requests until :meth:`reset`.
        baseline: The squad first shown, used as the reference for the change
            list so the user always sees the swaps versus where they started.
        current: The squad currently displayed.
    """

    def __init__(self, bootstrap: dict, projections: dict[int, Projection],
                 baseline: SquadSelection, meta: visualize.PitchMeta,
                 budget: int | None = None,
                 min_availability: float = 0.0) -> None:
        self.bootstrap = bootstrap
        self.projections = projections
        self.baseline = baseline
        self.current = baseline
        self.meta = meta
        self.budget = budget
        self.min_availability = min_availability
        self.banned: set[int] = set()
        self.required: set[int] = set()
        self._lock = threading.Lock()

    def reset(self) -> SquadSelection:
        """Clear every ban and requirement and return to the original squad."""
        with self._lock:
            self.banned.clear()
            self.required.clear()
            self.current = self.baseline
            return self.current

    def require(self, player_ids: Sequence[int]) -> SquadSelection:
        """Force players into the squad and rebuild the best team around them.

        Unlike :meth:`replace`, this does **not** hold the existing squad in
        place. Forcing in a premium asset usually means the rest of the squad
        has to be restructured to afford them, so the whole squad is re-solved
        subject to the required and banned sets. That is what "the best possible
        team containing this player" means; locking the other fourteen would
        either be infeasible or produce a much worse squad.

        Args:
            player_ids: Players that must appear in the squad.

        Returns:
            The best squad containing every required player.

        Raises:
            OptimizerError: if no legal squad contains them all — most often
                because they cost too much together, or three of them share a
                club with a fourth already required.
        """
        with self._lock:
            self.required.update(player_ids)
            # A player cannot be both wanted and rejected; the newer intent wins.
            self.banned -= self.required
            selection = optimize_squad(
                self.bootstrap,
                self.projections,
                budget=self.budget,
                locked=sorted(self.required),
                banned=sorted(self.banned),
                min_availability=self.min_availability,
            )
            self.current = selection
            return selection

    def unrequire(self, player_ids: Sequence[int]) -> SquadSelection:
        """Drop a requirement and re-solve without it."""
        with self._lock:
            self.required -= set(player_ids)
            selection = optimize_squad(
                self.bootstrap,
                self.projections,
                budget=self.budget,
                locked=sorted(self.required),
                banned=sorted(self.banned),
                min_availability=self.min_availability,
            )
            self.current = selection
            return selection

    def replace(self, replace_ids: Sequence[int],
                keep_ids: Sequence[int]) -> SquadSelection:
        """Re-solve with the given players banned and the rest kept.

        Args:
            replace_ids: Players to swap out. Added to the running ban list.
            keep_ids: Players to hold in the squad.

        Returns:
            The re-optimised squad.

        Raises:
            OptimizerError: if no squad satisfies the constraints — usually
                because too many players were banned at one position, or the
                kept players leave too little budget.
        """
        with self._lock:
            self.banned.update(replace_ids)
            # Explicitly rejecting a player overrides an earlier requirement.
            self.required -= set(replace_ids)
            locked = [pid for pid in keep_ids if pid not in self.banned]
            locked = sorted(set(locked) | self.required)
            selection = optimize_squad(
                self.bootstrap,
                self.projections,
                budget=self.budget,
                locked=locked,
                banned=sorted(self.banned),
                min_availability=self.min_availability,
            )
            self.current = selection
            return selection

    def changes(self) -> list[dict]:
        """Return the swaps from the baseline squad to the current one.

        Outgoing and incoming players are paired by position so the list reads
        naturally; where a swap crosses positions the pairing is arbitrary but
        the set of changes is still complete.
        """
        players = {e["id"]: e for e in self.bootstrap["elements"]}
        before, after = set(self.baseline.squad_ids), set(self.current.squad_ids)
        outgoing = sorted(before - after,
                          key=lambda p: (players[p]["element_type"], p))
        incoming = sorted(after - before,
                          key=lambda p: (players[p]["element_type"], p))

        def points(pid: int) -> float:
            projection = self.projections.get(pid)
            return projection.expected_points if projection else 0.0

        return [
            {
                "out": players[out]["web_name"],
                "in": players[into]["web_name"],
                "delta": points(into) - points(out),
            }
            for out, into in zip(outgoing, incoming)
        ]

    def render(self) -> str:
        """Render the current squad as an interactive fragment."""
        return visualize.render_squad_html(
            self.current, self.bootstrap, self.projections, self.meta,
            standalone=False, interactive=True,
            changes=self.changes(),
            baseline_points=self.baseline.predicted_points,
            required=self.required,
        )

    def render_page(self) -> str:
        """Render the current squad as a complete HTML document."""
        return visualize.render_squad_html(
            self.current, self.bootstrap, self.projections, self.meta,
            standalone=True, interactive=True,
            changes=self.changes(),
            baseline_points=self.baseline.predicted_points,
            required=self.required,
        )

    def body_only(self) -> str:
        """Return just the ``.wrap`` element, for in-place DOM replacement."""
        page = self.render()
        start = page.index('<div class="wrap">')
        end = page.rindex("</div>") + len("</div>")
        return page[start:end]


def _handler_factory(editor: SquadEditor):
    """Build a request handler bound to one :class:`SquadEditor`."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "FPLOptimizer/1.0"

        def log_message(self, fmt, *args):       # noqa: A003 - stdlib signature
            log.debug("%s - %s", self.address_string(), fmt % args)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # This server exists to be driven by its own page; nothing else
            # should be embedding or scripting it.
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):                        # noqa: N802 - stdlib signature
            if self.path == "/favicon.ico":
                # Browsers request this unprompted; answering it cleanly keeps
                # the console free of 404s that look like a real fault.
                self.send_response(204)
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                return
            if self.path not in ("/", "/index.html"):
                self._send(404, b"Not found", "text/plain; charset=utf-8")
                return
            page = editor.render_page().encode("utf-8")
            self._send(200, page, "text/html; charset=utf-8")

        def do_POST(self):                       # noqa: N802 - stdlib signature
            if self.path != "/api/optimise":
                self._send(404, b'{"error":"Not found"}', "application/json")
                return

            try:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._send(400, b'{"error":"Malformed request."}',
                           "application/json")
                return

            try:
                action = payload.get("action", "replace")
                if payload.get("revert") or action == "reset":
                    editor.reset()
                elif action == "require":
                    require = [int(x) for x in payload.get("require", [])]
                    if not require:
                        raise ValueError("Choose a player to add.")
                    if len(require) > MAX_SELECTION:
                        raise ValueError("That is more players than a squad holds.")
                    editor.require(require)
                elif action == "unrequire":
                    editor.unrequire([int(x) for x in payload.get("unrequire", [])])
                else:
                    replace = [int(x) for x in payload.get("replace", [])]
                    keep = [int(x) for x in payload.get("keep", [])]
                    if not replace:
                        raise ValueError("Select at least one player to replace.")
                    if len(replace) > MAX_SELECTION:
                        raise ValueError("That is more players than a squad holds.")
                    editor.replace(replace, keep)
            except OptimizerError as exc:
                message = (
                    f"No legal squad meets those constraints. {exc} "
                    "Usually this means the players you have pinned cost too "
                    "much together, or too many come from one club."
                )
                self._send(400, json.dumps({"error": message}).encode(),
                           "application/json")
                return
            except (ValueError, TypeError) as exc:
                self._send(400, json.dumps({"error": str(exc)}).encode(),
                           "application/json")
                return

            body = json.dumps({"html": editor.body_only()}).encode("utf-8")
            self._send(200, body, "application/json")

    return Handler


def serve(editor: SquadEditor, port: int = 8000,
          open_browser: bool = True) -> None:
    """Run the editor server until interrupted.

    Args:
        editor: The squad session to serve.
        port: TCP port on localhost.
        open_browser: Open the page automatically once the server is up.

    Raises:
        OSError: if the port is already in use.
    """
    server = HTTPServer(("127.0.0.1", port), _handler_factory(editor))
    url = f"http://127.0.0.1:{port}/"
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
