"""Render a squad as an HTML pitch view.

The terminal tables are good for scanning numbers but bad for seeing *shape* —
whether the defence is three or five, which club you are stacked on, where the
captaincy sits. This module lays the squad out on a pitch in its actual
formation, which makes those things obvious at a glance.

Output is a single self-contained HTML file: no build step, no external assets
beyond Google Fonts, and it renders correctly in both light and dark themes.

Use :func:`render_squad_html` for a standalone file to open in a browser, or
pass ``standalone=False`` for a document fragment suitable for embedding.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

from fpl.optimizer import SquadSelection, describe_selection
from fpl.scoring import POSITION_NAMES, Projection

#: Rows are drawn goalkeeper-first, matching FPL's own pick-team view.
ROW_ORDER = [1, 2, 3, 4]

STATUS_LABELS = {
    "d": "doubt",
    "i": "injured",
    "s": "suspended",
    "u": "unavailable",
    "n": "ineligible",
}


@dataclass
class PitchMeta:
    """Context shown above the pitch.

    Attributes:
        gameweek: Gameweek being shown.
        horizon: How many gameweeks the projection covers.
        model: Name of the model that produced the projection.
        deadline: Human-readable deadline string, if known.
        gameweek_shape: ``normal``, ``blank``, ``double`` or ``blank+double``.
        title: Page heading.
    """

    gameweek: int
    horizon: int = 1
    model: str = ""
    deadline: str | None = None
    gameweek_shape: str = "normal"
    title: str = "Suggested Squad"


def _player_card(row: dict, projection: Projection | None,
                 is_captain: bool, is_vice: bool,
                 interactive: bool = False) -> str:
    """Render one player as a shirt card.

    In interactive mode the card becomes a real button so it is reachable by
    keyboard and announced correctly, rather than a div with a click handler.
    """
    classes = ["player"]
    if is_captain:
        classes.append("is-captain")

    badge = ""
    if is_captain:
        badge = '<span class="armband" title="Captain">C</span>'
    elif is_vice:
        badge = '<span class="armband armband--vice" title="Vice-captain">V</span>'

    flag = STATUS_LABELS.get(row["status"], "")
    flag_html = f'<span class="flag">{html.escape(flag)}</span>' if flag else ""

    extras = []
    if projection is not None and projection.clean_sheet_probability is not None:
        extras.append(f'<span class="stat"><span class="stat-key">CS</span>'
                      f'{projection.clean_sheet_probability * 100:.0f}%</span>')
    if projection is not None and projection.expected_goal_involvement is not None:
        extras.append(f'<span class="stat"><span class="stat-key">xGI</span>'
                      f'{projection.expected_goal_involvement:.2f}</span>')
    if not extras:
        if row["set_piece"] != "-":
            marker = "pens" if "P" in row["set_piece"] else "set pieces"
            extras.append(f'<span class="stat stat--role">{marker}</span>')
        extras.append(f'<span class="stat"><span class="stat-key">own</span>'
                      f'{row["ownership"]:.0f}%</span>')

    inner = f"""
          {badge}
          <div class="shirt" aria-hidden="true"></div>
          <div class="name">{html.escape(row['name'])}</div>
          <div class="meta">
            <span class="club">{html.escape(row['team'])}</span>
            <span class="price">£{row['price']:.1f}</span>
          </div>
          <div class="points">{row['predicted']:.2f}</div>
          <div class="stats">{''.join(extras)}{flag_html}</div>"""

    if not interactive:
        return f'<div class="{" ".join(classes)}">{inner}</div>'

    classes.append("player--selectable")
    return (f'<button type="button" class="{" ".join(classes)}" '
            f'data-player-id="{row["id"]}" aria-pressed="false" '
            f'title="Click to swap {html.escape(row["name"])} out">'
            f'<span class="tick" aria-hidden="true"></span>{inner}</button>')


def _bench_card(row: dict, interactive: bool = False) -> str:
    """Render one bench player, deliberately quieter than a starter."""
    flag = STATUS_LABELS.get(row["status"], "")
    flag_html = f'<span class="flag">{html.escape(flag)}</span>' if flag else ""
    if interactive:
        return f"""
        <button type="button" class="bench-player bench-player--selectable"
                data-player-id="{row['id']}" aria-pressed="false"
                title="Click to swap {html.escape(row['name'])} out">
          <span class="bench-pos">{html.escape(row['position'])}</span>
          <span class="bench-name">{html.escape(row['name'])}</span>
          <span class="bench-club">{html.escape(row['team'])}</span>
          <span class="bench-price">£{row['price']:.1f}</span>
          <span class="bench-points">{row['predicted']:.2f}</span>
          {flag_html}
        </button>"""
    return f"""
        <div class="bench-player">
          <span class="bench-pos">{html.escape(row['position'])}</span>
          <span class="bench-name">{html.escape(row['name'])}</span>
          <span class="bench-club">{html.escape(row['team'])}</span>
          <span class="bench-price">£{row['price']:.1f}</span>
          <span class="bench-points">{row['predicted']:.2f}</span>
          {flag_html}
        </div>"""


def _styles() -> str:
    """Return the page stylesheet.

    Light is defined on bare ``:root``; dark is redefined both under
    ``prefers-color-scheme`` (guarded so an explicit light choice wins) and
    under ``[data-theme="dark"]`` (so an explicit dark choice wins). The pitch
    itself keeps its colours in both themes — a pitch is green either way.
    """
    return """
    :root {
      --bg: #EEF1EC;
      --surface: #FFFFFF;
      --ink: #10160F;
      --muted: #5B6B5E;
      --line: #D3DBD2;
      --pitch-deep: #0F4A2A;
      --pitch: #15653A;
      --chalk: rgba(255, 255, 255, 0.30);
      --on-pitch: #F4FAF4;
      --on-pitch-muted: #A9CDB4;
      --captain: #F0B429;
      --shirt: rgba(255, 255, 255, 0.13);
      --shirt-line: rgba(255, 255, 255, 0.30);
    }
    @media (prefers-color-scheme: dark) {
      :root:not([data-theme="light"]) {
        --bg: #0B0F0C;
        --surface: #141A15;
        --ink: #ECF1EA;
        --muted: #8B9C8E;
        --line: #232C24;
      }
    }
    :root[data-theme="dark"] {
      --bg: #0B0F0C;
      --surface: #141A15;
      --ink: #ECF1EA;
      --muted: #8B9C8E;
      --line: #232C24;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Barlow", system-ui, -apple-system, sans-serif;
      font-size: 16px;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }

    .wrap {
      max-width: 940px;
      margin: 0 auto;
      padding: 32px 20px 56px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    header { display: flex; flex-direction: column; gap: 6px; }

    .eyebrow {
      font-family: "Barlow", sans-serif;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .tag {
      padding: 2px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      letter-spacing: 0.08em;
    }
    .tag--alert { border-color: var(--captain); color: var(--captain); }

    h1 {
      font-family: "Barlow Condensed", "Barlow", sans-serif;
      font-size: clamp(34px, 6vw, 52px);
      font-weight: 700;
      letter-spacing: -0.01em;
      line-height: 1;
      margin: 0;
      text-wrap: balance;
    }

    .summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 1px;
      background: var(--line);
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
    }
    .tile { background: var(--surface); padding: 14px 16px; }
    .tile-key {
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .tile-value {
      font-family: "IBM Plex Mono", ui-monospace, monospace;
      font-size: 24px;
      font-weight: 500;
      font-variant-numeric: tabular-nums;
      margin-top: 2px;
    }

    .pitch {
      position: relative;
      background:
        repeating-linear-gradient(
          to bottom,
          rgba(255,255,255,0.022) 0 44px,
          rgba(0,0,0,0.022) 44px 88px
        ),
        linear-gradient(170deg, var(--pitch) 0%, var(--pitch-deep) 100%);
      border-radius: 12px;
      padding: 26px 16px 30px;
      display: flex;
      flex-direction: column;
      gap: 18px;
      overflow: hidden;
    }
    /* Centre circle and halfway line, drawn rather than imported. */
    .pitch::before {
      content: "";
      position: absolute;
      left: 50%;
      top: 50%;
      width: 168px;
      height: 168px;
      border: 1px solid var(--chalk);
      border-radius: 50%;
      transform: translate(-50%, -50%);
      pointer-events: none;
    }
    .pitch::after {
      content: "";
      position: absolute;
      left: 0;
      right: 0;
      top: 50%;
      border-top: 1px solid var(--chalk);
      pointer-events: none;
    }

    .row {
      position: relative;
      z-index: 1;
      display: flex;
      justify-content: center;
      gap: clamp(6px, 2vw, 22px);
      flex-wrap: wrap;
    }

    .player {
      position: relative;
      width: clamp(74px, 15vw, 104px);
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 3px;
      text-align: center;
      color: var(--on-pitch);
    }

    .shirt {
      width: 34px;
      height: 30px;
      background: var(--shirt);
      border: 1px solid var(--shirt-line);
      border-radius: 5px 5px 3px 3px;
      position: relative;
    }
    .shirt::before,
    .shirt::after {
      content: "";
      position: absolute;
      top: 0;
      width: 8px;
      height: 11px;
      background: inherit;
      border: inherit;
      border-radius: 3px;
    }
    .shirt::before { left: -8px; }
    .shirt::after { right: -8px; }

    .is-captain .shirt {
      border-color: var(--captain);
      background: rgba(240, 180, 41, 0.18);
    }

    .armband {
      position: absolute;
      top: -6px;
      right: 12px;
      z-index: 2;
      width: 19px;
      height: 19px;
      border-radius: 50%;
      background: var(--captain);
      color: #1A1200;
      font-family: "IBM Plex Mono", monospace;
      font-size: 11px;
      font-weight: 600;
      line-height: 19px;
      text-align: center;
    }
    .armband--vice {
      background: transparent;
      color: var(--captain);
      border: 1px solid var(--captain);
      line-height: 17px;
    }

    .name {
      font-family: "Barlow Condensed", "Barlow", sans-serif;
      font-size: 16px;
      font-weight: 600;
      line-height: 1.15;
      letter-spacing: 0.005em;
      margin-top: 3px;
    }
    .meta {
      display: flex;
      gap: 6px;
      font-size: 11px;
      color: var(--on-pitch-muted);
      font-variant-numeric: tabular-nums;
    }
    .club { text-transform: uppercase; letter-spacing: 0.06em; }
    .price { font-family: "IBM Plex Mono", monospace; }

    .points {
      font-family: "IBM Plex Mono", monospace;
      font-size: 15px;
      font-weight: 500;
      font-variant-numeric: tabular-nums;
    }

    .stats {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 1px;
      font-size: 10px;
      color: var(--on-pitch-muted);
      font-variant-numeric: tabular-nums;
    }
    .stat-key {
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-right: 3px;
      opacity: 0.75;
    }
    .stat--role { text-transform: uppercase; letter-spacing: 0.08em; }
    .flag {
      color: var(--captain);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .bench {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 6px 0;
    }
    .bench-head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      padding: 8px 16px 10px;
      border-bottom: 1px solid var(--line);
    }
    .bench-head span {
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .bench-player {
      display: grid;
      grid-template-columns: 44px 1fr auto 62px 62px;
      align-items: center;
      gap: 10px;
      padding: 9px 16px;
      border-bottom: 1px solid var(--line);
      font-variant-numeric: tabular-nums;
    }
    .bench-player:last-child { border-bottom: none; }
    .bench-pos {
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.08em;
      color: var(--muted);
    }
    .bench-name {
      font-family: "Barlow Condensed", "Barlow", sans-serif;
      font-size: 18px;
      font-weight: 600;
    }
    .bench-club {
      font-size: 11px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .bench-price,
    .bench-points {
      font-family: "IBM Plex Mono", monospace;
      font-size: 13px;
      text-align: right;
    }
    .bench-price { color: var(--muted); }

    footer {
      color: var(--muted);
      font-size: 13px;
      border-top: 1px solid var(--line);
      padding-top: 14px;
    }
    footer code {
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
    }

    @media (max-width: 560px) {
      .pitch { padding: 18px 6px 22px; }
      .row { gap: 4px; }
      .bench-player { grid-template-columns: 36px 1fr auto 54px; }
      .bench-club { display: none; }
    }

    /* --- Interactive mode ------------------------------------------------ */

    .controls {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px 16px;
    }
    .controls-text { flex: 1 1 220px; font-size: 14px; color: var(--muted); }
    .controls-text strong { color: var(--ink); font-weight: 600; }

    button.btn {
      font-family: "Barlow", sans-serif;
      font-size: 14px;
      font-weight: 600;
      letter-spacing: 0.02em;
      padding: 9px 16px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: var(--surface);
      color: var(--ink);
      cursor: pointer;
    }
    button.btn:hover:not(:disabled) { border-color: var(--muted); }
    button.btn:disabled { opacity: 0.45; cursor: not-allowed; }
    button.btn--primary {
      background: var(--pitch);
      border-color: var(--pitch);
      color: #F4FAF4;
    }
    button.btn--primary:hover:not(:disabled) { background: var(--pitch-deep); }

    :focus-visible { outline: 2px solid var(--captain); outline-offset: 2px; }

    button.player--selectable,
    button.bench-player--selectable {
      font: inherit;
      background: none;
      border: none;
      cursor: pointer;
      padding: 4px 2px;
      border-radius: 8px;
      text-align: inherit;
      color: inherit;
    }
    button.player--selectable { text-align: center; }
    button.bench-player--selectable {
      width: 100%;
      border-bottom: 1px solid var(--line);
      border-radius: 0;
      padding: 9px 16px;
    }
    button.player--selectable:hover { background: rgba(255, 255, 255, 0.07); }
    button.bench-player--selectable:hover { background: var(--bg); }

    .tick {
      position: absolute;
      top: -6px;
      left: 10px;
      z-index: 2;
      width: 17px;
      height: 17px;
      border-radius: 50%;
      border: 1px solid var(--on-pitch-muted);
      background: rgba(0, 0, 0, 0.25);
      opacity: 0;
    }
    button.player--selectable:hover .tick { opacity: 0.6; }

    [aria-pressed="true"].player--selectable {
      background: rgba(240, 180, 41, 0.16);
      box-shadow: inset 0 0 0 1px var(--captain);
    }
    [aria-pressed="true"] .tick {
      opacity: 1;
      background: var(--captain);
      border-color: var(--captain);
    }
    [aria-pressed="true"] .tick::after {
      content: "";
      position: absolute;
      left: 5px;
      top: 2px;
      width: 4px;
      height: 8px;
      border: solid #1A1200;
      border-width: 0 2px 2px 0;
      transform: rotate(45deg);
    }
    [aria-pressed="true"].bench-player--selectable {
      background: rgba(240, 180, 41, 0.12);
      box-shadow: inset 2px 0 0 var(--captain);
    }

    .changes {
      background: var(--surface);
      border: 1px solid var(--line);
      border-left: 3px solid var(--captain);
      border-radius: 10px;
      padding: 12px 16px;
      font-size: 14px;
      display: flex;
      flex-direction: column;
      gap: 5px;
    }
    .changes h2 {
      font-family: "Barlow", sans-serif;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
      margin: 0 0 2px;
    }
    .change-row { font-variant-numeric: tabular-nums; }
    .change-out { color: var(--muted); text-decoration: line-through; }
    .delta-up { color: var(--pitch); font-weight: 600; }
    :root[data-theme="dark"] .delta-up,
    :root:not([data-theme="light"]) .delta-up { color: #6FCF97; }
    @media (prefers-color-scheme: light) {
      :root:not([data-theme="dark"]) .delta-up { color: var(--pitch); }
    }
    .delta-down { color: #C2410C; font-weight: 600; }

    .is-busy { opacity: 0.5; pointer-events: none; }

    .error {
      background: var(--surface);
      border: 1px solid #C2410C;
      border-radius: 10px;
      padding: 12px 16px;
      font-size: 14px;
      color: #C2410C;
    }

    @media (prefers-reduced-motion: reduce) {
      * { transition: none !important; animation: none !important; }
    }
    """


def _script() -> str:
    """Client-side behaviour for the interactive pitch.

    Deliberately small and dependency-free. Selection state lives in the DOM
    (``aria-pressed``) rather than a parallel JavaScript structure, so the
    accessible state and the visual state cannot drift apart.

    The page never computes a squad itself — it posts the selected players to
    the local server, which runs the real optimizer and returns fully rendered
    markup. That keeps exactly one implementation of the optimisation logic.
    """
    return """
    (function () {
      var wrap = document.querySelector('.wrap');

      function selected() {
        return Array.from(
          document.querySelectorAll('[data-player-id][aria-pressed="true"]')
        ).map(function (el) { return Number(el.dataset.playerId); });
      }

      function refreshControls() {
        var count = selected().length;
        var button = document.getElementById('reoptimise');
        var reset = document.getElementById('reset');
        var text = document.getElementById('controls-text');
        if (!button) { return; }
        button.disabled = count === 0;
        if (reset) { reset.disabled = count === 0; }
        button.textContent = count > 1
          ? 'Replace ' + count + ' players'
          : 'Replace player';
        if (text) {
          text.innerHTML = count === 0
            ? 'Click any player to swap them out.'
            : '<strong>' + count + '</strong> selected \u2014 everyone else stays.';
        }
      }

      function toggle(el) {
        el.setAttribute('aria-pressed',
          el.getAttribute('aria-pressed') === 'true' ? 'false' : 'true');
        refreshControls();
      }

      function bind() {
        document.querySelectorAll('[data-player-id]').forEach(function (el) {
          el.addEventListener('click', function () { toggle(el); });
        });

        var button = document.getElementById('reoptimise');
        if (button) { button.addEventListener('click', reoptimise); }

        var reset = document.getElementById('reset');
        if (reset) {
          reset.addEventListener('click', function () {
            document.querySelectorAll('[data-player-id]').forEach(function (el) {
              el.setAttribute('aria-pressed', 'false');
            });
            refreshControls();
          });
        }

        var revert = document.getElementById('revert');
        if (revert) { revert.addEventListener('click', function () { post([], true); }); }

        refreshControls();
      }

      function reoptimise() { post(selected(), false); }

      function post(replace, revert) {
        var keep = Array.from(document.querySelectorAll('[data-player-id]'))
          .map(function (el) { return Number(el.dataset.playerId); })
          .filter(function (id) { return replace.indexOf(id) === -1; });

        wrap.classList.add('is-busy');
        fetch('/api/optimise', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ replace: replace, keep: keep, revert: !!revert })
        })
          .then(function (response) {
            if (!response.ok) { return response.json().then(function (e) { throw e; }); }
            return response.json();
          })
          .then(function (data) {
            wrap.outerHTML = data.html;
            wrap = document.querySelector('.wrap');
            bind();
          })
          .catch(function (error) {
            wrap.classList.remove('is-busy');
            var box = document.getElementById('error');
            if (box) {
              box.textContent = error && error.error
                ? error.error
                : 'Could not reach the optimizer. Is the server still running?';
              box.hidden = false;
            }
          });
      }

      bind();
    })();
    """


def render_squad_html(selection: SquadSelection, bootstrap: dict,
                      projections: dict[int, Projection],
                      meta: PitchMeta, standalone: bool = True,
                      interactive: bool = False,
                      changes: list[dict] | None = None,
                      baseline_points: float | None = None) -> str:
    """Render a squad selection as an HTML pitch view.

    Args:
        selection: The squad to draw.
        bootstrap: ``bootstrap-static`` payload.
        projections: Predicted points keyed by player ID.
        meta: Header context.
        standalone: Emit a complete HTML document. Set ``False`` for a fragment.
        interactive: Make players clickable and add the replace controls. Only
            meaningful when served by :mod:`fpl.server`, which supplies the
            endpoint the controls post to.
        changes: Swaps applied to reach this squad, each with ``out``, ``in``
            and ``delta`` keys, rendered as a change list.
        baseline_points: Projected points before those swaps, for the delta.

    Returns:
        HTML source.
    """
    rows = describe_selection(selection, bootstrap, projections)
    by_id = {row["id"]: row for row in rows}
    starters = selection.starting_ids

    pitch_rows = []
    players = {e["id"]: e for e in bootstrap["elements"]}
    for position in ROW_ORDER:
        in_row = [pid for pid in starters
                  if players[pid]["element_type"] == position]
        if not in_row:
            continue
        cards = "".join(
            _player_card(by_id[pid], projections.get(pid),
                         pid == selection.captain_id,
                         pid == selection.vice_captain_id,
                         interactive)
            for pid in in_row
        )
        pitch_rows.append(f'<div class="row">{cards}</div>')

    bench_cards = "".join(_bench_card(by_id[pid], interactive)
                          for pid in selection.bench_ids)
    bench_total = sum(by_id[pid]["predicted"] for pid in selection.bench_ids)

    horizon_label = (f"{meta.horizon} gameweeks" if meta.horizon > 1
                     else "1 gameweek")
    shape_tag = ""
    if meta.gameweek_shape != "normal":
        shape_tag = f'<span class="tag tag--alert">{html.escape(meta.gameweek_shape)}</span>'
    deadline_tag = (f'<span class="tag">{html.escape(meta.deadline)}</span>'
                    if meta.deadline else "")

    controls_html = ""
    if interactive:
        controls_html = """
      <section class="controls">
        <div class="controls-text" id="controls-text">
          Click any player to swap them out.
        </div>
        <button type="button" class="btn" id="reset" disabled>Clear</button>
        <button type="button" class="btn" id="revert">Reset squad</button>
        <button type="button" class="btn btn--primary" id="reoptimise" disabled>
          Replace player
        </button>
      </section>
      <div class="error" id="error" hidden></div>"""

    changes_html = ""
    if changes:
        rows = []
        for change in changes:
            delta = change.get("delta", 0.0)
            css = "delta-up" if delta >= 0 else "delta-down"
            rows.append(
                f'<div class="change-row">'
                f'<span class="change-out">{html.escape(change["out"])}</span>'
                f' &rarr; <strong>{html.escape(change["in"])}</strong> '
                f'<span class="{css}">{delta:+.2f}</span></div>'
            )
        total = ""
        if baseline_points is not None:
            difference = selection.predicted_points - baseline_points
            css = "delta-up" if difference >= 0 else "delta-down"
            total = (f'<div class="change-row">Squad total '
                     f'<span class="{css}">{difference:+.2f}</span> '
                     f'against the original.</div>')
        changes_html = (f'<section class="changes"><h2>Changes applied</h2>'
                        f'{"".join(rows)}{total}</section>')

    script_html = f"<script>{_script()}</script>" if interactive else ""

    body = f"""
    <div class="wrap">
      <header>
        <div class="eyebrow">
          <span>Gameweek {meta.gameweek}</span>
          <span class="tag">{html.escape(horizon_label)}</span>
          {shape_tag}
          {deadline_tag}
        </div>
        <h1>{html.escape(meta.title)}</h1>
      </header>

      {controls_html}
      {changes_html}

      <section class="summary">
        <div class="tile">
          <div class="tile-key">Formation</div>
          <div class="tile-value">{html.escape(selection.formation)}</div>
        </div>
        <div class="tile">
          <div class="tile-key">Squad cost</div>
          <div class="tile-value">£{selection.total_cost / 10:.1f}m</div>
        </div>
        <div class="tile">
          <div class="tile-key">Projected</div>
          <div class="tile-value">{selection.predicted_points:.1f}</div>
        </div>
        <div class="tile">
          <div class="tile-key">Bench</div>
          <div class="tile-value">{bench_total:.1f}</div>
        </div>
      </section>

      <section class="pitch">
        {''.join(pitch_rows)}
      </section>

      <section class="bench">
        <div class="bench-head">
          <span>Bench — in substitution order</span>
          <span>xPts</span>
        </div>
        {bench_cards}
      </section>

      <footer>
        Projected points include the captain doubled. Figures are estimates from
        the <code>{html.escape(meta.model or 'projection')}</code> model, not
        predictions — check team news before the deadline.
      </footer>
    </div>
    {script_html}"""

    head = f"""<title>{html.escape(meta.title)}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Barlow:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
    <style>{_styles()}</style>"""

    if not standalone:
        return head + body

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{head}
</head>
<body>
{body}
</body>
</html>
"""
