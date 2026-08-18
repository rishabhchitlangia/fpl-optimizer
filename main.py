#!/usr/bin/env python3
"""Fantasy Premier League team optimizer — command-line interface.

Fetches FPL data, projects points for every player, solves for the optimal
squad, and — if you supply your team ID — ranks transfer plans against the
squad you already own and flags any chip worth considering.

Examples:
    python main.py
    python main.py --team-id 1234567
    python main.py --team-id 1234567 --horizon 5 --free-transfers 2
    python main.py --refresh --model bayes
    python main.py --lock Haaland --ban Gabriel

Run ``python main.py --help`` for the full option list.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fpl import chips as chips_module
from fpl import data, optimizer, scoring, transfers

console = Console()

#: Status codes to a short human label, for the availability column.
STATUS_LABELS = {
    "a": "",
    "d": "doubt",
    "i": "injured",
    "s": "susp",
    "u": "unavail",
    "n": "inelig",
}


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="fpl-optimizer",
        description="Suggest an optimal FPL squad, transfers and chip usage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--team-id", type=int, default=os.environ.get("FPL_TEAM_ID"),
        help="Your FPL team ID (the number in your /entry/<id>/ URL). "
             "Falls back to the FPL_TEAM_ID environment variable. Without it, "
             "transfer suggestions are skipped.",
    )
    parser.add_argument(
        "--gameweek", type=int, default=None,
        help="Gameweek to optimise for. Defaults to the next one.",
    )
    parser.add_argument(
        "--horizon", type=int, default=1,
        help="How many gameweeks to project over. Longer horizons favour "
             "players with good fixture runs.",
    )
    parser.add_argument(
        "--model", choices=["auto", "bayes", "ep_next", "blended"], default="auto",
        help="Predicted-points model. 'auto' picks by season stage.",
    )
    parser.add_argument(
        "--free-transfers", type=int, default=1,
        help="Free transfers available this gameweek (1-5).",
    )
    parser.add_argument(
        "--max-transfers", type=int, default=transfers.DEFAULT_MAX_TRANSFERS,
        help="Largest transfer plan to evaluate.",
    )
    parser.add_argument(
        "--budget", type=float, default=None,
        help="Override the squad budget in millions, e.g. 100.0.",
    )
    parser.add_argument(
        "--lock", action="append", default=[], metavar="PLAYER",
        help="Force a player into the squad. Name or ID. Repeatable.",
    )
    parser.add_argument(
        "--ban", action="append", default=[], metavar="PLAYER",
        help="Exclude a player from the squad. Name or ID. Repeatable.",
    )
    parser.add_argument(
        "--max-ownership", type=float, default=None, metavar="PCT",
        help="Only pick players owned by fewer than this percentage of "
             "managers. Builds a differential squad. Costs expected points — "
             "it discards good players purely for being popular.",
    )
    parser.add_argument(
        "--min-availability", type=float, default=0.0,
        help="Exclude players below this availability, 0.0-1.0. Use 1.0 to "
             "avoid every doubtful player.",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Force a re-fetch of all cached API data.",
    )
    parser.add_argument(
        "--no-history", action="store_true",
        help="Skip per-player history (590 requests). Falls back to the weaker "
             "ep_next model.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging.",
    )
    return parser


def resolve_player(token: str, players: dict[int, dict]) -> int:
    """Resolve a CLI name or ID token to a player ID.

    Matching is case-insensitive against ``web_name`` first, then against the
    full name, and finally as a numeric ID.

    Raises:
        SystemExit: if the token matches no player, or is ambiguous.
    """
    if token.isdigit() and int(token) in players:
        return int(token)

    needle = token.casefold()
    exact = [p for p in players.values() if p["web_name"].casefold() == needle]
    if len(exact) == 1:
        return exact[0]["id"]

    partial = [
        p for p in players.values()
        if needle in p["web_name"].casefold()
        or needle in f"{p['first_name']} {p['second_name']}".casefold()
    ]
    if len(partial) == 1:
        return partial[0]["id"]
    if not partial:
        console.print(f"[red]No player matches {token!r}.[/red]")
        raise SystemExit(2)

    names = ", ".join(sorted(p["web_name"] for p in partial)[:10])
    console.print(f"[red]{token!r} is ambiguous — matches: {names}[/red]")
    raise SystemExit(2)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_squad(selection: optimizer.SquadSelection, bootstrap: dict,
                 projections: dict[int, scoring.Projection],
                 title: str) -> Table:
    """Render a squad as a rich table, starters above bench.

    Columns are kept narrow enough to fit an 80-column terminal. Availability
    is folded into the name as a marker rather than given its own column, and
    ``SP`` flags set-piece duty (``P`` penalties, ``S`` corners/free-kicks).
    """
    table = Table(title=title, title_style="bold", header_style="bold cyan",
                  show_lines=False, padding=(0, 1))
    table.add_column("Role", style="dim", width=7)
    table.add_column("Player", width=16, no_wrap=True)
    table.add_column("Pos", width=3)
    table.add_column("Club", width=4)
    table.add_column("Price", justify="right", width=6)
    table.add_column("xPts", justify="right", width=5)
    table.add_column("Own%", justify="right", width=5)
    table.add_column("SP", justify="center", width=2)

    rows = optimizer.describe_selection(selection, bootstrap, projections)
    for index, row in enumerate(rows):
        is_bench = row["role"] == "Bench"
        if is_bench and index and rows[index - 1]["role"] != "Bench":
            table.add_section()

        if "(C)" in row["role"]:
            role_text = Text("XI (C)", style="bold green")
        elif "(V)" in row["role"]:
            role_text = Text("XI (V)", style="green")
        elif is_bench:
            role_text = Text("Bench")
        else:
            role_text = Text("XI")

        flag = STATUS_LABELS.get(row["status"], "")
        name = row["name"][:14] + (" !" if flag else "")

        table.add_row(
            role_text,
            name,
            row["position"],
            row["team"],
            f"£{row['price']:.1f}m",
            f"{row['predicted']:.2f}",
            f"{row['ownership']:.1f}",
            row["set_piece"],
            style="dim" if is_bench else None,
        )
    return table


def render_transfer_plans(plans, bootstrap: dict, free_transfers: int) -> Table:
    """Render candidate transfer plans ranked by net gain."""
    table = Table(title="Transfer plans (ranked by net points gain)",
                  title_style="bold", header_style="bold cyan")
    table.add_column("#", justify="right", width=3)
    table.add_column("Moves", width=44)
    table.add_column("Gross", justify="right", width=7)
    table.add_column("Hit", justify="right", width=6)
    table.add_column("Net", justify="right", width=7)

    for plan in plans[:6]:
        if plan.n_transfers == 0:
            moves = "[dim]No transfers (roll)[/dim]"
        else:
            moves = "\n".join(
                f"{m.out_name[:14]} → {m.in_name[:14]} "
                f"([dim]£{m.out_price:.1f}m→£{m.in_price:.1f}m[/dim])"
                for m in plan.moves
            )
        net_style = "green" if plan.net_gain > 0 else "dim"
        table.add_row(
            str(plan.n_transfers),
            moves,
            f"{plan.gross_gain:+.2f}",
            f"-{plan.hit_cost:.0f}" if plan.hit_cost else "—",
            f"[{net_style}]{plan.net_gain:+.2f}[/{net_style}]",
        )
    return table


def render_chips(flags) -> Table:
    """Render chip flags with their triggering metric."""
    table = Table(title="Chip flags", title_style="bold", header_style="bold cyan")
    table.add_column("Chip", width=15)
    table.add_column("Verdict", width=14)
    table.add_column("Reason")

    for flag in flags:
        if not flag.available:
            verdict = Text("unavailable", style="dim")
            reason = Text("Not playable this gameweek (already used, or outside "
                          "its window)", style="dim")
        elif flag.recommend:
            verdict = Text("CONSIDER", style="bold green")
            reason = Text(flag.reason)
        else:
            verdict = Text("hold", style="dim")
            reason = Text(flag.reason, style="dim")
        table.add_row(flag.label, verdict, reason)
    return table


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # --- Fetch --------------------------------------------------------- #
    try:
        with console.status("Fetching FPL data..."):
            bootstrap = data.get_bootstrap(refresh=args.refresh)
            fixtures = data.get_fixtures(refresh=args.refresh)
    except data.FPLDataError as exc:
        console.print(f"[red]Could not reach the FPL API:[/red] {exc}")
        return 1

    players = data.player_lookup(bootstrap)
    state = data.get_game_state(bootstrap)
    gameweek = args.gameweek or data.target_gameweek(bootstrap)
    horizon = list(range(gameweek, min(gameweek + args.horizon, 39)))

    summaries: dict[int, dict] = {}
    if not args.no_history:
        with console.status("Loading player history...") as status:
            def progress(done: int, total: int) -> None:
                status.update(f"Loading player history... {done}/{total}")
            try:
                summaries = data.get_all_element_summaries(
                    list(players), refresh=args.refresh, progress_callback=progress,
                )
            except data.FPLDataError as exc:
                console.print(f"[yellow]Player history unavailable ({exc}); "
                              f"falling back to ep_next.[/yellow]")

    # --- Header -------------------------------------------------------- #
    shape = chips_module.gameweek_shape(bootstrap, fixtures, gameweek)
    season_note = (f"{state.finished_gws} gameweeks played"
                   if state.season_started else "pre-season, no gameweeks played")
    header = (
        f"[bold]Gameweek {gameweek}[/bold]  ·  {shape['fixtures']} fixtures "
        f"([{'yellow' if shape['label'] != 'normal' else 'green'}]{shape['label']}"
        f"[/])  ·  horizon {len(horizon)} GW\n"
        f"[dim]{season_note}[/dim]"
    )
    console.print(Panel(header, title="FPL Optimizer", border_style="blue"))

    if shape["label"] != "normal":
        teams = data.team_lookup(bootstrap)
        if shape["blank_teams"]:
            names = ", ".join(teams[t]["short_name"] for t in shape["blank_teams"])
            console.print(f"[yellow]Blank:[/yellow] no fixture for {names}")
        if shape["double_teams"]:
            names = ", ".join(teams[t]["short_name"] for t in shape["double_teams"])
            console.print(f"[yellow]Double:[/yellow] two fixtures for {names}")

    # --- Project ------------------------------------------------------- #
    scorer = scoring.build_scorer(bootstrap, fixtures, summaries, args.model)
    projections = scorer.project(horizon)
    console.print(f"[dim]Model: {scorer.name}[/dim]\n")

    locked = [resolve_player(t, players) for t in args.lock]
    banned = [resolve_player(t, players) for t in args.ban]

    if args.max_ownership is not None:
        popular = [pid for pid, p in players.items()
                   if float(p.get("selected_by_percent") or 0) >= args.max_ownership
                   and pid not in locked]
        banned.extend(popular)
        console.print(
            f"[yellow]Differential mode:[/yellow] excluding {len(popular)} players "
            f"owned by {args.max_ownership}%+ of managers. "
            f"[dim]This deliberately gives up expected points.[/dim]\n"
        )
    budget = int(round(args.budget * 10)) if args.budget else None

    # --- Optimal squad ------------------------------------------------- #
    try:
        with console.status("Solving for the optimal squad..."):
            ideal = optimizer.optimize_squad(
                bootstrap, projections, budget=budget,
                locked=locked, banned=banned,
                min_availability=args.min_availability,
            )
    except optimizer.OptimizerError as exc:
        console.print(f"[red]Optimizer failed:[/red] {exc}")
        return 1

    console.print(render_squad(ideal, bootstrap, projections,
                               "Optimal squad (from scratch)"))
    console.print(
        f"Cost [bold]£{ideal.total_cost / 10:.1f}m[/bold]  ·  "
        f"formation [bold]{ideal.formation}[/bold]  ·  "
        f"projected [bold green]{ideal.predicted_points:.2f}[/bold green] pts "
        f"[dim](captain doubled)[/dim]\n"
    )

    # --- Transfers ----------------------------------------------------- #
    plans = None
    current = None
    if args.team_id:
        source_gw = gameweek - 1 if not state.season_started else (state.current_gw or gameweek)
        try:
            with console.status(f"Fetching your squad (team {args.team_id})..."):
                picks = data.get_entry_picks(int(args.team_id), max(source_gw, 1),
                                             refresh=args.refresh)
            current = transfers.parse_picks(picks, max(source_gw, 1))
        except data.FPLDataError as exc:
            console.print(
                f"[yellow]Could not load squad for team {args.team_id}: {exc}[/yellow]\n"
                f"[dim]Before the season starts there is no squad to read, and "
                f"unknown team IDs return 404. Transfer suggestions skipped.[/dim]\n"
            )

        if current:
            positions = data.position_lookup(bootstrap)
            owned = [p for p in current.player_ids if p in players]
            try:
                mine = optimizer.pick_starting_xi(owned, players, projections, positions)
                console.print(render_squad(mine, bootstrap, projections,
                                           f"Your squad (as at GW{current.gameweek})"))
                console.print(
                    f"Bank [bold]£{current.bank / 10:.1f}m[/bold]  ·  "
                    f"projected [bold]{mine.predicted_points:.2f}[/bold] pts\n"
                )
            except optimizer.OptimizerError as exc:
                console.print(f"[yellow]Could not read your squad: {exc}[/yellow]")

            with console.status("Evaluating transfer plans..."):
                plans = transfers.evaluate_transfer_plans(
                    bootstrap, projections, current,
                    free_transfers=args.free_transfers,
                    max_transfers=args.max_transfers,
                    locked=locked, banned=banned,
                )
            console.print(render_transfer_plans(plans, bootstrap, args.free_transfers))
            console.print(Panel(transfers.recommend(plans, args.free_transfers),
                                border_style="green", title="Recommendation"))
    else:
        console.print(
            "[dim]No team ID supplied — pass --team-id or set FPL_TEAM_ID to get "
            "transfer suggestions for your own squad.[/dim]\n"
        )

    # --- Chips --------------------------------------------------------- #
    chip_selection = ideal
    rebuild_gain = None
    suggested = 0
    if plans:
        best = transfers.best_plan(plans)
        chip_selection = best.selection
        suggested = len(ideal.squad_ids) - len(set(ideal.squad_ids) & set(current.player_ids))
        rebuild_gain = ideal.predicted_points - best.baseline_points

    flags = chips_module.evaluate_chips(
        bootstrap, fixtures, projections, chip_selection, gameweek,
        chips_used=current.chips_used if current else (),
        rebuild_gain=rebuild_gain,
        suggested_transfers=suggested,
    )
    console.print(render_chips(flags))

    console.print(
        "\n[dim]Projections are estimates, not predictions. See RULES.md for the "
        "rules this is built on and scoring.py for the model's known limits.[/dim]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
