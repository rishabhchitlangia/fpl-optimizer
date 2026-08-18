"""Squad selection via mixed-integer linear programming (PuLP).

The central design decision here is that **the objective optimises the starting
XI, not the 15-man squad**. Only 11 players score in a normal gameweek, so
maximising the sum over all 15 would happily spend £30m on a strong bench. The
formulation therefore carries two binary variables per player — "in squad" and
"in starting XI" — and weights bench contribution down to
:data:`BENCH_WEIGHT`, which keeps the bench from being pure deadweight without
letting it drive selection.

Captaincy is part of the same program: a third binary marks the captain and
contributes their points a second time, so the solver trades off captaincy
against selection rather than picking it as an afterthought.

All constraints are read from :mod:`fpl.rules` values sourced from the live
game config — see RULES.md §1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import pulp

from fpl.scoring import POSITION_NAMES, Projection

log = logging.getLogger(__name__)

#: Weight applied to bench players' predicted points in the objective. Low
#: enough that the bench never outbids the XI, high enough to break ties toward
#: a bench that might actually return something (and to make Bench Boost viable).
BENCH_WEIGHT = 0.12

#: Points deducted per transfer beyond the free allocation. See RULES.md §3.
TRANSFER_HIT_COST = 4.0


class OptimizerError(RuntimeError):
    """Raised when no squad satisfies the given constraints."""


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass
class SquadSelection:
    """An optimised squad with its starting XI resolved.

    Attributes:
        squad_ids: All 15 selected player IDs.
        starting_ids: The 11 starters.
        bench_ids: The 4 bench players, in recommended auto-sub order (the
            reserve goalkeeper first, then outfielders by descending points).
        captain_id: Player to captain.
        vice_captain_id: Player to vice-captain.
        formation: Starting XI shape as ``"DEF-MID-FWD"``.
        total_cost: Squad cost in FPL tenths of a million.
        predicted_points: Expected points from the XI, captain doubled.
        transfers_out: IDs leaving the current squad, if one was supplied.
        transfers_in: IDs joining the squad, if one was supplied.
        transfer_hits: Points deducted for exceeding free transfers.
    """

    squad_ids: list[int]
    starting_ids: list[int]
    bench_ids: list[int]
    captain_id: int
    vice_captain_id: int
    formation: str
    total_cost: int
    predicted_points: float
    transfers_out: list[int] = field(default_factory=list)
    transfers_in: list[int] = field(default_factory=list)
    transfer_hits: float = 0.0

    @property
    def net_predicted_points(self) -> float:
        """Predicted points after subtracting any transfer point hits."""
        return self.predicted_points - self.transfer_hits


# --------------------------------------------------------------------------- #
# Starting XI
# --------------------------------------------------------------------------- #


def _valid_formations(positions: dict[int, dict]) -> list[tuple[int, int, int]]:
    """Enumerate legal (DEF, MID, FWD) starting combinations.

    Derived from the game's own per-position min/max play bounds rather than
    hard-coded, so it stays correct if FPL ever changes them. See RULES.md §1.
    """
    def bounds(position_id: int) -> tuple[int, int]:
        spec = positions[position_id]
        return spec["squad_min_play"], spec["squad_max_play"]

    def_min, def_max = bounds(2)
    mid_min, mid_max = bounds(3)
    fwd_min, fwd_max = bounds(4)
    gk_min, _ = bounds(1)
    outfield = 11 - gk_min

    return [
        (d, m, f)
        for d in range(def_min, def_max + 1)
        for m in range(mid_min, mid_max + 1)
        for f in range(fwd_min, fwd_max + 1)
        if d + m + f == outfield
    ]


def pick_starting_xi(squad_ids: Sequence[int], players: dict[int, dict],
                     projections: dict[int, Projection],
                     positions: dict[int, dict]) -> SquadSelection:
    """Choose the best legal starting XI, captain and bench from a fixed squad.

    Exhaustive over the legal formations (there are only eight). For a fixed
    formation the optimum is simply the highest-projected players at each
    position, so this is exact, not a heuristic.

    Captain is the highest-projected starter and vice-captain the second
    highest, which is optimal when captaincy simply doubles a score.

    Args:
        squad_ids: The 15 player IDs to choose from.
        players: Player payloads keyed by ID.
        projections: Projections keyed by player ID.
        positions: ``element_types`` keyed by ID.

    Returns:
        A :class:`SquadSelection` describing the XI, bench, captain and shape.

    Raises:
        OptimizerError: if the squad cannot field a legal XI.
    """
    def points(pid: int) -> float:
        proj = projections.get(pid)
        return proj.expected_points if proj else 0.0

    by_position: dict[int, list[int]] = {1: [], 2: [], 3: [], 4: []}
    for pid in squad_ids:
        by_position[players[pid]["element_type"]].append(pid)
    for pid_list in by_position.values():
        pid_list.sort(key=points, reverse=True)

    if not by_position[1]:
        raise OptimizerError("Squad contains no goalkeeper.")

    best: tuple[float, tuple[int, int, int], list[int]] | None = None
    for d, m, f in _valid_formations(positions):
        if len(by_position[2]) < d or len(by_position[3]) < m or len(by_position[4]) < f:
            continue
        xi = (by_position[1][:1] + by_position[2][:d]
              + by_position[3][:m] + by_position[4][:f])
        total = sum(points(pid) for pid in xi)
        if best is None or total > best[0]:
            best = (total, (d, m, f), xi)

    if best is None:
        raise OptimizerError("No legal formation available from this squad.")

    total, (d, m, f), xi = best
    ordered_xi = sorted(xi, key=points, reverse=True)
    captain = ordered_xi[0]
    vice = ordered_xi[1] if len(ordered_xi) > 1 else captain

    # Bench order: reserve goalkeeper occupies its own slot and is listed first;
    # the outfield three are ordered by projection, since that is the order in
    # which FPL will try to auto-sub them.
    bench = [pid for pid in squad_ids if pid not in set(xi)]
    bench_gk = [p for p in bench if players[p]["element_type"] == 1]
    bench_out = sorted((p for p in bench if players[p]["element_type"] != 1),
                       key=points, reverse=True)

    return SquadSelection(
        squad_ids=list(squad_ids),
        starting_ids=xi,
        bench_ids=bench_gk + bench_out,
        captain_id=captain,
        vice_captain_id=vice,
        formation=f"{d}-{m}-{f}",
        total_cost=sum(players[pid]["now_cost"] for pid in squad_ids),
        predicted_points=total + points(captain),
    )


# --------------------------------------------------------------------------- #
# Squad optimisation
# --------------------------------------------------------------------------- #


def optimize_squad(bootstrap: dict, projections: dict[int, Projection],
                   budget: int | None = None,
                   current_squad: Sequence[int] | None = None,
                   selling_prices: dict[int, int] | None = None,
                   bank: int = 0,
                   free_transfers: int = 1,
                   max_transfers: int | None = None,
                   apply_hits: bool = True,
                   locked: Iterable[int] = (),
                   banned: Iterable[int] = (),
                   min_availability: float = 0.0) -> SquadSelection:
    """Select the highest-scoring legal squad under FPL's constraints.

    Solves a single MILP over three binary variable families — squad
    membership, XI membership and captaincy — so selection, formation and
    captaincy are decided jointly rather than in sequence.

    Args:
        bootstrap: ``bootstrap-static`` payload, for players, clubs and rules.
        projections: Predicted points keyed by player ID.
        budget: Total spend limit in tenths of a million. Defaults to the
            game's own ``squad_total_spend`` (£100.0m). Ignored when
            ``current_squad`` is given, since then spend is constrained by the
            bank plus sale proceeds instead.
        current_squad: Existing squad, to optimise transfers rather than build
            from scratch.
        selling_prices: Sale value per owned player in tenths of a million.
            Defaults to current price, which overstates value for players who
            have risen — see RULES.md §3 on the 50% sell-on fee.
        bank: Cash in hand, in tenths of a million.
        free_transfers: Free transfers available this gameweek.
        max_transfers: Hard cap on transfers. ``None`` means unlimited, which
            is what a Wildcard or Free Hit gives you.
        apply_hits: Charge :data:`TRANSFER_HIT_COST` per transfer beyond the
            free allocation. Set ``False`` when modelling a Wildcard/Free Hit.
        locked: Players that must be in the squad.
        banned: Players that must not be.
        min_availability: Exclude players below this availability threshold.
            0.0 already excludes the injured and suspended, whose projection is
            zero; raise it to also avoid doubtful players.

    Returns:
        The optimal :class:`SquadSelection`.

    Raises:
        OptimizerError: if the constraints admit no feasible squad.
    """
    rules = bootstrap["game_config"]["rules"]
    positions = {t["id"]: t for t in bootstrap["element_types"]}
    players = {e["id"]: e for e in bootstrap["elements"]}

    squad_size = rules["squad_squadsize"]
    squad_play = rules["squad_squadplay"]
    club_limit = rules["squad_team_limit"]
    budget = rules["squad_total_spend"] if budget is None else budget

    locked_set, banned_set = set(locked), set(banned)
    current = set(current_squad or ())
    selling_prices = selling_prices or {}

    def points(pid: int) -> float:
        proj = projections.get(pid)
        return proj.expected_points if proj else 0.0

    # Candidate pool: drop players who cannot help, to keep the program small.
    # Anyone already owned stays in the pool so the solver can choose to keep
    # them even if they are currently flagged.
    candidates = [
        pid for pid, p in players.items()
        if pid not in banned_set
        and (pid in current or pid in locked_set
             or (projections.get(pid) and projections[pid].availability >= min_availability))
    ]
    if not candidates:
        raise OptimizerError("No candidate players available.")

    problem = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    in_squad = pulp.LpVariable.dicts("squad", candidates, cat="Binary")
    starting = pulp.LpVariable.dicts("start", candidates, cat="Binary")
    is_captain = pulp.LpVariable.dicts("captain", candidates, cat="Binary")

    # --- Objective -------------------------------------------------------- #
    # XI at full weight, bench discounted, captain counted a second time.
    objective = pulp.lpSum(
        points(pid) * starting[pid]
        + BENCH_WEIGHT * points(pid) * (in_squad[pid] - starting[pid])
        + points(pid) * is_captain[pid]
        for pid in candidates
    )

    # --- Transfer accounting ---------------------------------------------- #
    hits = None
    if current:
        kept = pulp.lpSum(in_squad[pid] for pid in candidates if pid in current)
        transfers_made = len(current) - kept
        if max_transfers is not None:
            problem += transfers_made <= max_transfers, "max_transfers"
        if apply_hits:
            # hits >= transfers - free, hits >= 0. Maximising with a negative
            # coefficient drives hits to its lower bound, so this is exact.
            hits = pulp.LpVariable("transfer_hits", lowBound=0, cat="Continuous")
            problem += hits >= transfers_made - free_transfers, "hit_floor"
            objective -= TRANSFER_HIT_COST * hits

    problem += objective

    # --- Squad composition ------------------------------------------------ #
    problem += pulp.lpSum(in_squad[pid] for pid in candidates) == squad_size, "squad_size"
    problem += pulp.lpSum(starting[pid] for pid in candidates) == squad_play, "xi_size"

    for position_id, spec in positions.items():
        pool = [pid for pid in candidates if players[pid]["element_type"] == position_id]
        problem += (pulp.lpSum(in_squad[pid] for pid in pool) == spec["squad_select"],
                    f"select_{position_id}")
        problem += (pulp.lpSum(starting[pid] for pid in pool) >= spec["squad_min_play"],
                    f"min_play_{position_id}")
        problem += (pulp.lpSum(starting[pid] for pid in pool) <= spec["squad_max_play"],
                    f"max_play_{position_id}")

    # A starter must be in the squad; a captain must be a starter.
    for pid in candidates:
        problem += starting[pid] <= in_squad[pid], f"start_implies_squad_{pid}"
        problem += is_captain[pid] <= starting[pid], f"captain_implies_start_{pid}"
    problem += pulp.lpSum(is_captain[pid] for pid in candidates) == 1, "one_captain"

    # --- Club limit ------------------------------------------------------- #
    for team_id in {p["team"] for p in players.values()}:
        pool = [pid for pid in candidates if players[pid]["team"] == team_id]
        if pool:
            problem += (pulp.lpSum(in_squad[pid] for pid in pool) <= club_limit,
                        f"club_limit_{team_id}")

    # --- Budget ----------------------------------------------------------- #
    if current:
        # Spend on incoming players is funded by the bank plus what we raise
        # selling players we own, at their selling price (not their list price).
        incoming = pulp.lpSum(players[pid]["now_cost"] * in_squad[pid]
                              for pid in candidates if pid not in current)
        raised = pulp.lpSum(selling_prices.get(pid, players[pid]["now_cost"])
                            * (1 - in_squad[pid])
                            for pid in candidates if pid in current)
        problem += incoming <= bank + raised, "budget"
    else:
        problem += (pulp.lpSum(players[pid]["now_cost"] * in_squad[pid]
                               for pid in candidates) <= budget, "budget")

    # --- Locks and bans --------------------------------------------------- #
    for pid in locked_set:
        if pid in in_squad:
            problem += in_squad[pid] == 1, f"locked_{pid}"

    # --- Solve ------------------------------------------------------------ #
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise OptimizerError(
            f"No feasible squad found (solver status: {pulp.LpStatus[status]}). "
            "Constraints may be too tight — check budget, locks and bans."
        )

    chosen = [pid for pid in candidates if in_squad[pid].value() > 0.5]

    # Re-derive the XI from the chosen squad. The MILP already picked one, but
    # routing through pick_starting_xi guarantees the bench order and formation
    # string are produced by a single code path.
    selection = pick_starting_xi(chosen, players, projections, positions)

    if current:
        selection.transfers_out = sorted(current - set(chosen))
        selection.transfers_in = sorted(set(chosen) - current)
        if apply_hits and hits is not None:
            selection.transfer_hits = TRANSFER_HIT_COST * max(0.0, hits.value() or 0.0)

    return selection


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #


def describe_selection(selection: SquadSelection, bootstrap: dict,
                       projections: dict[int, Projection]) -> list[dict]:
    """Flatten a selection into row dicts for tabular display.

    Returns one row per squad member with name, club, position, price,
    projection and role (``Starter``/``Bench``, with captain markers).
    """
    players = {e["id"]: e for e in bootstrap["elements"]}
    teams = {t["id"]: t for t in bootstrap["teams"]}
    starters = set(selection.starting_ids)

    order = selection.starting_ids + selection.bench_ids
    rows = []
    for pid in order:
        player = players[pid]
        proj = projections.get(pid)
        if pid == selection.captain_id:
            role = "Starter (C)"
        elif pid == selection.vice_captain_id:
            role = "Starter (V)"
        elif pid in starters:
            role = "Starter"
        else:
            role = "Bench"
        rows.append({
            "id": pid,
            "name": player["web_name"],
            "position": POSITION_NAMES[player["element_type"]],
            "team": teams[player["team"]]["short_name"],
            "price": player["now_cost"] / 10.0,
            "predicted": proj.expected_points if proj else 0.0,
            "role": role,
            "status": player.get("status", "a"),
            "news": player.get("news", ""),
        })
    return rows
