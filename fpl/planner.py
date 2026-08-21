"""Multi-gameweek transfer planning.

The single-gameweek optimizer answers "what is the best squad this week". That
is not how FPL is actually played. The skill is in the sequence: rolling a
transfer now to make two next week, moving a week early for a fixture swing,
accepting a worse squad today because it sets up a better one in three weeks.

This module plans a **path** — a squad for each of the next *N* gameweeks, with
the transfers that link them — by solving one mixed-integer program across the
whole horizon rather than a chain of greedy weekly decisions.

The program
-----------
For each gameweek *t* and player *p*:

``squad[p][t]``
    Is *p* in the squad in gameweek *t*?
``start[p][t]``, ``captain[p][t]``
    As in the weekly optimizer.
``transfer_in[p][t]``, ``transfer_out[p][t]``
    Movements between *t-1* and *t*, tied to the squad variables by
    ``squad[p][t] - squad[p][t-1] = transfer_in - transfer_out``.

Free transfers accumulate with the game's own rules (RULES.md §3): one earned
per gameweek, banked up to five, and everything beyond costs four points::

    free_used[t]  <= free_available[t]
    free_used[t]  <= transfers[t]
    hits[t]        = transfers[t] - free_used[t]
    free_available[t+1] <= free_available[t] - free_used[t] + 1
    free_available[t+1] <= 5

The recursion is an inequality rather than an equality, which is safe here: more
banked transfers is never worse, so maximisation drives it to the true value.

Deliberate simplifications
--------------------------
* **Prices are held fixed** across the horizon. Modelling price changes would
  make the budget constraint depend on the plan itself, and the 50% sell-on fee
  means a rise is worth half its face value anyway. Documented in RULES.md §5.
* **Chips are not planned.** Wildcards and Free Hits change the transfer rules
  entirely; planning them belongs with the blank/double calendar, which is not
  knowable this far ahead. :mod:`fpl.chips` still flags them week to week.
* **The candidate pool is pruned.** Every player at every gameweek would put
  tens of thousands of binaries in front of CBC. The pool is the current squad
  plus the best few dozen players per position over the horizon, which is
  standard practice for this problem and is reported in the result so the
  approximation is never silent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import pulp

from fpl.optimizer import BENCH_WEIGHT, TRANSFER_HIT_COST, OptimizerError
from fpl.scoring import POSITION_NAMES, Projection

log = logging.getLogger(__name__)

#: Maximum free transfers that can be banked (RULES.md §3).
MAX_FREE_TRANSFERS = 5

#: Candidate pool size per position, before adding the current squad. Tuned so
#: a five-gameweek plan solves in seconds rather than minutes.
POOL_PER_POSITION = {1: 12, 2: 40, 3: 40, 4: 25}

#: Solver time limit in seconds. A plan that is 99% of optimal now beats a
#: proven optimum after the deadline has passed.
SOLVE_TIME_LIMIT = 120


@dataclass
class GameweekPlan:
    """One gameweek within a plan.

    Attributes:
        gameweek: The gameweek number.
        squad_ids: The 15 players held.
        starting_ids: The 11 who start.
        captain_id: Who wears the armband.
        transfers_in: Players brought in for this gameweek.
        transfers_out: Players sold for this gameweek.
        hits: Points paid for exceeding free transfers.
        free_transfers_available: Free transfers held going into the gameweek.
        predicted_points: Expected points, captain doubled, before hits.
        squad_cost: Squad cost in tenths of a million.
    """

    gameweek: int
    squad_ids: list[int]
    starting_ids: list[int]
    captain_id: int
    transfers_in: list[int] = field(default_factory=list)
    transfers_out: list[int] = field(default_factory=list)
    hits: float = 0.0
    free_transfers_available: int = 1
    predicted_points: float = 0.0
    squad_cost: int = 0

    @property
    def net_points(self) -> float:
        """Predicted points after paying for transfers."""
        return self.predicted_points - self.hits


@dataclass
class TransferPath:
    """A plan across several gameweeks.

    Attributes:
        gameweeks: One :class:`GameweekPlan` per gameweek, in order.
        total_points: Sum of predicted points across the horizon.
        total_hits: Points paid for transfers across the horizon.
        pool_size: How many players the solver was allowed to consider.
        status: Solver status string.
    """

    gameweeks: list[GameweekPlan]
    total_points: float
    total_hits: float
    pool_size: int
    status: str = "Optimal"

    @property
    def net_points(self) -> float:
        """Total points after hits — the number the plan is chosen on."""
        return self.total_points - self.total_hits


def _build_pool(bootstrap: dict, projections: dict[int, Projection],
                gameweeks: Sequence[int],
                current_squad: Sequence[int],
                banned: set[int]) -> list[int]:
    """Select the candidate players the solver may use.

    Ranked by total projected points across the horizon, taking the best few per
    position, and always including the current squad so holding is possible.
    """
    players = {e["id"]: e for e in bootstrap["elements"]}

    def horizon_points(pid: int) -> float:
        projection = projections.get(pid)
        if not projection:
            return 0.0
        return sum(projection.per_gameweek.get(gw, 0.0) for gw in gameweeks)

    pool: set[int] = {pid for pid in current_squad
                      if pid in players and pid not in banned}

    for position, limit in POOL_PER_POSITION.items():
        ranked = sorted(
            (pid for pid, p in players.items()
             if p["element_type"] == position and pid not in banned
             and projections.get(pid) and projections[pid].availability > 0),
            key=horizon_points, reverse=True,
        )
        pool.update(ranked[:limit])

    return sorted(pool)


def plan_transfers(bootstrap: dict, projections: dict[int, Projection],
                   current_squad: Sequence[int],
                   gameweeks: Sequence[int],
                   free_transfers: int = 1,
                   bank: int = 0,
                   selling_prices: dict[int, int] | None = None,
                   locked: Sequence[int] = (),
                   banned: Sequence[int] = (),
                   max_hits_per_gameweek: int = 3,
                   time_limit: int = SOLVE_TIME_LIMIT) -> TransferPath:
    """Plan squads and transfers across several gameweeks at once.

    Args:
        bootstrap: ``bootstrap-static`` payload.
        projections: Predicted points keyed by player ID; must carry
            ``per_gameweek`` values covering the horizon.
        current_squad: The 15 players held now.
        gameweeks: Gameweeks to plan, in order.
        free_transfers: Free transfers available for the first gameweek.
        bank: Cash in hand, in tenths of a million.
        selling_prices: Sale value per owned player. Defaults to current price.
        locked: Players that must be held for the whole horizon.
        banned: Players that may never be selected.
        max_hits_per_gameweek: Cap on extra transfers in any one gameweek, to
            stop the solver proposing wild single-week rebuilds.
        time_limit: Solver time limit in seconds.

    Returns:
        The best :class:`TransferPath` found.

    Raises:
        OptimizerError: if no feasible plan exists, or the horizon is empty.
    """
    if not gameweeks:
        raise OptimizerError("No gameweeks to plan.")

    rules = bootstrap["game_config"]["rules"]
    positions = {t["id"]: t for t in bootstrap["element_types"]}
    players = {e["id"]: e for e in bootstrap["elements"]}

    squad_size = rules["squad_squadsize"]
    squad_play = rules["squad_squadplay"]
    club_limit = rules["squad_team_limit"]

    banned_set = set(banned)
    locked_set = set(locked)
    current = [pid for pid in current_squad if pid in players]
    selling_prices = selling_prices or {}

    pool = _build_pool(bootstrap, projections, gameweeks, current, banned_set)
    pool = sorted(set(pool) | (locked_set & set(players)))
    if len(pool) < squad_size:
        raise OptimizerError("Not enough candidate players to field a squad.")

    weeks = list(gameweeks)
    problem = pulp.LpProblem("fpl_transfer_path", pulp.LpMaximize)

    squad = pulp.LpVariable.dicts("squad", (pool, weeks), cat="Binary")
    start = pulp.LpVariable.dicts("start", (pool, weeks), cat="Binary")
    captain = pulp.LpVariable.dicts("captain", (pool, weeks), cat="Binary")
    moved_in = pulp.LpVariable.dicts("in", (pool, weeks), cat="Binary")
    moved_out = pulp.LpVariable.dicts("out", (pool, weeks), cat="Binary")

    free_available = pulp.LpVariable.dicts(
        "free", weeks, lowBound=1, upBound=MAX_FREE_TRANSFERS, cat="Integer")
    free_used = pulp.LpVariable.dicts(
        "free_used", weeks, lowBound=0, upBound=MAX_FREE_TRANSFERS, cat="Integer")
    hits = pulp.LpVariable.dicts("hits", weeks, lowBound=0, cat="Integer")

    def points(pid: int, gw: int) -> float:
        projection = projections.get(pid)
        return projection.per_gameweek.get(gw, 0.0) if projection else 0.0

    # --- Objective -------------------------------------------------------- #
    problem += (
        pulp.lpSum(
            points(pid, gw) * start[pid][gw]
            + BENCH_WEIGHT * points(pid, gw) * (squad[pid][gw] - start[pid][gw])
            + points(pid, gw) * captain[pid][gw]
            for pid in pool for gw in weeks
        )
        - TRANSFER_HIT_COST * pulp.lpSum(hits[gw] for gw in weeks)
    )

    # --- Per-gameweek squad legality -------------------------------------- #
    for gw in weeks:
        problem += pulp.lpSum(squad[pid][gw] for pid in pool) == squad_size
        problem += pulp.lpSum(start[pid][gw] for pid in pool) == squad_play
        problem += pulp.lpSum(captain[pid][gw] for pid in pool) == 1

        for position, spec in positions.items():
            in_position = [pid for pid in pool
                           if players[pid]["element_type"] == position]
            problem += (pulp.lpSum(squad[pid][gw] for pid in in_position)
                        == spec["squad_select"])
            problem += (pulp.lpSum(start[pid][gw] for pid in in_position)
                        >= spec["squad_min_play"])
            problem += (pulp.lpSum(start[pid][gw] for pid in in_position)
                        <= spec["squad_max_play"])

        for team_id in {players[pid]["team"] for pid in pool}:
            at_club = [pid for pid in pool if players[pid]["team"] == team_id]
            problem += pulp.lpSum(squad[pid][gw] for pid in at_club) <= club_limit

        for pid in pool:
            problem += start[pid][gw] <= squad[pid][gw]
            problem += captain[pid][gw] <= start[pid][gw]

        # Budget. Prices are held fixed, so the same limit applies every week.
        budget = bank + sum(
            selling_prices.get(pid, players[pid]["now_cost"]) for pid in current)
        problem += (pulp.lpSum(players[pid]["now_cost"] * squad[pid][gw]
                               for pid in pool) <= budget)

    # --- Transfer continuity ---------------------------------------------- #
    for index, gw in enumerate(weeks):
        for pid in pool:
            previous = (squad[pid][weeks[index - 1]] if index
                        else (1 if pid in current else 0))
            problem += squad[pid][gw] - previous == moved_in[pid][gw] - moved_out[pid][gw]
            # A player cannot arrive and leave in the same gameweek.
            problem += moved_in[pid][gw] + moved_out[pid][gw] <= 1

        transfers = pulp.lpSum(moved_in[pid][gw] for pid in pool)
        problem += free_used[gw] <= free_available[gw]
        problem += free_used[gw] <= transfers
        problem += hits[gw] >= transfers - free_used[gw]
        problem += transfers - free_used[gw] <= max_hits_per_gameweek

        if index == 0:
            problem += free_available[gw] == min(free_transfers, MAX_FREE_TRANSFERS)
        else:
            earlier = weeks[index - 1]
            problem += (free_available[gw]
                        <= free_available[earlier] - free_used[earlier] + 1)

    # --- Locks ------------------------------------------------------------ #
    for pid in locked_set & set(pool):
        for gw in weeks:
            problem += squad[pid][gw] == 1

    # --- Solve ------------------------------------------------------------ #
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    label = pulp.LpStatus[status]
    if label not in ("Optimal", "Not Solved"):
        raise OptimizerError(
            f"No feasible transfer path (solver status: {label}). "
            "Try a shorter horizon, or release some locks."
        )

    # --- Read the plan back ----------------------------------------------- #
    plans: list[GameweekPlan] = []
    previous_squad = set(current)
    total_points = total_hits = 0.0

    for gw in weeks:
        chosen = [pid for pid in pool if (squad[pid][gw].value() or 0) > 0.5]
        starters = [pid for pid in pool if (start[pid][gw].value() or 0) > 0.5]
        skipper = next((pid for pid in pool
                        if (captain[pid][gw].value() or 0) > 0.5), None)
        week_hits = TRANSFER_HIT_COST * round(hits[gw].value() or 0.0)
        week_points = (sum(points(pid, gw) for pid in starters)
                       + (points(skipper, gw) if skipper else 0.0))

        plans.append(GameweekPlan(
            gameweek=gw,
            squad_ids=sorted(chosen),
            starting_ids=sorted(starters),
            captain_id=skipper if skipper is not None else starters[0],
            transfers_in=sorted(set(chosen) - previous_squad),
            transfers_out=sorted(previous_squad - set(chosen)),
            hits=week_hits,
            free_transfers_available=int(round(free_available[gw].value() or 1)),
            predicted_points=week_points,
            squad_cost=sum(players[pid]["now_cost"] for pid in chosen),
        ))

        total_points += week_points
        total_hits += week_hits
        previous_squad = set(chosen)

    return TransferPath(
        gameweeks=plans,
        total_points=total_points,
        total_hits=total_hits,
        pool_size=len(pool),
        status=label,
    )


def describe_path(path: TransferPath, bootstrap: dict) -> list[dict]:
    """Flatten a path into row dicts for tabular display."""
    players = {e["id"]: e for e in bootstrap["elements"]}
    rows = []
    for plan in path.gameweeks:
        moves = " · ".join(
            f"{players[out]['web_name']} → {players[into]['web_name']}"
            for out, into in zip(
                sorted(plan.transfers_out,
                       key=lambda p: players[p]["element_type"]),
                sorted(plan.transfers_in,
                       key=lambda p: players[p]["element_type"]))
        ) or "—"
        rows.append({
            "gameweek": plan.gameweek,
            "moves": moves,
            "transfers": len(plan.transfers_in),
            "free": plan.free_transfers_available,
            "hits": plan.hits,
            "points": plan.predicted_points,
            "net": plan.net_points,
            "captain": players[plan.captain_id]["web_name"],
            "cost": plan.squad_cost / 10.0,
        })
    return rows
