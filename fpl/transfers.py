"""Transfer planning against an existing squad.

Given the squad you already own, this module answers the only question that
matters week to week: *is a transfer worth making, and is it worth taking a hit
for?*

Method
------
Rather than scoring individual swaps in isolation, transfer plans are evaluated
by re-running the squad optimizer with a hard cap of *n* transfers, for each
*n* from zero upward. The plan for ``n = 0`` is the "roll it" baseline. Every
plan is then scored **net of point hits**, so a two-transfer plan that gains 5
points but costs a 4-point hit ranks below a one-transfer plan that gains 3.

This matters because transfers interact: selling an expensive player can fund
two upgrades elsewhere, and a pairwise swap analysis cannot see that. The
optimizer can.

Selling prices
--------------
FPL applies a 50% sell-on fee to profit (RULES.md §3), so a player who has
risen is worth less than their list price. Purchase prices are only exposed by
the authenticated ``my-team`` endpoint, which this project does not use. When
purchase prices are unknown we fall back to current price, which slightly
**overstates** available funds. Supply ``purchase_prices`` to remove the
approximation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from fpl.optimizer import (
    TRANSFER_HIT_COST,
    OptimizerError,
    SquadSelection,
    optimize_squad,
    pick_starting_xi,
)
from fpl.scoring import Projection

log = logging.getLogger(__name__)

#: Default ceiling on transfers to evaluate. Beyond about four the hits make
#: plans self-defeating outside a Wildcard.
DEFAULT_MAX_TRANSFERS = 4


# --------------------------------------------------------------------------- #
# Current squad parsing
# --------------------------------------------------------------------------- #


@dataclass
class CurrentSquad:
    """A manager's existing squad, as read from the public API.

    Attributes:
        player_ids: The 15 owned players.
        bank: Cash in hand, in tenths of a million.
        squad_value: Squad value excluding bank, in tenths of a million.
        captain_id: Currently captained player, if known.
        vice_captain_id: Currently vice-captained player, if known.
        gameweek: The gameweek this squad was read from.
        chips_used: Names of chips already played this season.
    """

    player_ids: list[int]
    bank: int = 0
    squad_value: int = 0
    captain_id: int | None = None
    vice_captain_id: int | None = None
    gameweek: int | None = None
    chips_used: list[str] = field(default_factory=list)


def parse_picks(picks_payload: dict, gameweek: int | None = None) -> CurrentSquad:
    """Convert an ``entry/{id}/event/{gw}/picks`` payload into a CurrentSquad.

    Args:
        picks_payload: The raw API response.
        gameweek: Gameweek the picks were fetched for, recorded on the result.

    Returns:
        The parsed squad.
    """
    picks = picks_payload.get("picks", [])
    history = picks_payload.get("entry_history", {}) or {}
    captain = next((p["element"] for p in picks if p.get("is_captain")), None)
    vice = next((p["element"] for p in picks if p.get("is_vice_captain")), None)

    return CurrentSquad(
        player_ids=[p["element"] for p in picks],
        bank=history.get("bank", 0),
        squad_value=history.get("value", 0),
        captain_id=captain,
        vice_captain_id=vice,
        gameweek=gameweek,
        chips_used=[c for c in (picks_payload.get("active_chip"),) if c],
    )


def selling_price(now_cost: int, purchase_price: int | None) -> int:
    """Return what a player sells for, applying FPL's 50% sell-on fee.

    Profit is halved and rounded **down** to the nearest £0.1m; losses are
    borne in full. See RULES.md §3.

    Args:
        now_cost: Current price in tenths of a million.
        purchase_price: What you paid, in tenths of a million. ``None`` means
            unknown, in which case current price is returned unchanged.

    Returns:
        Selling price in tenths of a million.

    Examples:
        >>> selling_price(74, 70)   # bought at 7.0, now 7.4 -> sells for 7.2
        72
        >>> selling_price(66, 70)   # a fall is taken in full
        66
    """
    if purchase_price is None:
        return now_cost
    profit = now_cost - purchase_price
    if profit <= 0:
        return now_cost
    return purchase_price + profit // 2


def build_selling_prices(squad: CurrentSquad, players: dict[int, dict],
                         purchase_prices: dict[int, int] | None = None) -> dict[int, int]:
    """Compute selling prices for every player in a squad."""
    purchase_prices = purchase_prices or {}
    return {
        pid: selling_price(players[pid]["now_cost"], purchase_prices.get(pid))
        for pid in squad.player_ids
        if pid in players
    }


# --------------------------------------------------------------------------- #
# Transfer plans
# --------------------------------------------------------------------------- #


@dataclass
class TransferMove:
    """A single player swap within a plan."""

    out_id: int
    in_id: int
    out_name: str
    in_name: str
    out_price: float
    in_price: float
    points_delta: float


@dataclass
class TransferPlan:
    """A candidate set of transfers, scored net of hits.

    Attributes:
        n_transfers: How many transfers the plan makes.
        moves: The individual swaps, paired by position for readability.
        gross_gain: Predicted-points improvement before hits.
        hit_cost: Points deducted for exceeding free transfers.
        net_gain: ``gross_gain - hit_cost``. This is the number to rank on.
        selection: The resulting squad.
        baseline_points: Predicted points of the unchanged squad.
    """

    n_transfers: int
    moves: list[TransferMove]
    gross_gain: float
    hit_cost: float
    net_gain: float
    selection: SquadSelection
    baseline_points: float

    @property
    def is_worthwhile(self) -> bool:
        """Whether the plan beats leaving the squad alone."""
        return self.net_gain > 1e-6


def _pair_moves(selection: SquadSelection, players: dict[int, dict],
                projections: dict[int, Projection]) -> list[TransferMove]:
    """Pair transfers out with transfers in, matching by position.

    The optimizer returns unordered in/out sets. Pairing them by position makes
    the output readable ("Gabriel -> Saliba") without changing the plan; where
    a plan swaps across positions the pairing is arbitrary but the totals hold.
    """
    def points(pid: int) -> float:
        proj = projections.get(pid)
        return proj.expected_points if proj else 0.0

    outs = sorted(selection.transfers_out,
                  key=lambda p: (players[p]["element_type"], -points(p)))
    ins = sorted(selection.transfers_in,
                 key=lambda p: (players[p]["element_type"], -points(p)))

    moves = []
    for out_id, in_id in zip(outs, ins):
        moves.append(TransferMove(
            out_id=out_id,
            in_id=in_id,
            out_name=players[out_id]["web_name"],
            in_name=players[in_id]["web_name"],
            out_price=players[out_id]["now_cost"] / 10.0,
            in_price=players[in_id]["now_cost"] / 10.0,
            points_delta=points(in_id) - points(out_id),
        ))
    return moves


def evaluate_transfer_plans(bootstrap: dict, projections: dict[int, Projection],
                            squad: CurrentSquad,
                            free_transfers: int = 1,
                            max_transfers: int = DEFAULT_MAX_TRANSFERS,
                            purchase_prices: dict[int, int] | None = None,
                            locked: Sequence[int] = (),
                            banned: Sequence[int] = ()) -> list[TransferPlan]:
    """Evaluate transfer plans of increasing size and rank them by net gain.

    Plan ``n = 0`` is always included as the "make no transfers" baseline, so
    the caller can see whether any plan actually beats rolling the transfer.

    Args:
        bootstrap: ``bootstrap-static`` payload.
        projections: Predicted points keyed by player ID.
        squad: The manager's current squad.
        free_transfers: Free transfers available this gameweek (1-5).
        max_transfers: Largest plan to consider.
        purchase_prices: Known purchase prices, for exact selling prices.
        locked: Players that must be kept.
        banned: Players that must not be bought.

    Returns:
        Plans sorted by descending net gain. Always non-empty (the baseline is
        always present); plans the solver could not build are skipped.
    """
    players = {e["id"]: e for e in bootstrap["elements"]}
    positions = {t["id"]: t for t in bootstrap["element_types"]}
    prices = build_selling_prices(squad, players, purchase_prices)

    owned = [pid for pid in squad.player_ids if pid in players]
    if len(owned) != len(squad.player_ids):
        log.warning("%d squad players are not in the current player list; "
                    "they may have left the league.",
                    len(squad.player_ids) - len(owned))

    baseline_selection = pick_starting_xi(owned, players, projections, positions)
    baseline_points = baseline_selection.predicted_points

    plans: list[TransferPlan] = [
        TransferPlan(
            n_transfers=0,
            moves=[],
            gross_gain=0.0,
            hit_cost=0.0,
            net_gain=0.0,
            selection=baseline_selection,
            baseline_points=baseline_points,
        )
    ]

    for n in range(1, max_transfers + 1):
        try:
            # Hits are deliberately NOT charged inside the objective here.
            # Letting the solver internalise them collapses the ladder: at a cap
            # of 3 it would simply return the best 1-transfer plan again, hiding
            # the comparison. Instead we solve for the best plan *at exactly this
            # many transfers* and charge the hit afterwards, so the caller can
            # see what each extra transfer actually buys.
            selection = optimize_squad(
                bootstrap,
                projections,
                current_squad=owned,
                selling_prices=prices,
                bank=squad.bank,
                free_transfers=free_transfers,
                max_transfers=n,
                apply_hits=False,
                locked=locked,
                banned=banned,
            )
        except OptimizerError as exc:
            log.info("No feasible %d-transfer plan: %s", n, exc)
            continue

        actual = len(selection.transfers_out)
        if actual == 0:
            # No change improves the squad even for free; the baseline covers it.
            continue
        hit_cost = TRANSFER_HIT_COST * max(0, actual - free_transfers)
        gross = selection.predicted_points - baseline_points
        plans.append(TransferPlan(
            n_transfers=actual,
            moves=_pair_moves(selection, players, projections),
            gross_gain=gross,
            hit_cost=hit_cost,
            net_gain=gross - hit_cost,
            selection=selection,
            baseline_points=baseline_points,
        ))

    # Deduplicate: different caps often yield the same plan.
    seen: set[tuple] = set()
    unique: list[TransferPlan] = []
    for plan in sorted(plans, key=lambda p: (-p.net_gain, p.n_transfers)):
        key = (tuple(sorted(plan.selection.transfers_out)),
               tuple(sorted(plan.selection.transfers_in)))
        if key in seen:
            continue
        seen.add(key)
        unique.append(plan)
    return unique


def best_plan(plans: Sequence[TransferPlan]) -> TransferPlan:
    """Return the highest net-gain plan, preferring fewer transfers on a tie."""
    return min(plans, key=lambda p: (-p.net_gain, p.n_transfers))


def recommend(plans: Sequence[TransferPlan], free_transfers: int) -> str:
    """Produce a one-line recommendation for the CLI.

    Deliberately conservative: a hit is only recommended when it clears a
    margin, because a single gameweek's projection is not precise enough to
    justify -4 on a coin-flip.
    """
    best = best_plan(plans)
    if best.n_transfers == 0 or not best.is_worthwhile:
        banked = min(free_transfers + 1, 5)
        return (f"Hold. No transfer improves the squad this week — "
                f"roll your free transfer (you would bank {banked}).")
    if best.hit_cost > 0:
        if best.net_gain < 2.0:
            return (f"Marginal. The best plan takes a -{best.hit_cost:.0f} hit for only "
                    f"+{best.net_gain:.2f} net. Probably not worth it — consider rolling.")
        return (f"Take the hit: {best.n_transfers} transfers for "
                f"-{best.hit_cost:.0f}, net +{best.net_gain:.2f} points.")
    return (f"Make {best.n_transfers} free transfer(s) for "
            f"+{best.net_gain:.2f} predicted points.")
