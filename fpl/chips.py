"""Heuristic chip flags.

These are **flags, not a strategy engine.** Each chip is checked against a small
number of transparent conditions and reports whether this gameweek looks like a
candidate, along with the number that triggered it. Chip timing over a full
season is a planning problem this module deliberately does not attempt — it
answers "is there something here worth a look?" and leaves the call to you.

Chip availability is read from the game's own ``chips`` array rather than
assumed, so the gameweek windows and the two-sets-of-four structure stay correct
without hard-coding. See RULES.md §4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from fpl.optimizer import SquadSelection
from fpl.scoring import FixtureContext, Projection

#: Bench projection (all four players, one gameweek) above which Bench Boost is
#: worth considering. A typical bench returns 8-12; 18+ means the bench is
#: unusually strong or is enjoying a double gameweek.
BENCH_BOOST_THRESHOLD = 18.0

#: Captain projection above which Triple Captain is flagged. The chip is worth
#: exactly one extra copy of the captain's score, so this is the bar for that
#: extra copy being worth burning a chip on.
TRIPLE_CAPTAIN_THRESHOLD = 9.0

#: Squad members without a fixture that makes Free Hit worth considering.
FREE_HIT_BLANK_THRESHOLD = 4

#: Transfers the optimizer wants before a Wildcard looks more sensible than
#: paying hits.
WILDCARD_TRANSFER_THRESHOLD = 5

#: Net predicted gain a full rebuild must clear before a Wildcard is flagged.
WILDCARD_GAIN_THRESHOLD = 12.0

#: Human-readable names for the API's chip identifiers.
CHIP_LABELS = {
    "bboost": "Bench Boost",
    "3xc": "Triple Captain",
    "freehit": "Free Hit",
    "wildcard": "Wildcard",
}


@dataclass
class ChipFlag:
    """A recommendation (or non-recommendation) for one chip.

    Attributes:
        chip: API chip identifier, e.g. ``"bboost"``.
        label: Human-readable name.
        available: Whether the chip can be played this gameweek at all.
        recommend: Whether the heuristic thinks it is worth considering.
        metric: The number the decision was based on.
        reason: Plain-English explanation, shown in the CLI.
    """

    chip: str
    label: str
    available: bool
    recommend: bool
    metric: float
    reason: str


def available_chips(bootstrap: dict, gameweek: int,
                    chips_used: Sequence[str] = ()) -> set[str]:
    """Return chip identifiers playable in a given gameweek.

    Reads the ``chips`` array's ``start_event``/``stop_event`` windows, so the
    two-sets-of-four structure and the Gameweek 1 lockout on Wildcard and Free
    Hit are respected automatically.

    Args:
        bootstrap: ``bootstrap-static`` payload.
        gameweek: Gameweek to test.
        chips_used: Chips already played, which are excluded.

    Returns:
        Set of playable chip names.
    """
    used = set(chips_used)
    playable = set()
    for chip in bootstrap.get("chips", []):
        name = chip["name"]
        if name in used:
            continue
        start = chip.get("start_event")
        stop = chip.get("stop_event")
        if start is not None and gameweek < start:
            continue
        if stop is not None and gameweek > stop:
            continue
        playable.add(name)
    return playable


def evaluate_chips(bootstrap: dict, fixtures: Sequence[dict],
                   projections: dict[int, Projection],
                   selection: SquadSelection,
                   gameweek: int,
                   chips_used: Sequence[str] = (),
                   rebuild_gain: float | None = None,
                   suggested_transfers: int = 0) -> list[ChipFlag]:
    """Flag chips worth considering for a gameweek.

    Args:
        bootstrap: ``bootstrap-static`` payload.
        fixtures: Fixture list, for blank/double detection.
        projections: Predicted points keyed by player ID.
        selection: The squad under consideration (yours, or the optimal one).
        gameweek: Gameweek being planned.
        chips_used: Chips already played this season.
        rebuild_gain: Predicted gain from an unconstrained rebuild, used for the
            Wildcard check. ``None`` skips that part of the test.
        suggested_transfers: How many transfers the optimizer wants to make.

    Returns:
        One :class:`ChipFlag` per chip, including unavailable ones so the CLI
        can explain why a chip is not an option.
    """
    context = FixtureContext(fixtures)
    players = {e["id"]: e for e in bootstrap["elements"]}
    playable = available_chips(bootstrap, gameweek, chips_used)

    def points(pid: int) -> float:
        proj = projections.get(pid)
        return proj.expected_points if proj else 0.0

    squad_teams = {pid: players[pid]["team"] for pid in selection.squad_ids
                   if pid in players}
    blanking = [pid for pid, team in squad_teams.items()
                if context.fixture_count(team, gameweek) == 0]
    doubling = [pid for pid, team in squad_teams.items()
                if context.fixture_count(team, gameweek) >= 2]

    flags: list[ChipFlag] = []

    # --- Bench Boost ------------------------------------------------------ #
    bench_points = sum(points(pid) for pid in selection.bench_ids)
    bench_doubles = sum(1 for pid in selection.bench_ids if pid in doubling)
    bb_reason = f"Bench projects {bench_points:.1f} pts"
    if bench_doubles:
        bb_reason += f", {bench_doubles} bench player(s) have a double gameweek"
    bb_recommend = bench_points >= BENCH_BOOST_THRESHOLD
    if not bb_recommend:
        bb_reason += f" — below the {BENCH_BOOST_THRESHOLD:.0f} pt bar for the chip"
    flags.append(ChipFlag("bboost", CHIP_LABELS["bboost"], "bboost" in playable,
                          bb_recommend and "bboost" in playable,
                          bench_points, bb_reason))

    # --- Triple Captain --------------------------------------------------- #
    captain_points = points(selection.captain_id)
    captain_name = players.get(selection.captain_id, {}).get("web_name", "captain")
    captain_doubles = selection.captain_id in doubling
    tc_reason = f"{captain_name} projects {captain_points:.1f} pts"
    if captain_doubles:
        tc_reason += " with a double gameweek"
    tc_recommend = captain_points >= TRIPLE_CAPTAIN_THRESHOLD
    if not tc_recommend:
        tc_reason += (f" — the chip adds ~{captain_points:.1f} pts, "
                      f"below the {TRIPLE_CAPTAIN_THRESHOLD:.0f} pt bar")
    flags.append(ChipFlag("3xc", CHIP_LABELS["3xc"], "3xc" in playable,
                          tc_recommend and "3xc" in playable,
                          captain_points, tc_reason))

    # --- Free Hit --------------------------------------------------------- #
    fh_recommend = len(blanking) >= FREE_HIT_BLANK_THRESHOLD
    if blanking:
        fh_reason = f"{len(blanking)} squad player(s) have no fixture this gameweek"
        if not fh_recommend:
            fh_reason += f" — under the {FREE_HIT_BLANK_THRESHOLD} player bar"
    else:
        fh_reason = "Every squad player has a fixture; nothing to rescue"
    flags.append(ChipFlag("freehit", CHIP_LABELS["freehit"], "freehit" in playable,
                          fh_recommend and "freehit" in playable,
                          float(len(blanking)), fh_reason))

    # --- Wildcard --------------------------------------------------------- #
    wc_metric = rebuild_gain if rebuild_gain is not None else 0.0
    wc_recommend = (
        suggested_transfers >= WILDCARD_TRANSFER_THRESHOLD
        and wc_metric >= WILDCARD_GAIN_THRESHOLD
    )
    if rebuild_gain is None:
        wc_reason = "Not assessed (no current squad supplied)"
    elif wc_recommend:
        wc_reason = (f"A full rebuild gains {wc_metric:.1f} pts and wants "
                     f"{suggested_transfers} transfers — cheaper than paying hits")
    else:
        wc_reason = (f"A full rebuild gains {wc_metric:.1f} pts across "
                     f"{suggested_transfers} transfers — not enough to burn the chip")
    flags.append(ChipFlag("wildcard", CHIP_LABELS["wildcard"], "wildcard" in playable,
                          wc_recommend and "wildcard" in playable,
                          wc_metric, wc_reason))

    return flags


def gameweek_shape(bootstrap: dict, fixtures: Sequence[dict],
                   gameweek: int) -> dict:
    """Describe whether a gameweek is a blank, a double, or normal.

    Counts fixtures per club directly from the fixture list, which reflects the
    live schedule and is more reliable than any published calendar. See
    RULES.md §6.

    Returns:
        A dict with ``fixtures``, ``blank_teams``, ``double_teams`` and a
        ``label`` of ``"blank"``, ``"double"``, ``"blank+double"`` or
        ``"normal"``.
    """
    context = FixtureContext(fixtures)
    teams = [t["id"] for t in bootstrap["teams"]]
    blanks = [t for t in teams if context.fixture_count(t, gameweek) == 0]
    doubles = [t for t in teams if context.fixture_count(t, gameweek) >= 2]
    total = sum(context.fixture_count(t, gameweek) for t in teams) // 2

    if blanks and doubles:
        label = "blank+double"
    elif blanks:
        label = "blank"
    elif doubles:
        label = "double"
    else:
        label = "normal"

    return {
        "fixtures": total,
        "blank_teams": blanks,
        "double_teams": doubles,
        "label": label,
    }
