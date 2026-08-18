"""Predicted-points models.

Every model implements :class:`PlayerScorer` and returns a
:class:`Projection` per player over a horizon of gameweeks, so models are
interchangeable — see :func:`build_scorer` for automatic selection and the
``--model`` CLI flag for manual override.

Three models ship here:

``EPNextScorer``
    The simple baseline: FPL's own ``ep_next`` with a fixture-difficulty
    adjustment. Cheap and honest once the season is underway.

``BayesianRateScorer``
    The real projection model. Builds a recency-weighted, shrunk
    points-per-90 rate from each player's multi-season history, multiplies by
    projected minutes and a fixture multiplier, and sums over every fixture a
    player's club has in the gameweek (which handles doubles and blanks for
    free). This is what runs pre-season, where ``ep_next`` is useless.

``BlendedScorer``
    Weighted combination of the two, with the weight on current-season
    evidence growing as gameweeks are played.

Why not just use ``ep_next``
----------------------------
Pre-season ``ep_next`` saturates at 4.0 — Haaland (£15.5m) and Gabriel (£8.0m)
share the cap, so it cannot rank the top of the market at all. See RULES.md §8.

Modelling caveats (stated plainly, since they bound how much to trust output)
----------------------------------------------------------------------------
* Points-per-90 is treated as linear in minutes. Appearance points are not
  (1 pt under 60 minutes, 2 at 60+), so cameo-heavy players are slightly
  overrated. Small effect relative to the rest of the error.
* Seasons before 2025/26 predate defensive contribution points, so they
  understate defenders and defensive midfielders. Recency weighting
  (:data:`SEASON_DECAY`) is the mitigation — the most recent season carries
  roughly twice the weight of the one before it.
* Bonus points are inherited implicitly through historical totals rather than
  modelled from BPS. See RULES.md §2.3 for why.
* Fixture difficulty uses FPL's own per-fixture FDR. The finer-grained
  ``strength_attack_*`` / ``strength_defence_*`` team fields are all zero
  pre-season, so they are used only when populated.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Model constants. Exposed as module-level names so they are easy to tune.
# --------------------------------------------------------------------------- #

#: Geometric decay applied per season going backwards when averaging historical
#: rates. 0.5 means the most recent season counts double the one before it.
SEASON_DECAY = 0.5

#: Shrinkage strength for points-per-90, in units of "pseudo-90s". A player with
#: this many 90s of history gets a rate half-way between their own and the
#: price-implied prior. Guards against small-sample heroes.
RATE_SHRINKAGE_90S = 8.0

#: Same idea for projected minutes.
MINUTES_SHRINKAGE_GAMES = 6.0

#: Fixture sensitivity by position. Clean sheets are binary and swingy, so
#: defensive assets swing harder on fixture quality than attackers do.
FIXTURE_BETA = {1: 0.25, 2: 0.25, 3: 0.18, 4: 0.18}

#: Neutral fixture difficulty. FDR runs 1 (easiest) to 5 (hardest).
NEUTRAL_FDR = 3.0

#: Gameweeks of current-season evidence needed for it to carry half the weight
#: in :class:`BlendedScorer`.
BLEND_HALF_LIFE_GWS = 6.0

#: Minutes last season required before a player informs the price->rate prior.
PRIOR_FIT_MIN_MINUTES = 900

#: ``status`` codes that mean a player cannot play at all.
UNAVAILABLE_STATUSES = {"i", "s", "u", "n"}

# --- Set-piece premiums, in points per 90 --------------------------------- #
# Derivation: the Premier League awards roughly 100 penalties across 380
# matches, i.e. ~0.13 per team per match, converted at ~78%. A nailed taker
# therefore scores ~0.10 penalty goals per match. Multiplying by the value of a
# goal for that position (plus a little for the bonus points a goal tends to
# drag along, less a little for the -2 on a miss) gives the figures below.
PENALTY_TAKER_PREMIUM = {1: 0.0, 2: 0.70, 3: 0.60, 4: 0.50}

#: Second-choice takers only convert when the first choice is off the pitch.
SECONDARY_PENALTY_FRACTION = 0.25

#: Premium for the designated corner/free-kick taker, from the extra assists
#: set-piece delivery produces (~0.07 assists per match x 3 points).
SET_PIECE_CREATOR_PREMIUM = 0.20

#: Ownership percentage at which the crowd is treated as confident a player
#: starts. Used only as a *floor* on the start-rate prior — see
#: ``BayesianRateScorer._expected_minutes`` for why the asymmetry matters.
OWNERSHIP_NAILED_PIVOT = 20.0

#: Highest start rate ownership alone will imply.
OWNERSHIP_START_CEILING = 0.92

#: Sensitivity of the no-history prior to club strength, per point of the
#: game's 1-5 ``strength_overall`` scale.
TEAM_STRENGTH_BETA = 0.08

POSITION_NAMES = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Projection:
    """A per-player predicted-points projection over a gameweek horizon.

    Attributes:
        player_id: FPL element ID.
        expected_points: Total predicted points across the whole horizon.
        per_gameweek: Predicted points keyed by gameweek.
        fixtures_per_gameweek: Fixture count keyed by gameweek. 0 marks a blank,
            2+ a double.
        points_per_90: The underlying rate the projection was built from.
        expected_minutes: Projected minutes per fixture.
        availability: Probability the player features at all, 0.0-1.0.
        model: Name of the model that produced this.
    """

    player_id: int
    expected_points: float
    per_gameweek: dict[int, float] = field(default_factory=dict)
    fixtures_per_gameweek: dict[int, int] = field(default_factory=dict)
    points_per_90: float = 0.0
    expected_minutes: float = 0.0
    availability: float = 1.0
    model: str = ""


# --------------------------------------------------------------------------- #
# Fixture context
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TeamFixture:
    """One fixture from a specific club's point of view."""

    gameweek: int
    opponent_id: int
    is_home: bool
    difficulty: int


class FixtureContext:
    """Indexes the fixture list by club and gameweek.

    Counting fixtures per club per gameweek is how blanks and doubles are
    detected: zero fixtures is a blank, two or more is a double. This is more
    reliable than any published calendar because it reflects the live schedule.
    """

    def __init__(self, fixtures: Sequence[dict]) -> None:
        self._by_team_gw: dict[tuple[int, int], list[TeamFixture]] = defaultdict(list)
        for fx in fixtures:
            gw = fx.get("event")
            if gw is None:
                # Postponed with no new date assigned yet.
                continue
            self._by_team_gw[(fx["team_h"], gw)].append(
                TeamFixture(gw, fx["team_a"], True, fx.get("team_h_difficulty", 3))
            )
            self._by_team_gw[(fx["team_a"], gw)].append(
                TeamFixture(gw, fx["team_h"], False, fx.get("team_a_difficulty", 3))
            )

    def fixtures_for(self, team_id: int, gameweek: int) -> list[TeamFixture]:
        """Return a club's fixtures in a gameweek (empty list means a blank)."""
        return self._by_team_gw.get((team_id, gameweek), [])

    def fixture_count(self, team_id: int, gameweek: int) -> int:
        """Return how many fixtures a club has in a gameweek."""
        return len(self.fixtures_for(team_id, gameweek))

    def blank_teams(self, gameweek: int) -> set[int]:
        """Return clubs with no fixture in the gameweek."""
        return {t for (t, gw), f in self._by_team_gw.items() if gw == gameweek and not f} or {
            t for t in self._all_teams() if self.fixture_count(t, gameweek) == 0
        }

    def double_teams(self, gameweek: int) -> set[int]:
        """Return clubs with two or more fixtures in the gameweek."""
        return {t for t in self._all_teams() if self.fixture_count(t, gameweek) >= 2}

    def _all_teams(self) -> set[int]:
        return {team for team, _ in self._by_team_gw}


def fixture_multiplier(difficulty: int, element_type: int) -> float:
    """Scale expected output by fixture difficulty.

    FDR 3 is neutral (multiplier 1.0); easier fixtures scale up, harder down.
    The gradient is position-dependent — see :data:`FIXTURE_BETA`.

    Args:
        difficulty: FPL fixture difficulty rating, 1 (easiest) to 5 (hardest).
        element_type: Position ID, 1=GKP through 4=FWD.

    Returns:
        A multiplier, roughly 0.75-1.25.
    """
    beta = FIXTURE_BETA.get(element_type, 0.2)
    return 1.0 + beta * (NEUTRAL_FDR - difficulty) / 2.0


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #


def availability(player: dict) -> float:
    """Return the probability a player is available to feature, 0.0-1.0.

    Uses FPL's ``status`` flag and ``chance_of_playing_next_round``. Injured,
    suspended and unregistered players return 0.0 so the optimizer will never
    select them.
    """
    status = player.get("status", "a")
    if status in UNAVAILABLE_STATUSES:
        return 0.0
    chance = player.get("chance_of_playing_next_round")
    if chance is not None:
        return max(0.0, min(1.0, chance / 100.0))
    # status 'd' (doubtful) with no stated percentage: treat as a coin-flip
    # leaning available, rather than silently assuming full fitness.
    return 0.75 if status == "d" else 1.0


# --------------------------------------------------------------------------- #
# Historical rate estimation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HistoricalRates:
    """Recency-weighted historical aggregates for one player.

    Attributes:
        weighted_90s: Recency-weighted count of 90-minute blocks played.
        weighted_points: Recency-weighted total points.
        weighted_starts: Recency-weighted starts.
        weighted_games: Recency-weighted games available to start.
        raw_seasons: How many prior seasons contributed.
    """

    weighted_90s: float
    weighted_points: float
    weighted_starts: float
    weighted_games: float
    raw_seasons: int


def historical_rates(summary: dict | None, decay: float = SEASON_DECAY) -> HistoricalRates:
    """Aggregate a player's ``history_past`` with geometric recency weighting.

    ``history_past`` is ordered oldest-first, so the last entry is the most
    recent season and receives weight 1.0, the one before ``decay``, and so on.

    Args:
        summary: An ``element-summary`` payload, or ``None`` if unavailable.
        decay: Per-season geometric decay factor.

    Returns:
        Weighted aggregates; all-zero if there is no usable history.
    """
    if not summary or not summary.get("history_past"):
        return HistoricalRates(0.0, 0.0, 0.0, 0.0, 0)

    seasons = summary["history_past"]
    w90 = wpts = wstarts = wgames = 0.0
    for offset, season in enumerate(reversed(seasons)):
        weight = decay ** offset
        minutes = float(season.get("minutes") or 0)
        if minutes <= 0:
            continue
        w90 += weight * minutes / 90.0
        wpts += weight * float(season.get("total_points") or 0)
        wstarts += weight * float(season.get("starts") or 0)
        # A full PL season is 38 games; used as the denominator for start rate.
        wgames += weight * 38.0

    return HistoricalRates(w90, wpts, wstarts, wgames, len(seasons))


# --------------------------------------------------------------------------- #
# Price-implied priors
# --------------------------------------------------------------------------- #


class PricePrior:
    """Maps price to an expected points-per-90 rate, fitted per position.

    Player price is the single best pre-season signal for someone with no
    Premier League history — a promoted-club player or an overseas signing.
    FPL sets prices from its own expectations and the market's, so price
    encodes information no historical record can.

    The fit is an ordinary least-squares line of points-per-90 against price,
    estimated only from players with a substantial minutes sample. Positions
    are fitted separately because their scoring rates differ structurally.
    """

    def __init__(self, players: Sequence[dict], summaries: dict[int, dict]) -> None:
        self._coeffs: dict[int, tuple[float, float]] = {}
        self._fallback: dict[int, float] = {}
        self._fit(players, summaries)

    def _fit(self, players: Sequence[dict], summaries: dict[int, dict]) -> None:
        import numpy as np

        by_position: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for player in players:
            summary = summaries.get(player["id"])
            if not summary or not summary.get("history_past"):
                continue
            recent = summary["history_past"][-1]
            minutes = float(recent.get("minutes") or 0)
            if minutes < PRIOR_FIT_MIN_MINUTES:
                continue
            rate = float(recent.get("total_points") or 0) / (minutes / 90.0)
            by_position[player["element_type"]].append((player["now_cost"] / 10.0, rate))

        for position, observations in by_position.items():
            rates = [r for _, r in observations]
            self._fallback[position] = float(np.mean(rates)) if rates else 3.0
            if len(observations) < 5:
                continue
            prices = np.array([p for p, _ in observations])
            values = np.array(rates)
            slope, intercept = np.polyfit(prices, values, 1)
            self._coeffs[position] = (float(slope), float(intercept))

    def rate_for(self, player: dict) -> float:
        """Return the price-implied points-per-90 prior for a player."""
        position = player["element_type"]
        price = player["now_cost"] / 10.0
        if position in self._coeffs:
            slope, intercept = self._coeffs[position]
            return max(0.5, slope * price + intercept)
        return max(0.5, self._fallback.get(position, 3.0))


def set_piece_premium(player: dict) -> float:
    """Return the points-per-90 premium a player earns from set-piece duty.

    Penalty duty is the dominant term; corner and free-kick duty adds a smaller
    creative premium. Both are read from the ``*_order`` fields, which FPL
    refreshes for the new season and which therefore carry information no
    historical record can — a player who has just been handed penalties has
    never shown it in their past returns.

    Args:
        player: A bootstrap ``elements`` entry.

    Returns:
        Additional points per 90 minutes. Zero for players with no set-piece
        role, and always zero for goalkeepers.
    """
    element_type = player["element_type"]
    if element_type == 1:
        return 0.0

    premium = 0.0
    penalty_order = player.get("penalties_order")
    if penalty_order == 1:
        premium += PENALTY_TAKER_PREMIUM.get(element_type, 0.0)
    elif penalty_order == 2:
        premium += (PENALTY_TAKER_PREMIUM.get(element_type, 0.0)
                    * SECONDARY_PENALTY_FRACTION)

    takes_corners = player.get("corners_and_indirect_freekicks_order") == 1
    takes_freekicks = player.get("direct_freekicks_order") == 1
    if takes_corners or takes_freekicks:
        premium += SET_PIECE_CREATOR_PREMIUM

    return premium


def team_strength_multiplier(team: dict) -> float:
    """Scale a no-history prior by club quality.

    Price already encodes much of this, but not all of it — a £5.5m midfielder
    at a title contender is not the same asset as a £5.5m midfielder at a
    promoted club. Uses ``strength_overall_home``/``away``, which are populated
    pre-season even though the finer attack/defence splits are not.

    Args:
        team: A bootstrap ``teams`` entry.

    Returns:
        A multiplier centred on 1.0 for a mid-table club.
    """
    home = team.get("strength_overall_home") or 3
    away = team.get("strength_overall_away") or 3
    return 1.0 + TEAM_STRENGTH_BETA * ((home + away) / 2.0 - 3.5)


# --------------------------------------------------------------------------- #
# Scorer interface
# --------------------------------------------------------------------------- #


class PlayerScorer(ABC):
    """Interface for predicted-points models.

    Implement :meth:`project` to plug in a different model; everything
    downstream (optimizer, transfers, chips) consumes :class:`Projection`
    objects and neither knows nor cares which model produced them.
    """

    name: str = "abstract"

    @abstractmethod
    def project(self, gameweeks: Sequence[int]) -> dict[int, Projection]:
        """Return a projection per player over the given gameweeks."""

    def project_one(self, gameweeks: Sequence[int]) -> dict[int, float]:
        """Convenience wrapper returning only total expected points per player."""
        return {pid: p.expected_points for pid, p in self.project(gameweeks).items()}


# --------------------------------------------------------------------------- #
# Model 1: ep_next baseline
# --------------------------------------------------------------------------- #


class EPNextScorer(PlayerScorer):
    """FPL's own ``ep_next`` with a fixture-difficulty adjustment.

    Simple and reasonable in-season. Pre-season, ``ep_next`` is saturated and
    this model should not be trusted — see the module docstring.

    ``ep_next`` is FPL's expected points for the *next* gameweek specifically,
    so applying it across a multi-gameweek horizon assumes the rate holds. The
    fixture multiplier and per-gameweek fixture count still vary properly.
    """

    name = "ep_next"

    def __init__(self, bootstrap: dict, fixtures: Sequence[dict]) -> None:
        self.players = bootstrap["elements"]
        self.context = FixtureContext(fixtures)

    def project(self, gameweeks: Sequence[int]) -> dict[int, Projection]:
        projections: dict[int, Projection] = {}
        for player in self.players:
            base = float(player.get("ep_next") or 0.0)
            avail = availability(player)
            per_gw: dict[int, float] = {}
            counts: dict[int, int] = {}
            total = 0.0

            for gw in gameweeks:
                team_fixtures = self.context.fixtures_for(player["team"], gw)
                counts[gw] = len(team_fixtures)
                points = sum(
                    base * fixture_multiplier(fx.difficulty, player["element_type"])
                    for fx in team_fixtures
                ) * avail
                per_gw[gw] = points
                total += points

            projections[player["id"]] = Projection(
                player_id=player["id"],
                expected_points=total,
                per_gameweek=per_gw,
                fixtures_per_gameweek=counts,
                points_per_90=base,
                expected_minutes=90.0 * avail,
                availability=avail,
                model=self.name,
            )
        return projections


# --------------------------------------------------------------------------- #
# Model 2: the real projection model
# --------------------------------------------------------------------------- #


class BayesianRateScorer(PlayerScorer):
    """Multi-season shrunk rate model with minutes and fixture adjustment.

    For each player and gameweek::

        points = availability
               * (expected_minutes / 90)
               * points_per_90
               * sum over the club's fixtures of fixture_multiplier(fdr)

    Because the final term sums over *actual* fixtures, a blank gameweek
    contributes zero and a double contributes both matches automatically.

    ``points_per_90`` is an empirical-Bayes estimate: the player's own
    recency-weighted historical rate, shrunk toward a price-implied prior in
    proportion to how little history they have. A player with one good half-season
    is pulled toward what their price says; a four-season regular is trusted on
    their own record. Players with no Premier League history at all — promoted
    clubs, overseas signings — fall back entirely to the price prior.

    Expected minutes are estimated the same way, from historical start rate
    shrunk toward a price-implied expectation, then scaled by availability.
    """

    name = "bayes-rate"

    def __init__(self, bootstrap: dict, fixtures: Sequence[dict],
                 summaries: dict[int, dict]) -> None:
        self.players = bootstrap["elements"]
        self.context = FixtureContext(fixtures)
        self.summaries = summaries
        self.teams = {t["id"]: t for t in bootstrap["teams"]}
        self.prior = PricePrior(self.players, summaries)
        self._minutes_prior = self._fit_minutes_prior()

    def _fit_minutes_prior(self) -> dict[int, float]:
        """Return a per-position mean start rate, used when history is thin."""
        by_position: dict[int, list[float]] = defaultdict(list)
        for player in self.players:
            rates = historical_rates(self.summaries.get(player["id"]))
            if rates.weighted_games > 0:
                by_position[player["element_type"]].append(
                    rates.weighted_starts / rates.weighted_games
                )
        return {
            pos: (sum(vals) / len(vals) if vals else 0.5)
            for pos, vals in by_position.items()
        }

    def _prior_reliance(self, rates: HistoricalRates) -> float:
        """Return how much weight the prior carries for a player, 0.0-1.0.

        This is the shrinkage weight ``K / (N + K)``: 1.0 for a player with no
        history at all, approaching 0.0 for a long-serving regular.
        """
        return RATE_SHRINKAGE_90S / (rates.weighted_90s + RATE_SHRINKAGE_90S)

    def _points_per_90(self, player: dict) -> float:
        """Shrink a player's historical rate toward their price-implied prior.

        The price prior is nudged by club strength, since price alone does not
        fully separate a mid-priced player at a strong club from one at a weak
        club.

        The set-piece premium is scaled by :meth:`_prior_reliance` rather than
        applied flat. This matters: an established penalty taker's past returns
        *already contain* their penalty goals, so adding a premium on top would
        double-count them. A player with no history has nothing to double-count,
        so they receive it in full — which is exactly the case the premium
        exists to cover, a new signing just handed the duty.

        Known limitation: an established player who has only *just* inherited
        penalties gets a smaller bump than they deserve, because historical
        set-piece orders are not exposed by the API and the change cannot be
        detected.
        """
        rates = historical_rates(self.summaries.get(player["id"]))
        team = self.teams.get(player["team"], {})
        prior_rate = self.prior.rate_for(player) * team_strength_multiplier(team)

        numerator = rates.weighted_points + RATE_SHRINKAGE_90S * prior_rate
        denominator = rates.weighted_90s + RATE_SHRINKAGE_90S
        base = numerator / denominator

        return base + set_piece_premium(player) * self._prior_reliance(rates)

    def _expected_minutes(self, player: dict) -> float:
        """Estimate minutes per fixture, before availability is applied.

        Start rate is shrunk toward the positional mean, then converted to
        minutes assuming starters average 85 minutes and non-starting
        appearances contribute a short cameo.
        """
        rates = historical_rates(self.summaries.get(player["id"]))
        position_prior = self._minutes_prior.get(player["element_type"], 0.5)

        # A price-relative nudge: within a position, expensive players start.
        price_prior = self.prior.rate_for(player)
        prior_start_rate = min(0.95, max(0.15, position_prior * (price_prior / 4.0)))

        # Ownership is applied as a FLOOR, never a ceiling, and only to minutes
        # -- never to the points rate. The asymmetry and the restriction are
        # both deliberate:
        #   * High ownership is strong evidence a player is nailed; millions of
        #     managers have collectively checked the team news. Low ownership is
        #     weak evidence of anything, since differentials exist by definition.
        #   * Feeding ownership into the points rate would simply reproduce the
        #     crowd's opinion of quality and push every squad toward the
        #     template, surrendering the upside of being different. Feeding it
        #     into minutes captures the part the crowd genuinely knows -- who
        #     starts -- without copying its view of who is good.
        ownership = float(player.get("selected_by_percent") or 0.0)
        ownership_floor = min(OWNERSHIP_START_CEILING,
                              ownership / OWNERSHIP_NAILED_PIVOT * OWNERSHIP_START_CEILING)
        prior_start_rate = max(prior_start_rate, ownership_floor)

        observed_starts = rates.weighted_starts
        observed_games = rates.weighted_games
        start_rate = (
            observed_starts + MINUTES_SHRINKAGE_GAMES * prior_start_rate
        ) / (observed_games + MINUTES_SHRINKAGE_GAMES)

        # 85 minutes for a start, plus a modest cameo share for non-starts.
        return start_rate * 85.0 + (1.0 - start_rate) * 12.0

    def project(self, gameweeks: Sequence[int]) -> dict[int, Projection]:
        projections: dict[int, Projection] = {}
        for player in self.players:
            avail = availability(player)
            rate = self._points_per_90(player)
            minutes = self._expected_minutes(player)
            per_fixture = avail * (minutes / 90.0) * rate

            per_gw: dict[int, float] = {}
            counts: dict[int, int] = {}
            total = 0.0
            for gw in gameweeks:
                team_fixtures = self.context.fixtures_for(player["team"], gw)
                counts[gw] = len(team_fixtures)
                points = sum(
                    per_fixture * fixture_multiplier(fx.difficulty, player["element_type"])
                    for fx in team_fixtures
                )
                per_gw[gw] = points
                total += points

            projections[player["id"]] = Projection(
                player_id=player["id"],
                expected_points=total,
                per_gameweek=per_gw,
                fixtures_per_gameweek=counts,
                points_per_90=rate,
                expected_minutes=minutes,
                availability=avail,
                model=self.name,
            )
        return projections


# --------------------------------------------------------------------------- #
# Model 3: blend
# --------------------------------------------------------------------------- #


class BlendedScorer(PlayerScorer):
    """Weighted blend of current-season and historical models.

    The weight on current-season evidence grows as gameweeks are played::

        w = finished_gws / (finished_gws + BLEND_HALF_LIFE_GWS)

    so pre-season the projection is entirely historical, and by roughly
    gameweek 6 the two carry equal weight. This avoids the twin failure modes
    of trusting a saturated pre-season ``ep_next`` and of still leaning on last
    season's numbers in March.
    """

    name = "blended"

    def __init__(self, historical: PlayerScorer, current: PlayerScorer,
                 finished_gws: int) -> None:
        self.historical = historical
        self.current = current
        self.weight_current = finished_gws / (finished_gws + BLEND_HALF_LIFE_GWS)

    def project(self, gameweeks: Sequence[int]) -> dict[int, Projection]:
        hist = self.historical.project(gameweeks)
        curr = self.current.project(gameweeks)
        w = self.weight_current

        blended: dict[int, Projection] = {}
        for pid, h in hist.items():
            c = curr.get(pid)
            if c is None:
                blended[pid] = h
                continue
            per_gw = {
                gw: (1 - w) * h.per_gameweek.get(gw, 0.0) + w * c.per_gameweek.get(gw, 0.0)
                for gw in set(h.per_gameweek) | set(c.per_gameweek)
            }
            blended[pid] = Projection(
                player_id=pid,
                expected_points=(1 - w) * h.expected_points + w * c.expected_points,
                per_gameweek=per_gw,
                fixtures_per_gameweek=h.fixtures_per_gameweek,
                points_per_90=(1 - w) * h.points_per_90 + w * c.points_per_90,
                expected_minutes=h.expected_minutes,
                availability=h.availability,
                model=f"{self.name}(w_current={w:.2f})",
            )
        return blended


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def build_scorer(bootstrap: dict, fixtures: Sequence[dict],
                 summaries: dict[int, dict] | None = None,
                 model: str = "auto") -> PlayerScorer:
    """Construct a scorer, choosing a model automatically by default.

    Args:
        bootstrap: ``bootstrap-static`` payload.
        fixtures: Fixture list.
        summaries: Per-player ``element-summary`` payloads. Required for the
            historical models; without them only ``ep_next`` is available.
        model: One of ``"auto"``, ``"ep_next"``, ``"bayes"``, ``"blended"``.

    Returns:
        A ready-to-use :class:`PlayerScorer`.

    Raises:
        ValueError: on an unknown model name.
    """
    from fpl.data import get_game_state

    state = get_game_state(bootstrap)
    summaries = summaries or {}

    if model == "ep_next":
        return EPNextScorer(bootstrap, fixtures)
    if model == "bayes":
        return BayesianRateScorer(bootstrap, fixtures, summaries)
    if model == "blended":
        return BlendedScorer(
            BayesianRateScorer(bootstrap, fixtures, summaries),
            EPNextScorer(bootstrap, fixtures),
            state.finished_gws,
        )
    if model != "auto":
        raise ValueError(f"Unknown model {model!r}")

    # Auto: without history we have no choice; pre-season ep_next is useless, so
    # go purely historical; thereafter blend current-season evidence in.
    if not summaries:
        log.warning("No player history available; falling back to ep_next.")
        return EPNextScorer(bootstrap, fixtures)
    if not state.season_started:
        return BayesianRateScorer(bootstrap, fixtures, summaries)
    return BlendedScorer(
        BayesianRateScorer(bootstrap, fixtures, summaries),
        EPNextScorer(bootstrap, fixtures),
        state.finished_gws,
    )
