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

from fpl import dixon_coles

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
#: defensive assets swing harder on fixture quality than attackers do. Used by
#: the fixture-difficulty baseline; the Dixon-Coles path uses
#: :data:`POSITION_DEFENSIVE_SHARE` instead.
FIXTURE_BETA = {1: 0.25, 2: 0.25, 3: 0.18, 4: 0.18}

#: How much of each position's fixture sensitivity is defensive rather than
#: attacking, used to combine clean-sheet and expected-goals multipliers.
#:
#: Goalkeepers are below 1.0 on purpose: a hard fixture costs them clean-sheet
#: points but hands them save points, and the two partly cancel. Forwards are at
#: 0.0 — they earn nothing from a clean sheet (RULES.md §2.1).
POSITION_DEFENSIVE_SHARE = {1: 0.70, 2: 0.60, 3: 0.15, 4: 0.0}

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
OWNERSHIP_NAILED_PIVOT = 30.0

#: Price above a position's cheapest player, in millions, at which ownership is
#: fully trusted as a starting signal. Below this the signal is faded out: a
#: heavily-owned minimum-price player is bench fodder that many managers hold
#: precisely so it never plays, which is the opposite of what ownership means
#: for a premium asset.
OWNERSHIP_PRICE_CONFIDENCE_SPAN = 1.5

#: Highest start rate ownership alone will imply.
OWNERSHIP_START_CEILING = 0.92

#: Sensitivity of the no-history prior to club strength, per point of the
#: game's 1-5 ``strength_overall`` scale.
TEAM_STRENGTH_BETA = 0.08

#: Per-season decay applied when looking for a player's *peak* minutes level.
#: Deliberately gentler than :data:`SEASON_DECAY`: the recency-weighted mean
#: answers "what does this player do lately", whereas the peak answers "what is
#: this player capable of when fit and in favour", and capability fades more
#: slowly than form.
PEAK_SEASON_DECAY = 0.8

#: How much of a past peak we are willing to project forward. Below 1.0 because
#: a player who once played every minute may not reclaim that role.
PEAK_TRUST = 0.85

#: Minutes a starter is assumed to play, allowing for substitutions.
STARTER_MINUTES = 85.0

#: Minutes a non-starting appearance contributes on average.
CAMEO_MINUTES = 12.0

#: Players who joined their club on or after this date are treated as new
#: signings whose historical minutes were earned elsewhere.
NEW_SIGNING_CUTOFF = "2026-06-01"

#: Multiplier on minutes shrinkage for new signings. Their past minutes are
#: real but were earned in a different squad, so their new role is genuinely
#: more uncertain and the prior should carry more weight.
NEW_SIGNING_SHRINKAGE_FACTOR = 2.5

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
        clean_sheet_probability: Probability the player's club concedes zero
            across the horizon's fixtures, from the Dixon-Coles fit. ``None``
            when no results model is available. For a double gameweek this is
            the probability of a clean sheet in *at least one* fixture.
        expected_goal_involvement: Expected goals plus assists for this player
            over the horizon. ``None`` when no results model is available.
    """

    player_id: int
    expected_points: float
    per_gameweek: dict[int, float] = field(default_factory=dict)
    fixtures_per_gameweek: dict[int, int] = field(default_factory=dict)
    points_per_90: float = 0.0
    expected_minutes: float = 0.0
    availability: float = 1.0
    model: str = ""
    clean_sheet_probability: float | None = None
    expected_goal_involvement: float | None = None


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


class FixtureModel:
    """Turns a fixture into a multiplier on a player's baseline scoring rate.

    Two sources, blended:

    * the **fixture-difficulty baseline**, FPL's own 1-5 FDR, always available;
    * a fitted **Dixon-Coles** model (:mod:`fpl.dixon_coles`), available only
      once enough of this season's results are in.

    The blend is governed by :func:`fpl.dixon_coles.blend_weight`, so the
    handover is gradual: FDR alone pre-season, roughly even at six gameweeks,
    mostly Dixon-Coles thereafter.

    Both sources return a multiplier **normalised to 1.0 for an average
    fixture**, which is what keeps this compatible with the rest of the model.
    The projection scales a player's *empirically observed* points-per-90, and
    that rate already contains the clean sheets and goals they historically
    earned. Adding absolute clean-sheet points on top would count them twice.
    So the Dixon-Coles output is used to say *how much better or worse than
    usual* this particular fixture is — never to reconstruct points from
    scratch.
    """

    def __init__(self, context: "FixtureContext",
                 fit: "dixon_coles.DixonColesFit | None" = None) -> None:
        self.context = context
        self.fit = fit
        self.weight = dixon_coles.blend_weight(fit.matches_per_team) if fit else 0.0
        self._average_clean_sheet = fit.league_average_clean_sheet() if fit else None
        self._average_goals = fit.league_average_goals() if fit else None

    @property
    def has_results_model(self) -> bool:
        """Whether a fitted results model is contributing at all."""
        return self.fit is not None and self.weight > 0.0

    def clean_sheet_probability(self, team_id: int,
                                fixture: "TeamFixture") -> float | None:
        """Return P(clean sheet) for a club in a fixture, or ``None`` unfitted."""
        if self.fit is None:
            return None
        return self.fit.project(team_id, fixture.opponent_id,
                                fixture.is_home).clean_sheet_probability

    def expected_goals(self, team_id: int,
                       fixture: "TeamFixture") -> float | None:
        """Return a club's expected goals in a fixture, or ``None`` unfitted."""
        if self.fit is None:
            return None
        return self.fit.project(team_id, fixture.opponent_id,
                                fixture.is_home).expected_goals_for

    def multiplier(self, element_type: int, team_id: int,
                   fixture: "TeamFixture") -> float:
        """Return the scoring multiplier for a player in a fixture."""
        baseline = fixture_multiplier(fixture.difficulty, element_type)
        if not self.has_results_model:
            return baseline

        clean_sheet = self.clean_sheet_probability(team_id, fixture)
        goals = self.expected_goals(team_id, fixture)
        if clean_sheet is None or goals is None:
            return baseline

        defensive = clean_sheet / self._average_clean_sheet if self._average_clean_sheet else 1.0
        attacking = goals / self._average_goals if self._average_goals else 1.0

        share = POSITION_DEFENSIVE_SHARE.get(element_type, 0.0)
        modelled = share * defensive + (1.0 - share) * attacking

        return self.weight * modelled + (1.0 - self.weight) * baseline


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
        minutes_per_game: Minutes per league game for each prior season, most
            recent first. Used to recover a player's peak involvement, which the
            recency-weighted mean hides when a recent season was lost to injury.
    """

    weighted_90s: float
    weighted_points: float
    weighted_starts: float
    weighted_games: float
    raw_seasons: int
    minutes_per_game: tuple[float, ...] = ()

    @property
    def weighted_minutes(self) -> float:
        """Recency-weighted total minutes."""
        return self.weighted_90s * 90.0

    def peak_minutes_per_game(self) -> float:
        """Return the best minutes-per-game level this player has sustained.

        Each season's level is discounted by :data:`PEAK_SEASON_DECAY` per year
        of age and by :data:`PEAK_TRUST`, so an old peak counts for less than a
        recent one but is not erased by a single wrecked season.
        """
        if not self.minutes_per_game:
            return 0.0
        return max(
            level * (PEAK_SEASON_DECAY ** age) * PEAK_TRUST
            for age, level in enumerate(self.minutes_per_game)
        )


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
    per_season: list[float] = []

    for offset, season in enumerate(reversed(seasons)):
        weight = decay ** offset
        minutes = float(season.get("minutes") or 0)
        # Recorded even when zero: a season sat out is information about the
        # mean, and skipping it here would silently inflate the average.
        per_season.append(minutes / 38.0)
        if minutes <= 0:
            continue
        w90 += weight * minutes / 90.0
        wpts += weight * float(season.get("total_points") or 0)
        wstarts += weight * float(season.get("starts") or 0)
        # A full PL season is 38 games; used as the denominator for start rate.
        wgames += weight * 38.0

    return HistoricalRates(w90, wpts, wstarts, wgames, len(seasons),
                           tuple(per_season))


def historical_goal_involvement_per_90(summary: dict | None,
                                       decay: float = SEASON_DECAY) -> float:
    """Return recency-weighted expected goal involvements per 90 minutes.

    Prefers FPL's own ``expected_goal_involvements`` (xG + xA) where recorded,
    since it is far less noisy than raw goals plus assists, and falls back to
    actual goals and assists for seasons predating the expected-stats feed.
    """
    if not summary or not summary.get("history_past"):
        return 0.0

    weighted_involvement = weighted_90s = 0.0
    for offset, season in enumerate(reversed(summary["history_past"])):
        minutes = float(season.get("minutes") or 0)
        if minutes <= 0:
            continue
        weight = decay ** offset
        expected = season.get("expected_goal_involvements")
        if expected in (None, "", 0, "0.00"):
            involvement = (float(season.get("goals_scored") or 0)
                           + float(season.get("assists") or 0))
        else:
            involvement = float(expected)
        weighted_involvement += weight * involvement
        weighted_90s += weight * minutes / 90.0

    return weighted_involvement / weighted_90s if weighted_90s > 0 else 0.0


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
                 summaries: dict[int, dict],
                 fixture_model: FixtureModel | None = None) -> None:
        self.players = bootstrap["elements"]
        self.context = FixtureContext(fixtures)
        self.fixture_model = fixture_model or FixtureModel(self.context)
        self.summaries = summaries
        self.teams = {t["id"]: t for t in bootstrap["teams"]}
        self.prior = PricePrior(self.players, summaries)
        self._minutes_prior = self._fit_minutes_prior()
        self._position_min_price = self._find_position_min_prices()

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

    def _find_position_min_prices(self) -> dict[int, int]:
        """Return the cheapest available price in each position."""
        minima: dict[int, int] = {}
        for player in self.players:
            position = player["element_type"]
            price = player["now_cost"]
            if position not in minima or price < minima[position]:
                minima[position] = price
        return minima

    def _ownership_confidence(self, player: dict) -> float:
        """How far a player's price implies they are bought in order to play.

        Ownership means opposite things at opposite ends of the price range. A
        heavily-owned £12m midfielder is owned because managers expect him to
        start and score. A heavily-owned £4.0m goalkeeper is owned as bench
        fodder — held *because* he is cheap, often precisely so he never plays.
        Reading the second as evidence of nailed-on minutes puts backup keepers
        in the starting XI.

        Returns 0.0 at a position's minimum price, rising to 1.0 once a player
        is :data:`OWNERSHIP_PRICE_CONFIDENCE_SPAN` above it.
        """
        floor_price = self._position_min_price.get(player["element_type"],
                                                   player["now_cost"])
        excess = (player["now_cost"] - floor_price) / 10.0
        return max(0.0, min(1.0, excess / OWNERSHIP_PRICE_CONFIDENCE_SPAN))

    def _is_new_signing(self, player: dict) -> bool:
        """Whether a player joined their current club in the latest window."""
        joined = player.get("team_join_date") or ""
        return joined >= NEW_SIGNING_CUTOFF

    def _expected_minutes(self, player: dict) -> float:
        """Estimate minutes per fixture, before availability is applied.

        Worked in minutes-per-game space rather than as a start/bench
        dichotomy, because minutes-per-game is what the projection actually
        needs and it is directly observable per season.

        Three corrections matter, all of which need more than one season of
        history to compute:

        1. **Peak capability floor.** The recency-weighted mean is the right
           estimate of recent involvement, but it badly understates a player
           whose most recent season was lost to injury or a fallen-out-of-favour
           spell — and recency weighting makes that *worse*, since the wrecked
           season carries the highest weight. So the estimate is floored at the
           player's decayed peak: what they have proven they can sustain when
           fit and in the team. A player with three full seasons and one blank
           is projected on the three, not the blank.

        2. **New-signing uncertainty.** Minutes earned at a previous club are
           real evidence but weaker evidence — the role at the new club is
           unproven. Shrinkage toward the price-implied prior is widened for
           these players rather than trusting the old club's numbers outright.

        3. **Ownership floor**, as documented below.

        None of this predicts the literal starting XI. No public data source
        gives confirmed line-ups before the deadline; this estimates expected
        minutes, which is what the points projection needs.
        """
        rates = historical_rates(self.summaries.get(player["id"]))
        position_prior = self._minutes_prior.get(player["element_type"], 0.5)

        # A price-relative nudge: within a position, expensive players start.
        price_prior = self.prior.rate_for(player)
        prior_start_rate = min(0.95, max(0.15, position_prior * (price_prior / 4.0)))
        prior_minutes = (prior_start_rate * STARTER_MINUTES
                         + (1.0 - prior_start_rate) * CAMEO_MINUTES)

        shrinkage = MINUTES_SHRINKAGE_GAMES
        if self._is_new_signing(player):
            shrinkage *= NEW_SIGNING_SHRINKAGE_FACTOR

        shrunk = ((rates.weighted_minutes + shrinkage * prior_minutes)
                  / (rates.weighted_games + shrinkage))

        # Capability floor. Also shrunk for new signings, since a peak reached
        # at another club is likewise weaker evidence here.
        peak = rates.peak_minutes_per_game()
        if self._is_new_signing(player):
            peak *= 0.8
        minutes = max(shrunk, min(peak, STARTER_MINUTES))

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
        # ...and faded out at the cheap end, where ownership signals bench
        # fodder rather than a starting berth. See _ownership_confidence.
        ownership = float(player.get("selected_by_percent") or 0.0)
        ownership_share = min(1.0, ownership / OWNERSHIP_NAILED_PIVOT)
        ownership_floor = (ownership_share
                           * self._ownership_confidence(player)
                           * OWNERSHIP_START_CEILING
                           * STARTER_MINUTES)

        return min(90.0, max(minutes, ownership_floor))

    def project(self, gameweeks: Sequence[int]) -> dict[int, Projection]:
        projections: dict[int, Projection] = {}
        model_name = self.name
        if self.fixture_model.has_results_model:
            model_name = f"{self.name}+dixon-coles(w={self.fixture_model.weight:.2f})"

        for player in self.players:
            avail = availability(player)
            rate = self._points_per_90(player)
            minutes = self._expected_minutes(player)
            per_fixture = avail * (minutes / 90.0) * rate
            involvement_rate = historical_goal_involvement_per_90(
                self.summaries.get(player["id"]))

            per_gw: dict[int, float] = {}
            counts: dict[int, int] = {}
            total = 0.0
            involvement = 0.0
            # Probability of conceding in every fixture, so the complement is
            # "kept at least one clean sheet" -- the right framing for a double.
            concede_all = 1.0
            saw_clean_sheet_data = False

            for gw in gameweeks:
                team_fixtures = self.context.fixtures_for(player["team"], gw)
                counts[gw] = len(team_fixtures)
                points = 0.0

                for fixture in team_fixtures:
                    multiplier = self.fixture_model.multiplier(
                        player["element_type"], player["team"], fixture)
                    points += per_fixture * multiplier

                    goals = self.fixture_model.expected_goals(player["team"], fixture)
                    if goals is not None:
                        league_goals = self.fixture_model._average_goals or 1.0
                        involvement += (involvement_rate * (minutes / 90.0) * avail
                                        * goals / league_goals)

                    clean_sheet = self.fixture_model.clean_sheet_probability(
                        player["team"], fixture)
                    if clean_sheet is not None:
                        saw_clean_sheet_data = True
                        concede_all *= (1.0 - clean_sheet)

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
                model=model_name,
                clean_sheet_probability=(1.0 - concede_all) if saw_clean_sheet_data else None,
                expected_goal_involvement=involvement if self.fixture_model.has_results_model else None,
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

    # Fit the results model once and share it. Returns None until enough of
    # this season has been played, in which case FixtureModel falls back to FDR.
    context = FixtureContext(fixtures)
    fit = dixon_coles.fit(fixtures, [t["id"] for t in bootstrap["teams"]])
    fixture_model = FixtureModel(context, fit)

    if model == "ep_next":
        return EPNextScorer(bootstrap, fixtures)
    if model == "bayes":
        return BayesianRateScorer(bootstrap, fixtures, summaries, fixture_model)
    if model == "blended":
        return BlendedScorer(
            BayesianRateScorer(bootstrap, fixtures, summaries, fixture_model),
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
        return BayesianRateScorer(bootstrap, fixtures, summaries, fixture_model)
    return BlendedScorer(
        BayesianRateScorer(bootstrap, fixtures, summaries, fixture_model),
        EPNextScorer(bootstrap, fixtures),
        state.finished_gws,
    )
