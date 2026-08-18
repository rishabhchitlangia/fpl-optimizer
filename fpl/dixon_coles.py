"""A simplified Dixon-Coles goals model fitted to this season's results.

Estimates a per-club attack and defence strength from observed scorelines, plus
a league-wide home advantage, then uses them to produce the two quantities FPL
scoring actually turns on: **clean sheet probability** and **expected goals
scored**.

The model
---------
Goals are modelled as Poisson with club-specific rates::

    lambda_home = exp(attack[home] + defence[away] + home_advantage)
    lambda_away = exp(attack[away] + defence[home])

``attack`` is centred on zero for identifiability. ``defence`` is signed so that
a *negative* value means a club concedes fewer goals than average.

Dixon and Coles' contribution over plain Poisson is the ``tau`` correction,
which reweights the four lowest scorelines (0-0, 1-0, 0-1, 1-1) to fix the known
under-dispersion of independent Poissons at low scores. That correction is not
incidental here — those are precisely the scorelines that decide clean sheets,
so it directly sharpens the quantity we care about most.

Matches are weighted by :data:`TIME_DECAY_PER_DAY` so recent form counts for
more than August.

Why this is regularised so heavily
----------------------------------
The model has 41 free parameters (20 attack + 20 defence + home advantage) and a
gameweek supplies only 20 goal observations. Fitting that unpenalised after one
or two gameweeks produces nonsense — a club that wins 4-0 gets an attack
strength implying they will score four every week. Every fit therefore carries
an L2 penalty toward league average whose weight decays as matches accumulate,
and callers should additionally blend against a fixture-difficulty baseline via
:func:`blend_weight` until roughly six gameweeks have been played.

Pre-season this module produces nothing usable and says so: :meth:`fit` returns
``None`` below :data:`MIN_MATCHES_TO_FIT` and the caller falls back to FDR.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

#: Goal totals above this are lumped together when building score matrices.
MAX_GOALS = 10

#: Exponential decay applied per day of match age. ~0.0065 halves the weight of
#: a match roughly every 15 weeks, which keeps a full season relevant while
#: letting recent form lead.
TIME_DECAY_PER_DAY = 0.0065

#: Strength of the L2 pull toward league average, in log-likelihood units.
#:
#: Deliberately a FIXED penalty rather than one scaled by dataset size. A fixed
#: prior is self-correcting: the likelihood term grows as matches accumulate, so
#: the penalty dominates a three-gameweek sample and fades to a light touch over
#: a full season, with no schedule to tune. Scaling it by match count instead
#: crushes the parameters no matter how much evidence arrives — an earlier
#: version did exactly that and flattened the whole league toward average.
#:
#: Calibrated against simulated seasons with known parameters: at 1.0 the fit
#: recovers home advantage almost exactly (0.266 against a true 0.26) with an
#: attack-strength correlation of 0.93 over a full season, while still damping a
#: three-gameweek sample. Higher values visibly compress the league toward
#: average; lower values let three gameweeks of noise through.
REGULARISATION_STRENGTH = 1.0

#: Fewest finished matches before a fit is attempted at all.
MIN_MATCHES_TO_FIT = 20

#: Matches per club at which the model carries half the weight against the
#: fixture-difficulty baseline. Roughly six gameweeks.
BLEND_HALF_LIFE_MATCHES = 6.0

#: Bounds on the low-score correlation parameter. Empirically rho sits near
#: -0.1; the bounds stop a small sample driving it somewhere unphysical.
RHO_BOUNDS = (-0.15, 0.15)

#: Pull on rho toward zero. Much stronger than the team-strength penalty because
#: rho is identified by only four scorelines and is correspondingly noisy — on
#: simulated data it otherwise pinned itself to its bound, which cuts the
#: modelled probability of a 0-0 by a third and corrupts exactly the clean-sheet
#: numbers this model exists to produce.
RHO_REGULARISATION = 250.0


@dataclass
class TeamStrength:
    """Fitted strength parameters for one club.

    Attributes:
        team_id: FPL team ID.
        attack: Log-scale attacking strength, centred on zero. Positive is
            better than average.
        defence: Log-scale defensive strength. **Negative is better** — it
            reduces the opponent's expected goals.
        matches: How many finished matches informed this estimate.
    """

    team_id: int
    attack: float
    defence: float
    matches: int


@dataclass
class MatchProjection:
    """Expected goals and derived probabilities for one fixture, one side.

    Attributes:
        team_id: The club this projection is for.
        opponent_id: Their opponent.
        is_home: Whether this club is at home.
        expected_goals_for: Expected goals this club scores.
        expected_goals_against: Expected goals they concede.
        clean_sheet_probability: Probability they concede zero.
        expected_goals_conceded_points: Expected FPL penalty from goals
            conceded, as a negative number, applying the "-1 per 2 conceded"
            rule from RULES.md §2.1.
    """

    team_id: int
    opponent_id: int
    is_home: bool
    expected_goals_for: float
    expected_goals_against: float
    clean_sheet_probability: float
    expected_goals_conceded_points: float


@dataclass
class DixonColesFit:
    """A fitted model.

    Attributes:
        strengths: Fitted parameters keyed by team ID.
        home_advantage: League-wide log-scale home advantage.
        rho: Dixon-Coles low-score correction parameter.
        matches_used: Total finished matches the fit consumed.
        matches_per_team: Average finished matches per club, which drives how
            far the fit should be trusted.
        converged: Whether the optimiser reported success.
    """

    strengths: dict[int, TeamStrength]
    home_advantage: float
    rho: float
    matches_used: int
    matches_per_team: float
    converged: bool = True

    def expected_goals(self, home_id: int, away_id: int) -> tuple[float, float]:
        """Return ``(home_expected_goals, away_expected_goals)`` for a fixture."""
        home = self.strengths.get(home_id)
        away = self.strengths.get(away_id)
        if home is None or away is None:
            return (1.35, 1.15)          # league-average fallback
        lambda_home = math.exp(home.attack + away.defence + self.home_advantage)
        lambda_away = math.exp(away.attack + home.defence)
        return (lambda_home, lambda_away)

    def score_matrix(self, home_id: int, away_id: int) -> np.ndarray:
        """Return the joint scoreline probability matrix, tau correction applied.

        Element ``[x, y]`` is the probability of the fixture finishing
        ``x`` goals to ``y``.
        """
        lambda_home, lambda_away = self.expected_goals(home_id, away_id)
        goals = np.arange(MAX_GOALS + 1)

        home_pmf = np.exp(-lambda_home) * lambda_home ** goals / _factorials(goals)
        away_pmf = np.exp(-lambda_away) * lambda_away ** goals / _factorials(goals)
        matrix = np.outer(home_pmf, away_pmf)

        # Dixon-Coles low-score correction.
        matrix[0, 0] *= 1.0 - lambda_home * lambda_away * self.rho
        matrix[0, 1] *= 1.0 + lambda_home * self.rho
        matrix[1, 0] *= 1.0 + lambda_away * self.rho
        matrix[1, 1] *= 1.0 - self.rho

        total = matrix.sum()
        return matrix / total if total > 0 else matrix

    def project(self, team_id: int, opponent_id: int,
                is_home: bool) -> MatchProjection:
        """Project one club's outcomes in one fixture.

        Clean sheet probability is read off the corrected score matrix rather
        than as ``exp(-lambda)``, so it inherits the tau correction — which is
        the whole reason for preferring Dixon-Coles over plain Poisson here.
        """
        home_id, away_id = (team_id, opponent_id) if is_home else (opponent_id, team_id)
        matrix = self.score_matrix(home_id, away_id)

        if is_home:
            clean_sheet = float(matrix[:, 0].sum())
            conceded_distribution = matrix.sum(axis=0)
            goals_for, goals_against = self.expected_goals(home_id, away_id)
        else:
            clean_sheet = float(matrix[0, :].sum())
            conceded_distribution = matrix.sum(axis=1)
            goals_against, goals_for = self.expected_goals(home_id, away_id)

        # -1 point per two goals conceded, remainders discarded (RULES.md 2.1).
        penalty = -sum(prob * (conceded // 2)
                       for conceded, prob in enumerate(conceded_distribution))

        return MatchProjection(
            team_id=team_id,
            opponent_id=opponent_id,
            is_home=is_home,
            expected_goals_for=goals_for,
            expected_goals_against=goals_against,
            clean_sheet_probability=clean_sheet,
            expected_goals_conceded_points=float(penalty),
        )

    def league_average_clean_sheet(self) -> float:
        """Mean clean sheet probability across every possible fixture.

        Used as the normaliser so a fixture multiplier sits at 1.0 for an
        average matchup rather than at some arbitrary scale.
        """
        ids = list(self.strengths)
        if len(ids) < 2:
            return 0.30
        values = [
            self.project(team, opponent, is_home).clean_sheet_probability
            for team in ids for opponent in ids if team != opponent
            for is_home in (True, False)
        ]
        return float(np.mean(values))

    def league_average_goals(self) -> float:
        """Mean expected goals scored across every possible fixture."""
        ids = list(self.strengths)
        if len(ids) < 2:
            return 1.25
        values = [
            self.project(team, opponent, is_home).expected_goals_for
            for team in ids for opponent in ids if team != opponent
            for is_home in (True, False)
        ]
        return float(np.mean(values))


_FACTORIAL_CACHE: dict[int, np.ndarray] = {}


def _factorials(goals: np.ndarray) -> np.ndarray:
    """Return factorials for a goal-count array, cached by length."""
    key = len(goals)
    if key not in _FACTORIAL_CACHE:
        _FACTORIAL_CACHE[key] = np.array([math.factorial(int(g)) for g in goals],
                                         dtype=float)
    return _FACTORIAL_CACHE[key]


def _tau(home_goals: int, away_goals: int, lambda_home: float,
         lambda_away: float, rho: float) -> float:
    """Dixon-Coles correction factor for a single scoreline.

    Adjusts only the four lowest scorelines; everything else is unchanged.
    """
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_home * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_away * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def finished_matches(fixtures: Sequence[dict]) -> list[dict]:
    """Return fixtures that have a final score recorded.

    A fixture is only usable once both scores are present. Note that FPL does
    not finalise a gameweek until 09:00 UK the morning after its last match
    (RULES.md §2.4), but scorelines themselves settle at full time, so this is
    safe to read as soon as ``finished`` is set.
    """
    return [
        f for f in fixtures
        if f.get("finished")
        and f.get("team_h_score") is not None
        and f.get("team_a_score") is not None
    ]


def fit(fixtures: Sequence[dict], team_ids: Sequence[int],
        now: datetime | None = None,
        decay_per_day: float = TIME_DECAY_PER_DAY) -> DixonColesFit | None:
    """Fit the model to finished fixtures by penalised maximum likelihood.

    Args:
        fixtures: The full fixture list; unfinished fixtures are ignored.
        team_ids: Every club in the league, so clubs yet to play still get
            parameters (at league average).
        now: Reference time for match-age decay. Defaults to the current time.
        decay_per_day: Exponential decay rate per day of match age.

    Returns:
        A :class:`DixonColesFit`, or ``None`` if there are fewer than
        :data:`MIN_MATCHES_TO_FIT` finished matches — pre-season, and for the
        opening gameweek, that is the expected outcome and callers must fall
        back to fixture difficulty.
    """
    from scipy.optimize import minimize

    matches = finished_matches(fixtures)
    if len(matches) < MIN_MATCHES_TO_FIT:
        log.info("Only %d finished matches; need %d to fit Dixon-Coles.",
                 len(matches), MIN_MATCHES_TO_FIT)
        return None

    ids = list(team_ids)
    index = {team_id: i for i, team_id in enumerate(ids)}
    n = len(ids)
    now = now or datetime.now(timezone.utc)

    home_idx, away_idx, home_goals, away_goals, weights = [], [], [], [], []
    for match in matches:
        if match["team_h"] not in index or match["team_a"] not in index:
            continue
        home_idx.append(index[match["team_h"]])
        away_idx.append(index[match["team_a"]])
        home_goals.append(int(match["team_h_score"]))
        away_goals.append(int(match["team_a_score"]))
        weights.append(_match_weight(match, now, decay_per_day))

    home_idx = np.array(home_idx)
    away_idx = np.array(away_idx)
    home_goals = np.array(home_goals, dtype=float)
    away_goals = np.array(away_goals, dtype=float)
    weights = np.array(weights, dtype=float)

    played = np.zeros(n)
    for i in home_idx:
        played[i] += 1
    for i in away_idx:
        played[i] += 1
    matches_per_team = float(played.mean()) if n else 0.0

    def negative_log_likelihood(params: np.ndarray) -> float:
        attack = params[:n]
        defence = params[n:2 * n]
        home_advantage = params[2 * n]
        rho = params[2 * n + 1]

        # Centre attack for identifiability.
        attack = attack - attack.mean()

        lambda_home = np.exp(attack[home_idx] + defence[away_idx] + home_advantage)
        lambda_away = np.exp(attack[away_idx] + defence[home_idx])
        lambda_home = np.clip(lambda_home, 1e-6, 12.0)
        lambda_away = np.clip(lambda_away, 1e-6, 12.0)

        log_likelihood = (
            home_goals * np.log(lambda_home) - lambda_home
            + away_goals * np.log(lambda_away) - lambda_away
        )

        # tau applies only where both sides scored 0 or 1.
        low = (home_goals <= 1) & (away_goals <= 1)
        if low.any():
            tau = np.array([
                _tau(int(h), int(a), lh, la, rho)
                for h, a, lh, la in zip(home_goals[low], away_goals[low],
                                        lambda_home[low], lambda_away[low])
            ])
            tau = np.clip(tau, 1e-6, None)
            log_likelihood[low] += np.log(tau)

        penalty = (REGULARISATION_STRENGTH * (np.sum(attack ** 2) + np.sum(defence ** 2))
                   + RHO_REGULARISATION * rho ** 2)
        return -float(np.sum(weights * log_likelihood)) + penalty

    initial = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05]])
    bounds = ([(-2.0, 2.0)] * n + [(-2.0, 2.0)] * n
              + [(-1.0, 1.0), RHO_BOUNDS])

    result = minimize(negative_log_likelihood, initial, method="L-BFGS-B",
                      bounds=bounds)

    params = result.x
    attack = params[:n] - params[:n].mean()
    defence = params[n:2 * n]

    strengths = {
        team_id: TeamStrength(team_id, float(attack[i]), float(defence[i]),
                              int(played[i]))
        for team_id, i in index.items()
    }

    if not result.success:
        log.warning("Dixon-Coles optimiser did not converge cleanly: %s",
                    result.message)

    return DixonColesFit(
        strengths=strengths,
        home_advantage=float(params[2 * n]),
        rho=float(params[2 * n + 1]),
        matches_used=len(matches),
        matches_per_team=matches_per_team,
        converged=bool(result.success),
    )


def _match_weight(match: dict, now: datetime, decay_per_day: float) -> float:
    """Return the time-decay weight for one match."""
    kickoff = match.get("kickoff_time")
    if not kickoff:
        return 1.0
    try:
        played_at = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
    except ValueError:
        return 1.0
    age_days = max(0.0, (now - played_at).total_seconds() / 86400.0)
    return math.exp(-decay_per_day * age_days)


def blend_weight(matches_per_team: float) -> float:
    """How far to trust the fitted model against a fixture-difficulty baseline.

    Returns 0.0 with no matches played, rising toward 1.0 as evidence
    accumulates and passing 0.5 at :data:`BLEND_HALF_LIFE_MATCHES`.
    """
    if matches_per_team <= 0:
        return 0.0
    return matches_per_team / (matches_per_team + BLEND_HALF_LIFE_MATCHES)
