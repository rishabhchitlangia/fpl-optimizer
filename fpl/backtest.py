"""Point-in-time evaluation of the projection model.

The model was built and tuned without ever being checked against outcomes.
This module closes that gap: it rewinds the data to how it looked at the end
of some gameweek, projects forward from there, and compares the projection
with what actually happened.

Rewinding matters. A scorer handed today's payloads sees results and minutes
from the very gameweeks it is asked to predict, and will look far better than
it is. :func:`rewind` strips both: per-player history after the cutoff, and
the scores of fixtures played after it, so the fitted results model sees only
matches that had finished at the time.

The player list leaks too, less obviously. ``bootstrap-static`` carries today's
price, today's ownership and today's injury flags, and all three have absorbed
what happened after the cutoff: a player who scored heavily since has risen in
price and been bought, and one injured last week is flagged as unavailable for
matches he in fact played. :func:`rewind_bootstrap` restores price and
ownership from the per-match record, which keeps both, and clears availability
flags, which cannot be recovered. ``ep_next`` cannot be recovered either, so
the ``ep_next`` and blended models cannot be evaluated here at all; an earlier
run that tried reported a correlation of 0.78, all of it hindsight.

Metrics
-------
Correlation and mean absolute error describe the fit over the whole player
pool, but neither answers the question a manager actually has, which is
whether the players the model likes go on to score. ``top_n`` does: it is the
mean actual points per gameweek of the players the model ranked highest. The
number to compare it against is ``pool_mean``, the same figure over everyone
eligible. The gap between them is the model's edge, in points per player per
gameweek, and it is the only metric here worth tuning against.

Baselines are not optional. A model that cannot beat "rank players by the
points they have already scored" is not earning its complexity, and until
this module existed there was no way to know which side of that line it fell.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import Callable, Sequence

from fpl import scoring

log = logging.getLogger(__name__)

#: A player needs this share of the available minutes before the cutoff to
#: count as pickable. Without a floor the pool fills with academy players who
#: never appear, which flatters every ranking equally and hides real
#: differences between them.
MIN_MINUTES_SHARE = 0.5


def rewind(summaries: dict[int, dict], fixtures: Sequence[dict],
           cutoff: int) -> tuple[dict[int, dict], list[dict]]:
    """Return the data as it stood once gameweek ``cutoff`` had finished.

    Args:
        summaries: ``element-summary`` payloads keyed by player ID.
        fixtures: The full fixture list.
        cutoff: The last gameweek whose results are known.

    Returns:
        Truncated ``(summaries, fixtures)``. Fixtures after the cutoff keep
        their difficulty and kickoff but lose their scores and finished flag,
        so :mod:`fpl.dixon_coles` fits on the right matches.
    """
    rewound_summaries: dict[int, dict] = {}
    for pid, summary in summaries.items():
        copy = dict(summary)
        copy["history"] = [m for m in (summary.get("history") or [])
                           if (m.get("round") or 0) <= cutoff]
        rewound_summaries[pid] = copy

    rewound_fixtures: list[dict] = []
    for fixture in fixtures:
        copy = dict(fixture)
        if (fixture.get("event") or 10 ** 6) > cutoff:
            copy["finished"] = False
            copy["finished_provisional"] = False
            copy["team_h_score"] = None
            copy["team_a_score"] = None
        rewound_fixtures.append(copy)

    return rewound_summaries, rewound_fixtures


def rewind_bootstrap(bootstrap: dict, summaries: dict[int, dict],
                     cutoff: int) -> dict:
    """Return the player list as it stood once gameweek ``cutoff`` had finished.

    Args:
        bootstrap: Today's ``bootstrap-static`` payload.
        summaries: Full ``element-summary`` payloads keyed by player ID.
        cutoff: The last gameweek whose results are known.

    Returns:
        A copy in which each player's ``now_cost`` and ``selected_by_percent``
        come from his last match on or before the cutoff, every player is
        marked available, and ``ep_next`` is blanked. Players with no match by
        the cutoff keep today's price, the best estimate available.
    """
    managers = float(bootstrap.get("total_players") or 0)
    elements: list[dict] = []
    for element in bootstrap.get("elements", []):
        copy = dict(element)
        played = [m for m in ((summaries.get(element["id"]) or {}).get("history") or [])
                  if (m.get("round") or 0) <= cutoff]
        if played:
            last = played[-1]
            if last.get("value"):
                copy["now_cost"] = last["value"]
            if managers and last.get("selected") is not None:
                copy["selected_by_percent"] = str(
                    round(100.0 * float(last["selected"]) / managers, 1))
        copy["status"] = "a"
        copy["news"] = ""
        copy["chance_of_playing_next_round"] = None
        copy["chance_of_playing_this_round"] = None
        copy["ep_next"] = "0.0"
        elements.append(copy)

    rewound = dict(bootstrap)
    rewound["elements"] = elements
    return rewound


def actual_points(summary: dict, gameweeks: Sequence[int]) -> float:
    """Points a player actually scored across ``gameweeks``."""
    wanted = set(gameweeks)
    return float(sum(m.get("total_points") or 0
                     for m in (summary.get("history") or [])
                     if (m.get("round") or 0) in wanted))


def eligible(summary: dict, cutoff: int,
             min_share: float = MIN_MINUTES_SHARE) -> bool:
    """Whether a player had played enough by ``cutoff`` to be pickable."""
    if cutoff <= 0:
        return False
    played = sum(m.get("minutes") or 0 for m in (summary.get("history") or [])
                 if (m.get("round") or 0) <= cutoff)
    return played >= 90.0 * cutoff * min_share


@dataclass
class Result:
    """One walk-forward window's scores.

    Attributes:
        cutoff: Last gameweek the ranking could see.
        targets: Gameweeks predicted.
        n: Players in the pool.
        correlation: Pearson correlation of prediction against outcome.
        rank_correlation: Spearman correlation, which is what selection
            actually depends on — only the order matters when picking a squad.
        mae: Mean absolute error over the window, in points.
        top_n: Mean actual points per gameweek of the top-ranked players.
        pool_mean: Mean actual points per gameweek over the whole pool.
        slope: Least-squares slope of outcome on prediction. 1.0 means a
            predicted gap between two players is worth what it says; below 1
            the model is over-dispersed and its gaps are exaggerated. Only
            meaningful for predictors on the points scale -- a baseline that
            ranks by minutes has a slope with no interpretation.
        top_decile_excess: Mean prediction minus mean outcome, per gameweek,
            over the highest-predicted tenth of the pool. Positive means the
            players the model likes most are being over-sold.
    """

    cutoff: int
    targets: list[int]
    n: int
    correlation: float
    rank_correlation: float
    mae: float
    top_n: float
    pool_mean: float
    slope: float = 0.0
    top_decile_excess: float = 0.0

    @property
    def edge(self) -> float:
        """Points per player per gameweek over an average pool player."""
        return self.top_n - self.pool_mean


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else 0.0


def _ranks(values: Sequence[float]) -> list[float]:
    """Fractional ranks, averaging ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return _pearson(_ranks(xs), _ranks(ys))


def evaluate(predictions: dict[int, float], summaries: dict[int, dict],
             cutoff: int, targets: Sequence[int],
             top_n: int = 20, min_share: float = MIN_MINUTES_SHARE) -> Result | None:
    """Score one ranking against what the target gameweeks actually produced.

    Args:
        predictions: Predicted points over ``targets``, keyed by player ID.
            Only the ordering and scale matter; any ranking can be scored.
        summaries: Full (un-rewound) summaries, used for outcomes.
        cutoff: Last gameweek the prediction could see.
        targets: Gameweeks being predicted.
        top_n: How many of the highest-ranked players to score.
        min_share: Minutes-share floor for pool membership.

    Returns:
        A :class:`Result`, or ``None`` if fewer than ``top_n`` players qualify.
    """
    rows: list[tuple[float, float]] = []
    for pid, summary in summaries.items():
        if pid not in predictions or not eligible(summary, cutoff, min_share):
            continue
        rows.append((predictions[pid], actual_points(summary, targets)))

    if len(rows) < max(top_n, 30):
        return None

    predicted = [r[0] for r in rows]
    outcome = [r[1] for r in rows]
    weeks = len(targets)
    ordered = sorted(rows, key=lambda r: -r[0])
    best = ordered[:top_n]

    mean_p, mean_a = statistics.mean(predicted), statistics.mean(outcome)
    spread = sum((p - mean_p) ** 2 for p in predicted)
    slope = (sum((p - mean_p) * (a - mean_a) for p, a in rows) / spread
             if spread else 0.0)
    decile = ordered[:max(1, len(ordered) // 10)]
    excess = (statistics.mean(p for p, _ in decile)
              - statistics.mean(a for _, a in decile)) / weeks

    return Result(
        cutoff=cutoff,
        targets=list(targets),
        n=len(rows),
        correlation=_pearson(predicted, outcome),
        rank_correlation=_spearman(predicted, outcome),
        mae=statistics.mean(abs(p - a) for p, a in rows),
        top_n=statistics.mean(a for _, a in best) / weeks,
        pool_mean=statistics.mean(outcome) / weeks,
        slope=slope,
        top_decile_excess=excess,
    )


def walk_forward(bootstrap: dict, fixtures: Sequence[dict],
                 summaries: dict[int, dict],
                 predictor: Callable[[dict, list[dict], dict[int, dict], list[int]],
                                     dict[int, float]],
                 cutoffs: Sequence[int], horizon: int = 3,
                 last_gameweek: int | None = None,
                 top_n: int = 20) -> list[Result]:
    """Evaluate a predictor at each cutoff, predicting the next ``horizon`` weeks.

    Args:
        bootstrap: ``bootstrap-static`` payload.
        fixtures: Full fixture list.
        summaries: Full ``element-summary`` payloads.
        predictor: Called with ``(rewound_bootstrap, rewound_fixtures,
            rewound_summaries, targets)`` and returning predicted points per
            player ID.
        cutoffs: Gameweeks to rewind to.
        horizon: How many gameweeks ahead to predict.
        last_gameweek: Highest gameweek with results. Inferred when omitted.
        top_n: Size of the top bucket.

    Returns:
        One :class:`Result` per cutoff that produced a scorable window.
    """
    if last_gameweek is None:
        last_gameweek = max((m.get("round") or 0)
                            for s in summaries.values()
                            for m in (s.get("history") or [])) if summaries else 0

    results: list[Result] = []
    for cutoff in cutoffs:
        targets = list(range(cutoff + 1, min(cutoff + 1 + horizon, last_gameweek + 1)))
        if not targets:
            continue
        rewound_summaries, rewound_fixtures = rewind(summaries, fixtures, cutoff)
        predictions = predictor(rewind_bootstrap(bootstrap, summaries, cutoff),
                                rewound_fixtures, rewound_summaries, targets)
        result = evaluate(predictions, summaries, cutoff, targets, top_n=top_n)
        if result:
            results.append(result)
    return results


def model_predictor(model: str = "auto") -> Callable:
    """A predictor backed by :func:`fpl.scoring.build_scorer`.

    Only the historical model can be evaluated. ``ep_next`` is blanked by
    :func:`rewind_bootstrap` because its past values are not recoverable.
    """
    if model in ("ep_next", "blended"):
        raise ValueError(
            f"{model!r} cannot be backtested: ep_next is only published for "
            "the coming gameweek, so any past evaluation would read hindsight.")

    def predict(bootstrap, fixtures, summaries, targets):
        scorer = scoring.build_scorer(bootstrap, fixtures, summaries, model)
        return {pid: p.expected_points
                for pid, p in scorer.project(targets).items()}
    return predict


def _history_sum(summary: dict, field: str) -> float:
    return float(sum(m.get(field) or 0 for m in (summary.get("history") or [])))


def baseline_predictor(kind: str) -> Callable:
    """A trivial ranking, for the model to be measured against.

    ``total``   — points scored so far. The one to beat.
    ``form``    — points per game over the last four gameweeks.
    ``minutes`` — minutes played; tests how much of the signal is just availability.
    ``price``   — list price at the cutoff, i.e. what the game itself thought
                  a player was worth.
    """
    def predict(bootstrap, fixtures, summaries, targets):
        players = {e["id"]: e for e in bootstrap["elements"]}
        out: dict[int, float] = {}
        for pid, summary in summaries.items():
            player = players.get(pid, {})
            history = summary.get("history") or []
            if kind == "total":
                value = _history_sum(summary, "total_points")
            elif kind == "form":
                recent = history[-4:]
                value = (sum(m.get("total_points") or 0 for m in recent)
                         / len(recent)) if recent else 0.0
            elif kind == "minutes":
                value = _history_sum(summary, "minutes")
            elif kind == "price":
                value = float(player.get("now_cost") or 0)
            else:
                raise ValueError(f"Unknown baseline {kind!r}")
            out[pid] = value * len(targets)
        return out
    return predict


def summarise(results: Sequence[Result]) -> dict[str, float]:
    """Average a walk-forward run's windows into one row of headline figures."""
    if not results:
        return {}
    return {
        "rank": statistics.mean(r.rank_correlation for r in results),
        "edge": statistics.mean(r.edge for r in results),
        "slope": statistics.mean(r.slope for r in results),
        "top_decile_excess": statistics.mean(r.top_decile_excess for r in results),
        "mae": statistics.mean(r.mae for r in results),
        "windows": float(len(results)),
    }


def main() -> int:
    """Print the model against its baselines: ``python -m fpl.backtest``."""
    from fpl import data

    bootstrap = data.get_bootstrap()
    fixtures = data.get_fixtures()
    summaries = data.get_all_element_summaries(
        [e["id"] for e in bootstrap["elements"]])
    played = max(((m.get("round") or 0) for s in summaries.values()
                  for m in (s.get("history") or [])), default=0)
    if played < 5:
        print("Not enough of the season has been played to backtest.")
        return 1

    for horizon in (1, 3):
        cutoffs = list(range(4, played - horizon + 1))
        print(f"\nHorizon {horizon} gameweek(s), cutoffs GW{cutoffs[0]}-GW{cutoffs[-1]}")
        print(f"  {'predictor':18s} {'rank':>6} {'edge':>6} {'slope':>6} {'top-10% excess':>15}")
        rows = [(f"baseline: {kind}", baseline_predictor(kind), False)
                for kind in ("total", "form", "minutes", "price")]
        rows.append(("the model", model_predictor("bayes"), True))
        for label, predictor, on_points_scale in rows:
            summary = summarise(walk_forward(bootstrap, fixtures, summaries,
                                             predictor, cutoffs, horizon=horizon))
            if not summary:
                continue
            calibration = (f"{summary['slope']:6.2f} {summary['top_decile_excess']:+15.2f}"
                           if on_points_scale else f"{'-':>6} {'-':>15}")
            print(f"  {label:18s} {summary['rank']:6.3f} {summary['edge']:+6.2f} {calibration}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
