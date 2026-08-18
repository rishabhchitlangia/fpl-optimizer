"""Fetching and local caching of Fantasy Premier League API data.

The FPL API is public and unauthenticated. This module is the only place in the
project that talks to it; everything downstream consumes plain dicts returned
from here.

Endpoints used
--------------
``bootstrap-static``
    Players (``elements``), clubs (``teams``), positions (``element_types``),
    gameweeks (``events``), chips, and the game's own rule/scoring config.
``fixtures``
    All 380 fixtures with per-side difficulty ratings.
``element-summary/{player_id}``
    Per-player match history for the current season plus **full totals for every
    previous season** (``history_past``). One request per player, so it is
    fetched concurrently and cached hard.
``entry/{team_id}/event/{gw}/picks``
    A manager's squad for a given gameweek.

Caching
-------
Responses are written to ``data/cache`` as JSON alongside a fetch timestamp.
Reads are served from cache while fresh; see :func:`get_bootstrap` and friends
for the per-endpoint TTLs. Pass ``refresh=True`` (or call :func:`refresh_all`)
to force a re-fetch.

Why TTLs differ: fixtures and player prices move daily, whereas a completed
season's history never changes.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://fantasy.premierleague.com/api"

#: Project root, i.e. the directory containing ``fpl/``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# Cache lifetimes. Prices change at 00:00 UK daily, so a few hours is plenty
# for bootstrap; fixtures are rescheduled rarely; per-player season history is
# effectively immutable once a season ends.
TTL_BOOTSTRAP = timedelta(hours=6)
TTL_FIXTURES = timedelta(hours=12)
TTL_ELEMENT_SUMMARY = timedelta(days=1)
TTL_PICKS = timedelta(hours=1)

REQUEST_TIMEOUT = 30
MAX_RETRIES = 4
#: Concurrency for the 590-request element-summary sweep. Kept modest to stay
#: polite to a free public API — this is not a service we pay for.
MAX_WORKERS = 8


class FPLDataError(RuntimeError):
    """Raised when the FPL API cannot be reached or returns unusable data."""


# --------------------------------------------------------------------------- #
# Low-level HTTP + cache primitives
# --------------------------------------------------------------------------- #


def _cache_path(key: str) -> Path:
    """Return the on-disk cache location for a given cache key."""
    return CACHE_DIR / f"{key}.json"


def _read_cache(key: str, ttl: timedelta) -> Any | None:
    """Return cached payload for ``key`` if present and younger than ``ttl``.

    Returns ``None`` on a miss, an expired entry, or an unreadable file. A
    corrupt cache file is never fatal — we just re-fetch.
    """
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with path.open() as fh:
            envelope = json.load(fh)
        fetched_at = datetime.fromisoformat(envelope["fetched_at"])
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        log.debug("Discarding unreadable cache entry %s", path)
        return None

    if datetime.now(timezone.utc) - fetched_at > ttl:
        return None
    return envelope["payload"]


def _write_cache(key: str, payload: Any) -> None:
    """Persist ``payload`` under ``key`` with a UTC fetch timestamp.

    Written via a temporary file and atomically renamed so a crash mid-write
    cannot leave a truncated cache entry behind.
    """
    envelope = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
    path = _cache_path(key)
    # Keys may contain '/' (e.g. "element-summary/123"), so create the full
    # parent chain rather than just CACHE_DIR.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as fh:
        json.dump(envelope, fh)
    tmp.replace(path)


def cache_age(key: str) -> timedelta | None:
    """Return how long ago ``key`` was fetched, or ``None`` if not cached."""
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with path.open() as fh:
            fetched_at = datetime.fromisoformat(json.load(fh)["fetched_at"])
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        return None
    return datetime.now(timezone.utc) - fetched_at


def _get(endpoint: str, session: requests.Session | None = None) -> Any:
    """GET ``endpoint`` from the FPL API with retries and exponential backoff.

    Raises:
        FPLDataError: if all retries fail or the response is not valid JSON.
    """
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    caller = session or requests
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = caller.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                backoff = 2 ** attempt
                log.debug("GET %s failed (%s); retrying in %ss", url, exc, backoff)
                time.sleep(backoff)

    raise FPLDataError(f"Failed to fetch {url} after {MAX_RETRIES} attempts: {last_error}")


def _fetch_cached(key: str, endpoint: str, ttl: timedelta, refresh: bool) -> Any:
    """Return ``endpoint``'s payload, served from cache unless stale/``refresh``."""
    if not refresh:
        cached = _read_cache(key, ttl)
        if cached is not None:
            return cached
    payload = _get(endpoint)
    _write_cache(key, payload)
    return payload


# --------------------------------------------------------------------------- #
# Public endpoint accessors
# --------------------------------------------------------------------------- #


def get_bootstrap(refresh: bool = False) -> dict:
    """Return the ``bootstrap-static`` payload.

    Contains ``elements`` (players), ``teams``, ``element_types`` (positions),
    ``events`` (gameweeks), ``chips`` and ``game_config``.
    """
    return _fetch_cached("bootstrap-static", "bootstrap-static/", TTL_BOOTSTRAP, refresh)


def get_fixtures(refresh: bool = False) -> list[dict]:
    """Return all fixtures for the season, including difficulty ratings."""
    return _fetch_cached("fixtures", "fixtures/", TTL_FIXTURES, refresh)


def get_element_summary(player_id: int, refresh: bool = False,
                        session: requests.Session | None = None) -> dict:
    """Return one player's history, including ``history_past`` for prior seasons."""
    key = f"element-summary/{player_id}"
    if not refresh:
        cached = _read_cache(key, TTL_ELEMENT_SUMMARY)
        if cached is not None:
            return cached
    payload = _get(f"element-summary/{player_id}/", session=session)
    _write_cache(key, payload)
    return payload


def get_all_element_summaries(player_ids: Iterable[int], refresh: bool = False,
                              progress_callback=None) -> dict[int, dict]:
    """Fetch element summaries for many players concurrently.

    This is the expensive call in the project — roughly 590 requests on a cold
    cache, a handful on a warm one. Individual failures are logged and skipped
    rather than aborting the sweep, since the scoring model degrades gracefully
    when a player has no history.

    Args:
        player_ids: Player IDs to fetch.
        refresh: Bypass the cache and re-fetch everything.
        progress_callback: Optional ``callable(done, total)`` invoked as results
            arrive, for CLI progress reporting.

    Returns:
        Mapping of player ID to summary payload, omitting any that failed.
    """
    ids = list(player_ids)
    summaries: dict[int, dict] = {}

    # Serve warm-cache entries synchronously so a fully warm cache costs no
    # threads and no network at all.
    outstanding: list[int] = []
    if not refresh:
        for pid in ids:
            cached = _read_cache(f"element-summary/{pid}", TTL_ELEMENT_SUMMARY)
            if cached is not None:
                summaries[pid] = cached
            else:
                outstanding.append(pid)
    else:
        outstanding = ids

    if progress_callback:
        progress_callback(len(summaries), len(ids))

    if outstanding:
        with requests.Session() as session:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = {
                    pool.submit(get_element_summary, pid, refresh, session): pid
                    for pid in outstanding
                }
                for future in as_completed(futures):
                    pid = futures[future]
                    try:
                        summaries[pid] = future.result()
                    except (FPLDataError, OSError) as exc:
                        log.warning("Skipping player %s: %s", pid, exc)
                    if progress_callback:
                        progress_callback(len(summaries), len(ids))

    return summaries


def get_entry_picks(team_id: int, gameweek: int, refresh: bool = False) -> dict:
    """Return a manager's squad for a gameweek.

    Args:
        team_id: The numeric FPL team ID, from the ``/entry/<id>/`` URL.
        gameweek: Gameweek number, 1-38.

    Raises:
        FPLDataError: if the team ID is unknown or the gameweek has not started
            (the API returns 404 for a squad that does not exist yet).
    """
    key = f"entry-{team_id}-gw{gameweek}-picks"
    return _fetch_cached(key, f"entry/{team_id}/event/{gameweek}/picks/",
                         TTL_PICKS, refresh)


def get_entry(team_id: int, refresh: bool = False) -> dict:
    """Return a manager's entry metadata (name, overall rank, chips used)."""
    return _fetch_cached(f"entry-{team_id}", f"entry/{team_id}/", TTL_PICKS, refresh)


# --------------------------------------------------------------------------- #
# Convenience views over the raw payloads
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GameState:
    """Where the season currently stands.

    Attributes:
        current_gw: The gameweek in progress, or ``None`` before the season starts.
        next_gw: The next gameweek accepting transfers, or ``None`` once finished.
        finished_gws: How many gameweeks have been completed. Drives the choice
            of scoring model — see :mod:`fpl.scoring`.
        season_started: Whether any gameweek has been completed.
    """

    current_gw: int | None
    next_gw: int | None
    finished_gws: int
    season_started: bool


def get_game_state(bootstrap: dict) -> GameState:
    """Derive the current season state from a bootstrap payload."""
    events = bootstrap["events"]
    current = next((e["id"] for e in events if e.get("is_current")), None)
    upcoming = next((e["id"] for e in events if e.get("is_next")), None)
    finished = sum(1 for e in events if e.get("finished"))
    return GameState(
        current_gw=current,
        next_gw=upcoming,
        finished_gws=finished,
        season_started=finished > 0,
    )


def target_gameweek(bootstrap: dict) -> int:
    """Return the gameweek we should be optimising for.

    That is the next gameweek still accepting transfers; if the season is over
    we fall back to the final gameweek so the tool still runs.
    """
    state = get_game_state(bootstrap)
    if state.next_gw is not None:
        return state.next_gw
    if state.current_gw is not None:
        return state.current_gw
    return max(e["id"] for e in bootstrap["events"])


def team_lookup(bootstrap: dict) -> dict[int, dict]:
    """Return clubs keyed by team ID."""
    return {t["id"]: t for t in bootstrap["teams"]}


def player_lookup(bootstrap: dict) -> dict[int, dict]:
    """Return players keyed by player ID."""
    return {e["id"]: e for e in bootstrap["elements"]}


def position_lookup(bootstrap: dict) -> dict[int, dict]:
    """Return positions keyed by ``element_type`` ID."""
    return {t["id"]: t for t in bootstrap["element_types"]}


def refresh_all(include_player_history: bool = True, progress_callback=None) -> dict:
    """Force a re-fetch of every cached endpoint.

    Args:
        include_player_history: Also re-fetch all per-player summaries. This is
            ~590 requests and takes a minute or two; skip it if you only need
            fresh prices and fixtures.
        progress_callback: Forwarded to :func:`get_all_element_summaries`.

    Returns:
        A summary dict with counts of what was refreshed.
    """
    bootstrap = get_bootstrap(refresh=True)
    fixtures = get_fixtures(refresh=True)
    result = {
        "players": len(bootstrap["elements"]),
        "teams": len(bootstrap["teams"]),
        "fixtures": len(fixtures),
        "summaries": 0,
    }
    if include_player_history:
        ids = [e["id"] for e in bootstrap["elements"]]
        summaries = get_all_element_summaries(ids, refresh=True,
                                              progress_callback=progress_callback)
        result["summaries"] = len(summaries)
    return result
