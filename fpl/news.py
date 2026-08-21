"""Parsing FPL's player news into structured availability.

FPL publishes a short free-text status for flagged players — ``"Ankle injury -
Expected back 23 Aug"``, ``"Suspended until 29 Aug"``, ``"Groin injury -
Unknown return date"``. The structured fields alongside it (``status``,
``chance_of_playing_next_round``) describe only the *next* gameweek, so a model
that reads those alone treats every unavailable player as unavailable forever.

That is wrong in a way that matters for multi-gameweek planning. A player due
back in three days and one due back in three months are both ``status='i'`` with
``chance_of_playing_next_round = 0``, yet the first is a perfectly good pick for
the gameweek after next. This module recovers the return date so availability
can be evaluated per gameweek rather than once.

Only FPL's own text is parsed here. Press-conference reporting, beat writers and
the rest live outside the API and are not attempted — see the README for why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

#: Days after a return date during which a player is discounted for match
#: fitness. Players coming back from a lay-off are frequently benched, given
#: 20 minutes, or held back a week beyond the published date.
RETURN_RAMP_DAYS = 14.0

#: Availability multiplier on the day a player is due back, rising to 1.0 across
#: :data:`RETURN_RAMP_DAYS`.
RETURN_INITIAL_FACTOR = 0.55

#: How stale a news item may be before its "75% chance" style figure is treated
#: as unreliable. FPL updates flags frequently; a percentage that has not moved
#: in this long usually means nobody has revisited it.
STALE_NEWS_DAYS = 21.0

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

#: "Expected back 23 Aug", "Suspended until 6 Sep".
_RETURN_PATTERN = re.compile(
    r"(?:expected back|suspended until|out until)\s+(\d{1,2})\s+([A-Za-z]{3})",
    re.IGNORECASE,
)

_INDEFINITE = "unknown return date"


@dataclass(frozen=True)
class NewsFlag:
    """Structured form of a player's news string.

    Attributes:
        raw: The original text, for display.
        kind: ``injury``, ``suspension``, ``departed``, ``fitness`` or ``none``.
        return_date: When the player is expected available, if stated.
        indefinite: True when FPL says the return date is unknown — materially
            worse than a distant known date, because there is no point at which
            to start counting them back in.
        added: When the news was published.
    """

    raw: str = ""
    kind: str = "none"
    return_date: date | None = None
    indefinite: bool = False
    added: datetime | None = None

    @property
    def is_flagged(self) -> bool:
        """Whether there is any news at all."""
        return self.kind != "none"

    def age_days(self, now: datetime | None = None) -> float | None:
        """How long ago the news was published, in days."""
        if self.added is None:
            return None
        now = now or datetime.now(timezone.utc)
        return (now - self.added).total_seconds() / 86400.0

    def is_stale(self, now: datetime | None = None) -> bool:
        """Whether the item is old enough that its percentages are suspect."""
        age = self.age_days(now)
        return age is not None and age > STALE_NEWS_DAYS


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse FPL's ISO timestamps, which carry a ``Z`` suffix."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _resolve_year(day: int, month: int, anchor: datetime | None) -> date | None:
    """Turn a day and month with no year into a real date.

    FPL omits the year, so it has to be inferred. The news item's own publication
    timestamp is the natural anchor: a return date is always *after* the news
    announcing it, so pick the first year in which that holds. This handles the
    December-to-January rollover without any special-casing.
    """
    reference = (anchor or datetime.now(timezone.utc)).date()
    for year in (reference.year, reference.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue                      # e.g. 29 Feb in a non-leap year
        if candidate >= reference:
            return candidate
    return None


def parse_news(player: dict) -> NewsFlag:
    """Extract structured availability information from a player's news.

    Args:
        player: A bootstrap ``elements`` entry.

    Returns:
        A :class:`NewsFlag`. Unparseable or absent news yields ``kind="none"``,
        which callers treat as "no information", never as "available".
    """
    raw = (player.get("news") or "").strip()
    if not raw:
        return NewsFlag()

    added = _parse_timestamp(player.get("news_added"))
    lowered = raw.lower()

    if "has joined" in lowered or "has returned to" in lowered:
        kind = "departed"
    elif "suspend" in lowered:
        kind = "suspension"
    elif "lack of match fitness" in lowered:
        kind = "fitness"
    elif "injury" in lowered or "knock" in lowered:
        kind = "injury"
    else:
        kind = "other"

    return_date = None
    match = _RETURN_PATTERN.search(raw)
    if match:
        month = MONTHS.get(match.group(2).lower())
        if month:
            return_date = _resolve_year(int(match.group(1)), month, added)

    return NewsFlag(
        raw=raw,
        kind=kind,
        return_date=return_date,
        indefinite=_INDEFINITE in lowered,
        added=added,
    )


def availability_multiplier(flag: NewsFlag, when: datetime | None) -> float | None:
    """Return the availability implied by the news at a point in time.

    Args:
        flag: Parsed news.
        when: The moment being evaluated, normally a fixture's kick-off. If
            ``None``, no date reasoning is possible and the caller falls back to
            FPL's structured fields.

    Returns:
        A multiplier in ``[0.0, 1.0]``, or ``None`` when the news says nothing
        about this moment and the caller should decide by other means.

    A player is unavailable before their return date, then ramps back toward
    full availability across :data:`RETURN_RAMP_DAYS` rather than snapping to
    1.0 — returning players are routinely eased in, or miss the published date
    entirely.
    """
    if flag.kind == "departed":
        return 0.0
    if when is None or flag.return_date is None:
        return None

    fixture_day = when.date()
    if fixture_day < flag.return_date:
        return 0.0

    days_back = (fixture_day - flag.return_date).days
    if days_back >= RETURN_RAMP_DAYS:
        return 1.0
    progress = days_back / RETURN_RAMP_DAYS
    return RETURN_INITIAL_FACTOR + (1.0 - RETURN_INITIAL_FACTOR) * progress


def describe(flag: NewsFlag, when: datetime | None = None) -> str:
    """Return a short human-readable summary for the CLI and pitch view."""
    if not flag.is_flagged:
        return ""
    if flag.kind == "departed":
        return "left the league"
    if flag.indefinite:
        return "out, no return date"
    if flag.return_date:
        if when and when.date() < flag.return_date:
            return f"out until {flag.return_date:%d %b}"
        return f"back {flag.return_date:%d %b}"
    return flag.kind
