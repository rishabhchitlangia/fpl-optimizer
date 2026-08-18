# FPL Team Optimizer

A Fantasy Premier League squad optimizer for the **2026/27** season. Fetches
live data from the public FPL API, projects points for every player, and solves
for the optimal squad using linear programming — then tells you which transfers
are worth making and whether a chip is worth playing.

Built against researched, cited rules — see **[RULES.md](RULES.md)**, which is
the source of truth for every constraint in the code.

---

## Quick start

```bash
git clone <your-repo-url> fpl-optimizer
cd fpl-optimizer

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py
```

First run fetches ~590 player histories and takes a minute or two. Everything is
cached to `data/cache/`, so subsequent runs are near-instant.

---

## Plugging in your team ID

Your team ID is the number in your FPL URL. Log in at
[fantasy.premierleague.com](https://fantasy.premierleague.com), click
**Pick Team** or **Points**, and read the address bar:

```
https://fantasy.premierleague.com/entry/1234567/event/1
                                        ^^^^^^^
                                        this is your team ID
```

Then either pass it per-run:

```bash
python main.py --team-id 1234567
```

…or set it once so you never have to type it again:

```bash
export FPL_TEAM_ID=1234567
python main.py
```

Without a team ID the tool still builds the optimal squad from scratch; it just
skips the transfer suggestions, since it has nothing to compare against.

> **Note:** squad data only exists once a gameweek has been played. Before the
> season starts, the API returns 404 for your picks and the tool will say so and
> carry on.

---

## What it does

```
python main.py --team-id 1234567 --horizon 5 --free-transfers 2
```

1. **Fetches** players, fixtures, and per-player multi-season history (cached).
2. **Projects** points for every player over your chosen horizon.
3. **Optimises** the best possible 15-man squad under budget, position and
   3-per-club constraints.
4. **Compares** it against your actual squad and ranks transfer plans by points
   gained *net of the −4 hit* per extra transfer.
5. **Flags** any chip worth considering this gameweek.

Output is a set of tables: optimal squad, your squad, a transfer ladder, and
chip verdicts.

### The transfer ladder

The most useful output. For each number of transfers, it shows the best plan and
what it nets you after hits:

```
  # │ Moves                                 │  Gross │  Hit │    Net
  1 │ Obi → Haaland                         │  +3.66 │   —  │  +3.66
  2 │ Stach → B.Fernandes, Obi → Haaland    │  +5.75 │  -4  │  +1.75
  3 │ …                                     │  +5.86 │  -8  │  -2.14
  0 │ No transfers (roll)                   │  +0.00 │   —  │  +0.00
```

Here the second transfer gains 2.09 gross but costs 4 — so one transfer wins.
That comparison is the whole point.

---

## Options

| Flag | Default | What it does |
|---|---|---|
| `--team-id` | `$FPL_TEAM_ID` | Your FPL team ID |
| `--gameweek` | next GW | Gameweek to optimise for |
| `--horizon` | `1` | Gameweeks to project over. Longer favours good fixture runs |
| `--model` | `auto` | `auto`, `bayes`, `ep_next`, or `blended` |
| `--free-transfers` | `1` | Free transfers you have (1–5) |
| `--max-transfers` | `4` | Largest transfer plan to evaluate |
| `--budget` | `100.0` | Override squad budget, in millions |
| `--lock` | — | Force a player in. Name or ID. Repeatable |
| `--ban` | — | Exclude a player. Repeatable |
| `--max-ownership` | — | Only pick players below this ownership %. Differential mode |
| `--min-availability` | `0.0` | Drop players below this fitness. `1.0` avoids all doubts |
| `--refresh` | off | Force re-fetch of all cached data |
| `--no-history` | off | Skip the 590-request history sweep (weaker model) |

```bash
python main.py --lock Haaland --ban Gabriel --horizon 6
python main.py --refresh --min-availability 1.0
python main.py --max-ownership 15          # differential squad
```

The squad table's `Own%` column shows ownership and `SP` flags set-piece duty
(`P` penalties, `S` corners/free-kicks), so you can see at a glance how close to
the template a suggestion sits.

`--max-ownership` builds a deliberately differential squad. It costs expected
points by construction — it discards good players purely for being popular — so
it is a rank-variance tool, not a better squad. Use it when you need to catch up,
not when you are ahead.

---

## How the projection model works

**The problem:** FPL's own `ep_next` field saturates at 4.0. Pre-season, Haaland
(£15.5m) and Gabriel (£8.0m) both sit at the cap — it cannot rank the top of the
market at all. A squad built on it is legal but competitively worthless.

**The solution** (`scoring.py`, `BayesianRateScorer`):

```
points = availability
       × (projected_minutes / 90)
       × points_per_90
       × Σ over the club's fixtures: fixture_multiplier(difficulty)
```

- **`points_per_90`** is an empirical-Bayes estimate. A player's own
  recency-weighted historical rate (each season back counts half the previous
  one) is shrunk toward a **price-implied prior** in proportion to how little
  history they have. A four-season regular is trusted on their record; a
  half-season wonder is pulled toward what their price implies. Players with no
  Premier League history — promoted clubs, overseas signings — fall back
  entirely to the price prior, which is fitted per position by least squares.
- **Projected minutes** come from historical start rate, shrunk the same way,
  scaled by FPL's injury/availability flags.
- **Summing over actual fixtures** means blank gameweeks contribute zero and
  double gameweeks contribute both matches, automatically.

Two further signals matter most in the opening gameweeks, when history is
thinnest:

- **Set-piece duty.** Designated penalty takers get a points-per-90 premium
  derived from the base rate (the league awards ~0.13 penalties per team per
  match at ~78% conversion). Crucially this is scaled by how much the model is
  leaning on the prior: an established taker's past returns *already contain*
  their penalties, so they get almost none of it, while a new signing just
  handed the duty gets it in full. Corner and free-kick takers get a smaller
  creative premium.
- **Ownership**, applied as a **floor on expected minutes and never to the
  points rate.** Both restrictions are deliberate. High ownership is strong
  evidence a player is nailed — millions of managers have collectively checked
  the team news — but low ownership is weak evidence of anything, since
  differentials exist by definition. And feeding ownership into the *points*
  rate would just reproduce the crowd's opinion of quality and drag every squad
  toward the template. Into minutes, it captures the part the crowd genuinely
  knows: who starts.

Three models ship, and `auto` picks between them by season stage: pure
historical pre-season, then blending current-season evidence in as gameweeks
accumulate (equal weight at roughly GW6).

### Swapping in your own model

Subclass `PlayerScorer` and implement one method:

```python
from fpl.scoring import PlayerScorer, Projection

class MyScorer(PlayerScorer):
    name = "my-model"

    def project(self, gameweeks):
        return {player_id: Projection(...) for ...}
```

Everything downstream consumes `Projection` objects and neither knows nor cares
which model produced them.

### Known limits

Stated plainly, because they bound how far to trust the output:

- Points-per-90 is treated as linear in minutes; appearance points are not
  (1 pt under 60 minutes, 2 at 60+), so cameo players are mildly overrated.
- Seasons before 2025/26 predate defensive contribution points and understate
  defenders and defensive midfielders. Recency weighting mitigates this.
- Bonus points are inherited through historical totals, not modelled from BPS —
  BPS prediction needs pre-match Opta data the free API doesn't expose.
- Selling prices default to current price. FPL's 50% sell-on fee means this
  slightly **overstates** your available funds; purchase prices are only on the
  authenticated endpoint, which this project doesn't use.
- The model does not know about managerial changes, tactical shifts, or the
  transfer window beyond what price already reflects.
- An established player who has only *just* inherited penalties gets a smaller
  set-piece bump than they deserve: historical set-piece orders are not exposed
  by the API, so a change of duty cannot be detected.

---

## Why the optimizer optimises the XI, not the squad

Only 11 players score in a normal gameweek. Maximising the sum over all 15 will
happily spend £30m on a strong bench. So the MILP carries **two binaries per
player** — "in squad" and "in starting XI" — and weights bench contribution down
to 12%. Captaincy is a third binary in the same program, so it's traded off
against selection rather than bolted on afterward.

All constraints are read from the game's own config at runtime, not hard-coded,
so they stay correct if FPL changes them.

---

## Project layout

```
fpl/
  data.py         API client: fetching, caching, retries, concurrency
  scoring.py      Predicted-points models (pluggable)
  optimizer.py    MILP squad selection and starting-XI picker
  transfers.py    Transfer plan generation and ranking, net of hits
  chips.py        Chip heuristics and blank/double gameweek detection
main.py           CLI
tests/            45 tests, run against live cached data
RULES.md          Researched 2026/27 rules, with citations
data/cache/       Cached API responses (gitignored)
```

---

## Running the tests

```bash
python -m unittest discover -s tests -v
```

45 tests covering the sell-on fee, formation enumeration, squad legality
(size, positions, budget, 3-per-club), captain selection, transfer hit
arithmetic, and chip availability windows. They run against live cached data,
so they double as an assertion that the API still matches RULES.md — if FPL
changes a rule mid-season, `test_game_rules_match_rules_md` fails.

---

## Data source

All data comes from the free, unauthenticated public FPL API:

| Endpoint | Used for |
|---|---|
| `/api/bootstrap-static/` | Players, clubs, positions, gameweeks, chips, rules |
| `/api/fixtures/` | Fixture difficulty, blank/double detection |
| `/api/element-summary/{id}/` | Multi-season player history |
| `/api/entry/{team_id}/event/{gw}/picks/` | Your squad |

No API key, no login. Be reasonable with `--refresh`; the history sweep is 590
requests and is throttled to 8 concurrent.

---

## Caveat

Projections are estimates, not predictions. This tool is a decision aid — it
will not tell you that a manager just resigned or that a player picked up a
knock in training an hour ago. Use it alongside the news, not instead of it.
