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

### Keeping data current during the transfer window

Squads change daily while the window is open — players leave, new signings are
added to the game, prices move. Two things keep you current:

**Cache expiry is automatic.** Prices and the player list refresh every 6 hours,
fixtures every 12, per-player season history daily. Just running the tool
normally picks up most changes.

**Force it when it matters** — before a deadline, or after a transfer you know
about:

```bash
python main.py --refresh
```

That re-fetches everything including all player histories, so it takes a minute
or two. For a quicker refresh of just prices and the player list, delete the one
cache file instead:

```bash
rm data/cache/bootstrap-static.json      # Windows: del data\cache\bootstrap-static.json
```

**Departed players are excluded automatically.** FPL marks anyone who has left
the league with `can_select: false`, and the model reads that directly, so they
project zero points and the optimizer will never suggest one. New signings
appear in the player list as soon as FPL adds them and are picked up on the next
refresh — with no Premier League history, they are projected from their price
and club strength until they have played (see the model section).

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
| `--serve` | — | Open an interactive pitch for swapping players (port 8000) |
| `--replace` | — | Swap a player out and re-optimise. Repeatable |
| `--html` | — | Also write an HTML pitch view to this path |
| `--refresh` | off | Force re-fetch of all cached data |
| `--no-history` | off | Skip the 590-request history sweep (weaker model) |

```bash
python main.py --lock Haaland --ban Gabriel --horizon 6
python main.py --refresh --min-availability 1.0
python main.py --max-ownership 15          # differential squad
```

### Editing the squad interactively

If you disagree with a pick, click it:

```bash
python main.py --serve
```

That opens a pitch in your browser where every player is clickable. Select the
ones you want gone, press **Replace**, and the squad is re-solved around your
choices — the players you kept stay put, the rejected ones are swapped for the
best legal alternatives, and a change list shows what moved and what it cost.

The click goes to a small local server that runs the **real optimizer**. That
round-trip is the point: a static page cannot run a mixed-integer solver, so a
purely client-side version could only offer precomputed swaps. Going through
the server means every suggestion is genuinely optimal under the same
constraints as the command line — one implementation of the logic, not two.

Rejections accumulate. Once you have said you don't want a player they stay out
until you press **Reset squad**, so the optimizer can't immediately re-sign
someone you just rejected.

The server binds to `127.0.0.1` only and is meant for local use while you plan a
gameweek. It is not hardened for exposure to a network.

For the same thing without a browser:

```bash
python main.py --replace Haaland --replace Raya
```

### Seeing the team on a pitch

Terminal tables are good for numbers but bad for *shape* — whether the defence
is three or five, which club you are stacked on, where the captaincy sits:

```bash
python main.py --html squad.html && open squad.html
```

That writes a self-contained HTML page laying the XI out in its actual
formation, with the bench below it. No build step, no server, works offline,
and it follows your system light/dark theme. Nothing to install.

There is no third-party site that takes this tool's output — to actually enter
the team you use [fantasy.premierleague.com](https://fantasy.premierleague.com)
and its Pick Team page.

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
- **Projected minutes** are worked in minutes-per-game space across every
  season of history, with three corrections that need more than one season to
  compute:
  - a **peak-capability floor**, because recency weighting *over*-weights a
    season lost to injury. A player with three full seasons and one blank is
    projected on the three, not the blank — the peak is discounted by age, so
    it fades without being erased;
  - **widened uncertainty for new signings** (detected via `team_join_date`),
    whose minutes were earned in a different squad. This pulls them toward the
    price-implied prior in whichever direction that lies: a proven starter who
    moves loses some certainty, an expensive signing who was a bit-part player
    elsewhere gains some;
  - **availability**, from FPL's injury and suspension flags.
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

  It is also **faded out at the cheap end**, because ownership means opposite
  things at opposite ends of the price range. A heavily-owned £12m midfielder is
  owned because managers expect him to start; a heavily-owned £4.0m keeper is
  bench fodder, held *because* he is cheap and often specifically so he never
  plays. Reading the second as evidence of nailed-on minutes put backup keepers
  in the starting XI until it was fixed.

Three models ship, and `auto` picks between them by season stage: pure
historical pre-season, then blending current-season evidence in as gameweeks
accumulate (equal weight at roughly GW6).

### The results model (Dixon-Coles)

Once enough of the season has been played, a simplified **Dixon-Coles** model is
fitted to actual scorelines and takes over from FPL's fixture-difficulty ratings.

Goals are Poisson with per-club attack and defence strengths plus a league-wide
home advantage:

```
lambda_home = exp(attack[home] + defence[away] + home_advantage)
lambda_away = exp(attack[away] + defence[home])
```

Dixon and Coles' addition over plain Poisson is the `tau` correction, which
reweights the four lowest scorelines (0-0, 1-0, 0-1, 1-1). That is not
incidental here — those are exactly the scorelines that decide clean sheets, so
it sharpens the quantity the model exists to produce. Matches are time-decayed
so recent form leads.

From the fit come the two outputs that matter:

- **Clean sheet probability**, read off the corrected score matrix (not
  `exp(-lambda)`, so it inherits the tau correction). Over a double gameweek it
  is the probability of a clean sheet in *at least one* fixture.
- **Expected goal involvement**, from the player's historical xG+xA per 90
  scaled by their club's expected goals in that specific fixture.

**It will not fit early, by design.** The model has 41 free parameters and a
gameweek supplies 20 goal observations, so an unpenalised fit after two
gameweeks produces nonsense — a club that wins 4-0 gets an attack strength
implying four goals every week. Three guards apply:

1. No fit at all below 20 finished matches; the CLI says so and falls back to
   fixture difficulty.
2. A fixed L2 penalty toward league average. Fixed rather than
   dataset-scaled, so it is self-correcting: it dominates a three-gameweek
   sample and fades over a season.
3. Even once fitted, output is blended against the fixture-difficulty baseline,
   reaching even weight at roughly six gameweeks.

**The double-counting guard.** The projection scales a player's *observed*
points-per-90, which already contains the clean sheets and goals they
historically earned. So the results model is used only to say how much better or
worse than average a given fixture is — its multipliers are normalised to 1.0
across the league — never to reconstruct points from scratch. There is a test
asserting exactly this.

Because the model cannot be fitted pre-season, it is validated against
**simulated seasons with known parameters**. Those tests caught two real bugs
that live data could not have: regularisation crushing every club toward
average, and `rho` pinning itself to its bound on noise.

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
- **Nothing here predicts the literal starting XI.** No public data source gives
  confirmed line-ups before the deadline, and the API exposes only per-season
  aggregates for past years — not per-match records — so "did he start the last
  six games of last season" is not computable. The model estimates *expected
  minutes*, which is what a points projection needs, not a team sheet. Check the
  press conferences yourself.

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
  dixon_coles.py  Poisson goals model: clean sheets and expected goals
  optimizer.py    MILP squad selection and starting-XI picker
  transfers.py    Transfer plan generation and ranking, net of hits
  chips.py        Chip heuristics and blank/double gameweek detection
  visualize.py    HTML pitch view of a squad
  server.py       Local server for the interactive squad editor
main.py           CLI
tests/            106 tests, live data + simulated seasons + live server
RULES.md          Researched 2026/27 rules, with citations
data/cache/       Cached API responses (gitignored)
```

---

## Running the tests

```bash
python -m unittest discover -s tests -v
```

106 tests covering the sell-on fee, formation enumeration, squad legality
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
