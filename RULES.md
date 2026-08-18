# Fantasy Premier League — Rules Reference (2026/27 season)

**Compiled:** 18 August 2026
**Season:** 2026/27 (confirmed via the FPL API's own static-content path, which resolves to
`.../plfpl-production/2026_27/`)
**Season state at time of writing:** pre-season. Gameweek 1 deadline is
**2026-08-21 17:30 UTC**; no gameweek has been played yet.

This document is the **source of truth** for all logic in this project. Where the
live FPL API exposes a machine-readable value, that value is treated as
authoritative and quoted directly — those rows are marked **[API]**. Rules that
only exist in prose (thresholds, divisors, chip semantics) are corroborated
against at least two independent sources and marked **[WEB]**.

> **Sanity-check note.** Anything marked **[API]** you can verify yourself in
> ten seconds: `curl -s https://fantasy.premierleague.com/api/bootstrap-static/ | python3 -m json.tool`
> and look under `game_config.scoring`, `game_config.rules`, `element_types`, and `chips`.

---

## 1. Squad rules

| Rule | Value | Source |
|---|---|---|
| Squad size | 15 players | `game_config.rules.squad_squadsize = 15` **[API]** |
| Starting XI | 11 players | `game_config.rules.squad_squadplay = 11` **[API]** |
| Budget | **£100.0m** | `game_config.rules.squad_total_spend = 1000` with `ui_currency_multiplier = 10` **[API]** |
| Max players per real club | **3** | `game_config.rules.squad_team_limit = 3` **[API]** |
| Goalkeepers | 2 selected, exactly 1 must start | `element_types[1]`: `squad_select=2, squad_min_play=1, squad_max_play=1` **[API]** |
| Defenders | 5 selected, 3–5 start | `element_types[2]`: `squad_select=5, squad_min_play=3, squad_max_play=5` **[API]** |
| Midfielders | 5 selected, 2–5 start | `element_types[3]`: `squad_select=5, squad_min_play=2, squad_max_play=5` **[API]** |
| Forwards | 3 selected, 1–3 start | `element_types[4]`: `squad_select=3, squad_min_play=1, squad_max_play=3` **[API]** |

### Valid starting XI formations

A formation is valid iff it uses exactly 1 GK and satisfies DEF ∈ [3,5],
MID ∈ [2,5], FWD ∈ [1,3], with the outfield ten summing to 10. Enumerating
every combination that meets those bounds (written DEF-MID-FWD):

| Formation | DEF | MID | FWD | Valid? |
|---|---|---|---|---|
| 3-4-3 | 3 | 4 | 3 | ✅ |
| 3-5-2 | 3 | 5 | 2 | ✅ |
| 4-3-3 | 4 | 3 | 3 | ✅ |
| 4-4-2 | 4 | 4 | 2 | ✅ |
| 4-5-1 | 4 | 5 | 1 | ✅ |
| 5-2-3 | 5 | 2 | 3 | ✅ |
| 5-3-2 | 5 | 3 | 2 | ✅ |
| 5-4-1 | 5 | 4 | 1 | ✅ |

**8 valid formations.** (3-3-4 and 4-2-4 are excluded because max FWD is 3;
2-x-x is excluded because min DEF is 3.)

**Implementation note:** the optimizer should *not* hard-code this list. Enforce
the min/max bounds per position as linear constraints and the legal formations
fall out automatically. The table is here for human sanity-checking only.

### Bench

The 4 non-starting players form the bench, in a manager-chosen order. The
substitute goalkeeper occupies a dedicated slot; the other three are ordered
1–3 and auto-subbed in if a starter records 0 minutes, provided the
substitution keeps the formation legal.

### Captaincy

- **Captain** scores **double** points. **[WEB]**
- **Vice-captain** takes over if the captain plays 0 minutes.
  `game_config.rules.sys_vice_captain_enabled = true` **[API]**

---

## 2. Scoring

### 2.1 Points table

The values below are quoted **verbatim** from `game_config.scoring` in the live
API **[API]**, cross-checked against a published 2026/27 scoring guide **[WEB]**.

| Action | GK | DEF | MID | FWD |
|---|---|---|---|---|
| Playing 1–59 minutes | 1 | 1 | 1 | 1 |
| Playing 60+ minutes | 2 | 2 | 2 | 2 |
| **Goal scored** | **10** | **6** | **5** | **4** |
| Assist | 3 | 3 | 3 | 3 |
| Clean sheet (60+ mins) | 4 | 4 | 1 | 0 |
| **Defensive contribution** | — | **2** | **2** | **2** |
| Every 3 shot saves | 1 | — | — | — |
| Penalty save | 5 | — | — | — |
| Penalty miss | −2 | −2 | −2 | −2 |
| Every 2 goals conceded | −1 | −1 | 0 | 0 |
| Yellow card | −1 | −1 | −1 | −1 |
| Red card | −3 | −3 | −3 | −3 |
| Own goal | −2 | −2 | −2 | −2 |
| Each bonus point | 1 | 1 | 1 | 1 |

> ⚠️ **Verify this one.** A **goalkeeper goal is worth 10 points**, not the 6 it
> was in earlier seasons. This comes straight from the API
> (`scoring.goals_scored = {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}`) and is
> corroborated by a published 2026/27 guide. It is a genuine change and it is
> also almost never relevant — flagged here because it is the kind of thing
> that looks like a bug in the code.

**Divisor gotcha.** The API stores `saves: 1` and `goals_conceded: -1`, but these
are *per group*, not per event: **1 point per 3 saves** (8 saves = 2 pts, the
third arrives at 9) and **−1 per 2 goals conceded**. Both accumulate in complete
groups only; remainders are discarded. The API does not encode the divisor, so
this is **[WEB]**-sourced and hard-coded in `scoring.py`.

**Appearance points** are also not in the API scoring block — `long_play: 2` /
`short_play: 1` are, which is the same thing: 1 pt for any appearance under 60
minutes, 2 pts for 60+ **[API]**.

### 2.2 Defensive contribution ("DefCon") — unchanged for 2026/27

Introduced in 2025/26 and **carried into 2026/27 with no changes to thresholds
or point values** **[WEB]**.

| Position | Stats counted | Threshold | Award |
|---|---|---|---|
| Defender | **CBIT** — clearances, blocks, interceptions, tackles | **10+** in a match | +2 |
| Midfielder | **CBIRT** — CBIT **plus ball recoveries** | **12+** in a match | +2 |
| Forward | **CBIRT** | **12+** in a match | +2 |
| Goalkeeper | n/a | — | **0** (`scoring.defensive_contribution.GKP = 0` **[API]**) |

- The award is **capped at +2 per player per match** — hitting 20 CBIT still
  scores 2, not 4. **[WEB]**
- It is **independent of the clean sheet** — a defender can concede three and
  still bank the 2 points.
- Note the asymmetry: defenders get a *lower* threshold **and** a *smaller* stat
  basket (no recoveries). This is deliberate — it targets centre-backs and
  defensive midfielders.
- The API exposes per-player `defensive_contribution` and
  `defensive_contribution_per_90` fields, so historical DefCon rate is directly
  observable. Useful for a future custom scoring model.

### 2.3 Bonus Points System (BPS)

At the end of each match the three highest BPS scores earn **3, 2 and 1 bonus
points** respectively. Ties share the higher award. **[WEB]**

BPS is a separate, much finer-grained Opta-driven index (passes, tackles,
key passes, big chances created, errors, etc.) — it is *not* the same as the
points table above.

**2026/27 BPS changes** (all **[WEB]**, corroborated across two sources):

1. **The −1 BPS for being tackled/dispossessed while dribbling is removed.**
   Directly helps dribble-heavy wingers and attacking full-backs.
2. **Clearances, blocks and interceptions now earn 1 BPS per *three* actions**
   (previously per two). This was done specifically to reduce double-counting
   between BPS and DefCon points.
3. **Goalkeeper saves restructured** by shot location and quality:
   - Save from a shot **inside the box** → 3 BPS
   - Save from a shot **outside the box** → 0 BPS (previously scored)
   - Any other save → 2 BPS
   - Saving a **"big chance"** → +1 BPS on top
   Stated intent: improve the bonus prospects of goalkeepers, full-backs and
   attackers.

**Implementation note:** we do **not** model BPS in `scoring.py`. Predicting
bonus points requires per-match Opta event data the free API doesn't expose
pre-match. FPL's own `ep_next` already has expected bonus baked in, which is one
reason we start from it.

### 2.4 Live scoring and finalisation (new for 2026/27)

- Points now update **live during matches**, including appearance, goals,
  assists, clean sheets and DefCon. Mini-league standings and overall rank also
  update live. **[WEB]**
- **Projected bonus** appears after **20 minutes** of each match and moves as the
  game develops. **[WEB]**
- **Gameweek lockdown moved to 09:00 UK time the morning after the final match**
  (previously one hour after the final whistle). The extra window lets Opta
  review its data collection, which should mean fewer missed bonus/DefCon
  points at finalisation. **[WEB]**

> **Consequence for this project:** any score pulled from the API before the
> 09:00-next-day lockdown is provisional and can still move. A caching layer
> should not treat a just-finished gameweek as final.

---

## 3. Transfers

| Rule | Value | Source |
|---|---|---|
| Free transfers earned per gameweek | **1** | **[WEB]** |
| Max free transfers that can be banked | **5** | `max_extra_free_transfers = 4` (i.e. 1 base + 4 banked) **[API]**, corroborated **[WEB]** |
| Cost per transfer beyond your free ones | **−4 points** | **[WEB]** |
| Hard cap on transfers in one gameweek | **20** | `transfers_cap = 20` **[API]** |
| Sell-on fee | **50% of profit, rounded down** | `transfers_sell_on_fee = 0.5` **[API]** |
| Sell at purchase price? | **No** | `element_sell_at_purchase_price = false` **[API]** |

**Banking / rollover.** You gain 1 free transfer per gameweek. Unused ones roll
over and accumulate up to a **maximum of 5**. Once you are sitting on 5, further
gameweeks earn you nothing extra — use them or waste them.

**Point hits.** Each transfer beyond your available free transfers costs **−4**.
The hit is applied to that gameweek's score and is **permanent** — it is not
refunded if the transfer works out. Two extra transfers = −8, and so on.

**Sell-on fee, worked example.** Buy at £7.0m; the player rises to £7.4m. Profit
is £0.4m, you keep 50% = £0.2m, so your **selling price is £7.2m**. Rounding is
*down*, in £0.1m units — an odd £0.1m of profit is kept entirely by the game.
This means **squad value ≠ what you can actually spend**, and the optimizer must
use *selling price* for players you already own, not `now_cost`.

**Wildcard / Free Hit** override all of this — unlimited transfers, no hits.

---

## 4. Chips

Enumerated directly from the API's `chips` array **[API]**. There are **8 chips
in total, in two sets of four**:

| Chip | API name | Set 1 available | Set 2 available |
|---|---|---|---|
| Wildcard | `wildcard` | **GW2 – GW19** | GW20 – GW38 |
| Free Hit | `freehit` | **GW2 – GW19** | GW20 – GW38 |
| Bench Boost | `bboost` | GW1 – GW19 | GW20 – GW38 |
| Triple Captain | `3xc` | GW1 – GW19 | GW20 – GW38 |

### What each chip does **[WEB]**

- **Wildcard** — unlimited free transfers for one gameweek; the resulting squad
  is kept. No point hits. Does not carry your old squad back.
- **Free Hit** — unlimited free transfers for one gameweek only; your squad
  **reverts to the previous one** at the next deadline. The classic answer to a
  blank gameweek.
- **Bench Boost** — your 4 bench players' points **count** for that gameweek.
  All 15 score.
- **Triple Captain** — your captain scores **3×** instead of 2× for that
  gameweek.

### Chip rules and 2026/27 quirks

- **One chip per gameweek.** You cannot stack Bench Boost with Triple Captain.
- **The first set expires at the GW19 deadline** and does **not** carry over —
  unused first-half chips are simply lost. GW19 falls in early January 2027.
  **[WEB]** + **[API]** (`stop_event: 19` on every set-1 chip)
- **Wildcard and Free Hit are not available in Gameweek 1** (`start_event: 2`
  **[API]**). Bench Boost and Triple Captain *are* (`start_event: 1`). This makes
  sense — squad edits are unlimited and free before the GW1 deadline anyway.
- **The Assistant Manager chip has been REMOVED for 2026/27.** **[WEB]**,
  and confirmed structurally by the API: every `mng_*` scoring key
  (`mng_goals_scored`, `mng_win`, `mng_draw`, `mng_clean_sheets`,
  `mng_underdog_win`, …) is present but **set to 0** **[API]**, and no
  `manager` chip appears in the `chips` array. Do not build support for it.
- **No extra AFCON transfers this season** — the tournament falls in
  June/July 2027, outside the season. **[WEB]**

---

## 5. Price changes

- Player prices move based on **net transfer activity** since the last change.
  Heavily bought players **rise**; heavily sold players **fall**. **[WEB]**
- Changes are evaluated **once daily at 00:00 UK time**, and a player can move at
  most £0.1m per day. **[WEB]**
- The threshold is **dynamic**, scaled against the player's **ownership** — it is
  a proportion of owners transferring, not a flat count. Reported to sit around
  **~140,000 net transfers** for a typical mid-to-high-priced midfielder;
  cheap/low-owned players move on far fewer, premiums need more. **[WEB]**
- FPL now ships an **official price prediction tool** (new for 2026/27) showing
  likely risers and fallers from live transfer activity. It is explicitly
  described as a **guide only** and does not guarantee a change. **[WEB]**

**Useful API fields:** `cost_change_event`, `cost_change_event_fall`,
`cost_change_start`, `cost_change_start_fall`, `transfers_in_event`,
`transfers_out_event`, `price_change_percent`, `selected_by_percent`.

**Strategic relevance.** Price changes affect *team value*, not points. A rise
you hold is worth £0.05m of spending power per £0.1m (the 50% sell-on fee), so
chasing price rises is a second-order concern behind points. **This project
optimises points, not team value** — price data is used only to compute correct
selling prices and budget.

---

## 6. Blank and Double Gameweeks

| Term | Definition |
|---|---|
| **Blank Gameweek (BGW)** | Fewer than the normal 10 fixtures — at least two clubs have **no** Premier League fixture. Their players cannot score. |
| **Double Gameweek (DGW)** | More than 10 fixtures — at least two clubs play **twice** before the next deadline. Their players score in both. |

**[WEB]** for both definitions.

### What causes them

Premier League fixtures are postponed when they clash with **domestic cup ties**
(FA Cup and EFL Cup). When a club is in a cup round, its league fixture — and
therefore its opponent's — is pulled out of that gameweek, creating a **blank**.
Those postponed matches are later rescheduled into an existing gameweek,
creating a **double**. European and international commitments can contribute too.

### Typical 2026/27 pattern **[WEB]**

- One or two **blanks** in **late February / March**, around the FA Cup fifth
  round and quarter-finals.
- A larger **blank** around the FA Cup semi-finals, typically **GW32–GW34**.
- One or more **doubles** in the run-in, usually **GW34–GW37**, depending on how
  postponements pile up.

Exact gameweeks are **not knowable in advance** — they are confirmed only as the
Premier League reschedules, often just weeks ahead.

### Strategic implications

- **Blank** → Free Hit is the standard answer (field 11 players who actually
  have a fixture, then revert). Alternatively wildcard into DGW-heavy squads.
- **Double** → Bench Boost and Triple Captain are at their most valuable, since
  every player (or the captain) gets two matches.
- Chip planning is largely built *around* the BGW/DGW calendar, which is why the
  GW19 expiry of the first chip set matters: the juicy blanks and doubles are all
  in the **second** half, so first-set chips should be spent for their own sake
  rather than saved.

### Detecting them programmatically

The `/api/fixtures/` endpoint plus `bootstrap-static`'s `events` array is enough:
count fixtures per `team` per `event`. **0 fixtures = blank for that team;
2+ = double.** This is exactly what `chips.py` will use, and it is far more
reliable than any published calendar because it reflects the live schedule.

---

## 7. What this project does *not* model

Stated plainly so the scope is clear:

- **BPS / bonus point prediction** — needs pre-match Opta event data we don't have.
- **Price change prediction** — needs live net-transfer feeds; FPL's own tool is
  the better source, and it affects value not points.
- **Auto-substitution simulation** — the optimizer picks a starting XI; it does
  not simulate bench auto-subs.
- **Minutes / rotation risk** beyond the API's `chance_of_playing_next_round` and
  `status` flags.
- **Assistant Manager chip** — removed from the game for 2026/27.

---

## 8. Season-state caveat (important for Stage 2)

At the time of writing the season **has not started**. That means:

- `ep_next`, `form`, `points_per_game`, `total_points` and all the per-90 stats
  are **zero or near-zero** for every player.
- Any predicted-points model built on `ep_next` will produce **degenerate output
  until GW1 has been played** — expect the optimizer to return an essentially
  arbitrary (though rule-legal) squad.
- `scoring.py` should therefore expose a **pluggable** scorer interface, and it
  is worth having a price/ICT-based fallback heuristic for the pre-season and
  early-season window when `ep_next` carries no signal.

---

## Sources

**Primary (authoritative, machine-readable):**
- [FPL API — `bootstrap-static`](https://fantasy.premierleague.com/api/bootstrap-static/) — `game_config.scoring`, `game_config.rules`, `element_types`, `chips`, `events`. Fetched 18 Aug 2026.
- [FPL API — `fixtures`](https://fantasy.premierleague.com/api/fixtures/)

**Secondary (prose rules, thresholds, chip semantics):**
- [Fantasy Football Scout — FPL 2026/27: 5 rule changes + new features announced](https://www.fantasyfootballscout.co.uk/2026/07/20/fpl-2026-27-5-rule-changes-new-features-announced)
- [FPL Oracle — FPL 2026/27 Rule Changes Explained](https://fploracle.team/blog/fpl-2026-27-rule-changes-explained)
- [World In Sport — Complete Guide To FPL Scoring And Squad Rules For 2026/27](https://worldinsport.com/fantasy-premier-league-rules-scoring/)
- [Flashscore — Fantasy Premier League 2026/27: All rule changes and new features](https://www.flashscore.co.uk/news/football-premier-league-fantasy-premier-league-2026-27-all-rule-changes-and-new-features/hrmwAKL8/)
- [Premier League — How domestic cup ties cause Blank and Double Gameweeks in FPL](https://www.premierleague.com/en/news/4536133/how-domestic-cup-ties-cause-blank-and-double-gameweeks-in-fpl)
- [Fantasy Football Scout — Preparing for an FPL Blank or Double Gameweek](https://www.fantasyfootballscout.co.uk/2026/07/20/preparing-for-an-fpl-blank-or-double-gameweek)
- [Premier League — FPL basics explained: How to make transfers](https://www.premierleague.com/en/news/2174907/fpl-basics-explained-how-to-make-transfers)
- [LiveFPL — FPL Price Changes: When They Happen & How to Predict Them](https://livefpl.com/blog/fpl-price-changes)
- [Draft Fantasy — FPL Scoring System 2026/27: Points Explained](https://www.draftfantasy.com/blog/scoring-points-df)

**Could not be used:** `https://fantasy.premierleague.com/help/rules` is a
client-side-rendered SPA and returns no rule text to a plain HTTP fetch. The API
endpoints above cover the same ground with better fidelity.
