# CLAUDE.md — "38-0": A Premier League Draft Game

> Working title: **38-0** ("Can you go 38-0?"). Predicted points is the game's score; a perfect PL season = 38 wins = **114 points**, so the scale *is* the aspiration: 90+ = title, ~114 = the Invincible/perfect chase. Rename freely.

This file orients Claude Code on what we're building and the decisions already made. **Read it fully before scaffolding.** Python-native project — no JavaScript/React.

> ⚠️ **DATA REALITY (read §3 first).** FBref's advanced stats were Opta-sourced and **removed 2026-01-20** — gone for good as a free source (confirmed). We build on **three** free/open sources instead: **Understat** (xG performance), **Transfermarkt via the dcaribou open dataset** (market value, international caps, clean positions, roster scaffolding), and **FBref basic** (optional cross-checks). Do not reintroduce FBref xG/progressive/defensive/keeper stats — they don't exist anymore.

---

## 1. What we're building

A **Python** game inspired by [82-0.com](https://www.82-0.com/), reworked for the **English Premier League**.

**Core loop:**
1. Pick a **formation** (4-3-3, 4-4-2, 3-5-2, …) up front; locked for the game — like **FIFA Ultimate Team draft mode**. Sets how many of each position to draft.
2. Eleven rounds (**roll-and-reroll**, à la 82-0's slot machine). Each round rolls a random **(club, season)**, e.g. *Leicester City, 2015/16*. **Stats hidden.** That squad is the round's candidate pool.
3. Draft one player from that club-season who fills an **open position slot** in the locked formation — football knowledge only.
4. When the XI is complete, aggregate its data and feed an **expected-points model** (§4). Predicted points = score.
5. A few **rerolls (skips)** to pass on a bad (club, season).

**Kept from 82-0:** roll-and-reroll, blind draft, brutal aggregate rating, perfect-season aspiration. **Changed:** the rating is a learned expected-points model; **formation constraints** prevent the "eleven strikers" degenerate.

---

## 2. Design decisions

### 2.1 Season-by-season, 2014/15 → present
The machine rolls a **(club, season)**. **Understat covers the PL from 2014/15**, so that's the floor (includes Leicester 2015/16). Not every club appears every season (promotion/relegation) — a thin newly-promoted squad is a real risk. `meta.json` lists valid cells.

### 2.2 Formation (locked up front) handles positional bias structurally
Choosing 4-3-3 forces 1 GK, 4 DEF, 3 MID, 3 FWD. Four buckets — **GK / DEF / MID / FWD** — as counts (sum 11):

| Formation | GK | DEF | MID | FWD |
|-----------|----|-----|-----|-----|
| 4-4-2 | 1 | 4 | 4 | 2 |
| 4-3-3 | 1 | 4 | 3 | 3 |
| 3-5-2 | 1 | 3 | 5 | 2 |
| 4-2-3-1 | 1 | 4 | 5 | 1 |
| 3-4-3 | 1 | 3 | 4 | 3 |
| 5-3-2 | 1 | 5 | 3 | 2 |

Formation player-chosen and fixed (FUT-style); keep in config. **Use Transfermarkt's clean `sub_position`** (Centre-Back, Goalkeeper, etc.) for bucketing — it's far cleaner than Understat's coarse position field. Multi-position players are eligible for multiple slots.

> **The data gap and how value closes it.** Understat is *offensive only* per player — no individual defending, nothing for keepers. **Transfermarkt market value fills exactly this gap:** it's a per-player quality signal that exists for defenders and goalkeepers too. So defenders/keepers are scored primarily on (position-season-normalized) market value, mids/ball-players also get **xGBuildup** credit, and forwards/creators are scored on xG/xA. This is cleaner than the earlier "borrow your old team's xGA" provenance idea — value is an *individual* signal.

---

## 3. Data sources (three, all free/open)

### A. Understat — xG performance (primary performance layer)
PL + 5 leagues, **every season since 2014/15**; last easy-to-scrape free xG source. Per-player-season: `games, time, goals, assists, shots, key_passes, xG, xA, npg, npxG, xGChain, xGBuildup, position, team`. Per-team: `xG, xGA, PPDA`. **xGA is team-level only** (not per player) — this is why we need value for defense.
Access via `soccerdata`'s Understat module (`sd.Understat`, cached DataFrames), or `understatapi`/`UnderData`. Pin one.

### B. Transfermarkt via dcaribou/transfermarkt-datasets — value, caps, positions, rosters
Clean, weekly-refreshed open dataset (Kaggle `davidcariboo/player-scores`, also data.world, direct CSV, DuckDB). 12 tables; the ones we use:
- **`players`** — clean `sub_position`/`position`, `date_of_birth`, `country_of_citizenship`, and **`international_caps` / `international_goals` / `current_national_team_id`**.
- **`player_valuations`** — market-value **time series** (date-stamped) → get value *as of each season*, not just current.
- **`appearances`** / **`clubs`** / **`competitions`** — which players were at which club in which PL season (roster scaffolding, minutes).
- Bigger alternative if needed: `salimt/football-datasets` (93k players, adds `player_national_team_performances`). dcaribou is the cleaner default.
- **Use the published dataset snapshot — do NOT scrape Transfermarkt directly** (ToS + reliability). Pin a snapshot date for reproducibility.

### C. FBref basic — optional cross-checks only
Surviving free basic tables (goals/assists/minutes/positions, clean sheets). Use only to sanity-check positions or pull GK clean sheets if we decide keepers need a performance signal (§4.5). Behind Cloudflare; don't depend on it.

### Entity resolution (the main integration cost — budget for it)
Understat ids ≠ Transfermarkt ids and there's no free crosswalk. Build a resolver:
1. **Deterministic first:** normalize names (unidecode, strip punctuation, handle "Son Heung-min"/"Heung-Min Son" ordering), then match **constrained to the same club + season** (both sources tell you who was at a club in a season — a strong disambiguator).
2. **Fuzzy fallback** (token-set ratio) for the rest, again club-season-scoped.
3. **Manual override file** (`resolver_overrides.csv`) for the residual ambiguous/failed matches — review it once; it's small.
Players who fail to match keep value/caps/position from TM but lack xG (and vice versa) — handle missing fields explicitly (see §4.3).

### Etiquette / attribution
Build the dataset **offline, once**; cache; throttle Understat (~6s/req). Credit **Understat** and **Transfermarkt (+ dataset author)**; note TM values are crowd-sourced estimates, not fees. Ship the "independent project, not affiliated with the Premier League" disclaimer.

---

## 4. The rating: expected-points model + value/caps  ⭐

Spine unchanged: **train on real teams, score the drafted XI with the same feature definitions.** The new sources let market value stand in for the per-player defense we can't measure.

### 4.1 Recommended structure: two stages
The honest problem is that the best defensive predictor (team **xGA**) is measurable for real teams but not attributable to drafted individuals. Bridge it with value:

- **Stage 1 — points model (well-grounded):** `points_per_game ~ Ridge(xGF_per_game, xGA_per_game)`. Train on **pooled Understat team-seasons across all 6 leagues since 2014/15 (~1,000+ rows)** for robustness; per-game target handles different season lengths. xG difference predicts points strongly — validate with leave-one-season/league-out CV (points RMSE + Spearman). ×38 → EPL points.
- **Stage 2 — defense bridge:** `xGA_per_game ~ (defensive market-value index)` learned across real teams, so a drafted XI's xGA can be **estimated from its defenders'/keeper's normalized values**. This is where Transfermarkt earns its place.

At scoring: `xGF_hat` from the XI's Understat offense; `xGA_hat` from Stage 2 on the XI's DEF/GK values; feed both into Stage 1.

*(Lighter single-stage alternative if Stage 2 is weak: `points_pg ~ Ridge(xGF_pg, defensive_value_index)` directly. Decide empirically from CV.)*

### 4.2 Market value as a quality signal — normalize hard
Raw value is biased by **position** (attackers > equal-quality defenders), **age** (young inflated, elite veterans deflated), and **market inflation** (values balloon year-on-year). So never use raw euros. Compute a **z-score within (position bucket × season)** from `player_valuations` at the season's date. That yields "how elite was this player *for their position, that year*" — comparable across positions and eras, which is exactly what a cross-era draft needs. This normalized value is the DEF/GK quality input (and a useful sanity signal for all positions).

### 4.3 Building the XI's features (train vs. score)
- **Offense `xGF_hat`:** `Σ (player npxG / games)` over the XI (clean, minimal double-counting; xA optional with care). The "sum of player stats" you originally wanted.
- **Defense `xGA_hat`:** Stage 2 applied to the **position-season-normalized values of the GK + DEF picks** (optionally DM). Primary signal. (Optional: blend in old club-season xGA "provenance" if CV shows it helps.)
- **Missing data:** if a drafted player has no Understat match, fall back to value-implied offense or exclude with a flag; if no value, impute position-season median. Make fallbacks explicit, not silent.
- Feature *definitions* are identical train/score; only *estimation* differs (measured for real teams, reconstructed for XIs). **Calibration check:** confirm summing real teams' actual XIs recovers their measured xGF, and that Stage 2's value→xGA mapping is sane; add a learned scaling factor if not.

### 4.4 International caps — secondary signal, with a real caveat
Use caps as a **minor pedigree/experience signal or tiebreaker**, not a core feature, because:
- The dataset's `international_caps` is a **current/career snapshot, not "caps as of that season"** → using it for a 2015/16 player leaks future caps. The national-team-games tables only cover major tournaments, so reconstructing as-of-season caps is incomplete. Treat caps as a rough static profile signal.
- **Nationality bias:** equal quality, fewer caps for players from weak/low-fixture federations or behind strong competition.
Normalize (e.g., within position) and keep the weight small. Good for a **"squad pedigree" sub-score** on the reveal screen (total normalized value + caps) and as flavor, rather than driving the core rating.

### 4.5 Keepers
With no individual keeper performance data, GK quality = **normalized market value** (primary). If that feels thin, optionally add **clean sheets from FBref basic** or the source team's goals-against as a light signal. Confirm in §9.

### 4.6 Scoring → output
1. Build `[xGF_hat, xGA_hat]`; `points = clip(ppg_hat * 38, 0, 114)`.
2. Tier: relegation (<35) · mid-table · European (~65+) · title (~85+) · **Invincible chase (→114)**.
3. Reveal screen: **attack (xGF) vs. defense (xGA) breakdown** + a **pedigree sub-score** (normalized squad value + caps) — the football analog of 82-0's category feedback, and it makes value/caps legible.

---

## 5. Architecture (Python-native)

```
38-0/
├── data/
│   ├── fetch_understat.py    # xG player+team season + schedule, 2014/15→present (cached)
│   ├── fetch_transfermarkt.py# load dcaribou snapshot: players, player_valuations, appearances, clubs
│   ├── resolve.py            # Understat↔Transfermarkt entity resolution + overrides file
│   └── build.py              # positions→buckets, per-game offense, season-dated normalized values,
│                             #   caps, eligible-players index per (club,season) → parquet/json
├── model/
│   ├── features.py           # measured (train) AND reconstructed-from-XI (score) feature defs + calibration
│   ├── train.py              # Stage 1 points~[xGF,xGA] (pooled leagues); Stage 2 xGA~value; CV; persist
│   └── score.py              # XI → [xGF_hat, xGA_hat] → points; + pedigree sub-score
├── game/
│   ├── state.py              # formation, slot tracking, roll, rerolls, drafted-id set
│   └── app.py                # Streamlit UI (§6)
├── artifacts/                # understat.parquet, transfermarkt.parquet, resolver_overrides.csv,
│                             #   ep_model.joblib, value_to_xga.joblib, meta.json
├── meta.json                 # formations, valid cells, feature spec, value/caps weights, tiers, rerolls
├── pyproject.toml            # pin soccerdata/understatapi, duckdb/pandas, scikit-learn, streamlit
└── CLAUDE.md
```

**Non-negotiables:** `features.py` owns measured + reconstructed feature defs + calibration (no silent train/score drift). The resolver's override file is checked in and reviewed.

---

## 6. UI: Streamlit
Pure-Python, shareable. CLI/Textual TUI fine as a first prototype. Streamlit reruns the whole script per interaction → keep all draft state in **`st.session_state`** (formation, round, open slots, drafted ids, rerolls, RNG seed); cache data/model with **`@st.cache_data` / `@st.cache_resource`**. **Blind draft:** names + positions only during selection; reveal stats, value, caps, and predicted points only at the end.
Screens: formation select (locked) → slot-machine roll with reroll → eligible-player picker → roster tracker → final reveal (points, tier, attack/defense breakdown, pedigree sub-score, disclaimer).

---

## 7. Build order
1. **Fetch:** Understat PL data; load dcaribou Transfermarkt snapshot; (optional FBref basic for positions/clean sheets).
2. **Resolve:** deterministic + fuzzy name match scoped by club-season; create the overrides file; report match rate.
3. **Build dataset:** position buckets, per-game offense, season-dated normalized market values, caps, eligible-players index.
4. **Model:** Stage 1 (pooled), Stage 2 (value→xGA), CV, **calibration check**; persist. **Pause here for review** (the value-based defense is the part to scrutinize).
5. **Game core (CLI first):** locked formation → 11-round roll-and-reroll → reconstruct features → predict. Make it fun first.
6. **Streamlit UI:** wrap core; blind draft + reveal; breakdown + pedigree sub-score; attribution/disclaimer.
7. **Tune:** tiers, value/caps weights, fallbacks so a great draft can flirt with 114 but it's hard.

Ship step 5 as the first playable.

---

## 8. Conventions & guardrails
- **Determinism:** seed the slot-machine RNG (session_state) for reproducible test games.
- **Identity:** Understat id ↔ Transfermarkt id only via the resolver; no drafting a player twice.
- **Value:** always season-dated and (position×season) normalized — never raw euros (inflation/position/age bias).
- **Caps:** career snapshot — secondary signal only; don't let it leak future caps into the headline rating.
- **Per-game everywhere;** pool leagues only on per-game scales.
- **Config external** → `meta.json`: formations, season range, feature spec, value/caps weights, tiers, rerolls.
- **Single source of truth for features** (#1 silent-bug risk): `features.py`.
- **Attribution + disclaimer** in the UI day one.

---

## 9. Open questions to surface (don't silently decide)
- **Defense estimation (§4.1–4.3):** two-stage (value→xGA) vs. single-stage (value index direct). Decide from CV — confirm with the user.
- **Value vs. performance weighting:** how much the normalized-value channel should move the score relative to xG. This is the dial that sets whether the game rewards *production* or *pedigree*.
- **Caps:** include in the headline rating at all, or confine to the pedigree sub-score given the career-snapshot leakage? (Lean: sub-score only.)
- **Keepers (§4.5):** value-only, or add FBref clean sheets / source-team goals-against?
- **Calibration scaling** if reconstructed XI features don't track measured team features.

*(Resolved: per-game normalization; roll-and-reroll one-per-round; player-chosen locked formation; Understat replaces FBref-advanced; 2014/15 floor; Transfermarkt (dcaribou) added for value + caps + clean positions, with value as the primary DEF/GK quality signal.)*
