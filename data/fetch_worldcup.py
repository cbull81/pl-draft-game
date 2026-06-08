"""
Build World Cup 2026 player dataset from actual announced squads.

Steps:
  1. Scrape Wikipedia 2026 FIFA World Cup squads page for all 48 teams
  2. Fuzzy-match each player to Transfermarkt by name + citizenship → market value + sub_position
  3. Normalise: value_z (within position, WC players), log_caps_z, goals_per_cap_z
  4. Compute per-player wc_score (additive index)
  5. Compute tier thresholds from team-index distribution → store in meta.json
  6. Export artifacts/wc_players.parquet
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process

ROOT = Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts"
META_PATH = ROOT / "meta.json"

WC_SQUADS_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"

# ── Position mapping ───────────────────────────────────────────────────────────
# Wikipedia positions: GK, DF, MF, FW
WIKI_POS_MAP = {
    "GK": ["GK"],
    "DF": ["DEF"],
    "MF": ["MID"],
    "FW": ["FWD"],
}

TM_SUB_POSITION_MAP = {
    "Goalkeeper":          ["GK"],
    "Centre-Back":         ["DEF"],
    "Left-Back":           ["DEF"],
    "Right-Back":          ["DEF"],
    "Left Wing-Back":      ["DEF", "MID"],
    "Right Wing-Back":     ["DEF", "MID"],
    "Defensive Midfield":  ["MID"],
    "Central Midfield":    ["MID"],
    "Attacking Midfield":  ["MID"],
    "Left Midfield":       ["MID"],
    "Right Midfield":      ["MID"],
    "Second Striker":      ["MID", "FWD"],
    "Centre-Forward":      ["FWD"],
    "Left Winger":         ["FWD", "MID"],
    "Right Winger":        ["FWD", "MID"],
}

# ── 48 WC teams: display name → TM citizenship string + flag + confederation ──
WC_TEAMS = [
    # UEFA (16)
    ("Austria",              "Austria",            "UEFA", "🇦🇹"),
    ("Belgium",              "Belgium",            "UEFA", "🇧🇪"),
    ("Bosnia & Herzegovina", "Bosnia-Herzegovina", "UEFA", "🇧🇦"),
    ("Croatia",              "Croatia",            "UEFA", "🇭🇷"),
    ("Czech Republic",       "Czech Republic",     "UEFA", "🇨🇿"),
    ("England",              "England",            "UEFA", "🏴󠁧󠁢󠁥󠁮󠁧󠁿"),
    ("France",               "France",             "UEFA", "🇫🇷"),
    ("Germany",              "Germany",            "UEFA", "🇩🇪"),
    ("Netherlands",          "Netherlands",        "UEFA", "🇳🇱"),
    ("Norway",               "Norway",             "UEFA", "🇳🇴"),
    ("Portugal",             "Portugal",           "UEFA", "🇵🇹"),
    ("Scotland",             "Scotland",           "UEFA", "🏴󠁧󠁢󠁳󠁣󠁴󠁿"),
    ("Spain",                "Spain",              "UEFA", "🇪🇸"),
    ("Sweden",               "Sweden",             "UEFA", "🇸🇪"),
    ("Switzerland",          "Switzerland",        "UEFA", "🇨🇭"),
    ("Turkey",               "Turkey",             "UEFA", "🇹🇷"),
    # CONMEBOL (6)
    ("Argentina", "Argentina", "CONMEBOL", "🇦🇷"),
    ("Brazil",    "Brazil",    "CONMEBOL", "🇧🇷"),
    ("Colombia",  "Colombia",  "CONMEBOL", "🇨🇴"),
    ("Ecuador",   "Ecuador",   "CONMEBOL", "🇪🇨"),
    ("Paraguay",  "Paraguay",  "CONMEBOL", "🇵🇾"),
    ("Uruguay",   "Uruguay",   "CONMEBOL", "🇺🇾"),
    # CONCACAF (6)
    ("Canada",        "Canada",        "CONCACAF", "🇨🇦"),
    ("Curaçao",       "Curacao",       "CONCACAF", "🇨🇼"),
    ("Haiti",         "Haiti",         "CONCACAF", "🇭🇹"),
    ("Mexico",        "Mexico",        "CONCACAF", "🇲🇽"),
    ("Panama",        "Panama",        "CONCACAF", "🇵🇦"),
    ("United States", "United States", "CONCACAF", "🇺🇸"),
    # CAF (10)
    ("Algeria",      "Algeria",      "CAF", "🇩🇿"),
    ("Cape Verde",   "Cape Verde",   "CAF", "🇨🇻"),
    ("DR Congo",     "DR Congo",     "CAF", "🇨🇩"),
    ("Egypt",        "Egypt",        "CAF", "🇪🇬"),
    ("Ghana",        "Ghana",        "CAF", "🇬🇭"),
    ("Ivory Coast",  "Cote d'Ivoire","CAF", "🇨🇮"),
    ("Morocco",      "Morocco",      "CAF", "🇲🇦"),
    ("Senegal",      "Senegal",      "CAF", "🇸🇳"),
    ("South Africa", "South Africa", "CAF", "🇿🇦"),
    ("Tunisia",      "Tunisia",      "CAF", "🇹🇳"),
    # AFC (9)
    ("Australia",    "Australia",    "AFC", "🇦🇺"),
    ("Iran",         "Iran",         "AFC", "🇮🇷"),
    ("Iraq",         "Iraq",         "AFC", "🇮🇶"),
    ("Japan",        "Japan",        "AFC", "🇯🇵"),
    ("Jordan",       "Jordan",       "AFC", "🇯🇴"),
    ("Qatar",        "Qatar",        "AFC", "🇶🇦"),
    ("Saudi Arabia", "Saudi Arabia", "AFC", "🇸🇦"),
    ("South Korea",  "Korea, South", "AFC", "🇰🇷"),
    ("Uzbekistan",   "Uzbekistan",   "AFC", "🇺🇿"),
    # OFC (1)
    ("New Zealand", "New Zealand", "OFC", "🇳🇿"),
]

# Wikipedia h3 text → our display name (handles slight spelling differences)
WIKI_NAME_OVERRIDES = {
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "Czech Republic":         "Czech Republic",
    "United States":          "United States",
    "South Korea":            "South Korea",
    "Ivory Coast":            "Ivory Coast",
    "DR Congo":               "DR Congo",
    "Curaçao":                "Curaçao",
    "Cape Verde":             "Cape Verde",
    "New Zealand":            "New Zealand",
}

TEAM_DISPLAY = {row[0]: row for row in WC_TEAMS}
TEAM_BY_CITIZENSHIP = {row[1]: row[0] for row in WC_TEAMS}
TM_CIT = {row[0]: row[1] for row in WC_TEAMS}
# TM uses both "Turkey" and "Türkiye" — search both to avoid missing players
TM_CIT_EXTRA: dict[str, list[str]] = {
    "Turkey": ["Türkiye"],
}


# ── Step 1: Scrape Wikipedia ───────────────────────────────────────────────────

def _clean_name(raw: str) -> str:
    """Strip footnotes, parenthetical suffixes (captain, c, vc), whitespace."""
    import re
    raw = str(raw)
    raw = re.sub(r'\[.*?\]', '', raw)                      # [1], [a]
    raw = re.sub(r'\s*\(captain\)', '', raw, flags=re.I)   # (captain)
    raw = re.sub(r'\s*\(\s*[Cc]\s*\)', '', raw)            # (c) / (C)
    raw = re.sub(r'\s*\(\s*[Vv][Cc]\s*\)', '', raw)        # (vc)
    raw = re.sub(r'\s+', ' ', raw)
    return raw.strip()


def _clean_pos(raw: str) -> str:
    """Extract position code: GK / DF / MF / FW."""
    s = _clean_name(raw)
    for code in ["GK", "DF", "MF", "FW"]:
        if code in s.upper():
            return code
    return "MF"  # fallback


def fetch_wikipedia_squads() -> pd.DataFrame:
    print("  Fetching Wikipedia squads page...")
    resp = requests.get(WC_SQUADS_URL,
                        headers={"User-Agent": "Mozilla/5.0 (research project)"},
                        timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    rows = []
    known_display = {row[0].lower() for row in WC_TEAMS}
    # Also add common aliases
    alias_map = {v.lower(): k for k, v in WIKI_NAME_OVERRIDES.items()}
    alias_map.update({k.lower(): k for k in TEAM_DISPLAY})

    current_team = None
    for tag in soup.find_all(["h2", "h3", "table"]):
        if tag.name in ("h2", "h3"):
            raw = tag.get_text(strip=True).replace("[edit]", "").strip()
            # Check override map first, then direct lookup
            display = WIKI_NAME_OVERRIDES.get(raw, raw)
            if display in TEAM_DISPLAY or display.lower() in alias_map:
                current_team = TEAM_DISPLAY.get(display, TEAM_DISPLAY.get(alias_map.get(display.lower(), display)))
                current_team = current_team[0] if current_team else display
            else:
                current_team = None

        elif tag.name == "table" and current_team:
            try:
                df = pd.read_html(str(tag), flavor="bs4")[0]
            except Exception:
                continue
            # Normalise column names
            df.columns = [str(c).strip() for c in df.columns]
            # Identify key columns
            player_col = next((c for c in df.columns if "player" in c.lower() or "name" in c.lower()), None)
            pos_col    = next((c for c in df.columns if "pos" in c.lower()), None)
            caps_col   = next((c for c in df.columns if "cap" in c.lower()), None)
            goals_col  = next((c for c in df.columns if "goal" in c.lower()), None)
            club_col   = next((c for c in df.columns if "club" in c.lower()), None)
            dob_col    = next((c for c in df.columns if "birth" in c.lower() or "dob" in c.lower() or "age" in c.lower()), None)

            if player_col is None:
                continue

            for _, r in df.iterrows():
                name = _clean_name(r.get(player_col, ""))
                if not name or name.lower() in ("player", "name"):
                    continue
                pos  = _clean_pos(r.get(pos_col, "MF") if pos_col else "MF")
                try:
                    caps = int(float(str(r.get(caps_col, 0)).split("[")[0])) if caps_col else 0
                except (ValueError, TypeError):
                    caps = 0
                try:
                    goals = int(float(str(r.get(goals_col, 0)).split("[")[0])) if goals_col else 0
                except (ValueError, TypeError):
                    goals = 0
                club = _clean_name(r.get(club_col, "")) if club_col else ""

                rows.append({
                    "wc_team":       current_team,
                    "player_name":   name,
                    "wiki_pos":      pos,
                    "international_caps":  caps,
                    "international_goals": goals,
                    "club_wiki":     club,
                })

            # One squad per heading — clear team so next table isn't reused
            current_team = None

    df = pd.DataFrame(rows)
    print(f"  Scraped {len(df)} players across {df['wc_team'].nunique()} teams")
    return df


# ── Step 2: Match to Transfermarkt ────────────────────────────────────────────

FUZZY_THRESHOLD = 75   # token-set-ratio minimum for a valid TM match


def _normalise_name(s: str) -> str:
    import unicodedata, re
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    return " ".join(s.split())


def match_to_transfermarkt(squads: pd.DataFrame) -> pd.DataFrame:
    tm = pd.read_parquet(ARTIFACTS / "tm_players_raw.parquet")
    tm["tm_id"] = tm["player_id"].astype(str)
    tm["_norm"] = tm["name"].apply(_normalise_name)

    # Load manual overrides: player_name + wc_team → correct tm_id
    overrides: dict[tuple[str, str], str] = {}
    override_path = ARTIFACTS / "wc_resolver_overrides.csv"
    if override_path.exists():
        ov = pd.read_csv(override_path, dtype=str)
        for _, r in ov.iterrows():
            if pd.notna(r.get("tm_id")) and r["tm_id"].strip():
                overrides[(r["player_name"].strip(), r["wc_team"].strip())] = r["tm_id"].strip()
        print(f"  Loaded {len(overrides)} WC resolver override(s)")

    matched_tm_id   = []
    matched_value   = []
    matched_sub_pos = []
    matched_type    = []

    for _, row in squads.iterrows():
        override_id = overrides.get((row["player_name"], row["wc_team"]))
        if override_id:
            if override_id.upper() == "NONE":
                # Player isn't in the TM snapshot — force unmatched rather than
                # let the fuzzy fallback latch onto an unrelated player.
                matched_tm_id.append(None)
                matched_value.append(None)
                matched_sub_pos.append(None)
                matched_type.append("override_unmatched")
                continue
            tm_row = tm[tm["tm_id"] == override_id]
            if not tm_row.empty:
                best = tm_row.iloc[0]
                matched_tm_id.append(override_id)
                matched_value.append(best.get("market_value_in_eur"))
                matched_sub_pos.append(best.get("sub_position"))
                matched_type.append("override")
                continue
        team = row["wc_team"]
        cit  = TM_CIT.get(team, team)
        cit_list = [cit] + TM_CIT_EXTRA.get(team, [])
        norm_q = _normalise_name(row["player_name"])

        # Restrict to same citizenship first (some teams have multiple TM citizenship strings)
        pool = tm[tm["country_of_citizenship"].isin(cit_list)]

        # Try exact first
        exact = pool[pool["_norm"] == norm_q]
        if not exact.empty:
            best = exact.sort_values("market_value_in_eur", ascending=False).iloc[0]
            matched_tm_id.append(str(best["player_id"]))
            matched_value.append(best.get("market_value_in_eur"))
            matched_sub_pos.append(best.get("sub_position"))
            matched_type.append("exact")
            continue

        # Fuzzy within citizenship pool
        if not pool.empty:
            choices = pool["_norm"].tolist()
            result = process.extractOne(norm_q, choices, scorer=fuzz.token_set_ratio)
            if result and result[1] >= FUZZY_THRESHOLD:
                idx = pool.index[choices.index(result[0])]
                best = pool.loc[idx]
                matched_tm_id.append(str(best["player_id"]))
                matched_value.append(best.get("market_value_in_eur"))
                matched_sub_pos.append(best.get("sub_position"))
                matched_type.append("fuzzy")
                continue

        # Fuzzy across all TM (last resort)
        all_choices = tm["_norm"].tolist()
        result = process.extractOne(norm_q, all_choices, scorer=fuzz.token_set_ratio)
        if result and result[1] >= 88:
            idx = tm.index[all_choices.index(result[0])]
            best = tm.loc[idx]
            matched_tm_id.append(str(best["player_id"]))
            matched_value.append(best.get("market_value_in_eur"))
            matched_sub_pos.append(best.get("sub_position"))
            matched_type.append("fuzzy_global")
            continue

        matched_tm_id.append(None)
        matched_value.append(None)
        matched_sub_pos.append(None)
        matched_type.append("unmatched")

    squads = squads.copy()
    squads["tm_id"]       = matched_tm_id
    squads["market_value_eur"] = matched_value
    squads["sub_position"] = matched_sub_pos
    squads["match_type"]  = matched_type

    n_exact   = (squads["match_type"] == "exact").sum()
    n_fuzzy   = squads["match_type"].str.startswith("fuzzy").sum()
    n_none    = (squads["match_type"] == "unmatched").sum()
    print(f"  TM match: {n_exact} exact, {n_fuzzy} fuzzy, {n_none} unmatched  ({100*(1-n_none/len(squads)):.0f}% matched)")
    return squads


# ── Step 3: Position buckets + normalised scores ───────────────────────────────

def assign_buckets(row) -> list[str]:
    if row.get("sub_position") and not pd.isna(row["sub_position"]):
        buckets = TM_SUB_POSITION_MAP.get(row["sub_position"], [])
        if buckets:
            return buckets
    return WIKI_POS_MAP.get(row.get("wiki_pos", "MF"), ["MID"])


def compute_wc_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["eligible_buckets"] = df.apply(assign_buckets, axis=1)
    df["primary_bucket"]   = df["eligible_buckets"].apply(lambda b: b[0] if b else "MID")

    # value_z: within position, across ALL WC players
    df["value_z"] = np.nan
    for bucket, grp in df.groupby("primary_bucket"):
        vals = grp["market_value_eur"].dropna()
        if len(vals) < 3:
            continue
        mu, sigma = vals.mean(), vals.std()
        if sigma == 0:
            continue
        mask = df["primary_bucket"] == bucket
        df.loc[mask, "value_z"] = (df.loc[mask, "market_value_eur"] - mu) / sigma

    # Impute missing value_z with 0 (position median)
    df["value_z"] = df["value_z"].fillna(0.0)

    # log_caps_z: across all WC players
    df["log_caps"] = np.log1p(df["international_caps"].fillna(0).clip(lower=0))
    mu_c, sd_c = df["log_caps"].mean(), df["log_caps"].std()
    df["caps_z"] = (df["log_caps"] - mu_c) / sd_c if sd_c > 0 else 0.0

    # goals_per_cap: only meaningful for MID/FWD; z-score within position bucket
    df["goals_per_cap"] = np.where(
        df["international_caps"].fillna(0) > 0,
        df["international_goals"].fillna(0) / df["international_caps"].clip(lower=1),
        0.0,
    )
    df["goals_pg_z"] = 0.0
    for bucket in ["MID", "FWD"]:
        mask = df["primary_bucket"] == bucket
        vals = df.loc[mask, "goals_per_cap"]
        mu_g, sd_g = vals.mean(), vals.std()
        if sd_g > 0:
            df.loc[mask, "goals_pg_z"] = (vals - mu_g) / sd_g
    # Only award positive goal contribution (don't penalise low scorers)
    df["goals_pg_z"] = df["goals_pg_z"].clip(lower=0)

    # Composite WC score
    df["wc_score"] = (
        0.60 * df["value_z"]
        + 0.25 * df["caps_z"]
        + 0.15 * df["goals_pg_z"]
    )

    df["eligible_buckets"] = df["eligible_buckets"].apply(lambda b: ",".join(b))
    return df


# ── Step 4: Tier thresholds from team-index distribution ─────────────────────

def compute_tier_thresholds(df: pd.DataFrame) -> dict:
    """
    For each WC team compute 'optimal XI score' = mean wc_score of top-11 players.
    Set tier thresholds at percentile cutoffs of those 48 scores.
    """
    team_scores = []
    for team, grp in df.groupby("wc_team"):
        top11 = grp.nlargest(11, "wc_score")
        team_scores.append(top11["wc_score"].mean())

    team_scores = sorted(team_scores)
    n = len(team_scores)

    # Roughly: top 8% champion, 17% finalist, 33% semi, 54% QF, 75% R16, rest group
    thresholds = {
        "World Cup Champions":  float(np.percentile(team_scores, 92)),
        "Finalists":            float(np.percentile(team_scores, 83)),
        "Semi-finalists":       float(np.percentile(team_scores, 67)),
        "Quarter-finalists":    float(np.percentile(team_scores, 46)),
        "Round of 16":          float(np.percentile(team_scores, 21)),
        "Group Stage Exit":     float("-inf"),
    }

    print(f"\n  Team index distribution (optimal XI scores):")
    for pct in [10, 25, 50, 75, 90]:
        print(f"    p{pct}: {np.percentile(team_scores, pct):.3f}")
    print(f"  Tier thresholds: {thresholds}")
    return thresholds


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Building WC 2026 player dataset (actual squads) ===\n")

    print("Step 1: Scraping Wikipedia squads...")
    squads = fetch_wikipedia_squads()

    # Check which teams we got
    got = set(squads["wc_team"].unique())
    expected = {row[0] for row in WC_TEAMS}
    missing = expected - got
    if missing:
        print(f"  WARNING: Missing teams from Wikipedia: {missing}")

    print("\nStep 2: Matching to Transfermarkt...")
    squads = match_to_transfermarkt(squads)

    print("\nStep 3: Computing position buckets and WC scores...")
    df = compute_wc_scores(squads)

    print(f"\n  Dataset: {len(df)} players, {df['wc_team'].nunique()} teams")
    print(f"  Players with market value: {df['market_value_eur'].notna().sum()}")
    print(f"  Score range: {df['wc_score'].min():.2f} – {df['wc_score'].max():.2f}")

    # Save
    out = ARTIFACTS / "wc_players.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved wc_players.parquet  shape={df.shape}")

    print("\nStep 4: Computing tier thresholds...")
    thresholds = compute_tier_thresholds(df)

    # Update meta.json
    if META_PATH.exists():
        with open(META_PATH) as f:
            meta = json.load(f)
    else:
        meta = {}

    meta["wc_teams"] = [
        {"name": name, "flag": flag, "confederation": conf}
        for name, _, conf, flag in WC_TEAMS
        if name in got
    ]
    meta["wc_tier_thresholds"] = thresholds
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nUpdated meta.json: {len(meta['wc_teams'])} teams, tier thresholds saved")

    # Quick sanity check
    print("\n  Top 10 teams by optimal XI score:")
    team_top = []
    for team, grp in df.groupby("wc_team"):
        score = grp.nlargest(11, "wc_score")["wc_score"].mean()
        team_top.append((team, round(score, 3)))
    for t, s in sorted(team_top, key=lambda x: -x[1])[:10]:
        print(f"    {t}: {s:.3f}")


if __name__ == "__main__":
    main()
