"""
Entity resolution: link Understat player IDs to Transfermarkt player IDs.

Strategy (§3 of CLAUDE.md):
  1. Deterministic: normalize names → exact match constrained to same club + season.
  2. Fuzzy fallback: token-set ratio, still club-season-scoped, threshold 85.
  3. Manual overrides: resolver_overrides.csv (checked in, reviewed once).

Output: artifacts/resolver.parquet  — columns: understat_id, tm_id, player_name_us, match_type
        artifacts/resolver_overrides.csv  — created empty if missing; fill manually for failures
"""

from pathlib import Path

import pandas as pd

try:
    from rapidfuzz.fuzz import token_set_ratio
    HAVE_RAPIDFUZZ = True
except ImportError:
    HAVE_RAPIDFUZZ = False
    print("WARNING: rapidfuzz not installed — fuzzy matching disabled. pip install rapidfuzz")

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
OVERRIDES_PATH = ARTIFACTS / "resolver_overrides.csv"

FUZZY_THRESHOLD = 85  # token_set_ratio score to accept a fuzzy player name match
CLUB_THRESHOLD = 70   # token_set_ratio score to accept a club name match


def normalize_name(name: str) -> str:
    """Lowercase, strip accents, remove punctuation, sort tokens for order-independence."""
    import unicodedata
    import re
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"[^a-z ]", "", name.lower())
    tokens = sorted(name.split())
    return " ".join(tokens)


def normalize_club(name: str) -> str:
    """Strip 'Football Club', 'FC', 'AFC', etc. for cross-source club matching."""
    import re
    if not isinstance(name, str):
        return ""
    name = name.lower()
    for pat in [r"\bassociation football club\b", r"\bfootball club\b", r"\bfc\b", r"\bafc\b"]:
        name = re.sub(pat, "", name)
    return re.sub(r"\s+", " ", name).strip()


def build_us_index(us_players: pd.DataFrame) -> pd.DataFrame:
    """Return Understat players with normalized names+clubs, one row per (player_id, team, season)."""
    df = us_players[["player_id", "player", "team", "season"]].drop_duplicates().copy()
    df["norm_name"] = df["player"].apply(normalize_name)
    df["norm_club"] = df["team"].apply(normalize_club)
    return df


def build_tm_index(tm_appearances: pd.DataFrame, tm_players: pd.DataFrame,
                   tm_clubs: pd.DataFrame, tm_games: pd.DataFrame) -> pd.DataFrame:
    """Return TM PL appearances with normalized names + integer start-year season + club name.

    tm_appearances has no season column — derive it by joining with tm_games on game_id.
    Filter to Premier League (competition_id == 'GB1') before joining.
    """
    TOP5 = ["GB1", "ES1", "IT1", "FR1", "L1"]
    pl_apps = tm_appearances[tm_appearances["competition_id"].isin(TOP5)].copy()

    # Add season (integer start year, e.g. 2014 for 2014/15) from games table
    season_map = (
        tm_games[tm_games["competition_id"].isin(TOP5)][["game_id", "season"]]
        .drop_duplicates("game_id")
    )
    pl_apps = pl_apps.merge(season_map, on="game_id", how="left")

    # One row per (player, club, season) is enough for the name-match index
    pl_apps = pl_apps.drop_duplicates(subset=["player_id", "player_club_id", "season"])

    apps = pl_apps.merge(
        tm_players[["player_id", "name", "last_name", "sub_position", "position",
                    "date_of_birth", "country_of_citizenship"]],
        on="player_id", how="left"
    )
    apps = apps.merge(
        tm_clubs[["club_id", "name"]].rename(columns={"name": "club_name"}),
        left_on="player_club_id", right_on="club_id", how="left"
    )
    apps["norm_name"] = apps["name"].apply(normalize_name)
    apps["norm_club"] = apps["club_name"].apply(normalize_club)
    return apps


def load_overrides() -> dict[str, str]:
    """Load manual overrides: {understat_player_id: tm_player_id}."""
    if not OVERRIDES_PATH.exists():
        pd.DataFrame(columns=["understat_id", "tm_id", "note"]).to_csv(OVERRIDES_PATH, index=False)
        print(f"  Created empty overrides file at {OVERRIDES_PATH}")
        return {}
    df = pd.read_csv(OVERRIDES_PATH)
    df = df.dropna(subset=["understat_id", "tm_id"])
    return dict(zip(df["understat_id"].astype(str), df["tm_id"].astype(str)))


def resolve(us_players: pd.DataFrame, tm_appearances: pd.DataFrame,
            tm_players: pd.DataFrame, tm_clubs: pd.DataFrame,
            tm_games: pd.DataFrame) -> pd.DataFrame:
    """
    Return a crosswalk DataFrame with columns:
      understat_id, tm_id, player_name_us, match_type (exact/fuzzy/override/unmatched)
    """
    overrides = load_overrides()
    us_idx = build_us_index(us_players)
    tm_idx = build_tm_index(tm_appearances, tm_players, tm_clubs, tm_games)

    results = []
    us_ids = us_idx["player_id"].unique()
    total = len(us_ids)
    exact = fuzzy = override = unmatched = 0

    # First pass: overrides
    for us_id in us_ids:
        us_id_str = str(us_id)
        if us_id_str in overrides:
            results.append({
                "understat_id": us_id,
                "tm_id": overrides[us_id_str],
                "player_name_us": us_idx[us_idx["player_id"] == us_id]["player"].iloc[0],
                "match_type": "override",
            })
            override += 1

    override_us_ids = {r["understat_id"] for r in results}
    remaining = [uid for uid in us_ids if uid not in override_us_ids]

    for us_id in remaining:
        us_rows = us_idx[us_idx["player_id"] == us_id]
        player_name = us_rows["player"].iloc[0]
        norm = us_rows["norm_name"].iloc[0]

        # Collect all (team, season) pairs this player appeared in
        matched_tm_id = None
        match_type = "unmatched"

        for _, us_row in us_rows.iterrows():
            # Season matching: Understat "2324" → start year 2023 == TM games.season integer
            us_season = str(us_row["season"])
            tm_season_year = int("20" + us_season[:2])
            us_norm_club = us_row["norm_club"]

            # Filter by season, then by club (fuzzy on normalized names)
            tm_season = tm_idx[tm_idx["season"] == tm_season_year]
            if HAVE_RAPIDFUZZ and not tm_season.empty and us_norm_club:
                club_scores = tm_season["norm_club"].apply(
                    lambda c: token_set_ratio(us_norm_club, c)
                )
                tm_cands = tm_season[club_scores >= CLUB_THRESHOLD]
            else:
                tm_cands = tm_season

            if tm_cands.empty:
                continue  # try next (team, season) row for this player

            # Exact name match
            exact_match = tm_cands[tm_cands["norm_name"] == norm]
            if not exact_match.empty:
                matched_tm_id = exact_match.iloc[0]["player_id"]
                match_type = "exact"
                break

            # Fuzzy name match
            if HAVE_RAPIDFUZZ:
                scores = tm_cands["norm_name"].apply(lambda n: token_set_ratio(norm, n))
                best_idx = scores.idxmax()
                if scores[best_idx] >= FUZZY_THRESHOLD:
                    matched_tm_id = tm_cands.loc[best_idx, "player_id"]
                    match_type = "fuzzy"
                    break

        results.append({
            "understat_id": us_id,
            "tm_id": matched_tm_id,
            "player_name_us": player_name,
            "match_type": match_type,
        })
        if match_type == "exact":
            exact += 1
        elif match_type == "fuzzy":
            fuzzy += 1
        else:
            unmatched += 1

    resolver = pd.DataFrame(results)
    print(f"\n  Resolution summary ({total} Understat players):")
    print(f"    Exact:    {exact:4d} ({100*exact/total:.1f}%)")
    print(f"    Fuzzy:    {fuzzy:4d} ({100*fuzzy/total:.1f}%)")
    print(f"    Override: {override:4d}")
    print(f"    Unmatched:{unmatched:4d} ({100*unmatched/total:.1f}%)")
    print(f"\n  Unmatched players (first 20):")
    print(resolver[resolver["match_type"] == "unmatched"]["player_name_us"].head(20).to_string())
    print(f"\n  Add entries to {OVERRIDES_PATH} to fix critical mismatches.")
    return resolver


def main():
    print("=== Loading data ===")
    us_players = pd.read_parquet(ARTIFACTS / "understat_players_all.parquet")
    tm_players = pd.read_parquet(ARTIFACTS / "tm_players_raw.parquet")
    tm_appearances = pd.read_parquet(ARTIFACTS / "tm_appearances_raw.parquet")
    tm_clubs = pd.read_parquet(ARTIFACTS / "tm_clubs_raw.parquet")
    tm_games = pd.read_parquet(ARTIFACTS / "tm_games_raw.parquet")

    print(f"  Understat PL players: {us_players['player_id'].nunique()} unique")
    print(f"  TM players (raw): {len(tm_players)}")
    print(f"  TM appearances (raw): {len(tm_appearances)}")
    print(f"  TM games (raw): {len(tm_games)}")

    print("\n=== Running entity resolution ===")
    resolver = resolve(us_players, tm_appearances, tm_players, tm_clubs, tm_games)
    resolver.to_parquet(ARTIFACTS / "resolver.parquet", index=False)
    print(f"\n  Saved resolver.parquet  shape={resolver.shape}")


if __name__ == "__main__":
    main()
