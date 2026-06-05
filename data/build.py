"""
Build the clean, merged player-season dataset from raw artifacts.

Steps:
  1. Load Understat PL players + resolver + TM players/valuations/appearances
  2. Merge: each player gets Understat xG stats + TM sub_position + season-dated market value + caps
  3. Map sub_position → position bucket (GK/DEF/MID/FWD); multi-position players eligible for multiple
  4. Compute per-game offense rates (xG/games, xa/games, etc.)
  5. Compute position×season normalized market value (z-score)
  6. Build (club, season) → eligible players index
  7. Export fbref_players.parquet (misleadingly named; keep for API compat) and update meta.json

Output: artifacts/players.parquet, updates meta.json valid_cells
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from model.features import age_value_weight

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
ROOT = Path(__file__).parent.parent
META_PATH = ROOT / "meta.json"

# ── Position bucket mapping (Transfermarkt sub_position → bucket) ─────────────
# TM sub_position is cleaner and more precise than Understat's coarse position field
TM_SUB_POSITION_MAP = {
    # GK
    "Goalkeeper": ["GK"],
    # DEF
    "Centre-Back": ["DEF"],
    "Left-Back": ["DEF"],
    "Right-Back": ["DEF"],
    "Left Wing-Back": ["DEF", "MID"],
    "Right Wing-Back": ["DEF", "MID"],
    # MID
    "Defensive Midfield": ["MID"],
    "Central Midfield": ["MID"],
    "Attacking Midfield": ["MID"],
    "Left Midfield": ["MID"],
    "Right Midfield": ["MID"],
    "Second Striker": ["MID", "FWD"],
    # FWD
    "Centre-Forward": ["FWD"],
    "Left Winger": ["FWD", "MID"],
    "Right Winger": ["FWD", "MID"],
}

MIN_MINUTES = 450  # filter low-sample players from game candidates


def map_buckets(sub_position: str) -> list[str]:
    if pd.isna(sub_position):
        return []
    return TM_SUB_POSITION_MAP.get(str(sub_position).strip(), [])


def season_str_to_start_year(season: str) -> int:
    """'2324' → 2023, '1415' → 2014."""
    return int("20" + season[:2]) if len(season) == 4 else int(season)


def get_season_valuation(player_id: str, season: str, valuations: pd.DataFrame) -> float | None:
    """
    Return the market value of a player closest to the START of a given season.
    Season end = May/June; season start = August. Use August 1 as the anchor.
    """
    start_year = season_str_to_start_year(season)
    anchor = pd.Timestamp(f"{start_year}-08-01")
    pv = valuations[valuations["player_id"] == player_id].copy()
    if pv.empty:
        return None
    pv["date"] = pd.to_datetime(pv["date"], errors="coerce")
    pv = pv.dropna(subset=["date"])
    if pv.empty:
        return None
    pv["delta"] = (pv["date"] - anchor).abs()
    row = pv.loc[pv["delta"].idxmin()]
    return float(row["market_value_in_eur"]) if pd.notna(row["market_value_in_eur"]) else None


def build_players() -> pd.DataFrame:
    print("  Loading artifacts...")
    us = pd.read_parquet(ARTIFACTS / "understat_players_pl.parquet")
    resolver = pd.read_parquet(ARTIFACTS / "resolver.parquet")
    tm_players = pd.read_parquet(ARTIFACTS / "tm_players_raw.parquet")
    tm_valuations = pd.read_parquet(ARTIFACTS / "tm_player_valuations_raw.parquet")

    print(f"  Understat rows: {len(us)}  Resolver rows: {len(resolver)}")

    # Merge Understat with resolver; drop resolver's understat_id (same as player_id)
    df = us.merge(
        resolver[["understat_id", "tm_id", "match_type"]],
        left_on="player_id", right_on="understat_id", how="left"
    ).drop(columns=["understat_id"])
    # Resolver tm_id is float64 (int + None → float when pandas builds the DataFrame).
    # Normalize to clean string IDs so downstream merges and dict lookups work.
    df["tm_id"] = df["tm_id"].apply(lambda x: str(int(x)) if pd.notna(x) else None)

    # Merge TM player profile (sub_position, caps, dob, nationality)
    # Cast to string to match resolver's tm_id (object dtype)
    # Exclude "position" — us already has it; sub_position is what we actually use
    tm_profile = tm_players[[
        "player_id", "sub_position",
        "country_of_citizenship", "date_of_birth",
    ]].rename(columns={"player_id": "tm_id"})
    tm_profile["tm_id"] = tm_profile["tm_id"].astype(str)
    # Add caps if column exists
    for cap_col in ["current_club_domestic_competition_id", "international_caps", "international_goals"]:
        if cap_col in tm_players.columns:
            tm_profile[cap_col] = tm_players[cap_col]

    df = df.merge(tm_profile, on="tm_id", how="left")

    # Position buckets from TM sub_position
    df["eligible_buckets"] = df["sub_position"].apply(map_buckets)

    # Fall back to Understat position for unmatched players
    # Understat position is coarse: "G S", "D S", "M S", "F S" or similar
    def fallback_buckets(row):
        if row["eligible_buckets"]:
            return row["eligible_buckets"]
        pos = str(row.get("position", "")).strip().upper()
        if pos.startswith("G"):
            return ["GK"]
        if pos.startswith("D"):
            return ["DEF"]
        if pos.startswith("M"):
            return ["MID"]
        if pos.startswith("F"):
            return ["FWD"]
        return []
    df["eligible_buckets"] = df.apply(fallback_buckets, axis=1)

    # Drop players with no bucket mapping
    df = df[df["eligible_buckets"].map(len) > 0].copy()

    # Per-game offense rates
    df["npxg_pg"] = df["np_xg"] / df["matches"].replace(0, np.nan)
    df["xa_pg"] = df["xa"] / df["matches"].replace(0, np.nan)
    df["xg_buildup_pg"] = df["xg_buildup"] / df["matches"].replace(0, np.nan)
    df["xg_chain_pg"] = df["xg_chain"] / df["matches"].replace(0, np.nan)

    # Season-dated market value (slow: one call per player-season)
    print(f"  Looking up season-dated valuations for {len(df)} player-season rows...")
    valuations_by_player = {
        str(pid): grp for pid, grp in tm_valuations.groupby("player_id")
    }

    def get_val(row):
        tm_id = row.get("tm_id")
        if pd.isna(tm_id):
            return np.nan
        grp = valuations_by_player.get(str(tm_id), pd.DataFrame())
        if grp.empty:
            return np.nan
        start_year = season_str_to_start_year(str(row["season"]))
        anchor = pd.Timestamp(f"{start_year}-08-01")
        grp = grp.copy()
        grp["date"] = pd.to_datetime(grp["date"], errors="coerce")
        grp = grp.dropna(subset=["date"])
        if grp.empty:
            return np.nan
        grp["delta"] = (grp["date"] - anchor).abs()
        row_val = grp.loc[grp["delta"].idxmin(), "market_value_in_eur"]
        return float(row_val) if pd.notna(row_val) else np.nan

    df["market_value_eur"] = df.apply(get_val, axis=1)

    # Normalize market value within (primary_bucket × season)
    # Use primary bucket (first element of eligible_buckets) for normalization group
    df["primary_bucket"] = df["eligible_buckets"].apply(lambda b: b[0] if b else None)

    for (bucket, season), grp in df.groupby(["primary_bucket", "season"]):
        vals = grp["market_value_eur"].dropna()
        if len(vals) < 3:
            continue
        mu, sigma = vals.mean(), vals.std()
        if sigma == 0:
            continue
        mask = (df["primary_bucket"] == bucket) & (df["season"] == season)
        df.loc[mask, "value_z"] = (df.loc[mask, "market_value_eur"] - mu) / sigma

    print(f"  Players with value: {df['market_value_eur'].notna().sum()} / {len(df)}")
    print(f"  Players with value_z: {df['value_z'].notna().sum()} / {len(df)}")

    # Age at season start (Aug 1 of start year), used to discount young-player potential premium
    def _age_at_season(row) -> float | None:
        try:
            dob = pd.Timestamp(row["date_of_birth"])
            start_year = int("20" + str(row["season"])[:2])
            return (pd.Timestamp(f"{start_year}-08-01") - dob).days / 365.25
        except Exception:
            return None

    df["age_at_season"] = df.apply(_age_at_season, axis=1)
    df["age_adj_value_z"] = df.apply(
        lambda row: (
            row["value_z"] * age_value_weight(row["age_at_season"])
            if pd.notna(row["value_z"]) and pd.notna(row["age_at_season"])
            else row["value_z"]
        ),
        axis=1,
    )
    print(f"  Players with age_adj_value_z: {df['age_adj_value_z'].notna().sum()} / {len(df)}")

    # Rename columns to clean API
    df = df.rename(columns={
        "player_id": "understat_id",
        "player": "player_name",
        "team": "club",
        "matches": "games",
        "position": "us_position",
    })

    # Keep only columns we'll use downstream
    keep = [
        "understat_id", "tm_id", "player_name", "club", "season",
        "games", "minutes",
        "goals", "np_goals", "xg", "np_xg", "assists", "xa",
        "xg_chain", "xg_buildup", "shots", "key_passes",
        "npxg_pg", "xa_pg", "xg_buildup_pg", "xg_chain_pg",
        "sub_position", "primary_bucket", "eligible_buckets",
        "market_value_eur", "value_z", "age_at_season", "age_adj_value_z",
        "match_type",
    ]
    # Add caps if present
    if "international_caps" in df.columns:
        keep.append("international_caps")
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    return df


def build_valid_cells(players: pd.DataFrame) -> list[dict]:
    """(club, season) cells with enough players to draft from."""
    cells = []
    for (club, season), grp in players.groupby(["club", "season"]):
        if len(grp) >= 5:
            cells.append({"club": club, "season": season})
    return cells


def main():
    print("=== Building player dataset ===")
    players = build_players()

    if players.empty:
        print("\nERROR: No players built — check fetch_understat.py and fetch_transfermarkt.py ran first.")
        return

    out_path = ARTIFACTS / "players.parquet"
    players.to_parquet(out_path, index=False)
    print(f"\nSaved players.parquet  shape={players.shape}")
    print(f"Columns: {players.columns.tolist()}")
    print(f"\nSample rows:\n{players.head(5).to_string(index=False)}")

    # Update meta.json with valid cells
    cells = build_valid_cells(players)
    meta_path = META_PATH
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}
    meta["valid_cells"] = cells
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nUpdated meta.json with {len(cells)} valid (club, season) cells")


if __name__ == "__main__":
    main()
