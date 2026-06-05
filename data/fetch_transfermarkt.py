"""
Download and cache the dcaribou/transfermarkt-datasets snapshot.
Run once; do NOT scrape Transfermarkt directly (ToS).

Downloads from the GitHub release of dcariboo/transfermarkt-datasets.
Tables fetched: players, player_valuations, appearances, clubs, competitions, games.

Outputs:
  artifacts/tm_players.parquet
  artifacts/tm_valuations.parquet
  artifacts/tm_appearances.parquet
  artifacts/tm_clubs.parquet
  artifacts/tm_competitions.parquet
  artifacts/tm_games.parquet

Attribution: Transfermarkt data via dcaribou/transfermarkt-datasets (David Cariboo).
  Market values are crowd-sourced estimates, not official transfer fees.
"""

from pathlib import Path

import pandas as pd
import io, requests

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# dcaribou/transfermarkt-datasets on GitHub — CSV files in the data/ directory
# We pin the main branch; if you want reproducibility, pin a specific commit SHA

# 1) point at the R2 bucket (not raw GitHub — data is DVC-tracked, not in git)
CSV_BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"

# 2) the files are gzipped
TABLES = {
    "players":           "players.csv.gz",
    "player_valuations": "player_valuations.csv.gz",
    "appearances":       "appearances.csv.gz",
    "clubs":             "clubs.csv.gz",
    "competitions":      "competitions.csv.gz",
    "games":             "games.csv.gz",
}

def download_csv(name: str, filename: str) -> pd.DataFrame:
    url = f"{CSV_BASE}/{filename}"
    print(f"  Downloading {name} from {url}")
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=300)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content), compression="gzip", low_memory=False)

TOP5_COMP_IDS = ["GB1", "ES1", "IT1", "FR1", "L1"]  # PL, La Liga, Serie A, Ligue 1, Bundesliga


def filter_to_relevant(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Narrow Transfermarkt data to the top-5 European leagues.
    Keeps all seasons; league filtering in build.py uses the Understat league column.
    """
    clubs       = tables["clubs"]
    players     = tables["players"]
    valuations  = tables["player_valuations"]
    appearances = tables["appearances"]
    comps       = tables["competitions"]
    games       = tables["games"]

    top5_club_ids = clubs[clubs["domestic_competition_id"].isin(TOP5_COMP_IDS)]["club_id"].unique()
    print(f"  Top-5 clubs: {len(top5_club_ids)}")

    top5_appearances = appearances[appearances["competition_id"].isin(TOP5_COMP_IDS)]
    top5_player_ids  = top5_appearances["player_id"].unique()
    print(f"  Top-5 associated players: {len(top5_player_ids)}")

    return {
        "players":          players[players["player_id"].isin(top5_player_ids)],
        "player_valuations": valuations[valuations["player_id"].isin(top5_player_ids)],
        "appearances":      top5_appearances,
        "clubs":            clubs[clubs["club_id"].isin(top5_club_ids)],
        "competitions":     comps,
        "games":            games,
    }


def inspect_columns(tables: dict[str, pd.DataFrame]):
    print("\n=== Column inspection ===")
    for name, df in tables.items():
        print(f"\n  {name}  shape={df.shape}")
        print(f"    cols: {df.columns.tolist()[:20]}")
        if "sub_position" in df.columns:
            print(f"    sub_position values: {sorted(df['sub_position'].dropna().unique())}")
        if "market_value_in_eur" in df.columns:
            print(f"    market_value_in_eur: {df['market_value_in_eur'].describe()}")


def main():
    print("=== Downloading Transfermarkt (dcaribou) dataset ===")
    raw = {}
    for name, filename in TABLES.items():
        try:
            raw[name] = download_csv(name, filename)
            raw[name].to_parquet(ARTIFACTS / f"tm_{name}_raw.parquet", index=False)
            print(f"    → shape={raw[name].shape}")
        except Exception as e:
            print(f"    ERROR downloading {name}: {e}")
            print(f"    Check the URL: {CSV_BASE}/{filename}")
            print("    You may need to check the actual file paths in the GitHub repo.")

    if len(raw) == 0:
        print("\nAll downloads failed. Check the GitHub URL structure — the repo may have changed.")
        print("Alternative: download the DuckDB file from the repo and extract tables manually.")
        return

    inspect_columns(raw)

    print("\n=== Filtering to PL-relevant data ===")
    try:
        filtered = filter_to_relevant(raw)
        for name, df in filtered.items():
            out_path = ARTIFACTS / f"tm_{name}.parquet"
            df.to_parquet(out_path, index=False)
            print(f"  Saved tm_{name}.parquet  shape={df.shape}")
    except Exception as e:
        print(f"  Filter step failed: {e}")
        print("  Saving unfiltered tables and continuing...")
        for name, df in raw.items():
            df.to_parquet(ARTIFACTS / f"tm_{name}.parquet", index=False)

    print("\nDone. Run data/resolve.py next.")


if __name__ == "__main__":
    main()
