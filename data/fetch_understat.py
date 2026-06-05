"""
Fetch and cache all Understat data needed for 38-0.
Run once (or to refresh); never called at game runtime.

Fetches per-player-season stats AND per-team-match stats for:
  - ENG-Premier League  (2014/15 → present) — game dataset
  - ESP-La Liga, GER-Bundesliga, ITA-Serie A, FRA-Ligue 1 (2014/15 → present)
    → pooled for Stage 1 model training (more rows = better fit)

Outputs (all in artifacts/):
  understat_players_pl.parquet   — PL player-season stats (xg, np_xg, xa, xg_buildup, …)
  understat_players_all.parquet  — all 5 leagues (same schema, used only for training)
  understat_team_matches.parquet — all 5 leagues, per-match home/away xG + points

Attribution: Data from Understat.com. Independent project, not affiliated with the PL.
"""

import time
from pathlib import Path

import pandas as pd
import soccerdata as sd

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

PL_LEAGUE = "ENG-Premier League"
ALL_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]
SEASONS = [
    "1415", "1516", "1617", "1718", "1819",
    "1920", "2021", "2122", "2223", "2324", "2425",
]
DELAY = 6  # seconds between requests — be polite to Understat


def fetch_player_stats(leagues: list[str], seasons: list[str]) -> pd.DataFrame:
    frames = []
    for league in leagues:
        print(f"  Player stats: {league}")
        us = sd.Understat(leagues=league, seasons=seasons)
        df = us.read_player_season_stats()
        df = df.reset_index()
        frames.append(df)
        time.sleep(DELAY)
    return pd.concat(frames, ignore_index=True)


def fetch_team_match_stats(leagues: list[str], seasons: list[str]) -> pd.DataFrame:
    frames = []
    for league in leagues:
        print(f"  Team match stats: {league}")
        us = sd.Understat(leagues=league, seasons=seasons)
        df = us.read_team_match_stats()
        df = df.reset_index()
        frames.append(df)
        time.sleep(DELAY)
    return pd.concat(frames, ignore_index=True)


def compute_team_season_stats(match_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-match team stats to per-season totals and per-game rates.
    Returns one row per (league, season, team) with xgf_pg, xga_pg, points_pg, games.
    """
    rows = []
    for _, match in match_df.iterrows():
        base = {"league": match["league"], "season": match["season"], "date": match["date"]}
        rows.append({**base, "team": match["home_team"],
                     "xg_for": match["home_xg"], "xg_against": match["away_xg"],
                     "goals_for": match["home_goals"], "goals_against": match["away_goals"],
                     "points": match["home_points"]})
        rows.append({**base, "team": match["away_team"],
                     "xg_for": match["away_xg"], "xg_against": match["home_xg"],
                     "goals_for": match["away_goals"], "goals_against": match["home_goals"],
                     "points": match["away_points"]})

    games_df = pd.DataFrame(rows)
    agg = games_df.groupby(["league", "season", "team"], as_index=False).agg(
        games=("points", "count"),
        xgf_total=("xg_for", "sum"),
        xga_total=("xg_against", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
        points_total=("points", "sum"),
    )
    agg["xgf_pg"] = agg["xgf_total"] / agg["games"]
    agg["xga_pg"] = agg["xga_total"] / agg["games"]
    agg["points_pg"] = agg["points_total"] / agg["games"]
    return agg


def main():
    print("=== Fetching Understat player stats (PL only) ===")
    pl_players = fetch_player_stats([PL_LEAGUE], SEASONS)
    pl_players.to_parquet(ARTIFACTS / "understat_players_pl.parquet", index=False)
    print(f"  Saved understat_players_pl.parquet  shape={pl_players.shape}")
    print(f"  Columns: {pl_players.columns.tolist()}")
    print(f"  Seasons: {sorted(pl_players['season'].unique())}")
    print(f"  Teams (sample): {sorted(pl_players['team'].unique())[:5]}")

    print("\n=== Fetching Understat player stats (all 5 leagues) ===")
    all_players = fetch_player_stats(ALL_LEAGUES, SEASONS)
    all_players.to_parquet(ARTIFACTS / "understat_players_all.parquet", index=False)
    print(f"  Saved understat_players_all.parquet  shape={all_players.shape}")

    print("\n=== Fetching Understat team match stats (all 5 leagues) ===")
    matches = fetch_team_match_stats(ALL_LEAGUES, SEASONS)
    matches.to_parquet(ARTIFACTS / "understat_team_matches.parquet", index=False)
    print(f"  Saved understat_team_matches.parquet  shape={matches.shape}")

    print("\n=== Computing team-season aggregates ===")
    team_seasons = compute_team_season_stats(matches)
    team_seasons.to_parquet(ARTIFACTS / "understat_team_seasons.parquet", index=False)
    print(f"  Saved understat_team_seasons.parquet  shape={team_seasons.shape}")
    print(f"  Season-league rows: {len(team_seasons)}")
    print(team_seasons[team_seasons["league"] == PL_LEAGUE].sort_values(
        ["season", "points_total"], ascending=[True, False]
    ).head(10).to_string(index=False))

    print("\nDone. Run data/fetch_transfermarkt.py next.")


if __name__ == "__main__":
    main()
