"""
Score a drafted XI → predicted points + breakdown.
Called by game/state.py after all 11 slots are filled.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model.features import (
    build_offensive_features,
    build_position_value_features,
    compute_xi_xgf,
    compute_xi_xga,
    season_max_games,
    squad_pedigree,
)

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
ROOT = Path(__file__).parent.parent

GAMES_BONUS_THRESHOLD = 30   # avg games above this earn +1 pt per game
CHAMPION_BONUS_PER = 2       # pts per champion-season player
CHAMPION_BONUS_CAP = 5       # max players counted for the bonus

TIER_THRESHOLDS = [
    ("Title Contenders",  80),
    ("European Football", 62),
    ("Top Half",          54),
    ("Bottom Half",       45),
    ("Relegation Battle",  0),
]


def _load_champions() -> dict[str, dict[str, str]]:
    """Return {league: {season: champion_club}} from meta.json."""
    meta_path = ROOT / "meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("league_champions", {})


def load_models():
    ep_path  = ARTIFACTS / "ep_model.joblib"
    xga_path = ARTIFACTS / "xga_model.joblib"
    xgf_path = ARTIFACTS / "xgf_model.joblib"
    if not ep_path.exists():
        raise FileNotFoundError("Stage 1 model not found — run model/train.py first")
    ep  = joblib.load(ep_path)
    xga = joblib.load(xga_path)["model"] if xga_path.exists() else None
    xgf = joblib.load(xgf_path)["model"] if xgf_path.exists() else None
    return ep["model"], xga, xgf


def tier(points: float) -> str:
    for label, threshold in TIER_THRESHOLDS:
        if points >= threshold:
            return label
    return "Relegation Battle"


def points_to_record(points: float, games: int = 38) -> str:
    pts = max(0, min(games * 3, round(points)))
    wins = pts // 3
    remainder = pts % 3
    draws = remainder
    losses = max(0, games - wins - draws)
    return f"{wins}W {draws}D {losses}L"


def score_xi(players: pd.DataFrame) -> dict:
    """
    players: 11-row DataFrame (one per drafted player).
    Returns: predicted_points, tier, record, attack/defense breakdown, pedigree.

    Scoring:
      base_pts  = ppg_hat * 38  (always a full season — no availability penalty)
      games_bonus = max(0, avg_normalized_games - 30)  (+1 pt per game above 30)
      champion_bonus = 2 * min(champion_players, 5)   (players from title-winning seasons)
    """
    ep_model, xga_model, xgf_model = load_models()

    off_feats = build_offensive_features(players)
    xgf_pg = compute_xi_xgf(players, xgf_model)
    xga_pg = compute_xi_xga(players, xga_model)

    if np.isnan(xga_pg):
        xga_pg = 1.2  # ~PL average fallback if Stage 2 not trained

    pos_value_feats = build_position_value_features(players)

    feat = pd.concat([
        pd.DataFrame([{"xgf_pg": xgf_pg, "xga_pg": xga_pg}]),
        pos_value_feats,
    ], axis=1)
    raw_ppg = float(ep_model.predict(feat)[0])

    # Normalize each player's games to a 38-game equivalent.
    def _norm_games(row) -> float:
        max_g = season_max_games(str(row.get("league", "")), str(row.get("season", "")))
        g = min(float(row.get("games") or 0), max_g)
        return g * 38.0 / max_g

    avg_games = float(players.apply(_norm_games, axis=1).mean())
    avg_games = max(1.0, avg_games)

    # Base: always a full 38-game season (no downward penalty)
    base_pts = raw_ppg * 38.0

    # Positive games bonus: +1 pt per average game above 30
    games_bonus = max(0.0, avg_games - GAMES_BONUS_THRESHOLD)

    # Champion bonus: players whose club won the league that season
    champions = _load_champions()
    champion_count = 0
    for _, row in players.iterrows():
        league = str(row.get("league", ""))
        club   = str(row.get("club", ""))
        season = str(row.get("season", ""))
        if champions.get(league, {}).get(season) == club:
            champion_count += 1
    champion_bonus = CHAMPION_BONUS_PER * min(champion_count, CHAMPION_BONUS_CAP)

    raw_pts = base_pts + games_bonus + champion_bonus
    predicted_points = float(np.clip(raw_pts, 0, 114))

    pedigree = squad_pedigree(players)

    return {
        "predicted_points": predicted_points,
        "tier": tier(predicted_points),
        "record": points_to_record(predicted_points),
        "avg_games": round(avg_games, 1),
        "games_bonus": round(games_bonus, 1),
        "champion_count": champion_count,
        "champion_bonus": champion_bonus,
        "breakdown": {
            "attack_xgf_pg": round(xgf_pg, 3),
            "defense_xga_pg": round(xga_pg, 3),
            "position_value": {c: round(float(v), 3) for c, v in pos_value_feats.iloc[0].items()},
            "ppg_hat": round(raw_ppg, 3),
            "base_pts": round(base_pts, 1),
            "games_bonus": round(games_bonus, 1),
            "champion_bonus": champion_bonus,
            "offensive_features": {c: round(float(v), 3) for c, v in off_feats.iloc[0].items()},
        },
        "pedigree": pedigree,
    }
