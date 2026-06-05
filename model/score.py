"""
Score a drafted XI → predicted points + breakdown.
Called by game/state.py after all 11 slots are filled.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model.features import (
    build_offensive_features,
    compute_xi_xgf,
    compute_xi_xga,
    season_max_games,
    squad_pedigree,
)

ARTIFACTS = Path(__file__).parent.parent / "artifacts"

GAMES_THRESHOLD = 30  # avg games played per player; above this, no availability penalty

TIER_THRESHOLDS = [
    ("Invincible Chase (38-0)", 105),
    ("Title Contender", 85),
    ("European Football", 65),
    ("Mid-Table", 45),
    ("Relegation Battle", 0),
]


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

    Per-game quality (ppg_hat) is estimated from individual per-game rates, then
    scaled by the XI's average games played — not a fixed 38. Drafting players who
    only appeared in 15 games contributes proportionally less to total points.
    """
    ep_model, xga_model, xgf_model = load_models()

    off_feats = build_offensive_features(players)
    xgf_pg = compute_xi_xgf(players, xgf_model)
    xga_pg = compute_xi_xga(players, xga_model)

    if np.isnan(xga_pg):
        xga_pg = 1.2  # ~PL average fallback if Stage 2 not trained

    val_col = "age_adj_value_z" if "age_adj_value_z" in players.columns else "value_z"
    squad_value_z = float(players[val_col].fillna(0).mean())

    feat = pd.DataFrame([{"xgf_pg": xgf_pg, "xga_pg": xga_pg, "squad_value_z": squad_value_z}])
    raw_ppg = float(ep_model.predict(feat)[0])

    # Normalize each player's games to a 38-game equivalent before averaging.
    # Bundesliga has 34-game seasons; Ligue 1 has 34-game seasons from 2023/24 onwards.
    def _norm_games(row) -> float:
        max_g = season_max_games(str(row.get("league", "")), str(row.get("season", "")))
        g = min(float(row.get("games") or 0), max_g)
        return g * 38.0 / max_g

    avg_games = float(players.apply(_norm_games, axis=1).mean())
    avg_games = max(1.0, avg_games)
    effective_games = 38.0 if avg_games >= GAMES_THRESHOLD else avg_games
    raw_pts = raw_ppg * effective_games
    predicted_points = float(np.clip(raw_pts, 0, 114))

    pedigree = squad_pedigree(players)

    return {
        "predicted_points": predicted_points,
        "tier": tier(predicted_points),
        "record": points_to_record(predicted_points, games=round(effective_games)),
        "avg_games": round(avg_games, 1),
        "effective_games": round(effective_games, 1),
        "breakdown": {
            "attack_xgf_pg": round(xgf_pg, 3),
            "defense_xga_pg": round(xga_pg, 3),
            "squad_value_z": round(squad_value_z, 3),
            "ppg_hat": round(raw_ppg, 3),
            "offensive_features": {c: round(float(v), 3) for c, v in off_feats.iloc[0].items()},
        },
        "pedigree": pedigree,
    }
