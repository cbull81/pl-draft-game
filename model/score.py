"""
Score a drafted XI → predicted points + breakdown.
Called by game/state.py after all 11 slots are filled.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model.features import (
    compute_xi_xgf,
    compute_xi_xga,
    squad_pedigree,
)

ARTIFACTS = Path(__file__).parent.parent / "artifacts"

TIER_THRESHOLDS = [
    ("Invincible Chase (38-0)", 105),
    ("Title Contender", 85),
    ("European Football", 65),
    ("Mid-Table", 45),
    ("Relegation Battle", 0),
]


def load_models():
    ep_path = ARTIFACTS / "ep_model.joblib"
    xga_path = ARTIFACTS / "xga_model.joblib"
    if not ep_path.exists():
        raise FileNotFoundError("Stage 1 model not found — run model/train.py first")
    ep = joblib.load(ep_path)
    xga = joblib.load(xga_path)["model"] if xga_path.exists() else None
    return ep["model"], xga


def tier(points: float) -> str:
    for label, threshold in TIER_THRESHOLDS:
        if points >= threshold:
            return label
    return "Relegation Battle"


def points_to_record(points: float) -> str:
    pts = max(0, min(114, round(points)))
    wins = pts // 3
    remainder = pts % 3
    draws = remainder
    losses = max(0, 38 - wins - draws)
    return f"{wins}W {draws}D {losses}L"


def score_xi(players: pd.DataFrame) -> dict:
    """
    players: 11-row DataFrame (one per drafted player).
    Returns: predicted_points, tier, record, attack/defense breakdown, pedigree.
    """
    ep_model, xga_model = load_models()

    xgf_pg = compute_xi_xgf(players)
    xga_pg = compute_xi_xga(players, xga_model)

    if np.isnan(xga_pg):
        # Stage 2 not available — fall back to league-average xGA
        xga_pg = 1.2  # ~PL average; replace with empirical value after training

    feat = pd.DataFrame([{"xgf_pg": xgf_pg, "xga_pg": xga_pg}])
    raw_ppg = float(ep_model.predict(feat)[0])
    raw_pts = raw_ppg * 38
    predicted_points = float(np.clip(raw_pts, 0, 114))

    pedigree = squad_pedigree(players)

    return {
        "predicted_points": predicted_points,
        "tier": tier(predicted_points),
        "record": points_to_record(predicted_points),
        "breakdown": {
            "attack_xgf_pg": round(xgf_pg, 3),
            "defense_xga_pg": round(xga_pg, 3),
            "ppg_hat": round(raw_ppg, 3),
        },
        "pedigree": pedigree,
    }
