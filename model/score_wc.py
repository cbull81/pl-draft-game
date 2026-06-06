"""
World Cup scoring — additive quality index.

No xG / Ridge model dependency. Score = weighted sum of:
  0.60 * value_z         (position-normalised market value)
  0.25 * caps_z          (log-normalised international caps)
  0.15 * goals_pg_z      (goals per cap, clipped >= 0)

Thresholds are distribution-based, computed by fetch_worldcup.py and
stored in meta.json under "wc_tier_thresholds".
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent

# ── Tier configuration ────────────────────────────────────────────────────────

WC_BG = {
    "World Cup Champions": "#fef3c7",
    "Finalists":           "#f0fdf4",
    "Semi-finalists":      "#dbeafe",
    "Quarter-finalists":   "#ede9fe",
    "Round of 16":         "#f9fafb",
    "Group Stage Exit":    "#fee2e2",
}

WC_ACCENT = {
    "World Cup Champions": "#d97706",
    "Finalists":           "#16a34a",
    "Semi-finalists":      "#3b82f6",
    "Quarter-finalists":   "#7c3aed",
    "Round of 16":         "#6b7280",
    "Group Stage Exit":    "#ef4444",
}

# Fallback thresholds if meta.json not available
_DEFAULT_THRESHOLDS = [
    ("World Cup Champions", 0.90),
    ("Finalists",           0.65),
    ("Semi-finalists",      0.35),
    ("Quarter-finalists",   0.05),
    ("Round of 16",        -0.25),
    ("Group Stage Exit",   -9.99),
]


def _load_thresholds() -> list[tuple[str, float]]:
    meta_path = ROOT / "meta.json"
    if not meta_path.exists():
        return _DEFAULT_THRESHOLDS
    with open(meta_path) as f:
        meta = json.load(f)
    thresholds = meta.get("wc_tier_thresholds", {})
    if not thresholds:
        return _DEFAULT_THRESHOLDS
    ordered = [
        "World Cup Champions", "Finalists", "Semi-finalists",
        "Quarter-finalists", "Round of 16", "Group Stage Exit",
    ]
    return [(t, float(thresholds.get(t, -9.99))) for t in ordered]


def wc_tier(score: float) -> str:
    for label, threshold in _load_thresholds():
        if score >= threshold:
            return label
    return "Group Stage Exit"


# ── Pixel art ─────────────────────────────────────────────────────────────────

def wc_tier_pixel_art(tier: str) -> dict:
    bg = "transparent"
    arts = {
        "World Cup Champions": {
            "rows": [
                "..GGGGG..",
                ".G.....G.",
                "GG.....GG",
                "G.......G",
                "GG.....GG",
                ".GGGGGGG.",
                "....G....",
                "...GGG...",
                ".GGGGGGG.",
            ],
            "palette": {".": bg, "G": "#fbbf24"},
            "label": "CHAMPIONS!",
        },
        "Finalists": {
            "rows": [
                "...S...",
                "..SSS..",
                ".SSSSS.",
                "SSSSSSS",
                ".SSSSS.",
                "..SSS..",
                "...S...",
            ],
            "palette": {".": bg, "S": "#94a3b8"},
            "label": "FINALISTS",
        },
        "Semi-finalists": {
            "rows": [
                "...B...",
                "..BBB..",
                ".BBBBB.",
                "BBBBBBB",
                ".BBBBB.",
                "..BBB..",
                "...B...",
            ],
            "palette": {".": bg, "B": "#60a5fa"},
            "label": "SEMI-FINALS",
        },
        "Quarter-finalists": {
            "rows": [
                "...P...",
                "..PPP..",
                ".PPPPP.",
                "PPPPPPP",
                ".PPPPP.",
                "..PPP..",
                "...P...",
            ],
            "palette": {".": bg, "P": "#a78bfa"},
            "label": "QUARTERS",
        },
        "Round of 16": {
            "rows": [
                "..KKK..",
                ".KWWWK.",
                "KWWKWWK",
                "KWKWKWK",
                "KWWKWWK",
                ".KWWWK.",
                "..KKK..",
            ],
            "palette": {".": bg, "W": "#f9fafb", "K": "#1f2937"},
            "label": "ROUND OF 16",
        },
        "Group Stage Exit": {
            "rows": [
                "RRRRRRR",
                "...R...",
                "...R...",
                "...R...",
                ".RRRRR.",
                "..RRR..",
                "...R...",
            ],
            "palette": {".": bg, "R": "#ef4444"},
            "label": "GROUP EXIT",
        },
    }
    return arts.get(tier, arts["Group Stage Exit"])


# ── Scoring ───────────────────────────────────────────────────────────────────

WEIGHTS = {"value": 0.60, "caps": 0.25, "goals": 0.15}


def _safe(v, fallback=0.0) -> float:
    if v is None:
        return fallback
    try:
        f = float(v)
        return fallback if np.isnan(f) else f
    except (TypeError, ValueError):
        return fallback


def score_wc_xi(players: pd.DataFrame) -> dict:
    """
    Score a drafted WC XI using the additive index.
    Expected columns in players: value_z (or wc_value_z), caps_z, goals_pg_z.
    Missing values are imputed at 0 (position median for this WC cohort).
    """
    df = players.copy()

    # Accept either column name
    if "value_z" not in df.columns and "wc_value_z" in df.columns:
        df["value_z"] = df["wc_value_z"]

    records = df.to_dict("records")

    player_scores = []
    position_value: dict[str, list] = {"GK": [], "DEF": [], "MID": [], "FWD": []}

    value_z_vals, caps_z_vals, goals_z_vals = [], [], []

    for p in records:
        vz = _safe(p.get("value_z"))
        cz = _safe(p.get("caps_z"))
        gz = _safe(p.get("goals_pg_z"))

        score = WEIGHTS["value"] * vz + WEIGHTS["caps"] * cz + WEIGHTS["goals"] * gz
        player_scores.append(score)
        value_z_vals.append(vz)
        caps_z_vals.append(cz)
        goals_z_vals.append(gz)

        bucket = p.get("drafted_bucket") or p.get("primary_bucket", "MID")
        if bucket in position_value:
            position_value[bucket].append(vz)

    team_score = float(np.mean(player_scores)) if player_scores else 0.0
    tier = wc_tier(team_score)

    total_caps = sum(
        int(_safe(p.get("international_caps"), 0))
        for p in records
    )

    pos_avg = {
        "gk_value_z":  float(np.mean(position_value["GK"]))  if position_value["GK"]  else 0.0,
        "def_value_z": float(np.mean(position_value["DEF"])) if position_value["DEF"] else 0.0,
        "mid_value_z": float(np.mean(position_value["MID"])) if position_value["MID"] else 0.0,
        "fwd_value_z": float(np.mean(position_value["FWD"])) if position_value["FWD"] else 0.0,
    }

    avg_value_z = float(np.mean(value_z_vals)) if value_z_vals else 0.0
    avg_caps_z  = float(np.mean(caps_z_vals))  if caps_z_vals  else 0.0
    avg_goals_z = float(np.mean(goals_z_vals)) if goals_z_vals else 0.0

    # Pedigree sub-score: mean value_z across squad
    pedigree = {
        "squad_value_z": round(avg_value_z, 3),
        "squad_caps_z":  round(avg_caps_z, 3),
    }

    return {
        "wc_index":   round(team_score, 3),
        "tier":       tier,
        "breakdown": {
            "avg_value_z":    round(avg_value_z, 3),
            "avg_caps_z":     round(avg_caps_z, 3),
            "avg_goals_pg_z": round(avg_goals_z, 3),
            "wc_index":       round(team_score, 3),
            "position_value": {k: round(v, 3) for k, v in pos_avg.items()},
        },
        "pedigree":   pedigree,
        "total_caps": total_caps,
    }
