"""
Feature engineering — SINGLE SOURCE OF TRUTH for both training and scoring.

Training (real team-seasons):
  xGF_pg = team's actual Understat xGF / games  (measured directly)
  xGA_pg = team's actual Understat xGA / games  (measured directly)

Scoring (drafted XI):
  xGF_pg = sum of outfield players' npxg_pg  (reconstructed from individual stats)
  xGA_pg = Stage 2 model applied to DEF/GK value_z  (estimated, since xGA is not per-player)

Both paths produce the same two-feature vector [xGF_pg, xGA_pg] fed into Stage 1.

Calibration: after building, confirm that summing a real team's starting XI's npxg_pg
recovers something close to the team's measured xGF_pg. Apply a scaling factor if needed.
"""

import numpy as np
import pandas as pd

# Age-weight parameters: logistic ramp, normalised so peak_def_age → 1.0.
# Young players carry a future-potential premium in market value that doesn't
# reflect current defensive ability; this discounts that premium.
_AGE_INFLECTION = 22.0   # age at which weight = 0.5 (before normalisation)
_AGE_K          = 0.5    # logistic steepness
_PEAK_DEF_AGE   = 27.0   # age at which weight is normalised to 1.0


def age_value_weight(age: float) -> float:
    """
    Discount factor in [0, 1] applied to value_z for defensive players.
    Ramps from ~0.13 at 18 up to 1.0 at peak_def_age (27), capped there.
    Values at or above peak age are not discounted.
    """
    raw = 1.0 / (1.0 + np.exp(-_AGE_K * (age - _AGE_INFLECTION)))
    norm = 1.0 / (1.0 + np.exp(-_AGE_K * (_PEAK_DEF_AGE - _AGE_INFLECTION)))
    return float(min(1.0, raw / norm))


def compute_xi_xgf(players: pd.DataFrame) -> float:
    """
    Estimate an XI's xGF rate as the sum of outfield players' npxg_pg.
    GKs are excluded (they don't generate expected goals in open play).
    Missing npxg_pg values → impute with position-bucket median (handled by caller).
    """
    outfield = players[players["primary_bucket"] != "GK"]
    vals = outfield["npxg_pg"].fillna(0)
    return float(vals.sum())


def compute_xi_xga(players: pd.DataFrame, stage2_model) -> float:
    """
    Estimate an XI's xGA rate using the Stage 2 model.
    stage2_model.predict() takes a 1-row DataFrame with the defensive value index.

    If stage2_model is None (not trained yet), returns NaN.
    """
    if stage2_model is None:
        return np.nan
    feat = build_defensive_value_index(players)
    return float(stage2_model.predict(feat)[0])


def build_defensive_value_index(players: pd.DataFrame) -> pd.DataFrame:
    """
    One-row feature vector for Stage 2: average age-adjusted value_z of DEF+GK players.
    Prefers age_adj_value_z (age-weighted) over raw value_z when available.
    Missing values imputed with 0 (= position-season average quality).
    """
    defensive = players[players["primary_bucket"].isin(["GK", "DEF"])]
    if defensive.empty:
        return pd.DataFrame([{"def_value_z": 0.0}])
    val_col = "age_adj_value_z" if "age_adj_value_z" in defensive.columns else "value_z"
    val = defensive[val_col].fillna(0).mean()
    return pd.DataFrame([{"def_value_z": float(val)}])


def compute_measured_team_features(team_season_df: pd.DataFrame) -> pd.DataFrame:
    """
    For training: return one row per (league, season, team) with [xGF_pg, xGA_pg].
    team_season_df should be understat_team_seasons.parquet (already aggregated).
    """
    return team_season_df[["league", "season", "team", "xgf_pg", "xga_pg", "points_pg", "games"]].copy()


def squad_pedigree(players: pd.DataFrame) -> dict:
    """
    Secondary signal shown on the reveal screen: total normalized value + caps.
    Not used in the headline rating.
    """
    value_z_mean = players["value_z"].fillna(0).mean()
    caps = 0
    if "international_caps" in players.columns:
        caps = players["international_caps"].fillna(0).sum()
    return {"squad_value_z": float(value_z_mean), "total_caps": int(caps)}
