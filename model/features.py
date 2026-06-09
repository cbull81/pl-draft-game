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


def season_max_games(league: str, season: str) -> int:
    """
    Max games in a full season for a league.
    Bundesliga: always 34 (18 teams since inception).
    Ligue 1: 34 from 2023/24 onwards (reduced from 20 to 18 teams), else 38.
    All others (PL, La Liga, Serie A): 38.
    """
    if league == "GER-Bundesliga":
        return 34
    if league == "FRA-Ligue 1":
        start_year = int("20" + season[:2]) if len(season) == 4 else int(season)
        if start_year >= 2023:
            return 34
    return 38


def age_value_weight(age: float) -> float:
    """
    Discount factor in [0, 1] applied to value_z for defensive players.
    Ramps from ~0.13 at 18 up to 1.0 at peak_def_age (27), capped there.
    Values at or above peak age are not discounted.
    """
    raw = 1.0 / (1.0 + np.exp(-_AGE_K * (age - _AGE_INFLECTION)))
    norm = 1.0 / (1.0 + np.exp(-_AGE_K * (_PEAK_DEF_AGE - _AGE_INFLECTION)))
    return float(min(1.0, raw / norm))


# Features fed into Stage 0 (xGF model). Order matters — must match training.
OFFENSIVE_COLS = ["npxg_pg", "xa_pg", "xg_buildup_pg", "key_passes_pg", "shots_pg"]


def build_offensive_features(players: pd.DataFrame) -> pd.DataFrame:
    """
    Sum of outfield players' per-game offensive stats (Stage 0 input).

    Pure per-game quality — no games weighting here. Availability is accounted for
    separately by scaling ppg_hat by avg games played (see score.py).
    GKs excluded. Missing values filled with 0.
    """
    outfield = players[players["primary_bucket"] != "GK"]
    row = {}
    for col in OFFENSIVE_COLS:
        vals = outfield[col].fillna(0) if col in outfield.columns else pd.Series(0.0, index=outfield.index)
        row[f"sum_{col}"] = float(vals.sum())
    return pd.DataFrame([row])[[f"sum_{c}" for c in OFFENSIVE_COLS]]


def compute_xi_xgf(players: pd.DataFrame, stage0_model=None) -> float:
    """
    Estimate an XI's xGF per game.
    With Stage 0 model: Ridge(Σnpxg, Σxa, Σbuildup) — accounts for creative midfielders via xA.
    Without: falls back to simple Σnpxg_pg.
    """
    if stage0_model is not None:
        feat = build_offensive_features(players)
        return float(stage0_model.predict(feat)[0])
    outfield = players[players["primary_bucket"] != "GK"]
    return float(outfield["npxg_pg"].fillna(0).sum())


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
    Uses drafted_bucket when present (scoring path) so a CDM drafted as DEF
    contributes to the defensive index; falls back to primary_bucket for training.
    """
    bucket_col = "drafted_bucket" if "drafted_bucket" in players.columns else "primary_bucket"
    defensive = players[players[bucket_col].isin(["GK", "DEF"])]
    if defensive.empty:
        return pd.DataFrame([{"def_value_z": 0.0}])
    val_col = "age_adj_value_z" if "age_adj_value_z" in defensive.columns else "value_z"
    val = defensive[val_col].fillna(0).mean()
    return pd.DataFrame([{"def_value_z": float(val)}])


# Position groups and their Stage 1 feature names. Order matches the model's feature vector.
POSITION_VALUE_GROUPS = [("GK", "gk_value_z"), ("DEF", "def_value_z"), ("MID", "mid_value_z"), ("FWD", "fwd_value_z")]


def build_position_value_features(players: pd.DataFrame) -> pd.DataFrame:
    """
    One-row feature vector of mean age_adj_value_z per position group (Stage 1 input).

    value_z is already position×season-normalised, so a defender's value_z is
    relative to other defenders that season — no cross-position inflation.
    Missing values imputed with 0 (position-season median quality).

    Uses drafted_bucket when present (scoring path) so the per-position values
    match exactly what the reveal screen displays. Falls back to primary_bucket
    for training (real-team data has no drafted_bucket).
    """
    val_col = "age_adj_value_z" if "age_adj_value_z" in players.columns else "value_z"
    bucket_col = "drafted_bucket" if "drafted_bucket" in players.columns else "primary_bucket"
    row = {}
    for bucket, feat_name in POSITION_VALUE_GROUPS:
        grp = players[players[bucket_col] == bucket][val_col]
        row[feat_name] = float(grp.fillna(0).mean()) if not grp.empty else 0.0
    return pd.DataFrame([row])[[feat for _, feat in POSITION_VALUE_GROUPS]]


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
