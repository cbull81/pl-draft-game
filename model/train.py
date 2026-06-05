"""
Train the two-stage expected-points model.

Stage 1: points_pg ~ Ridge(xGF_pg, xGA_pg)
  Training data: Understat team-seasons, all 5 leagues, 2014/15→present (~1,000 rows)
  Validation: leave-one-season-and-league-out CV → RMSE + Spearman

Stage 2: xGA_pg ~ Ridge(def_value_z)
  Training data: PL team-seasons only (where we have TM valuations)
  Joins team xGA (measured) to team's DEF+GK value_z (from players.parquet)

Output: artifacts/ep_model.joblib, artifacts/xga_model.joblib

Run after data/build.py.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, RidgeCV

from model.features import (
    build_defensive_value_index,
    build_offensive_features,
    build_position_value_features,
    compute_measured_team_features,
    POSITION_VALUE_GROUPS,
)

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
ALPHAS = np.logspace(-3, 3, 25)


def build_stage1_matrix() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Load Understat team-seasons from all 5 leagues.
    Returns X (xgf_pg, xga_pg, gk/def/mid/fwd_value_z), y (points_pg), league, season.

    Position-group value features: mean age_adj_value_z per bucket (≥450 min players).
    value_z is already position×season-normalised so cross-position inflation is not
    an issue. The model learns separate weights for each group.
    """
    ts = pd.read_parquet(ARTIFACTS / "understat_team_seasons.parquet")
    ts = ts.dropna(subset=["xgf_pg", "xga_pg", "points_pg"])
    ts = ts[ts["games"] >= 10]

    players = pd.read_parquet(ARTIFACTS / "players.parquet")
    qualified = players[players["minutes"].fillna(0) >= 450]

    for bucket, feat_name in POSITION_VALUE_GROUPS:
        grp = (
            qualified[qualified["primary_bucket"] == bucket]
            .groupby(["league", "season", "club"])["age_adj_value_z"]
            .apply(lambda x: x.fillna(0).mean())
            .reset_index()
            .rename(columns={"age_adj_value_z": feat_name, "club": "team"})
        )
        ts = ts.merge(grp, on=["league", "season", "team"], how="left")
        ts[feat_name] = ts[feat_name].fillna(0)

    value_feat_names = [feat for _, feat in POSITION_VALUE_GROUPS]
    X = ts[["xgf_pg", "xga_pg"] + value_feat_names]
    y = ts["points_pg"]
    return X, y, ts["league"], ts["season"], ts


def build_stage0_matrix() -> tuple[pd.DataFrame, pd.Series]:
    """
    PL team-seasons: measured xGF_pg ~ f(Σ individual offensive stats).
    Ridge is justified here — npxg, xa, and buildup are correlated.
    """
    players = pd.read_parquet(ARTIFACTS / "players.parquet")
    ts = pd.read_parquet(ARTIFACTS / "understat_team_seasons.parquet")
    pl = ts[ts["league"] == "ENG-Premier League"][["season", "team", "xgf_pg"]].dropna()

    rows = []
    for _, row in pl.iterrows():
        team_players = players[
            (players["club"] == row["team"]) &
            (players["season"] == row["season"]) &
            (players["primary_bucket"] != "GK") &
            (players["minutes"].fillna(0) >= 450)
        ]
        if team_players.empty:
            continue
        feat = build_offensive_features(team_players)
        feat["xgf_pg"] = row["xgf_pg"]
        rows.append(feat)

    if not rows:
        print("  WARNING: No Stage 0 training rows — run build.py first.")
        return pd.DataFrame(), pd.Series()

    df = pd.concat(rows, ignore_index=True).dropna(subset=["xgf_pg"])
    X_cols = [c for c in df.columns if c.startswith("sum_")]
    return df[X_cols], df["xgf_pg"]


def build_stage2_matrix() -> tuple[pd.DataFrame, pd.Series]:
    """
    PL team-seasons: measured xGA_pg ~ f(defensive DEF+GK value_z)
    """
    ts = pd.read_parquet(ARTIFACTS / "understat_team_seasons.parquet")
    pl = ts[ts["league"] == "ENG-Premier League"].copy()
    players = pd.read_parquet(ARTIFACTS / "players.parquet")

    rows = []
    for _, row in pl.iterrows():
        season = row["season"]
        team = row["team"]
        team_players = players[(players["club"] == team) & (players["season"] == season)]
        if team_players.empty:
            continue
        feat = build_defensive_value_index(team_players)
        feat["season"] = season
        feat["team"] = team
        feat["xga_pg"] = row["xga_pg"]
        rows.append(feat)

    if not rows:
        print("  WARNING: No Stage 2 training rows — TM data may not be merged yet.")
        return pd.DataFrame(), pd.Series()

    stage2_df = pd.concat(rows, ignore_index=True).dropna(subset=["def_value_z", "xga_pg"])
    X2 = stage2_df[["def_value_z"]]
    y2 = stage2_df["xga_pg"]
    return X2, y2


def loso_cv(X: pd.DataFrame, y: pd.Series, league_s: pd.Series, season_s: pd.Series):
    """Leave-one-season-and-league-out cross-validation."""
    groups = list(zip(league_s, season_s))
    unique_groups = list(set(groups))
    preds = pd.Series(index=y.index, dtype=float)

    for held in unique_groups:
        train_mask = [g != held for g in groups]
        test_mask = [g == held for g in groups]
        m = LinearRegression()
        m.fit(X[train_mask], y[train_mask])
        preds[test_mask] = m.predict(X[test_mask])

    rmse = float(np.sqrt(((y - preds) ** 2).mean()))
    rho, pval = spearmanr(y, preds)
    return rmse, float(rho), float(pval)


def main():
    print("=== Stage 0: xGF_pg ~ RidgeCV(Σnpxg, Σxa, Σbuildup) ===")
    X0, y0 = build_stage0_matrix()
    if X0.empty:
        print("  Skipping Stage 0 — no training data.")
        stage0 = None
    else:
        print(f"  Rows: {len(X0)}  Features: {X0.columns.tolist()}")
        stage0 = RidgeCV(alphas=ALPHAS)
        stage0.fit(X0, y0)
        preds0 = stage0.predict(X0)
        rho0, _ = spearmanr(y0, preds0)
        r2_0 = float(1 - ((y0 - preds0)**2).sum() / ((y0 - y0.mean())**2).sum())
        print(f"  Best alpha: {stage0.alpha_:.4f}")
        print(f"  Coefficients: {dict(zip(X0.columns, stage0.coef_))}")
        print(f"  In-sample  Spearman ρ: {rho0:.3f}  R²: {r2_0:.3f}")

    print("\n=== Stage 1: points_pg ~ OLS(xGF_pg, xGA_pg, gk/def/mid/fwd_value_z) ===")
    X1, y1, league_s, season_s, ts = build_stage1_matrix()
    print(f"  Rows: {len(X1)}  (leagues: {league_s.nunique()}, seasons: {season_s.nunique()})")
    print(f"  points_pg: {y1.min():.3f} – {y1.max():.3f}  mean={y1.mean():.3f}")

    print("\n  Leave-one-season/league-out CV...")
    rmse, rho, pval = loso_cv(X1, y1, league_s, season_s)
    print(f"  RMSE (ppg): {rmse:.4f}")
    print(f"  Spearman ρ: {rho:.3f}  (p={pval:.4f})")

    stage1 = LinearRegression()
    stage1.fit(X1, y1)
    feat_names = ["xGF_pg", "xGA_pg"] + [feat for _, feat in POSITION_VALUE_GROUPS]
    coefs1 = dict(zip(feat_names, stage1.coef_))
    print(f"\n  Intercept: {stage1.intercept_:.4f}")
    print(f"  Coefficients: {coefs1}")
    if coefs1.get("xGF_pg", 0) <= 0 or coefs1.get("xGA_pg", 0) >= 0:
        print("  WARNING: unexpected signs — xGF should be +, xGA should be –")

    print("\n=== Stage 2: xGA_pg ~ Ridge(def_value_z) ===")
    X2, y2 = build_stage2_matrix()
    if X2.empty:
        print("  Skipping Stage 2 — no training data. Run build.py with TM data first.")
        stage2 = None
        stage2_r2 = None
    else:
        print(f"  Rows: {len(X2)}")
        stage2 = LinearRegression()
        stage2.fit(X2, y2)
        preds2 = stage2.predict(X2)
        rho2, _ = spearmanr(y2, preds2)
        stage2_r2 = float(1 - ((y2 - preds2)**2).sum() / ((y2 - y2.mean())**2).sum())
        print(f"  In-sample Spearman ρ: {rho2:.3f}  R²: {stage2_r2:.3f}")
        print("  (Low R² here is expected — value is a noisy proxy for defense)")

    print("\n=== Persisting models ===")
    if stage0 is not None:
        joblib.dump({"model": stage0, "feature_cols": list(X0.columns)}, ARTIFACTS / "xgf_model.joblib")
        print("  Saved xgf_model.joblib")
    else:
        print("  xgf_model.joblib NOT saved (no Stage 0 data)")

    stage1_artifact = {
        "model": stage1,
        "feature_cols": ["xgf_pg", "xga_pg"] + [feat for _, feat in POSITION_VALUE_GROUPS],
        "coefs": coefs1,
        "loso_rmse_ppg": rmse,
        "loso_spearman": rho,
    }
    joblib.dump(stage1_artifact, ARTIFACTS / "ep_model.joblib")
    print("  Saved ep_model.joblib")

    if stage2 is not None:
        joblib.dump({"model": stage2, "r2": stage2_r2}, ARTIFACTS / "xga_model.joblib")
        print("  Saved xga_model.joblib")
    else:
        print("  xga_model.joblib NOT saved (no Stage 2 data)")

    print("\nDone. Review coefficient signs and CV metrics before moving on (§4.1 of CLAUDE.md).")


if __name__ == "__main__":
    main()
