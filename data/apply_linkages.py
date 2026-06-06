"""
Apply manual linkage fixes from artifacts/missing_linkages.csv → artifacts/resolver_overrides.csv,
then rebuild artifacts/players.parquet.

Handles two fix types:
  1. Direct tm_id:     row has a value in the `tm_id` column  → used as-is
  2. common_name:      row has a value in `common_name` but no tm_id
                       → fuzzy-matched against tm_players_raw by name, result reported for review

Workflow:
  1. Fill in `tm_id` and/or `common_name` in missing_linkages.csv
  2. Run:  python data/apply_linkages.py
  3. Review any common_name matches printed (auto-accepted if score ≥ 90)
  4. Rebuild is automatic; restart the web server afterwards
"""

from pathlib import Path
import re
import pandas as pd
import subprocess
import sys

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
LINKAGES  = ARTIFACTS / "missing_linkages.csv"
OVERRIDES = ARTIFACTS / "resolver_overrides.csv"

COMMON_NAME_COL  = "common_name"
AUTO_ACCEPT_THRESHOLD = 90   # fuzz score; matches below this are skipped with a warning


def _normalize(s: str) -> str:
    """Lowercase, strip accents crudely, remove punctuation for fuzzy comparison."""
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s))
    s = s.encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    return s.strip()


def _resolve_common_names(pending: pd.DataFrame, tm: pd.DataFrame) -> list[dict]:
    """
    For each row in `pending` (has common_name, no tm_id), find the best TM match.
    Returns list of dicts with understat_id, tm_id, player_name, score, matched_name.
    """
    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        print("  rapidfuzz not installed — skipping common_name resolution. pip install rapidfuzz")
        return []

    tm = tm.copy()
    tm["name_norm"] = tm["name"].apply(_normalize)
    choices = tm["name_norm"].tolist()
    tm_ids  = tm["player_id"].tolist()
    tm_names = tm["name"].tolist()

    resolved = []
    for _, row in pending.iterrows():
        query_raw = str(row[COMMON_NAME_COL]).strip()
        query = _normalize(query_raw)
        match = process.extractOne(query, choices, scorer=fuzz.token_sort_ratio)
        if match is None:
            print(f"  [SKIP] {row['player_name']} / '{query_raw}' — no match found")
            continue
        matched_norm, score, idx = match
        tm_id    = str(tm_ids[idx])
        tm_name  = tm_names[idx]
        if score >= AUTO_ACCEPT_THRESHOLD:
            print(f"  [AUTO] {row['player_name']} / '{query_raw}' → '{tm_name}' (id={tm_id}, score={score})")
            resolved.append({
                "understat_id": str(row["understat_id"]).strip(),
                "tm_id": tm_id,
                "note": f"common_name '{query_raw}' → '{tm_name}' (score={score})",
            })
        else:
            print(f"  [SKIP] {row['player_name']} / '{query_raw}' → best match '{tm_name}' (id={tm_id}, score={score}) — below threshold {AUTO_ACCEPT_THRESHOLD}")
    return resolved


def main():
    if not LINKAGES.exists():
        print(f"ERROR: {LINKAGES} not found.")
        sys.exit(1)

    raw = pd.read_csv(LINKAGES, dtype=str)

    # --- Direct tm_id fixes ---
    direct = raw[raw["tm_id"].notna() & (raw["tm_id"].str.strip() != "")].copy()
    direct["tm_id"] = direct["tm_id"].str.strip()
    direct["understat_id"] = direct["understat_id"].str.strip()
    print(f"Direct tm_id fixes:   {len(direct)}")

    # --- common_name rows (no tm_id) ---
    has_common = (
        raw.get(COMMON_NAME_COL, pd.Series(dtype=str)).notna() &
        (raw.get(COMMON_NAME_COL, pd.Series(dtype=str)).str.strip() != "")
    )
    pending = raw[has_common & (raw["tm_id"].isna() | (raw["tm_id"].str.strip() == ""))].copy()
    print(f"common_name to resolve: {len(pending)}")

    common_resolved = []
    if not pending.empty:
        print("\nResolving common_name rows against tm_players_raw...")
        tm = pd.read_parquet(ARTIFACTS / "tm_players_raw.parquet")
        common_resolved = _resolve_common_names(pending, tm)
        print(f"\nAuto-resolved {len(common_resolved)} of {len(pending)} common_name rows.")

    # --- Combine all new overrides ---
    new_direct = direct[["understat_id", "tm_id", "player_name"]].rename(columns={"player_name": "note"})
    new_direct["note"] = "direct: " + new_direct["note"]

    new_common = pd.DataFrame(common_resolved) if common_resolved else pd.DataFrame(columns=["understat_id","tm_id","note"])

    all_new = pd.concat([new_direct, new_common], ignore_index=True)

    if all_new.empty:
        print("\nNo fixes to apply.")
        return

    # --- Merge with existing overrides ---
    if OVERRIDES.exists():
        existing = pd.read_csv(OVERRIDES, dtype=str)
    else:
        existing = pd.DataFrame(columns=["understat_id", "tm_id", "note"])

    existing = existing[~existing["understat_id"].isin(all_new["understat_id"])].copy()
    combined = pd.concat([existing, all_new], ignore_index=True)
    combined.to_csv(OVERRIDES, index=False)
    print(f"\nUpdated {OVERRIDES}  ({len(combined)} total overrides, {len(all_new)} new/updated)")

    # --- Rebuild ---
    root = Path(__file__).parent.parent
    for script in ["data/resolve.py", "data/build.py"]:
        print(f"\nRunning {script} ...")
        result = subprocess.run([sys.executable, script], capture_output=False, cwd=root)
        if result.returncode != 0:
            print(f"{script} failed — check output above.")
            sys.exit(1)

    print("\nDone! Restart the web server to pick up the updated players.parquet.")


if __name__ == "__main__":
    main()
