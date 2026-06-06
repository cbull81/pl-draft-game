"""
World Cup 2026 game state — mirrors state.py but rolls national teams.
"""

import json
import random
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).parent.parent
META_PATH = ROOT / "meta.json"
ARTIFACTS = ROOT / "artifacts"

from game.state import FORMATIONS, _open_buckets  # reuse formation config

DEFAULT_REROLLS = 0


def new_wc_game(formation: str, seed: Optional[int] = None) -> dict:
    if formation not in FORMATIONS:
        raise ValueError(f"Unknown formation '{formation}'. Choose from: {list(FORMATIONS)}")
    slots = FORMATIONS[formation]
    return {
        "mode":         "wc",
        "formation":    formation,
        "slots":        slots.copy(),
        "filled":       {pos: 0 for pos in slots},
        "drafted":      [],
        "drafted_ids":  set(),       # (wc_team, tm_id) pairs — no dupes
        "rerolls_left": DEFAULT_REROLLS,
        "round":        0,
        "current_team": None,        # display name of currently rolled team
        "seed":         seed if seed is not None else random.randint(0, 2**32),
        "rng":          None,
        "complete":     False,
    }


def _get_rng(state: dict) -> random.Random:
    if state["rng"] is None:
        state["rng"] = random.Random(state["seed"])
    return state["rng"]


def _load_wc_teams() -> list[str]:
    if not META_PATH.exists():
        raise FileNotFoundError("meta.json not found — run data/fetch_worldcup.py first")
    with open(META_PATH) as f:
        meta = json.load(f)
    return [t["name"] for t in meta.get("wc_teams", [])]


def roll_team(state: dict) -> str:
    teams = _load_wc_teams()
    if not teams:
        raise ValueError("No WC teams found — run data/fetch_worldcup.py first")
    rng = _get_rng(state)
    team = rng.choice(teams)
    state["current_team"] = team
    return team


def reroll_team(state: dict) -> str:
    if state["rerolls_left"] <= 0:
        raise ValueError("No rerolls remaining")
    state["rerolls_left"] -= 1
    return roll_team(state)


def get_wc_candidates(state: dict, wc_df: pd.DataFrame) -> pd.DataFrame:
    if state["current_team"] is None:
        return pd.DataFrame()
    open_buckets = _open_buckets(state)

    # exclude already-drafted (by tm_id within wc context)
    drafted_ids = state["drafted_ids"]  # set of tm_id strings

    mask = (
        (wc_df["wc_team"] == state["current_team"])
        & (~wc_df["tm_id"].isin(drafted_ids))
        & wc_df["tm_id"].notna()
        & (wc_df["tm_id"] != "None")
    )
    pool = wc_df[mask].copy()

    # parse eligible_buckets
    pool["_buckets"] = pool["eligible_buckets"].fillna("").str.split(",").apply(
        lambda b: [x for x in b if x]
    )
    eligible = pool[pool["_buckets"].apply(
        lambda buckets: bool(set(buckets) & set(open_buckets))
    )].drop(columns=["_buckets"])

    return eligible.reset_index(drop=True)


def draft_wc_player(state: dict, player_row: dict, bucket: str) -> None:
    open_buckets = _open_buckets(state)
    if bucket not in open_buckets:
        raise ValueError(f"Bucket '{bucket}' is full or invalid")

    buckets = player_row.get("eligible_buckets", "")
    if isinstance(buckets, str):
        buckets = [b for b in buckets.split(",") if b]
    if bucket not in buckets:
        raise ValueError(f"Player not eligible for bucket '{bucket}'")

    tm_id = player_row.get("tm_id")
    if tm_id in state["drafted_ids"]:
        raise ValueError("Player already drafted")

    player_row = dict(player_row)
    player_row["drafted_bucket"] = bucket
    state["drafted"].append(player_row)
    state["drafted_ids"].add(tm_id)
    state["filled"][bucket] += 1
    state["round"] += 1
    if state["round"] == 11:
        state["complete"] = True
