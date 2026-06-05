"""
Game state: formation config, slot tracking, slot-machine roll, rerolls.
Used by both the CLI prototype and the Streamlit app.

All public state is held in a plain dict so Streamlit can stash it in
st.session_state without any serialisation surprises.
"""

import json
import random
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).parent.parent
META_PATH = ROOT / "meta.json"
ARTIFACTS = ROOT / "artifacts"

FORMATIONS: dict[str, dict[str, int]] = {
    "4-4-2":   {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
    "4-3-3":   {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
    "3-5-2":   {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
    "4-2-3-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},
    "3-4-3":   {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
    "5-3-2":   {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2},
}

DEFAULT_REROLLS = 3


def new_game(formation: str, seed: Optional[int] = None) -> dict:
    if formation not in FORMATIONS:
        raise ValueError(f"Unknown formation '{formation}'. Choose from: {list(FORMATIONS)}")
    slots = FORMATIONS[formation]
    return {
        "formation": formation,
        "slots": slots.copy(),
        "filled": {pos: 0 for pos in slots},
        "drafted": [],                   # list of player dicts
        "drafted_ids": set(),            # understat_id set — no dupes
        "rerolls_left": DEFAULT_REROLLS,
        "round": 0,                      # 0-indexed
        "current_cell": None,            # (club, season) currently rolled
        "seed": seed if seed is not None else random.randint(0, 2**32),
        "rng": None,
        "complete": False,
    }


def _get_rng(state: dict) -> random.Random:
    if state["rng"] is None:
        state["rng"] = random.Random(state["seed"])
    return state["rng"]


def _load_valid_cells() -> list[tuple[str, str]]:
    if not META_PATH.exists():
        raise FileNotFoundError("meta.json not found — run data/build.py first")
    with open(META_PATH) as f:
        meta = json.load(f)
    return [(c["club"], c["season"]) for c in meta["valid_cells"]]


def _open_buckets(state: dict) -> list[str]:
    return [pos for pos, total in state["slots"].items() if state["filled"][pos] < total]


def roll(state: dict) -> tuple[str, str]:
    cells = _load_valid_cells()
    rng = _get_rng(state)
    club, season = rng.choice(cells)
    state["current_cell"] = (club, season)
    return club, season


def reroll(state: dict) -> tuple[str, str]:
    if state["rerolls_left"] <= 0:
        raise ValueError("No rerolls remaining")
    state["rerolls_left"] -= 1
    return roll(state)


def get_candidates(state: dict, players_df: pd.DataFrame) -> pd.DataFrame:
    if state["current_cell"] is None:
        return pd.DataFrame()
    club, season = state["current_cell"]
    open_buckets = _open_buckets(state)

    mask = (
        (players_df["club"] == club)
        & (players_df["season"] == season)
        & (~players_df["understat_id"].isin(state["drafted_ids"]))
        & (players_df["minutes"].fillna(0) >= 450)
    )
    pool = players_df[mask].copy()

    eligible = pool[pool["eligible_buckets"].apply(
        lambda buckets: bool(set(buckets) & set(open_buckets))
    )]
    return eligible.reset_index(drop=True)


def draft_player(state: dict, player_row: dict, bucket: str) -> None:
    open_buckets = _open_buckets(state)
    if bucket not in open_buckets:
        raise ValueError(f"Bucket '{bucket}' is full or invalid")
    eligible = player_row.get("eligible_buckets", [])
    if bucket not in eligible:
        raise ValueError(f"Player not eligible for bucket '{bucket}'")
    if player_row["understat_id"] in state["drafted_ids"]:
        raise ValueError("Player already drafted")

    player_row = dict(player_row)
    player_row["drafted_bucket"] = bucket
    state["drafted"].append(player_row)
    state["drafted_ids"].add(player_row["understat_id"])
    state["filled"][bucket] += 1
    state["round"] += 1
    if state["round"] == 11:
        state["complete"] = True


def roster_summary(state: dict) -> str:
    lines = [f"Formation: {state['formation']}  |  Rerolls left: {state['rerolls_left']}"]
    lines.append(f"Round: {state['round'] + 1}/11\n")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        total = state["slots"].get(pos, 0)
        filled = state["filled"].get(pos, 0)
        players_in_slot = [
            f"{p['player_name']} (20{p['season'][:2]}-{p['season'][2:]})"
            for p in state["drafted"] if p["drafted_bucket"] == pos
        ]
        filled_str = ", ".join(players_in_slot) if players_in_slot else "—"
        open_str = f"({total - filled} open)"
        lines.append(f"  {pos:4s} [{filled}/{total}] {open_str}  {filled_str}")
    return "\n".join(lines)
