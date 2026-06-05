"""
CLI prototype for the 38-0 draft game.
Run: conda run -n pl-draft python -m game.cli

Names + positions shown during draft; stats hidden until final reveal.

Attribution: Data from Understat.com and Transfermarkt (dcaribou/transfermarkt-datasets).
  Independent project, not affiliated with the Premier League.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from game.state import (
    FORMATIONS,
    _open_buckets,
    draft_player,
    get_candidates,
    new_game,
    reroll,
    roll,
    roster_summary,
)
from model.score import score_xi


def choose_formation() -> str:
    print("\n╔══════════════════════════════════════╗")
    print("║    38-0: Can you go 38 and 0?         ║")
    print("╚══════════════════════════════════════╝")
    print("\nPick your formation (locked for the whole game):\n")
    names = list(FORMATIONS)
    for i, name in enumerate(names, 1):
        slots = FORMATIONS[name]
        breakdown = "  ".join(f"{v}{k}" for k, v in slots.items() if k != "GK")
        print(f"  [{i}] {name:8s}  GK + {breakdown}")
    while True:
        choice = input("\nFormation number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        print("  Enter a number from the list.")


def print_candidates(candidates: pd.DataFrame, show_stats: bool) -> None:
    if show_stats:
        print(f"\n  {'#':>3}  {'Player':<26}  {'Position':<20}  {'Slots':<14}  {'npxG/g':>7}  {'xA/g':>6}  {'val_z':>6}  {'g':>3}")
        print(f"  {'─'*3}  {'─'*26}  {'─'*20}  {'─'*14}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*3}")
        for i, row in candidates.iterrows():
            sub_pos = row.get("sub_position", "—") or "—"
            buckets = ", ".join(row["eligible_buckets"])
            npxg = row.get("npxg_pg")
            xa   = row.get("xa_pg")
            vz   = row.get("value_z")
            g    = int(row.get("games", 0) or 0)
            npxg_s = f"{npxg:.3f}" if pd.notna(npxg) else "—"
            xa_s   = f"{xa:.3f}"   if pd.notna(xa)   else "—"
            vz_s   = f"{vz:+.2f}" if pd.notna(vz)   else "—"
            print(f"  [{i+1:2d}]  {row['player_name']:<26}  {sub_pos:<20}  {buckets:<14}  {npxg_s:>7}  {xa_s:>6}  {vz_s:>6}  {g:>3}")
    else:
        print(f"\n  {'#':>3}  {'Player':<28}  {'Position':<22}  Eligible Slots")
        print(f"  {'─'*3}  {'─'*28}  {'─'*22}  {'─'*20}")
        for i, row in candidates.iterrows():
            sub_pos = row.get("sub_position", "—") or "—"
            buckets = ", ".join(row["eligible_buckets"])
            print(f"  [{i+1:2d}]  {row['player_name']:<28}  {sub_pos:<22}  {buckets}")


def print_model_inputs(xi_df: pd.DataFrame, result: dict) -> None:
    bd = result["breakdown"]
    print("\n  ── Model Inputs " + "─" * 39)
    print(f"\n  {'Player':<26}  {'Slot':<4}  {'npxG/g':>7}  {'xA/g':>6}  {'val_z':>6}")
    print(f"  {'─'*26}  {'─'*4}  {'─'*7}  {'─'*6}  {'─'*6}")
    for _, row in xi_df.sort_values("drafted_bucket").iterrows():
        bucket = row.get("drafted_bucket", "?")
        npxg = row.get("npxg_pg")
        xa   = row.get("xa_pg")
        vz   = row.get("value_z")
        npxg_s = f"{npxg:.3f}" if pd.notna(npxg) else "—"
        xa_s   = f"{xa:.3f}"   if pd.notna(xa)   else "—"
        vz_s   = f"{vz:+.2f}" if pd.notna(vz)   else "—"
        print(f"  {row['player_name']:<26}  {bucket:<4}  {npxg_s:>7}  {xa_s:>6}  {vz_s:>6}")

    defensive = xi_df[xi_df["primary_bucket"].isin(["GK", "DEF"])]
    mean_def_vz = defensive["value_z"].fillna(0).mean() if not defensive.empty else 0.0

    print(f"\n  xGF_hat  = Σ npxG/g (outfield)  →  {bd['attack_xgf_pg']:.4f}")
    print(f"  def_val_z = mean value_z (DEF+GK) →  {mean_def_vz:+.4f}")
    print(f"  xGA_hat  = Stage 2 (def_val_z)    →  {bd['defense_xga_pg']:.4f}  (lower = better)")
    print(f"  ppg_hat  = Stage 1 [xGF, xGA]     →  {bd['ppg_hat']:.4f}")
    print(f"  pts      = ppg × 38               →  {bd['ppg_hat'] * 38:.1f}")
    print("  " + "─" * 54)


def choose_bucket(state: dict, eligible_buckets: list[str]) -> str:
    open_b = _open_buckets(state)
    valid = [b for b in eligible_buckets if b in open_b]
    if len(valid) == 1:
        return valid[0]
    print(f"\n  Player eligible for: {', '.join(valid)}")
    while True:
        choice = input("  Which slot? ").strip().upper()
        if choice in valid:
            return choice
        print(f"  Choose from: {', '.join(valid)}")


def play():
    players_path = ROOT / "artifacts" / "players.parquet"
    if not players_path.exists():
        print("\nERROR: players.parquet not found.")
        print("Run:  conda run -n pl-draft python -m data.fetch_understat")
        print("      conda run -n pl-draft python -m data.fetch_transfermarkt")
        print("      conda run -n pl-draft python -m data.resolve")
        print("      conda run -n pl-draft python -m data.build")
        sys.exit(1)

    players_df = pd.read_parquet(players_path)
    formation = choose_formation()
    state = new_game(formation)
    show_stats = False
    print(f"\nLocked: {formation}  |  Rerolls: {state['rerolls_left']}")

    while not state["complete"]:
        print("\n" + "─" * 55)
        print(roster_summary(state))
        print("─" * 55)

        club, season = roll(state)
        season_display = f"20{season[:2]}/{season[2:]}"
        print(f"\n🎰  Rolled: {club}  ({season_display})")

        while True:
            candidates = get_candidates(state, players_df)
            if candidates.empty:
                print("  No eligible players for open slots in this roll.")
                if state["rerolls_left"] > 0:
                    print(f"  Auto-rerolling ({state['rerolls_left']} rerolls left)…")
                    club, season = reroll(state)
                    season_display = f"20{season[:2]}/{season[2:]}"
                    print(f"  → {club}  ({season_display})")
                else:
                    print("  ERROR: No rerolls left and no eligible players — this shouldn't happen.")
                    sys.exit(1)
                continue

            print(f"\n  Squad: {club}  {season_display}")
            print_candidates(candidates, show_stats)

            stats_label = "on" if show_stats else "off"
            print(f"\n  [r] Reroll ({state['rerolls_left']} left)  [?] Roster  [s] Stats ({stats_label})")
            choice = input("\n  Pick number: ").strip().lower()

            if choice == "?":
                print("\n" + roster_summary(state))
                continue
            if choice == "s":
                show_stats = not show_stats
                print(f"  Stats display: {'on' if show_stats else 'off'}")
                continue
            if choice == "r":
                if state["rerolls_left"] <= 0:
                    print("  No rerolls left!")
                    continue
                club, season = reroll(state)
                season_display = f"20{season[:2]}/{season[2:]}"
                print(f"\n  → Rerolled: {club}  ({season_display})")
                continue

            if not choice.isdigit() or not (1 <= int(choice) <= len(candidates)):
                print("  Invalid choice.")
                continue

            idx = int(choice) - 1
            player_row = candidates.iloc[idx].to_dict()
            bucket = choose_bucket(state, player_row["eligible_buckets"])
            draft_player(state, player_row, bucket)
            print(f"\n  ✓  {player_row['player_name']} → {bucket}")
            break

    # ── Final reveal ───────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("             FINAL XI — REVEAL")
    print("═" * 55)
    print(roster_summary(state))

    xi_df = pd.DataFrame(state["drafted"])
    result = score_xi(xi_df)

    print_model_inputs(xi_df, result)

    print(f"\n  Predicted points:  {result['predicted_points']:.1f} / 114")
    print(f"  Record:            {result['record']}")
    print(f"  Tier:              {result['tier']}")

    pg = result["pedigree"]
    print(f"\n  Squad value score:  {pg['squad_value_z']:+.2f}  (vs. position peers)")
    if pg.get("total_caps", 0) > 0:
        print(f"  Total intl. caps:   {pg['total_caps']}")

    print("\n─────────────────────────────────────────────────")
    print("Data: Understat.com | Transfermarkt (dcaribou/transfermarkt-datasets)")
    print("Independent project, not affiliated with the Premier League.")


if __name__ == "__main__":
    play()
