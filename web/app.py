"""
FastAPI web interface for 38-0.
Run: uvicorn web.app:app --reload
"""
import math
import sys
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from game.state import (
    FORMATIONS,
    _open_buckets,
    draft_player,
    get_candidates,
    new_game,
    reroll as state_reroll,
    roll as state_roll,
)
from game.wc_state import (
    draft_wc_player,
    get_wc_candidates,
    new_wc_game,
    reroll_team,
    roll_team,
)
from model.score import score_xi
from model.score_wc import score_wc_xi, wc_tier_pixel_art

app = FastAPI()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# ── Session store (in-memory) ─────────────────────────────────────────────────
_sessions: dict[str, dict] = {}


def _get_session(request: Request) -> dict | None:
    sid = request.cookies.get("sid")
    return _sessions.get(sid)


# ── Data (loaded once) ────────────────────────────────────────────────────────
_players_df: pd.DataFrame | None = None
_wc_df: pd.DataFrame | None = None


def _players() -> pd.DataFrame:
    global _players_df
    if _players_df is None:
        p = pd.read_parquet(ROOT / "artifacts" / "players.parquet")
        p["eligible_buckets"] = (
            p["eligible_buckets"].fillna("").str.split(",")
            .apply(lambda x: [s for s in x if s])
        )
        _players_df = p
    return _players_df


def _wc_players() -> pd.DataFrame:
    global _wc_df
    if _wc_df is None:
        wc = pd.read_parquet(ROOT / "artifacts" / "wc_players.parquet")
        wc["eligible_buckets"] = wc["eligible_buckets"].fillna("")
        # Attach flag + confederation from meta.json so player dicts carry them
        import json as _json
        try:
            with open(ROOT / "meta.json") as _f:
                _meta = _json.load(_f)
            _team_info = {t["name"]: t for t in _meta.get("wc_teams", [])}
            wc["flag"]          = wc["wc_team"].map(lambda t: _team_info.get(t, {}).get("flag", "🌍"))
            wc["confederation"] = wc["wc_team"].map(lambda t: _team_info.get(t, {}).get("confederation", ""))
        except Exception:
            wc["flag"] = "🌍"
            wc["confederation"] = ""
        _wc_df = wc
    return _wc_df


# ── Constants ─────────────────────────────────────────────────────────────────
LEAGUE_OPTIONS = [
    ("ENG-Premier League", "Premier League", "🏴󠁧󠁢󠁥󠁮󠁧󠁿"),
    ("ESP-La Liga",        "La Liga",        "🇪🇸"),
    ("GER-Bundesliga",     "Bundesliga",     "🇩🇪"),
    ("ITA-Serie A",        "Serie A",        "🇮🇹"),
    ("FRA-Ligue 1",        "Ligue 1",        "🇫🇷"),
]
LEAGUE_SHORT = {lid: name for lid, name, _ in LEAGUE_OPTIONS}

def _max_games(league: str, season: str) -> int:
    """Max league games for a given league+season. Season is '2324' format."""
    if league == "GER-Bundesliga":
        return 34
    if league == "FRA-Ligue 1":
        # Ligue 1 reduced from 20→18 teams (38→34 games) starting 2023/24
        start_year = int("20" + season[:2]) if len(season) == 4 else int(season)
        return 34 if start_year >= 2023 else 38
    return 38  # PL, La Liga, Serie A

BUCKET_COLOURS = {
    "GK":  "amber",
    "DEF": "blue",
    "MID": "green",
    "FWD": "red",
}


def _roll_and_auto_reroll(state: dict) -> tuple[str, str, str]:
    """Roll; if no eligible candidates exist, auto-consume a reroll."""
    pdf = _players()
    league, club, season = state_roll(state)
    candidates = get_candidates(state, pdf)
    while candidates.empty and state["rerolls_left"] > 0:
        league, club, season = state_reroll(state)
        candidates = get_candidates(state, pdf)
    return league, club, season


def _format_season(season: str) -> str:
    return f"20{season[:2]}/{season[2:]}"


def _roster_rows(state: dict) -> list[dict]:
    """Return per-position rows for the formation visual."""
    rows = []
    by_bucket: dict[str, list] = {b: [] for b in ["GK", "DEF", "MID", "FWD"]}
    for p in state["drafted"]:
        by_bucket[p["drafted_bucket"]].append(p)
    for bucket in ["FWD", "MID", "DEF", "GK"]:
        total = state["slots"].get(bucket, 0)
        filled = by_bucket[bucket]
        rows.append({
            "bucket": bucket,
            "colour": BUCKET_COLOURS[bucket],
            "total": total,
            "players": filled,
            "empty": total - len(filled),
        })
    return rows


def _valid_buckets(player_row: dict, state: dict) -> list[str]:
    open_b = _open_buckets(state)
    return [b for b in player_row["eligible_buckets"] if b in open_b]


def _wc_valid_buckets(player_row: dict, state: dict) -> list[str]:
    open_b = _open_buckets(state)
    buckets = player_row.get("eligible_buckets", "")
    if isinstance(buckets, str):
        buckets = [b for b in buckets.split(",") if b]
    return [b for b in buckets if b in open_b]


_wc_team_meta: dict[str, dict] = {}

def _load_wc_meta():
    global _wc_team_meta
    if not _wc_team_meta:
        import json
        with open(ROOT / "meta.json") as f:
            meta = json.load(f)
        _wc_team_meta = {t["name"]: t for t in meta.get("wc_teams", [])}
    return _wc_team_meta


def _wc_roll_auto(state: dict) -> tuple[str, str, str]:
    """Roll WC team; auto-reroll if no eligible candidates exist."""
    wc = _wc_players()
    team = roll_team(state)
    candidates = get_wc_candidates(state, wc)
    while candidates.empty and state["rerolls_left"] > 0:
        team = reroll_team(state)
        candidates = get_wc_candidates(state, wc)
    meta = _load_wc_meta()
    info = meta.get(team, {})
    return team, info.get("flag", "🌍"), info.get("confederation", "")


def _build_wc_candidate_ctx(candidates: pd.DataFrame, state: dict) -> list[dict]:
    out = []
    for _, row in candidates.iterrows():
        d = _sanitize(row.to_dict())
        d["valid_buckets"] = _wc_valid_buckets(d, state)
        out.append(d)
    return out


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "league_options": LEAGUE_OPTIONS,
        "formations": FORMATIONS,
    })


@app.get("/faq", response_class=HTMLResponse)
async def faq(request: Request):
    return templates.TemplateResponse(request, "faq.html", {})


@app.post("/start", response_class=HTMLResponse)
async def start(
    request: Request,
    league: str = Form(...),
    formation: str = Form(...),
):
    state = new_game(formation, league)
    league_r, club, season = _roll_and_auto_reroll(state)

    sid = str(uuid.uuid4())
    _sessions[sid] = state

    candidates = get_candidates(state, _players())
    ctx = {
        "state": state,
        "league_short": LEAGUE_SHORT,
        "league_display": LEAGUE_SHORT.get(league, league),
        "club": club,
        "season_display": _format_season(season),
        "candidates": _build_candidate_ctx(candidates, state),
        "roster_rows": _roster_rows(state),
        "bucket_colours": BUCKET_COLOURS,
        "max_games": _max_games(league_r, season),
        "is_partial": False,
    }
    resp = templates.TemplateResponse(request, "game.html", ctx)
    resp.set_cookie("sid", sid, httponly=True, samesite="lax")
    return resp


@app.post("/pick", response_class=HTMLResponse)
async def pick(
    request: Request,
    understat_id: str = Form(...),
    bucket: str = Form(...),
):
    state = _get_session(request)
    if state is None:
        return RedirectResponse("/", status_code=303)

    pdf = _players()
    uid = int(understat_id)
    # Filter by current club+season so multi-season players resolve correctly
    league_c, club_c, season_c = state["current_cell"]
    rows = pdf[
        (pdf["understat_id"] == uid) &
        (pdf["club"] == club_c) &
        (pdf["season"] == season_c)
    ]
    if rows.empty:
        rows = pdf[pdf["understat_id"] == uid]  # fallback (shouldn't happen)
    player_row = rows.iloc[0].to_dict()
    draft_player(state, player_row, bucket)

    if state["complete"]:
        xi_df = pd.DataFrame(state["drafted"])
        result = score_xi(xi_df)
        state["drafted"] = [_sanitize(p) for p in state["drafted"]]
        state["_result"] = result
        # Full-page redirect — HTMX follows HX-Redirect as a real navigation
        sid = request.cookies.get("sid")
        resp = Response(status_code=200, headers={"HX-Redirect": "/result"})
        resp.set_cookie("sid", sid, httponly=True, samesite="lax")
        return resp

    league_r, club, season = _roll_and_auto_reroll(state)
    candidates = get_candidates(state, pdf)

    cur_league, _, cur_season = state["current_cell"]
    ctx = {
        "state": state,
        "club": club,
        "season_display": _format_season(season),
        "candidates": _build_candidate_ctx(candidates, state),
        "roster_rows": _roster_rows(state),
        "bucket_colours": BUCKET_COLOURS,
        "max_games": _max_games(cur_league, cur_season),
        "is_partial": True,
    }
    return templates.TemplateResponse(request, "partials/round.html", ctx)


@app.post("/reroll", response_class=HTMLResponse)
async def reroll(request: Request):
    state = _get_session(request)
    if state is None:
        return RedirectResponse("/", status_code=303)
    if state["rerolls_left"] <= 0:
        return HTMLResponse("No rerolls left.", status_code=400)

    league_r, club, season = state_reroll(state)
    candidates = get_candidates(state, _players())
    cur_league, _, cur_season = state["current_cell"]

    ctx = {
        "state": state,
        "club": club,
        "season_display": _format_season(season),
        "candidates": _build_candidate_ctx(candidates, state),
        "roster_rows": _roster_rows(state),
        "bucket_colours": BUCKET_COLOURS,
        "max_games": _max_games(cur_league, cur_season),
        "is_partial": True,
    }
    return templates.TemplateResponse(request, "partials/round.html", ctx)


@app.get("/result", response_class=HTMLResponse)
async def show_result(request: Request):
    state = _get_session(request)
    if state is None or "_result" not in state:
        return RedirectResponse("/", status_code=303)
    result = state["_result"]
    return templates.TemplateResponse(request, "reveal.html", {
        "state": state,
        "result": result,
        "roster_rows": _roster_rows(state),
        "league_display": LEAGUE_SHORT.get(state["league"], state["league"]),
        "bucket_colours": BUCKET_COLOURS,
        "tier_art": _tier_pixel_art(result["tier"]),
    })


# ── World Cup 2026 routes ─────────────────────────────────────────────────────

@app.post("/wc/start", response_class=HTMLResponse)
async def wc_start(request: Request, formation: str = Form(...)):
    state = new_wc_game(formation)
    team, flag, confederation = _wc_roll_auto(state)

    sid = str(uuid.uuid4())
    _sessions[sid] = state

    candidates = get_wc_candidates(state, _wc_players())
    ctx = {
        "state": state,
        "team_name": team,
        "team_flag": flag,
        "team_confederation": confederation,
        "candidates": _build_wc_candidate_ctx(candidates, state),
        "roster_rows": _roster_rows(state),
        "bucket_colours": BUCKET_COLOURS,
        "is_partial": False,
    }
    resp = templates.TemplateResponse(request, "wc_game.html", ctx)
    resp.set_cookie("sid", sid, httponly=True, samesite="lax")
    return resp


@app.post("/wc/pick", response_class=HTMLResponse)
async def wc_pick(
    request: Request,
    tm_id: str = Form(...),
    wc_team: str = Form(...),
    bucket: str = Form(...),
):
    state = _get_session(request)
    if state is None:
        return RedirectResponse("/", status_code=303)

    wc = _wc_players()
    rows = wc[(wc["tm_id"] == tm_id) & (wc["wc_team"] == wc_team)]
    if rows.empty:
        rows = wc[wc["tm_id"] == tm_id]
    player_row = rows.iloc[0].to_dict()
    draft_wc_player(state, player_row, bucket)

    if state["complete"]:
        xi_df = pd.DataFrame(state["drafted"])
        result = score_wc_xi(xi_df)
        state["drafted"] = [_sanitize(p) for p in state["drafted"]]
        state["_result"] = result
        sid = request.cookies.get("sid")
        resp = Response(status_code=200, headers={"HX-Redirect": "/wc/result"})
        resp.set_cookie("sid", sid, httponly=True, samesite="lax")
        return resp

    team, flag, confederation = _wc_roll_auto(state)
    candidates = get_wc_candidates(state, wc)

    ctx = {
        "state": state,
        "team_name": team,
        "team_flag": flag,
        "team_confederation": confederation,
        "candidates": _build_wc_candidate_ctx(candidates, state),
        "roster_rows": _roster_rows(state),
        "bucket_colours": BUCKET_COLOURS,
        "is_partial": True,
    }
    return templates.TemplateResponse(request, "partials/wc_round.html", ctx)


@app.post("/wc/reroll", response_class=HTMLResponse)
async def wc_reroll(request: Request):
    state = _get_session(request)
    if state is None:
        return RedirectResponse("/", status_code=303)
    if state["rerolls_left"] <= 0:
        return HTMLResponse("No rerolls left.", status_code=400)

    team = reroll_team(state)
    meta = _load_wc_meta()
    info = meta.get(team, {})
    flag = info.get("flag", "🌍")
    confederation = info.get("confederation", "")
    candidates = get_wc_candidates(state, _wc_players())

    ctx = {
        "state": state,
        "team_name": team,
        "team_flag": flag,
        "team_confederation": confederation,
        "candidates": _build_wc_candidate_ctx(candidates, state),
        "roster_rows": _roster_rows(state),
        "bucket_colours": BUCKET_COLOURS,
        "is_partial": True,
    }
    return templates.TemplateResponse(request, "partials/wc_round.html", ctx)


@app.get("/wc/result", response_class=HTMLResponse)
async def wc_result(request: Request):
    state = _get_session(request)
    if state is None or "_result" not in state:
        return RedirectResponse("/", status_code=303)
    result = state["_result"]
    return templates.TemplateResponse(request, "wc_reveal.html", {
        "state": state,
        "result": result,
        "roster_rows": _roster_rows(state),
        "bucket_colours": BUCKET_COLOURS,
        "tier_art": wc_tier_pixel_art(result["tier"]),
    })


def _tier_pixel_art(tier: str) -> dict:
    _BG = {
        "Invincible": "#fef3c7",
        "Title":      "#dcfce7",
        "European":   "#dbeafe",
        "Mid":        "#f3f4f6",
    }
    tier_key = next((k for k in _BG if k in tier), "Relegated")
    bg = _BG.get(tier_key, "#fee2e2")

    arts: dict[str, dict] = {
        "Invincible": {
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
            "label": "INVINCIBLE",
        },
        "Title": {
            "rows": [
                ".EEEEE.",
                "E.....E",
                "E.....E",
                ".EEEEE.",
                "...E...",
                "..EEE..",
                ".EEEEE.",
            ],
            "palette": {".": bg, "E": "#4ade80"},
            "label": "CHAMPIONS",
        },
        "European": {
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
            "label": "EUROPE",
        },
        "Mid": {
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
            "label": "SURVIVED",
        },
        "Relegated": {
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
            "label": "RELEGATED",
        },
    }
    return arts.get(tier_key, arts["Relegated"])


def _sanitize(d: dict) -> dict:
    """Replace float NaN with None so Jinja2 'is not none' checks work cleanly."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}


def _build_candidate_ctx(candidates: pd.DataFrame, state: dict) -> list[dict]:
    out = []
    for _, row in candidates.iterrows():
        d = _sanitize(row.to_dict())
        d["valid_buckets"] = _valid_buckets(d, state)
        d["season_display"] = _format_season(str(d.get("season", "")))
        d["colour"] = BUCKET_COLOURS.get(d.get("primary_bucket", "MID"), "stone")
        out.append(d)
    return out
