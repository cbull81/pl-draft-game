"""
FastAPI web interface for Tacticos & Galacticos.
Run: uvicorn web.app:app --reload
"""
import logging
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

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

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("tacticos")

# ── App + templates ───────────────────────────────────────────────────────────
app = FastAPI()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


# ── Request logger middleware ─────────────────────────────────────────────────
class _RequestLogger(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            ms = (time.perf_counter() - t0) * 1000
            log.error("%-6s %-40s UNHANDLED %.0fms — %s",
                      request.method, request.url.path, ms, exc, exc_info=True)
            raise
        ms = (time.perf_counter() - t0) * 1000
        log.info("%-6s %-40s %d  %.0fms",
                 request.method, request.url.path, response.status_code, ms)
        return response

app.add_middleware(_RequestLogger)


# ── Global exception handlers ─────────────────────────────────────────────────
def _is_htmx(request: Request) -> bool:
    return request.headers.get("HX-Request") == "true"


def _error_fragment(message: str) -> HTMLResponse:
    """Inline error HTML for HTMX partial responses."""
    return HTMLResponse(
        f'<div style="padding:24px; text-align:center; font-family:sans-serif;">'
        f'<div style="color:#b91c1c; font-weight:700; margin-bottom:8px;">Something went wrong</div>'
        f'<div style="color:#6b7280; font-size:13px; margin-bottom:16px;">{message}</div>'
        f'<a href="/" style="font-size:12px; color:#1e3a8a;">← Start a new game</a>'
        f'</div>',
        status_code=500,
    )


@app.exception_handler(Exception)
async def _unhandled_exception(request: Request, exc: Exception):
    log.error("Unhandled exception — %s %s: %r",
              request.method, request.url.path, exc, exc_info=True)
    if _is_htmx(request):
        return _error_fragment("An unexpected error occurred.")
    try:
        return templates.TemplateResponse(
            request, "error.html", {"detail": str(exc)}, status_code=500
        )
    except Exception:
        return HTMLResponse("<h1>500 — Something went wrong</h1><p><a href='/'>Home</a></p>",
                            status_code=500)


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    log.warning("Form validation error — %s %s: %s",
                request.method, request.url.path, exc)
    if _is_htmx(request):
        return _error_fragment("Invalid form data.")
    return RedirectResponse("/", status_code=303)


# ── Startup validation ────────────────────────────────────────────────────────
_REQUIRED_ARTIFACTS = [
    ("artifacts", "players.parquet"),
    ("artifacts", "wc_players.parquet"),
    ("artifacts", "ep_model.joblib"),
    ("artifacts", "xgf_model.joblib"),
    ("artifacts", "xga_model.joblib"),
    ("",          "meta.json"),
]


@app.on_event("startup")
async def _startup():
    missing = [name for subdir, name in _REQUIRED_ARTIFACTS
               if not (ROOT / subdir / name).exists()]
    if missing:
        msg = f"Missing required artifacts: {missing} — run data/build.py then model/train.py"
        log.critical(msg)
        raise RuntimeError(msg)

    _init_analytics()

    # Eagerly load both datasets so first-request latency is predictable
    # and any parquet/schema errors surface immediately at boot.
    try:
        _players()
        _wc_players()
        log.info("Startup OK — %d PL players, %d WC players loaded",
                 len(_players_df), len(_wc_df))
    except Exception:
        log.critical("Failed to load player data at startup", exc_info=True)
        raise


# ── Club abbreviation filter ──────────────────────────────────────────────────
_CLUB_ABBR: dict[str, str] = {
    # England
    "Manchester City":          "Man City",
    "Manchester United":        "Man Utd",
    "Newcastle United":         "Newcastle",
    "West Bromwich Albion":     "West Brom",
    "Wolverhampton Wanderers":  "Wolves",
    "Queens Park Rangers":      "QPR",
    "Crystal Palace":           "C. Palace",
    "Nottingham Forest":        "Nott'm F.",
    "Sheffield United":         "Sheff Utd",
    # France
    "Paris Saint Germain":      "PSG",
    "Evian Thonon Gaillard":    "Evian",
    "GFC Ajaccio":              "GFC Ajaccio",
    "SC Bastia":                "Bastia",
    # Germany
    "Borussia Dortmund":        "Dortmund",
    "Borussia M.Gladbach":      "M'gladbach",
    "RasenBallsport Leipzig":   "RB Leipzig",
    "Eintracht Frankfurt":      "E. Frankfurt",
    "Fortuna Duesseldorf":      "Duesseldorf",
    "Greuther Fuerth":          "Gr. Fuerth",
    "Hamburger SV":             "Hamburg",
    "Bayer Leverkusen":         "Leverkusen",
    "VfB Stuttgart":            "Stuttgart",
    "FC Heidenheim":            "Heidenheim",
    "Holstein Kiel":            "Holstein",
    # Spain
    "Atletico Madrid":          "Atletico",
    "Athletic Club":            "Ath. Club",
    "Deportivo La Coruna":      "Deportivo",
    "Sporting Gijon":           "Sp. Gijon",
    "Real Valladolid":          "Valladolid",
    "Rayo Vallecano":           "Rayo",
    "Celta Vigo":               "Celta",
    "Real Sociedad":            "R. Sociedad",
    "Real Betis":               "Real Betis",
    # Italy
    "Parma Calcio 1913":        "Parma",
    "SPAL 2013":                "SPAL",
    "AC Milan":                 "AC Milan",
}


def _abbr_club(name: str) -> str:
    return _CLUB_ABBR.get(name, name)


templates.env.filters["abbr_club"] = _abbr_club

# ── Analytics (Supabase Postgres) ────────────────────────────────────────────
_DB_URL    = os.environ.get("DATABASE_URL", "")
_STATS_KEY = os.environ.get("STATS_KEY", "")


def _pg_conn():
    import psycopg2
    return psycopg2.connect(_DB_URL)


def _init_analytics() -> None:
    if not _DB_URL:
        log.info("DATABASE_URL not set — analytics disabled")
        return
    try:
        with _pg_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id        SERIAL PRIMARY KEY,
                    ts        DOUBLE PRECISION NOT NULL,
                    event     TEXT NOT NULL,
                    sid       TEXT,
                    mode      TEXT,
                    formation TEXT,
                    league    TEXT
                )
            """)
        log.info("Analytics DB ready (Supabase Postgres)")
    except Exception:
        log.warning("Analytics init failed — continuing without analytics", exc_info=True)


def _track(event: str, sid: str = None, mode: str = None,
           formation: str = None, league: str = None) -> None:
    if not _DB_URL:
        return
    try:
        with _pg_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO events (ts, event, sid, mode, formation, league) VALUES (%s,%s,%s,%s,%s,%s)",
                (time.time(), event, sid, mode, formation, league),
            )
    except Exception:
        log.warning("Analytics write failed", exc_info=True)


# ── Session store (in-memory, with TTL) ───────────────────────────────────────
_SESSION_TTL = 1800  # 30 minutes
_sessions: dict[str, dict] = {}

_PRODUCTION = os.environ.get("ENVIRONMENT") == "production"


def _new_sid() -> str:
    sid = str(uuid.uuid4())
    _sessions[sid] = {}
    return sid


def _prune_sessions() -> None:
    """Evict sessions older than _SESSION_TTL. Called on each new game start."""
    cutoff = time.time() - _SESSION_TTL
    expired = [sid for sid, s in _sessions.items()
               if s.get("_created_at", 0) < cutoff]
    for sid in expired:
        del _sessions[sid]
    if expired:
        log.info("Pruned %d expired session(s), %d active", len(expired), len(_sessions))


def _get_session(request: Request) -> dict | None:
    sid = request.cookies.get("sid")
    if not sid:
        log.debug("Request with no sid cookie")
        return None
    state = _sessions.get(sid)
    if state is None:
        log.warning("Unknown or expired session: %s", sid)
    return state


def _set_cookie(resp, sid: str) -> None:
    resp.set_cookie("sid", sid, httponly=True, samesite="lax",
                    secure=_PRODUCTION, max_age=_SESSION_TTL)


# ── Data (loaded once at startup) ─────────────────────────────────────────────
_players_df: pd.DataFrame | None = None
_wc_df: pd.DataFrame | None = None


def _players() -> pd.DataFrame:
    global _players_df
    if _players_df is None:
        log.info("Loading players.parquet …")
        p = pd.read_parquet(ROOT / "artifacts" / "players.parquet")
        p["eligible_buckets"] = (
            p["eligible_buckets"].fillna("").str.split(",")
            .apply(lambda x: [s for s in x if s])
        )
        _players_df = p
        log.info("players.parquet loaded: %d rows", len(p))
    return _players_df


def _wc_players() -> pd.DataFrame:
    global _wc_df
    if _wc_df is None:
        log.info("Loading wc_players.parquet …")
        wc = pd.read_parquet(ROOT / "artifacts" / "wc_players.parquet")
        wc["eligible_buckets"] = wc["eligible_buckets"].fillna("")
        import json as _json
        try:
            with open(ROOT / "meta.json") as _f:
                _meta = _json.load(_f)
            _team_info = {t["name"]: t for t in _meta.get("wc_teams", [])}
            wc["flag"]          = wc["wc_team"].map(lambda t: _team_info.get(t, {}).get("flag", "🌍"))
            wc["confederation"] = wc["wc_team"].map(lambda t: _team_info.get(t, {}).get("confederation", ""))
        except Exception:
            log.warning("Could not attach WC team meta (flag/confederation)", exc_info=True)
            wc["flag"] = "🌍"
            wc["confederation"] = ""
        _wc_df = wc
        log.info("wc_players.parquet loaded: %d rows", len(wc))
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
    if league == "GER-Bundesliga":
        return 34
    if league == "FRA-Ligue 1":
        start_year = int("20" + season[:2]) if len(season) == 4 else int(season)
        return 34 if start_year >= 2023 else 38
    return 38


BUCKET_COLOURS = {
    "GK":  "amber",
    "DEF": "blue",
    "MID": "green",
    "FWD": "red",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _roll_and_auto_reroll(state: dict) -> tuple[str, str, str]:
    """Roll; if no eligible candidates exist, auto-consume a reroll."""
    pdf = _players()
    league, club, season = state_roll(state)
    candidates = get_candidates(state, pdf)
    while candidates.empty and state["rerolls_left"] > 0:
        league, club, season = state_reroll(state)
        candidates = get_candidates(state, pdf)
    log.debug("Rolled %s %s %s (rerolls_left=%d)", league, club, season, state["rerolls_left"])
    return league, club, season


def _format_season(season: str) -> str:
    return f"20{season[:2]}/{season[2:]}"


def _roster_rows(state: dict) -> list[dict]:
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


_league_champions: dict[str, dict[str, str]] = {}


def _load_league_champions() -> dict[str, dict[str, str]]:
    global _league_champions
    if not _league_champions:
        import json
        with open(ROOT / "meta.json") as f:
            meta = json.load(f)
        _league_champions = meta.get("league_champions", {})
    return _league_champions


def _is_champion(league: str, club: str, season: str) -> bool:
    champs = _load_league_champions()
    return champs.get(league, {}).get(season) == club


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
    wc = _wc_players()
    team = roll_team(state)
    candidates = get_wc_candidates(state, wc)
    while candidates.empty and state["rerolls_left"] > 0:
        team = reroll_team(state)
        candidates = get_wc_candidates(state, wc)
    meta = _load_wc_meta()
    info = meta.get(team, {})
    log.debug("WC rolled %s (rerolls_left=%d)", team, state["rerolls_left"])
    return team, info.get("flag", "🌍"), info.get("confederation", "")


def _build_wc_candidate_ctx(candidates: pd.DataFrame, state: dict) -> list[dict]:
    out = []
    for _, row in candidates.iterrows():
        d = _sanitize(row.to_dict())
        d["valid_buckets"] = _wc_valid_buckets(d, state)
        out.append(d)
    return out


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


@app.get("/start")
async def start_get():
    return RedirectResponse("/", status_code=303)


@app.post("/start", response_class=HTMLResponse)
async def start(
    request: Request,
    league: str = Form(...),
    formation: str = Form(...),
):
    log.info("New game — league=%s formation=%s", league, formation)
    _prune_sessions()
    state = new_game(formation, league)
    state["_created_at"] = time.time()
    league_r, club, season = _roll_and_auto_reroll(state)

    sid = str(uuid.uuid4())
    _sessions[sid] = state
    _track("game_start", sid=sid, mode="pl", formation=formation, league=league)

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
        "is_champion": _is_champion(league_r, club, season),
        "is_partial": False,
    }
    resp = templates.TemplateResponse(request, "game.html", ctx)
    _set_cookie(resp, sid)
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

    # Validate form values before touching state
    try:
        uid = int(understat_id)
    except (ValueError, TypeError):
        log.warning("Invalid understat_id submitted: %r", understat_id)
        return HTMLResponse("Invalid player id.", status_code=400)

    if bucket not in ("GK", "DEF", "MID", "FWD"):
        log.warning("Invalid bucket submitted: %r", bucket)
        return HTMLResponse("Invalid position bucket.", status_code=400)

    pdf = _players()
    league_c, club_c, season_c = state["current_cell"]
    rows = pdf[
        (pdf["understat_id"] == uid) &
        (pdf["club"] == club_c) &
        (pdf["season"] == season_c)
    ]
    if rows.empty:
        rows = pdf[pdf["understat_id"] == uid]
    if rows.empty:
        log.warning("Player understat_id=%d not found in dataset", uid)
        return HTMLResponse("Player not found.", status_code=400)

    player_row = rows.iloc[0].to_dict()
    try:
        draft_player(state, player_row, bucket)
    except ValueError as exc:
        log.warning("draft_player rejected: %s", exc)
        return HTMLResponse(str(exc), status_code=400)

    log.info("Picked %s → %s (round %d)", player_row.get("player_name"), bucket, state["round"])

    if state["complete"]:
        xi_df = pd.DataFrame(state["drafted"])
        try:
            result = score_xi(xi_df)
        except Exception:
            log.error("score_xi failed", exc_info=True)
            return _error_fragment("Scoring failed — please start a new game.")
        log.info("Game complete — score=%.1f tier=%s", result["predicted_points"], result["tier"])
        state["drafted"] = [_sanitize(p) for p in state["drafted"]]
        state["_result"] = result
        sid = request.cookies.get("sid")
        resp = Response(status_code=200, headers={"HX-Redirect": "/result"})
        _set_cookie(resp, sid)
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
        "is_champion": _is_champion(cur_league, club, cur_season),
        "is_partial": True,
    }
    return templates.TemplateResponse(request, "partials/round.html", ctx)


@app.post("/reroll", response_class=HTMLResponse)
async def reroll(request: Request):
    state = _get_session(request)
    if state is None:
        return RedirectResponse("/", status_code=303)
    if state["rerolls_left"] <= 0:
        log.warning("Reroll attempted with none remaining (sid=%s)", request.cookies.get("sid"))
        return HTMLResponse("No rerolls left.", status_code=400)

    league_r, club, season = state_reroll(state)
    log.info("Reroll → %s %s %s (rerolls_left=%d)", league_r, club, season, state["rerolls_left"])
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
        "is_champion": _is_champion(cur_league, club, cur_season),
        "is_partial": True,
    }
    return templates.TemplateResponse(request, "partials/round.html", ctx)


@app.get("/result", response_class=HTMLResponse)
async def show_result(request: Request):
    state = _get_session(request)
    if state is None or "_result" not in state:
        log.info("Result page hit with no complete game in session — redirecting home")
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

@app.get("/wc/start")
async def wc_start_get():
    return RedirectResponse("/", status_code=303)


@app.post("/wc/start", response_class=HTMLResponse)
async def wc_start(request: Request, formation: str = Form(...)):
    log.info("New WC game — formation=%s", formation)
    _prune_sessions()
    state = new_wc_game(formation)
    state["_created_at"] = time.time()
    team, flag, confederation = _wc_roll_auto(state)

    sid = str(uuid.uuid4())
    _sessions[sid] = state
    _track("game_start", sid=sid, mode="wc", formation=formation)

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
    _set_cookie(resp, sid)
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

    if bucket not in ("GK", "DEF", "MID", "FWD"):
        log.warning("Invalid WC bucket submitted: %r", bucket)
        return HTMLResponse("Invalid position bucket.", status_code=400)

    wc = _wc_players()
    rows = wc[(wc["tm_id"] == tm_id) & (wc["wc_team"] == wc_team)]
    if rows.empty:
        rows = wc[wc["tm_id"] == tm_id]
    if rows.empty:
        log.warning("WC player tm_id=%r wc_team=%r not found", tm_id, wc_team)
        return HTMLResponse("Player not found.", status_code=400)

    player_row = rows.iloc[0].to_dict()
    try:
        draft_wc_player(state, player_row, bucket)
    except ValueError as exc:
        log.warning("draft_wc_player rejected: %s", exc)
        return HTMLResponse(str(exc), status_code=400)

    log.info("WC picked %s → %s (round %d)", player_row.get("player_name"), bucket, state["round"])

    if state["complete"]:
        xi_df = pd.DataFrame(state["drafted"])
        try:
            result = score_wc_xi(xi_df)
        except Exception:
            log.error("score_wc_xi failed", exc_info=True)
            return _error_fragment("Scoring failed — please start a new game.")
        log.info("WC game complete — tier=%s", result["tier"])
        state["drafted"] = [_sanitize(p) for p in state["drafted"]]
        state["_result"] = result
        sid = request.cookies.get("sid")
        resp = Response(status_code=200, headers={"HX-Redirect": "/wc/result"})
        _set_cookie(resp, sid)
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
        log.warning("WC reroll attempted with none remaining (sid=%s)", request.cookies.get("sid"))
        return HTMLResponse("No rerolls left.", status_code=400)

    team = reroll_team(state)
    meta = _load_wc_meta()
    info = meta.get(team, {})
    flag = info.get("flag", "🌍")
    confederation = info.get("confederation", "")
    log.info("WC reroll → %s (rerolls_left=%d)", team, state["rerolls_left"])
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
        log.info("WC result page hit with no complete game — redirecting home")
        return RedirectResponse("/", status_code=303)
    result = state["_result"]
    return templates.TemplateResponse(request, "wc_reveal.html", {
        "state": state,
        "result": result,
        "roster_rows": _roster_rows(state),
        "bucket_colours": BUCKET_COLOURS,
        "tier_art": wc_tier_pixel_art(result["tier"]),
    })


# ── Stats ─────────────────────────────────────────────────────────────────────

@app.get("/stats", response_class=HTMLResponse)
async def stats(request: Request, key: str = ""):
    if _STATS_KEY and key != _STATS_KEY:
        return HTMLResponse("Forbidden", status_code=403)

    active = len(_sessions)

    if not _DB_URL:
        return HTMLResponse(
            f"<pre>Active sessions: {active}\n\nAnalytics disabled (no DATABASE_URL).</pre>"
        )

    day_ago = time.time() - 86400
    with _pg_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM events WHERE event='game_start'")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM events WHERE event='game_start' AND ts>%s", (day_ago,))
        today = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM events WHERE event='game_start' AND mode='pl'")
        pl_ct = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM events WHERE event='game_start' AND mode='wc'")
        wc_ct = cur.fetchone()[0]
        cur.execute("""
            SELECT formation, COUNT(*) n FROM events
            WHERE event='game_start' AND formation IS NOT NULL
            GROUP BY formation ORDER BY n DESC
        """)
        fmts = cur.fetchall()
        cur.execute("""
            SELECT ts, mode, formation, league FROM events
            WHERE event='game_start' ORDER BY ts DESC LIMIT 30
        """)
        recent = cur.fetchall()

    def _row(ts, mode, formation, league):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        lbl = (LEAGUE_SHORT.get(league, league) or "") if mode == "pl" else "WC 2026"
        return f"<tr><td>{dt}</td><td>{mode.upper()}</td><td>{formation or '—'}</td><td>{lbl}</td></tr>"

    fmt_rows    = "".join(f"<tr><td>{f}</td><td>{n}</td></tr>" for f, n in fmts)
    recent_rows = "".join(_row(*r) for r in recent)

    html = f"""<!DOCTYPE html><html><head><title>T&amp;G Stats</title>
<style>
  body{{font-family:monospace;padding:24px;max-width:860px;margin:0 auto;color:#1e3a5f;}}
  h1{{margin-bottom:4px;}}p.sub{{color:#6b7280;margin-bottom:28px;}}
  h2{{margin:28px 0 10px;border-bottom:2px solid #e5e7eb;padding-bottom:4px;}}
  table{{border-collapse:collapse;width:100%;margin-bottom:16px;}}
  td,th{{border:1px solid #e5e7eb;padding:7px 12px;text-align:left;}}
  th{{background:#f9fafb;font-weight:700;}}
  .big{{font-size:2rem;font-weight:900;}}
  .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:8px;}}
  .card{{background:#f0f9ff;border:2px solid #bae6fd;border-radius:6px;padding:14px 16px;}}
  .card .lbl{{font-size:11px;color:#6b7280;margin-bottom:4px;}}
</style></head><body>
<h1>Tacticos &amp; Galacticos</h1>
<p class="sub">Live dashboard · refreshes on reload</p>
<div class="grid">
  <div class="card"><div class="lbl">ACTIVE NOW</div><div class="big">{active}</div></div>
  <div class="card"><div class="lbl">LAST 24H</div><div class="big">{today}</div></div>
  <div class="card"><div class="lbl">TOTAL STARTS</div><div class="big">{total}</div></div>
  <div class="card"><div class="lbl">PL / WC</div><div class="big">{pl_ct} / {wc_ct}</div></div>
</div>
<h2>Formations</h2>
<table><tr><th>Formation</th><th>Starts</th></tr>{fmt_rows}</table>
<h2>Recent 30 games</h2>
<table><tr><th>Time (UTC)</th><th>Mode</th><th>Formation</th><th>League</th></tr>{recent_rows}</table>
</body></html>"""
    return HTMLResponse(html)


# ── Pixel art ─────────────────────────────────────────────────────────────────
def _tier_pixel_art(tier: str) -> dict:
    _BG = {
        "Title":    "#dcfce7",
        "European": "#dbeafe",
        "Top":      "#f3f4f6",
        "Bottom":   "#fef9c3",
        "Relegated": "#fee2e2",
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
        "Top": {
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
            "label": "TOP HALF",
        },
        "Bottom": {
            "rows": [
                "YYYYYYY",
                "Y.....Y",
                "Y.....Y",
                "Y.....Y",
                "Y.....Y",
                "Y.....Y",
                "YYYYYYY",
            ],
            "palette": {".": bg, "Y": "#ca8a04"},
            "label": "BOTTOM HALF",
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
