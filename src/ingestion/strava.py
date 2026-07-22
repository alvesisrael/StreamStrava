"""
src/ingestion/strava.py
-----------------------
Fetches running activities and laps from Strava API.
Used after migrating from Garmin to Amazfit T-Rex 3
(Amazfit syncs automatically to Strava).

Strava rate limits: 100 req / 15 min · 1 000 req / day
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import requests

RESTING_HR = 45   # default resting HR for TRIMP calculation
MAX_HR     = 195  # overridden by garmin_insights.json if available

SPORT_MAP = {
    "Run":         "Run",
    "TrailRun":    "TrailRun",
    "VirtualRun":  "Run",
    "Treadmill":   "Run",
    "Hike":        "Hike",
    "Walk":        "Walk",
}


# ── Auth ──────────────────────────────────────────────────────────────────────

def get_token() -> str:
    """Return a valid Strava access token, refreshing if needed."""
    from src.auth.strava_auth import get_valid_token
    token = get_valid_token()
    if not token:
        raise RuntimeError(
            "Strava token inválido. Verifique data/token.json ou reconecte o Strava."
        )
    return token


# ── Rate-limit aware request ──────────────────────────────────────────────────

def _safe_get(url: str, headers: dict, params: dict | None = None, retries: int = 3) -> dict | list:
    for attempt in range(1, retries + 1):
        r = requests.get(url, headers=headers, params=params, timeout=15)
        if r.status_code == 200:
            usage = r.headers.get("X-RateLimit-Usage", "0,0")
            limit = r.headers.get("X-RateLimit-Limit", "100,1000")
            u15 = int(usage.split(",")[0])
            l15 = int(limit.split(",")[0])
            if u15 >= l15 * 0.90:
                print(f"  ⚠️  Rate limit 15min: {u15}/{l15} — pausando 60s...")
                time.sleep(60)
            return r.json()
        if r.status_code == 429:
            wait = int(r.headers.get("X-Retry-After", 900))
            print(f"  🚫 Rate limit atingido — aguardando {wait}s...")
            time.sleep(wait)
        else:
            print(f"  ⚠️  HTTP {r.status_code} (tentativa {attempt}/{retries}): {url}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    return {}


# ── Activities ────────────────────────────────────────────────────────────────

def fetch_strava_activities(
    after_ts: int | None = None,
    before_ts: int | None = None,
    per_page: int = 200,
) -> list[dict]:
    """
    Fetch all activities from Strava athlete/activities endpoint.
    after_ts / before_ts = Unix timestamps (optional).
    Returns list of raw Strava activity dicts.
    """
    token   = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    url     = "https://www.strava.com/api/v3/athlete/activities"
    results = []
    page    = 1

    while True:
        params: dict = {"per_page": per_page, "page": page}
        if after_ts:
            params["after"] = after_ts
        if before_ts:
            params["before"] = before_ts

        data = _safe_get(url, headers, params)
        if not isinstance(data, list) or not data:
            break
        results.extend(data)
        print(f"  📄 Página {page}: {len(data)} atividades ({len(results)} total)")
        if len(data) < per_page:
            break
        page += 1

    return results


def build_activity_row(a: dict, resting_hr: int = RESTING_HR, max_hr: int = MAX_HR) -> dict:
    """Map raw Strava activity dict to consolidated CSV schema."""
    dist_m  = float(a.get("distance") or 0)
    dist_km = dist_m / 1000.0
    mov_sec = float(a.get("moving_time") or 0)
    ela_sec = float(a.get("elapsed_time") or 0)
    elev    = float(a.get("total_elevation_gain") or 0)
    avg_hr  = a.get("average_heartrate")
    max_hr_ = a.get("max_heartrate")
    cadence = a.get("average_cadence")
    avg_spd = float(a.get("average_speed") or 0)   # m/s
    max_spd = float(a.get("max_speed") or 0)
    calories= a.get("calories")

    latlng  = a.get("start_latlng") or []
    lat     = latlng[0] if len(latlng) >= 2 else None
    lng     = latlng[1] if len(latlng) >= 2 else None

    map_obj = a.get("map") or {}
    s_poly  = map_obj.get("summary_polyline") or ""
    f_poly  = map_obj.get("polyline") or ""   # only in detailed endpoint

    # Pace
    pace_sec_km = (mov_sec / dist_km) if dist_km > 0 else None
    pace_fmt    = None
    if pace_sec_km:
        mm, ss   = divmod(int(pace_sec_km), 60)
        pace_fmt = f"{mm}:{ss:02d}"

    # TRIMP (Bannister)
    training_load = None
    if avg_hr and avg_hr > resting_hr and mov_sec > 0:
        training_load = round(
            (mov_sec / 60.0) * (avg_hr - resting_hr) / (max_hr - resting_hr), 2
        )

    # Efficiency index: dist_km / (avg_hr * min) * 1000
    efficiency_index = None
    if avg_hr and avg_hr > 0 and mov_sec > 0 and dist_km > 0:
        efficiency_index = round(dist_km / (avg_hr * (mov_sec / 60.0)) * 1000, 4)

    sport_raw = a.get("sport_type") or a.get("type") or "Run"
    sport     = SPORT_MAP.get(sport_raw, sport_raw)

    # start_date: Strava returns local time with Z suffix — strip to naive datetime
    start_local = a.get("start_date_local") or a.get("start_date") or ""
    start_local = start_local.replace("Z", "").replace("+00:00", "") if start_local else ""

    return {
        "id":                   a.get("id"),
        "name":                 a.get("name") or "",
        "sport_type":           sport,
        "start_date":           start_local,
        "distance_km":          round(dist_km, 4),
        "moving_time_sec":      round(mov_sec, 2),
        "elapsed_time_sec":     round(ela_sec, 2),
        "elevation_gain":       round(elev, 2),
        "average_heartrate":    avg_hr,
        "max_heartrate":        max_hr_,
        "average_cadence":      round(cadence, 1) if cadence else None,
        "average_speed":        round(avg_spd * 3.6, 4) if avg_spd else None,  # km/h
        "max_speed":            round(max_spd * 3.6, 4) if max_spd else None,
        "calories":             calories,
        "latitude":             lat,
        "longitude":            lng,
        "map_summary_polyline": s_poly,
        "map_polyline":         f_poly,
        "altitude_stream":      None,   # filled by enrich step if needed
        "weather_temp":         None,
        "weather_feels_like":   None,
        "weather_humidity":     None,
        "weather_precipitation":None,
        "weather_rain":         None,
        "weather_wind_speed":   None,
        "weather_wind_gusts":   None,
        "weather_code":         None,
        "weather_condition":    None,
        "pace_sec_km":          round(pace_sec_km, 2) if pace_sec_km else None,
        "pace_formatted":       pace_fmt,
        "efficiency_index":     efficiency_index,
        "training_load":        training_load,
        "Intensidade":          None,   # filled by app from FC zones
    }


# ── Laps ──────────────────────────────────────────────────────────────────────

def fetch_laps_for_activity(activity_id: int, sport_type: str) -> list[dict]:
    """
    Fetch lap data for a single activity.
    Returns list of rows matching activity_laps_consolidated.csv schema.
    """
    token   = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    url     = f"https://www.strava.com/api/v3/activities/{activity_id}/laps"
    data    = _safe_get(url, headers)

    if not isinstance(data, list):
        return []

    rows = []
    for lap in data:
        dist_m  = float(lap.get("distance") or 0)
        dist_km = dist_m / 1000.0
        mov_sec = float(lap.get("moving_time") or 0)
        ela_sec = float(lap.get("elapsed_time") or 0)
        avg_spd = float(lap.get("average_speed") or 0)
        max_spd = float(lap.get("max_speed") or 0)

        pace_sec_km = (mov_sec / dist_km) if dist_km > 0 else None
        pace_fmt    = None
        if pace_sec_km:
            mm, ss   = divmod(int(pace_sec_km), 60)
            pace_fmt = f"{mm}:{ss:02d}"

        rows.append({
            "activity_id":           activity_id,
            "activity_sport_type":   sport_type,
            "lap_id":                lap.get("id"),
            "lap_index":             lap.get("lap_index"),
            "split":                 lap.get("split"),
            "name":                  lap.get("name") or "",
            "start_date":            lap.get("start_date_local") or lap.get("start_date") or "",
            "distance_m":            round(dist_m, 2),
            "distance_km":           round(dist_km, 4),
            "moving_time_sec":       round(mov_sec, 2),
            "elapsed_time_sec":      round(ela_sec, 2),
            "average_speed":         round(avg_spd * 3.6, 4) if avg_spd else None,
            "max_speed":             round(max_spd * 3.6, 4) if max_spd else None,
            "average_heartrate":     lap.get("average_heartrate"),
            "max_heartrate":         lap.get("max_heartrate"),
            "average_cadence":       lap.get("average_cadence"),
            "total_elevation_gain":  lap.get("total_elevation_gain"),
            "start_index":           lap.get("start_index"),
            "end_index":             lap.get("end_index"),
            "pace_sec_km":           round(pace_sec_km, 2) if pace_sec_km else None,
            "pace_formatted":        pace_fmt,
        })
    return rows
