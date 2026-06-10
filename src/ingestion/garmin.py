"""
src/ingestion/garmin.py
-----------------------
Fetches running activities from Garmin Connect using the garminconnect library.

Requires:
    pip install garminconnect garth
    Run garmin_login.py once to save tokens to ~/.garth/
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import pandas as pd

RESTING_HR  = 45
MAX_HR      = 195

# garminconnect 0.3.2 stores tokens as garmin_tokens.json inside this dir
TOKENS_DIR  = Path.home() / ".garminconnect"


def _get_client():
    try:
        from garminconnect import Garmin
    except ImportError as e:
        raise ImportError("Run: pip install garminconnect") from e

    tokens_file = TOKENS_DIR / "garmin_tokens.json"
    if not tokens_file.exists():
        raise FileNotFoundError(
            f"Garmin tokens not found at {tokens_file}\n"
            "Run: python garmin_login.py"
        )

    client = Garmin()
    client.login(str(TOKENS_DIR))
    return client


def fetch_garmin_activities(
    start_date: str | date,
    end_date: str | date,
    activity_type: str = "running",
    page_size: int = 200,   # kept for API compat; pagination is handled internally
) -> list[dict]:
    """Fetch all Garmin activities in [start_date, end_date].
    garminconnect 0.3.x handles pagination internally in get_activities_by_date."""
    client = _get_client()
    if isinstance(start_date, date):
        start_date = start_date.isoformat()
    if isinstance(end_date, date):
        end_date = end_date.isoformat()

    result = client.get_activities_by_date(
        startdate=start_date,
        enddate=end_date,
        activitytype=activity_type,
    )
    return result if isinstance(result, list) else result.get("activities", [])


def build_garmin_row(activity: dict) -> dict:
    """Map raw Garmin activity dict to consolidated CSV schema."""
    dist_m  = activity.get("distance_meters") or activity.get("distance") or 0
    dur_sec = activity.get("duration_seconds") or activity.get("duration") or 0
    avg_hr  = activity.get("avg_hr_bpm") or activity.get("averageHR") or 0
    max_hr  = activity.get("max_hr_bpm") or activity.get("maxHR")
    steps   = activity.get("steps") or 0
    elev    = activity.get("elevation_gain_meters") or activity.get("elevationGain") or 0
    dist_km = dist_m / 1000.0

    pace_sec_km = (dur_sec / dist_km) if dist_km > 0 else None
    pace_fmt    = None
    if pace_sec_km:
        mm, ss = divmod(int(pace_sec_km), 60)
        pace_fmt = f"{mm}:{ss:02d}"

    avg_cadence = None
    if steps and dur_sec > 0:
        avg_cadence = round((steps / (dur_sec / 60.0)) / 2.0, 1)

    training_load = None
    if avg_hr > RESTING_HR and dur_sec > 0:
        training_load = round(
            (dur_sec / 60.0) * (avg_hr - RESTING_HR) / (MAX_HR - RESTING_HR), 2
        )

    efficiency_index = None
    if avg_hr > 0 and dur_sec > 0 and dist_km > 0:
        efficiency_index = round(dist_km / (avg_hr * (dur_sec / 60.0)) * 1000, 4)

    return {
        "id":               activity.get("id") or activity.get("activityId"),
        "name":             activity.get("name") or activity.get("activityName") or "",
        "sport_type":       {
            "running": "Run", "trail_running": "TrailRun",
            "treadmill_running": "Run", "virtual_run": "Run",
        }.get(
            (activity.get("type") or activity.get("activityType", {}).get("typeKey", "running")).lower().replace(" ","_"),
            activity.get("type", "Run")
        ),
        "start_date":       activity.get("start_time") or activity.get("startTimeLocal") or "",
        "distance_km":      round(dist_km, 4),
        "moving_time_sec":  round(dur_sec, 2),
        "elapsed_time_sec": round(dur_sec, 2),
        "elevation_gain":   round(elev, 2),
        "average_heartrate":avg_hr if avg_hr else None,
        "max_heartrate":    max_hr,
        "average_cadence":  avg_cadence,
        "average_speed":    round(dist_km / (dur_sec / 3600.0), 4) if (dist_km > 0 and dur_sec > 0) else None,
        "max_speed":        None,
        "calories":         activity.get("calories"),
        "latitude":         None, "longitude": None,
        "map_summary_polyline": None, "map_polyline": None, "altitude_stream": None,
        "weather_temp": None, "weather_feels_like": None, "weather_humidity": None,
        "weather_precipitation": None, "weather_rain": None,
        "weather_wind_speed": None, "weather_wind_gusts": None,
        "weather_code": None, "weather_condition": None,
        "pace_sec_km":       round(pace_sec_km, 2) if pace_sec_km else None,
        "pace_formatted":    pace_fmt,
        "efficiency_index":  efficiency_index,
        "training_load":     training_load,
    }


def enrich_with_strava_polylines(df: pd.DataFrame, access_token: str) -> pd.DataFrame:
    """Backfill map_polyline / altitude_stream / lat-lon from Strava (GPS only)."""
    import requests
    missing = df["map_polyline"].isna() | (df["map_polyline"] == "")
    to_fill = df[missing]
    if to_fill.empty:
        return df
    print(f"Fetching polylines for {len(to_fill)} activities from Strava...")
    headers = {"Authorization": f"Bearer {access_token}"}
    for idx, row in to_fill.iterrows():
        try:
            r = requests.get(
                f"https://www.strava.com/api/v3/activities/{row['id']}",
                headers=headers, timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                df.at[idx, "map_polyline"]         = data.get("map", {}).get("polyline")
                df.at[idx, "map_summary_polyline"] = data.get("map", {}).get("summary_polyline")
                latlng = data.get("start_latlng", [])
                if len(latlng) >= 2:
                    df.at[idx, "latitude"]  = latlng[0]
                    df.at[idx, "longitude"] = latlng[1]
        except Exception as exc:
            print(f"  Warning {row['id']}: {exc}")
    return df
