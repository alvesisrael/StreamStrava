"""
sync.py — PerformanceRun incremental sync
==========================================
Replaces main.py. Uses Garmin as primary source for metrics.
Strava is used only for map_polyline / altitude_stream / lat-lon (GPS).

Run:
    python sync.py                   # incremental (last activity → today)
    python sync.py --full            # full backfill (since 2025-03-01)
    python sync.py --date 2025-06-01 # from specific date
    python sync.py --insights-only   # only refresh garmin_insights.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent
CONSOLIDATED  = PROJECT_ROOT / "data" / "processed" / "activities_consolidated.csv"
INSIGHTS_FILE = PROJECT_ROOT / "data" / "processed" / "garmin_insights.json"
GARMIN_START  = "2025-03-01"   # earliest Garmin data available


# ─────────────────────────────────────────────────────────────────────────────
#  Activities sync
# ─────────────────────────────────────────────────────────────────────────────

def load_existing() -> pd.DataFrame:
    if CONSOLIDATED.exists():
        return pd.read_csv(CONSOLIDATED, sep=";", engine="python", on_bad_lines="skip",
                           parse_dates=["start_date"])
    return pd.DataFrame()


def get_last_date(df: pd.DataFrame) -> str | None:
    if df.empty or "start_date" not in df.columns:
        return None
    last = pd.to_datetime(df["start_date"]).max()
    return (last + timedelta(days=1)).strftime("%Y-%m-%d")


def fetch_and_merge(start: str, end: str, df_existing: pd.DataFrame) -> pd.DataFrame:
    from src.ingestion.garmin import fetch_garmin_activities, build_garmin_row

    print(f"[Garmin] Fetching activities {start} → {end} ...")
    raw = fetch_garmin_activities(start, end)
    if not raw:
        print("  No new activities found.")
        return df_existing

    print(f"  Found {len(raw)} activities.")
    new_rows = [build_garmin_row(a) for a in raw]
    df_new = pd.DataFrame(new_rows)
    df_new["start_date"] = pd.to_datetime(df_new["start_date"])

    if df_existing.empty:
        return df_new

    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=["id"], keep="last")
    return df_combined.sort_values("start_date").reset_index(drop=True)


def enrich_polylines(df: pd.DataFrame) -> pd.DataFrame:
    """Try to backfill GPS from Strava if token available."""
    try:
        from src.auth.strava_auth import get_valid_token
        from src.ingestion.garmin import enrich_with_strava_polylines
        token = get_valid_token()
        df = enrich_with_strava_polylines(df, token)
    except Exception as e:
        print(f"  [Strava GPS] Skipped: {e}")
    return df


def save_activities(df: pd.DataFrame) -> None:
    CONSOLIDATED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CONSOLIDATED, index=False, sep=";")
    print(f"✅ Saved {len(df)} activities to {CONSOLIDATED}")


# ─────────────────────────────────────────────────────────────────────────────
#  Garmin Insights refresh (race predictions, VO2max, RHR, body battery)
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_seconds(sec) -> str:
    """Convert total seconds to HH:MM:SS or MM:SS string."""
    if not sec:
        return "—"
    h, rem = divmod(int(sec), 3600)
    m, s   = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def refresh_garmin_insights() -> None:
    """Fetch live Garmin data and write garmin_insights.json."""
    print("[Garmin Insights] Refreshing ...")
    today     = date.today()
    start_30d = (today - timedelta(days=29)).isoformat()
    today_str = today.isoformat()

    insights: dict = {"updated_at": today_str}

    try:
        import garminconnect
        COOKIES_PATH = Path(os.path.expanduser("~/.garminconnect/cookies.json"))
        client = garminconnect.Garmin()

        if COOKIES_PATH.exists():
            try:
                with open(COOKIES_PATH) as _f:
                    _cookies_raw = _f.read()
                client.garth.loads(_cookies_raw)
            except Exception:
                pass

        try:
            client.login()
        except Exception:
            pass

        # ── Race Predictions ─────────────────────────────────────────────────
        try:
            rp_raw = client.get_race_predictions()
            rp_out: dict = {}
            items = rp_raw if isinstance(rp_raw, list) else [rp_raw]
            for item in items:
                if not isinstance(item, dict):
                    continue
                for dist_key, label in [
                    ("fiveK",        "5K"),
                    ("tenK",         "10K"),
                    ("halfMarathon", "half_marathon"),
                    ("marathon",     "marathon"),
                ]:
                    val = item.get(dist_key)
                    if isinstance(val, dict):
                        secs = val.get("timeInSeconds") or val.get("time") or 0
                        rp_out[label] = {"time": _fmt_seconds(secs), "time_seconds": int(secs)}
            if rp_out:
                insights["race_predictions"] = rp_out
                print(f"  OK Race predictions: {list(rp_out.keys())}")
        except Exception as e:
            print(f"  WARN Race predictions: {e}")

        # ── VO2max ────────────────────────────────────────────────────────────
        try:
            vo2_data = client.get_vo2max_trend(start_30d, today_str)
            if isinstance(vo2_data, list) and vo2_data:
                vals = []
                for v in vo2_data:
                    if isinstance(v, dict):
                        x = v.get("vo2MaxValue") or v.get("value")
                        if x is not None:
                            vals.append(float(x))
                if vals:
                    cur  = vals[-1]
                    prev = vals[-2] if len(vals) >= 2 else cur
                    insights["vo2max"] = {
                        "current":  cur,
                        "previous": prev,
                        "change":   round(cur - prev, 1),
                    }
                    print(f"  OK VO2max: {cur}")
        except Exception as e:
            print(f"  WARN VO2max: {e}")

        # ── Resting HR ────────────────────────────────────────────────────────
        try:
            rhr_raw = client.get_rhr_day(today_str)
            rhr_val = None
            if isinstance(rhr_raw, dict):
                rhr_val = (rhr_raw.get("restingHeartRate")
                           or (rhr_raw.get("allDayHR") or {}).get("restingHeartRate")
                           or rhr_raw.get("value"))
            elif isinstance(rhr_raw, (int, float)):
                rhr_val = int(rhr_raw)
            if rhr_val:
                insights["rhr_today"] = int(rhr_val)
                print(f"  OK RHR: {int(rhr_val)} bpm")
        except Exception as e:
            print(f"  WARN RHR: {e}")

        # ── Body Battery ─────────────────────────────────────────────────────
        try:
            bb_raw = client.get_body_battery(start_30d, today_str)
            bb_out = []
            for day in (bb_raw if isinstance(bb_raw, list) else []):
                if not isinstance(day, dict):
                    continue
                d   = day.get("date") or day.get("calendarDate", "")
                chg = day.get("charged")  or day.get("chargedValue",  0) or 0
                drn = day.get("drained")  or day.get("drainedValue",  0) or 0
                if d:
                    bb_out.append({"date": str(d), "charged": int(chg), "drained": int(drn)})
            if bb_out:
                insights["body_battery"] = bb_out
                print(f"  OK Body Battery: {len(bb_out)} days")
        except Exception as e:
            print(f"  WARN Body Battery: {e}")

    except Exception as e:
        print(f"  ERROR Could not connect to Garmin: {e}")

    INSIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INSIGHTS_FILE, "w") as f:
        json.dump(insights, f, ensure_ascii=False, indent=2)
    print(f"OK garmin_insights.json saved ({today_str})")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PerformanceRun sync")
    parser.add_argument("--full",          action="store_true", help="Full backfill from GARMIN_START")
    parser.add_argument("--date",          type=str,            help="Start date (YYYY-MM-DD)")
    parser.add_argument("--insights-only", action="store_true", help="Only refresh garmin_insights.json")
    args = parser.parse_args()

    if args.insights_only:
        refresh_garmin_insights()
        return

    df_existing = load_existing()
    today = date.today().isoformat()

    if args.full:
        start = GARMIN_START
    elif args.date:
        start = args.date
    else:
        start = get_last_date(df_existing) or GARMIN_START

    print(f"Sync window: {start} -> {today}")

    df = fetch_and_merge(start, today, df_existing)
    df = enrich_polylines(df)
    save_activities(df)

    # Always refresh insights after an activity sync
    refresh_garmin_insights()


if __name__ == "__main__":
    main()
