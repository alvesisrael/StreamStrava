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

# ── DB layer (optional — graceful fallback if src.db not available) ───────────
def _try_import_db():
    try:
        from src.db.queries import upsert_activities, upsert_laps, upsert_garmin_insights
        return upsert_activities, upsert_laps, upsert_garmin_insights
    except ImportError:
        return None, None, None


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

    # Secondary dedup: same start_date within same minute → keep longer activity
    # (Garmin sometimes creates two records for the same workout)
    df_combined["start_date"] = pd.to_datetime(df_combined["start_date"])
    df_combined = df_combined.sort_values("distance_km", ascending=False)
    df_combined["_ts_min"] = df_combined["start_date"].dt.floor("min")
    df_combined = df_combined.drop_duplicates(subset=["_ts_min"], keep="first")
    df_combined = df_combined.drop(columns=["_ts_min"])

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

    # ── Also write to SQLite ─────────────────────────────────────────────────
    upsert_act, _, _ = _try_import_db()
    if upsert_act:
        try:
            n = upsert_act(df)
            print(f"✅ SQLite: {n} activities upserted")
        except Exception as _e:
            print(f"  [SQLite] activities skipped: {_e}")


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


def _parse_rhr(raw) -> int | None:
    """Extract RHR int from any garminconnect response shape."""
    if isinstance(raw, (int, float)):
        return int(raw)
    if not isinstance(raw, dict):
        return None
    # garminconnect 0.3.x: allMetrics.metricsMap or direct keys
    mm = (raw.get("allMetrics") or {}).get("metricsMap") or {}
    for key in ("WELLNESS_RESTING_HEART_RATE", "restingHeartRate"):
        v = mm.get(key)
        if v:
            if isinstance(v, list) and v:
                return int(v[0].get("value", 0) or 0)
            if isinstance(v, (int, float)):
                return int(v)
    # flat keys
    for key in ("restingHeartRate", "value", "heartRate"):
        v = raw.get(key)
        if v:
            return int(v)
    return None


def refresh_garmin_insights() -> None:
    """Fetch live Garmin data and MERGE into garmin_insights.json.
    Preserves existing data for any field that fails to update."""
    print("[Garmin Insights] Refreshing ...")
    today     = date.today()
    start_30d = (today - timedelta(days=29)).isoformat()
    today_str = today.isoformat()

    # Load existing data as baseline — never overwrite with empty
    existing: dict = {}
    if INSIGHTS_FILE.exists():
        try:
            with open(INSIGHTS_FILE) as _f:
                existing = json.load(_f)
        except Exception:
            pass

    insights: dict = {**existing, "updated_at": today_str}

    try:
        from garminconnect import Garmin
        TOKENS_DIR = Path.home() / ".garminconnect"
        client = Garmin()
        client.login(str(TOKENS_DIR))

        # ── Race Predictions ─────────────────────────────────────────────────
        try:
            rp_raw = client.get_race_predictions()   # no args = latest
            rp_out: dict = {}
            if isinstance(rp_raw, dict):
                # garminconnect 0.3.x returns nested dict with distance keys
                # possible shapes: {fiveK:{time,timeInSeconds}, ...}
                #                  {racePredictions:[{distance,time,...},...]}
                #                  {predictions:{5K:{time,time_seconds},...}}
                preds = rp_raw.get("racePredictions") or rp_raw.get("predictions") or rp_raw
                if isinstance(preds, list):
                    for p in preds:
                        dist = str(p.get("distance", "")).upper().replace(" ", "")
                        secs = p.get("time") or p.get("timeInSeconds") or 0
                        label = {"5K": "5K", "10K": "10K",
                                 "HALFMARATHON": "half_marathon", "MARATHON": "marathon"}.get(dist)
                        if label and secs:
                            rp_out[label] = {"time": _fmt_seconds(secs), "time_seconds": int(secs)}
                elif isinstance(preds, dict):
                    # Try camelCase keys first (raw garminconnect)
                    for dist_key, label in [
                        ("fiveK", "5K"), ("tenK", "10K"),
                        ("halfMarathon", "half_marathon"), ("marathon", "marathon"),
                    ]:
                        val = preds.get(dist_key) or preds.get(label) or preds.get(label.replace("_", ""))
                        if isinstance(val, dict):
                            secs = val.get("timeInSeconds") or val.get("time_seconds") or val.get("time") or 0
                            if secs and not isinstance(secs, str):
                                rp_out[label] = {"time": _fmt_seconds(secs), "time_seconds": int(secs)}
                            elif isinstance(secs, str) and ":" in secs:
                                # already formatted — parse back
                                parts = secs.split(":")
                                total = sum(int(x) * (60 ** (len(parts)-1-i)) for i, x in enumerate(parts))
                                rp_out[label] = {"time": secs, "time_seconds": total}
            if rp_out:
                insights["race_predictions"] = rp_out
                print(f"  OK Race predictions: {list(rp_out.keys())}")
            else:
                print(f"  WARN Race predictions: empty response shape: {list(rp_raw.keys()) if isinstance(rp_raw, dict) else type(rp_raw)}")
        except Exception as e:
            print(f"  WARN Race predictions: {e}")

        # ── VO2max — from user profile (userData.vo2MaxRunning) ───────────────
        try:
            profile = client.get_user_profile()
            vo2 = None
            if isinstance(profile, dict):
                ud = profile.get("userData") or profile
                vo2 = ud.get("vo2MaxRunning") or ud.get("vo2Max")
            if vo2 is not None:
                prev_vo2 = (existing.get("vo2max") or {}).get("current", float(vo2))
                insights["vo2max"] = {
                    "current":  float(vo2),
                    "previous": float(prev_vo2),
                    "change":   round(float(vo2) - float(prev_vo2), 1),
                }
                print(f"  OK VO2max: {vo2}")
        except Exception as e:
            print(f"  WARN VO2max: {e}")

        # ── Resting HR — try today then fallback to yesterday ─────────────────
        try:
            rhr_val = None
            for try_date in [today_str, (today - timedelta(days=1)).isoformat()]:
                raw = client.get_rhr_day(try_date)
                rhr_val = _parse_rhr(raw)
                if rhr_val:
                    break
            if rhr_val:
                insights["rhr_today"] = rhr_val
                print(f"  OK RHR: {rhr_val} bpm")
            else:
                print("  WARN RHR: no data (device not synced yet?)")
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
                chg = day.get("charged") if day.get("charged") is not None else day.get("chargedValue", 0)
                drn = day.get("drained") if day.get("drained") is not None else day.get("drainedValue", 0)
                if d and chg is not None:
                    bb_out.append({"date": str(d), "charged": int(chg or 0), "drained": int(drn or 0)})
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

    # ── Also write to SQLite ─────────────────────────────────────────────────
    _, _, upsert_ins = _try_import_db()
    if upsert_ins:
        try:
            upsert_ins(insights)
            print("OK garmin_insights → SQLite")
        except Exception as _e:
            print(f"  [SQLite] insights skipped: {_e}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
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

    # Sync laps too
    try:
        from src.ingestion.garmin import fetch_garmin_laps
        from src.db.queries import upsert_laps as _ul
        _laps_df = fetch_garmin_laps(start, today)
        if not _laps_df.empty:
            _n = _ul(_laps_df)
            print(f"SQLite: {_n} laps upserted")
    except Exception as _le:
        print(f"  [laps] skipped: {_le}")

    # Always refresh insights after an activity sync
    refresh_garmin_insights()


if __name__ == "__main__":
    main()
