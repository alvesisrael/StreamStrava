"""
sync.py — PerformanceRun incremental sync
==========================================
Replaces main.py. Uses Garmin as primary source for metrics.
Strava is used only for map_polyline / altitude_stream / lat-lon (GPS).

Run:
    python sync.py                  # incremental (last activity → today)
    python sync.py --full           # full backfill (since 2025-03-01)
    python sync.py --date 2025-06-01  # from specific date
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent
CONSOLIDATED  = PROJECT_ROOT / "data" / "processed" / "activities_consolidated.csv"
GARMIN_START  = "2025-03-01"   # earliest Garmin data available


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

    # Merge: deduplicate by id, prefer new rows
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


def save(df: pd.DataFrame) -> None:
    CONSOLIDATED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CONSOLIDATED, index=False, sep=";")
    print(f"✅ Saved {len(df)} activities to {CONSOLIDATED}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PerformanceRun sync")
    parser.add_argument("--full",  action="store_true", help="Full backfill from GARMIN_START")
    parser.add_argument("--date",  type=str,            help="Start date (YYYY-MM-DD)")
    args = parser.parse_args()

    df_existing = load_existing()
    today = date.today().isoformat()

    if args.full:
        start = GARMIN_START
    elif args.date:
        start = args.date
    else:
        start = get_last_date(df_existing) or GARMIN_START

    print(f"Sync window: {start} → {today}")

    df = fetch_and_merge(start, today, df_existing)
    df = enrich_polylines(df)
    save(df)


if __name__ == "__main__":
    main()
