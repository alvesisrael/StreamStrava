"""
migrate.py — One-time migration from CSV/JSON flat files → SQLite.

Run from project root:
    python -m src.db.migrate

Safe to re-run: uses INSERT OR REPLACE, so existing data is updated in place.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

# ── Make sure project root is on sys.path ─────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.db.schema import DB_PATH, init_db
from src.db.queries import (
    upsert_activities,
    upsert_laps,
    upsert_garmin_insights,
    save_training_plan,
)

BASE = PROJECT_ROOT / "data" / "processed"


# ── Migration helpers ─────────────────────────────────────────────────────────

def migrate_activities() -> int:
    csv = BASE / "activities_consolidated.csv"
    if not csv.exists():
        print(f"  SKIP activities — {csv.name} not found")
        return 0

    df = pd.read_csv(
        csv, sep=";", encoding="utf-8-sig",
        low_memory=False, on_bad_lines="skip",
    )
    # Strip UTF-8 BOM from column names
    df.columns = [c.lstrip("﻿").lstrip("﻿") for c in df.columns]

    count = upsert_activities(df)
    print(f"  ✓ activities: {count:,} rows → SQLite")
    return count


def migrate_laps() -> int:
    csv = BASE / "activity_laps_consolidated.csv"
    if not csv.exists():
        print(f"  SKIP laps — {csv.name} not found")
        return 0

    df = pd.read_csv(
        csv, sep=";", encoding="utf-8-sig",
        low_memory=False, on_bad_lines="skip",
    )
    df.columns = [c.lstrip("﻿").lstrip("﻿") for c in df.columns]

    count = upsert_laps(df)
    print(f"  ✓ activity_laps: {count:,} rows → SQLite")
    return count


def migrate_garmin_insights() -> None:
    jf = BASE / "garmin_insights.json"
    if not jf.exists():
        print(f"  SKIP garmin_insights — {jf.name} not found")
        return

    with open(jf, encoding="utf-8") as f:
        insights = json.load(f)

    upsert_garmin_insights(insights)
    print("  ✓ garmin_insights → SQLite")


def migrate_training_plan() -> None:
    jf = BASE / "training_plan.json"
    if not jf.exists():
        print("  SKIP training_plan — training_plan.json not found (created on first import)")
        return

    with open(jf, encoding="utf-8") as f:
        plan = json.load(f)

    save_training_plan(plan)
    print(f"  ✓ training_plan: {len(plan)} workouts → SQLite")


# ── Validation ────────────────────────────────────────────────────────────────

def validate(db_path: Path = DB_PATH) -> None:
    print("\nValidation:")
    conn = sqlite3.connect(str(db_path))
    for tbl in ["activities", "activity_laps", "garmin_insights", "training_plan", "conversations"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl:<22} {count:>6} rows")
    conn.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print(f"PerformanceRun — SQLite migration")
    print(f"Database: {DB_PATH}\n")

    init_db()
    print("Tables created (or already exist).\n")

    print("Migrating data...")
    migrate_activities()
    migrate_laps()
    migrate_garmin_insights()
    migrate_training_plan()

    validate()
    print("\nDone! ✓")
    print(f"\nTo connect manually:")
    print(f"  sqlite3 \"{DB_PATH}\"")
    print(f"\nQuick test queries:")
    print("  SELECT COUNT(*), sport_type FROM activities GROUP BY sport_type;")
    print("  SELECT date, title, distance_km FROM training_plan ORDER BY date;")
    print("  SELECT updated_at, vo2max_current, rhr_today FROM garmin_insights ORDER BY updated_at DESC LIMIT 5;")


if __name__ == "__main__":
    main()
