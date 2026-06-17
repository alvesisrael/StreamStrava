"""
schema.py — SQLite schema and connection management for PerformanceRun.

Tables
------
activities        : one row per Garmin/Strava activity
activity_laps     : laps/splits for each activity
garmin_insights   : VO2max, RHR, race predictions, body battery (latest snapshot)
training_plan     : imported workouts (from coach screenshots)
conversations     : Groq assistant chat history (all tabs, all sessions)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

# ── Absolute path to the database file ───────────────────────────────────────
DB_PATH: Path = (
    Path(__file__).parent.parent.parent / "data" / "processed" / "performancerun.db"
)

# ── CREATE TABLE statements ──────────────────────────────────────────────────
_DDL: list[str] = [
    # ── activities ────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS activities (
        id                    TEXT PRIMARY KEY,
        name                  TEXT,
        sport_type            TEXT,
        start_date            TEXT,
        distance_km           REAL,
        moving_time_sec       REAL,
        elapsed_time_sec      REAL,
        elevation_gain        REAL,
        average_heartrate     REAL,
        max_heartrate         REAL,
        average_cadence       REAL,
        average_speed         REAL,
        max_speed             REAL,
        calories              REAL,
        latitude              REAL,
        longitude             REAL,
        map_summary_polyline  TEXT,
        map_polyline          TEXT,
        altitude_stream       TEXT,
        weather_temp          REAL,
        weather_feels_like    REAL,
        weather_humidity      REAL,
        weather_precipitation REAL,
        weather_rain          REAL,
        weather_wind_speed    REAL,
        weather_wind_gusts    REAL,
        weather_code          REAL,
        weather_condition     TEXT,
        pace_sec_km           REAL,
        pace_formatted        TEXT,
        efficiency_index      REAL,
        training_load         REAL,
        suffer_score          REAL,
        pr_count              INTEGER,
        achievement_count     INTEGER,
        Intensidade           TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_act_date  ON activities(start_date)",
    "CREATE INDEX IF NOT EXISTS idx_act_sport ON activities(sport_type)",
    "CREATE INDEX IF NOT EXISTS idx_act_ds    ON activities(start_date, sport_type)",

    # ── activity_laps ─────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS activity_laps (
        lap_id                TEXT PRIMARY KEY,
        activity_id           TEXT,
        activity_sport_type   TEXT,
        lap_index             INTEGER,
        split                 INTEGER,
        name                  TEXT,
        start_date            TEXT,
        distance_m            REAL,
        distance_km           REAL,
        moving_time_sec       REAL,
        elapsed_time_sec      REAL,
        average_speed         REAL,
        max_speed             REAL,
        average_heartrate     REAL,
        max_heartrate         REAL,
        average_cadence       REAL,
        total_elevation_gain  REAL,
        start_index           INTEGER,
        end_index             INTEGER,
        pace_sec_km           REAL,
        pace_formatted        TEXT,
        FOREIGN KEY (activity_id) REFERENCES activities(id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_laps_act ON activity_laps(activity_id)",

    # ── garmin_insights ───────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS garmin_insights (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        updated_at              TEXT NOT NULL,
        vo2max_current          REAL,
        vo2max_previous         REAL,
        vo2max_change           REAL,
        rhr_today               INTEGER,
        race_pred_5k            TEXT,
        race_pred_5k_secs       INTEGER,
        race_pred_10k           TEXT,
        race_pred_10k_secs      INTEGER,
        race_pred_half          TEXT,
        race_pred_half_secs     INTEGER,
        race_pred_marathon      TEXT,
        race_pred_marathon_secs INTEGER,
        body_battery            TEXT
    )
    """,

    # ── training_plan ─────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS training_plan (
        date           TEXT PRIMARY KEY,
        title          TEXT,
        training_type  TEXT,
        course         TEXT,
        distance_km    REAL,
        elevation_gain TEXT,
        intensity      TEXT,
        description    TEXT,
        blocks         TEXT,
        training_load  REAL
    )
    """,

    # ── conversations ─────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id    TEXT    NOT NULL,
        timestamp     TEXT    DEFAULT (datetime('now')),
        role          TEXT    NOT NULL,
        content       TEXT    NOT NULL,
        tab_context   TEXT,
        data_snapshot TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_conv_ts      ON conversations(timestamp DESC)",
]


# ── Public API ────────────────────────────────────────────────────────────────

def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Return an open SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except Exception:
        pass
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Create all tables and indexes (idempotent — safe to call repeatedly)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    # WAL mode improves read concurrency on native filesystems.
    # Falls back silently if the FS doesn't support it (e.g. network mounts).
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except Exception:
        pass
    with conn:
        for stmt in _DDL:
            conn.execute(stmt)
    conn.close()
