"""
queries.py — Reusable query functions for the PerformanceRun SQLite database.

All functions use the DB_PATH from schema.py by default and accept an optional
db_path override for testing.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.db.schema import DB_PATH, get_connection, init_db


# ═══════════════════════════════════════════════════════════════════════════════
#  ACTIVITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_activities(
    sport_type: Optional[str | list] = None,
    since: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Return activities table as DataFrame.

    Columns match activities_consolidated.csv (minus computed cols like MesAno).
    """
    if not db_path.exists():
        return pd.DataFrame()

    where: list[str] = []
    params: list = []

    if sport_type:
        if isinstance(sport_type, str):
            where.append("sport_type = ?")
            params.append(sport_type)
        else:
            ph = ",".join("?" * len(sport_type))
            where.append(f"sport_type IN ({ph})")
            params.extend(sport_type)

    if since:
        where.append("start_date >= ?")
        params.append(since)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    conn = get_connection(db_path)
    df = pd.read_sql_query(
        f"SELECT * FROM activities {where_sql} ORDER BY start_date",
        conn, params=params,
    )
    conn.close()
    return df


def upsert_activities(df: pd.DataFrame, db_path: Path = DB_PATH) -> int:
    """Insert-or-replace activities from a DataFrame into the DB.

    Returns the number of rows written.
    """
    if df.empty:
        return 0

    init_db(db_path)

    DB_COLS = [
        "id", "name", "sport_type", "start_date", "distance_km",
        "moving_time_sec", "elapsed_time_sec", "elevation_gain",
        "average_heartrate", "max_heartrate", "average_cadence",
        "average_speed", "max_speed", "calories", "latitude", "longitude",
        "map_summary_polyline", "map_polyline", "altitude_stream",
        "weather_temp", "weather_feels_like", "weather_humidity",
        "weather_precipitation", "weather_rain", "weather_wind_speed",
        "weather_wind_gusts", "weather_code", "weather_condition",
        "pace_sec_km", "pace_formatted", "efficiency_index",
        "training_load", "suffer_score", "pr_count", "achievement_count",
        "Intensidade",
    ]

    # Strip BOM from column names if present (CSV artefact)
    df = df.copy()
    df.columns = [c.lstrip("﻿") for c in df.columns]

    available = [c for c in DB_COLS if c in df.columns]
    df_ins = df[available].copy()

    if "start_date" in df_ins.columns:
        df_ins["start_date"] = df_ins["start_date"].astype(str)

    cols_sql = ", ".join(f'"{c}"' for c in available)
    ph = ", ".join("?" * len(available))
    sql = f"INSERT OR REPLACE INTO activities ({cols_sql}) VALUES ({ph})"

    rows = [tuple(r) for r in df_ins.itertuples(index=False)]
    conn = get_connection(db_path)
    with conn:
        conn.executemany(sql, rows)
    conn.close()
    return len(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  LAPS
# ═══════════════════════════════════════════════════════════════════════════════

def get_laps(
    activity_id: Optional[str] = None,
    sport_type: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Return activity_laps as DataFrame."""
    if not db_path.exists():
        return pd.DataFrame()

    where: list[str] = []
    params: list = []

    if activity_id:
        where.append("activity_id = ?")
        params.append(str(activity_id))
    if sport_type:
        where.append("activity_sport_type = ?")
        params.append(sport_type)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    conn = get_connection(db_path)
    df = pd.read_sql_query(
        f"SELECT * FROM activity_laps {where_sql} ORDER BY activity_id, lap_index",
        conn, params=params,
    )
    conn.close()
    return df


def upsert_laps(df: pd.DataFrame, db_path: Path = DB_PATH) -> int:
    """Insert-or-replace laps from a DataFrame. Returns rows written."""
    if df.empty:
        return 0

    init_db(db_path)

    DB_COLS = [
        "lap_id", "activity_id", "activity_sport_type", "lap_index", "split",
        "name", "start_date", "distance_m", "distance_km", "moving_time_sec",
        "elapsed_time_sec", "average_speed", "max_speed", "average_heartrate",
        "max_heartrate", "average_cadence", "total_elevation_gain",
        "start_index", "end_index", "pace_sec_km", "pace_formatted",
    ]

    df = df.copy()
    df.columns = [c.lstrip("﻿") for c in df.columns]

    available = [c for c in DB_COLS if c in df.columns]
    df_ins = df[available].copy()
    if "start_date" in df_ins.columns:
        df_ins["start_date"] = df_ins["start_date"].astype(str)

    cols_sql = ", ".join(f'"{c}"' for c in available)
    ph = ", ".join("?" * len(available))
    sql = f"INSERT OR REPLACE INTO activity_laps ({cols_sql}) VALUES ({ph})"

    rows = [tuple(r) for r in df_ins.itertuples(index=False)]
    conn = get_connection(db_path)
    with conn:
        conn.executemany(sql, rows)
    conn.close()
    return len(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  GARMIN INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

def get_garmin_insights(db_path: Path = DB_PATH) -> dict:
    """Return latest garmin_insights row as a dict matching garmin_insights.json."""
    if not db_path.exists():
        return {}

    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT * FROM garmin_insights ORDER BY updated_at DESC LIMIT 1"
    ).fetchone()
    conn.close()

    if row is None:
        return {}

    d = dict(row)
    result: dict = {"updated_at": d.get("updated_at")}

    # vo2max
    if d.get("vo2max_current") is not None:
        result["vo2max"] = {
            "current":  d["vo2max_current"],
            "previous": d.get("vo2max_previous"),
            "change":   d.get("vo2max_change"),
        }

    # rhr
    if d.get("rhr_today"):
        result["rhr_today"] = d["rhr_today"]

    # race predictions
    rp: dict = {}
    for col_prefix, label in [
        ("race_pred_5k",       "5K"),
        ("race_pred_10k",      "10K"),
        ("race_pred_half",     "half_marathon"),
        ("race_pred_marathon", "marathon"),
    ]:
        t = d.get(col_prefix)
        s = d.get(col_prefix + "_secs")
        if t:
            rp[label] = {"time": t, "time_seconds": s or 0}
    if rp:
        result["race_predictions"] = rp

    # body battery
    raw_bb = d.get("body_battery")
    if raw_bb:
        try:
            result["body_battery"] = json.loads(raw_bb)
        except Exception:
            pass

    return result


def upsert_garmin_insights(insights: dict, db_path: Path = DB_PATH) -> None:
    """Insert a garmin insights dict into the DB (replaces old rows for same date)."""
    if not insights:
        return

    init_db(db_path)

    vo2 = insights.get("vo2max") or {}
    rhr = insights.get("rhr_today")
    rp  = insights.get("race_predictions") or {}
    bb  = insights.get("body_battery") or []

    def _rp_time(label: str) -> Optional[str]:
        return (rp.get(label) or {}).get("time")

    def _rp_secs(label: str) -> Optional[int]:
        v = (rp.get(label) or {}).get("time_seconds")
        return int(v) if v else None

    conn = get_connection(db_path)
    with conn:
        # Delete any existing row for the same updated_at date (keep one per day)
        conn.execute(
            "DELETE FROM garmin_insights WHERE updated_at = ?",
            (insights.get("updated_at"),),
        )
        conn.execute(
            """
            INSERT INTO garmin_insights (
                updated_at,
                vo2max_current, vo2max_previous, vo2max_change,
                rhr_today,
                race_pred_5k, race_pred_5k_secs,
                race_pred_10k, race_pred_10k_secs,
                race_pred_half, race_pred_half_secs,
                race_pred_marathon, race_pred_marathon_secs,
                body_battery
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                insights.get("updated_at"),
                vo2.get("current"), vo2.get("previous"), vo2.get("change"),
                rhr,
                _rp_time("5K"),           _rp_secs("5K"),
                _rp_time("10K"),          _rp_secs("10K"),
                _rp_time("half_marathon"),_rp_secs("half_marathon"),
                _rp_time("marathon"),     _rp_secs("marathon"),
                json.dumps(bb, ensure_ascii=False) if bb else None,
            ),
        )
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING PLAN
# ═══════════════════════════════════════════════════════════════════════════════

def get_training_plan(db_path: Path = DB_PATH) -> list:
    """Return the full training plan as a list of dicts (matches training_plan.json)."""
    if not db_path.exists():
        return []

    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM training_plan ORDER BY date"
    ).fetchall()
    conn.close()

    result = []
    for row in rows:
        d = dict(row)
        raw_blocks = d.get("blocks")
        if raw_blocks:
            try:
                d["blocks"] = json.loads(raw_blocks)
            except Exception:
                d["blocks"] = []
        else:
            d["blocks"] = []
        result.append(d)
    return result


def save_training_plan(plan: list, db_path: Path = DB_PATH) -> None:
    """Upsert every workout in the plan list into the DB (keyed by date)."""
    if not plan:
        return

    init_db(db_path)
    conn = get_connection(db_path)
    with conn:
        for item in plan:
            blocks = item.get("blocks") or []
            conn.execute(
                """
                INSERT OR REPLACE INTO training_plan (
                    date, title, training_type, course, distance_km,
                    elevation_gain, intensity, description, blocks, training_load
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    item.get("date"),
                    item.get("title"),
                    item.get("training_type"),
                    item.get("course"),
                    item.get("distance_km"),
                    item.get("elevation_gain"),
                    item.get("intensity"),
                    item.get("description"),
                    json.dumps(blocks, ensure_ascii=False) if blocks else None,
                    item.get("training_load"),
                ),
            )
    conn.close()


def delete_plan_entry(date_str: str, db_path: Path = DB_PATH) -> None:
    """Remove a training plan entry by date."""
    if not db_path.exists():
        return
    conn = get_connection(db_path)
    with conn:
        conn.execute("DELETE FROM training_plan WHERE date = ?", (date_str,))
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVERSATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def save_message(
    session_id: str,
    role: str,
    content: str,
    tab_context: Optional[str] = None,
    data_snapshot: Optional[dict] = None,
    db_path: Path = DB_PATH,
) -> None:
    """Persist a single chat message to the conversations table."""
    init_db(db_path)
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO conversations
                (session_id, role, content, tab_context, data_snapshot)
            VALUES (?,?,?,?,?)
            """,
            (
                session_id,
                role,
                content,
                tab_context,
                json.dumps(data_snapshot, ensure_ascii=False, default=str)
                if data_snapshot
                else None,
            ),
        )
    conn.close()


def get_conversation_history(
    session_id: str,
    limit: int = 30,
    db_path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    """Return the most recent messages for a session, oldest first.

    Returns a list of {"role": ..., "content": ...} dicts (Groq-compatible).
    """
    if not db_path.exists():
        return []

    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT role, content
        FROM conversations
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (session_id, limit),
    ).fetchall()
    conn.close()

    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def list_sessions(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Return a summary of all conversation sessions (most recent first)."""
    if not db_path.exists():
        return []

    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT
            session_id,
            COUNT(*)        AS message_count,
            MIN(timestamp)  AS started_at,
            MAX(timestamp)  AS last_active,
            MAX(tab_context) AS last_tab
        FROM conversations
        GROUP BY session_id
        ORDER BY last_active DESC
        LIMIT 100
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
