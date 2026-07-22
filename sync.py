"""
sync.py — PerformanceRun sync via Strava
=========================================
Fonte única de dados: Strava API.
O Amazfit T-Rex 3 sincroniza automaticamente com o Strava após cada treino.

Uso:
    python sync.py                   # incremental (desde o último treino)
    python sync.py --full            # backfill completo desde STRAVA_START
    python sync.py --date 2025-06-01 # a partir de uma data específica
    python sync.py --laps-only       # só atualiza laps das atividades existentes

Nota: garmin_insights.json é preservado (dados históricos do Garmin).
      Ele não é mais atualizado — serve apenas para consulta no dashboard.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent
CONSOLIDATED   = PROJECT_ROOT / "data" / "processed" / "activities_consolidated.csv"
LAPS_CSV       = PROJECT_ROOT / "data" / "processed" / "activity_laps_consolidated.csv"
INSIGHTS_FILE  = PROJECT_ROOT / "data" / "processed" / "garmin_insights.json"
STRAVA_START   = "2025-01-01"   # data mais antiga para backfill completo


# ── DB layer (opcional — graceful fallback) ───────────────────────────────────
def _try_import_db():
    try:
        from src.db.queries import upsert_activities, upsert_laps
        return upsert_activities, upsert_laps
    except ImportError:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _date_to_ts(d: str | date) -> int:
    """Converte data YYYY-MM-DD para Unix timestamp (início do dia UTC)."""
    if isinstance(d, str):
        d = date.fromisoformat(d)
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def load_existing_activities() -> pd.DataFrame:
    if CONSOLIDATED.exists():
        return pd.read_csv(
            CONSOLIDATED, sep=";", engine="python",
            on_bad_lines="skip", parse_dates=["start_date"]
        )
    return pd.DataFrame()


def load_existing_laps() -> pd.DataFrame:
    if LAPS_CSV.exists():
        return pd.read_csv(LAPS_CSV, sep=";", engine="python", on_bad_lines="skip")
    return pd.DataFrame()


def get_last_date(df: pd.DataFrame) -> str:
    """Retorna o dia após a última atividade no CSV, ou STRAVA_START."""
    if df.empty or "start_date" not in df.columns:
        return STRAVA_START
    last = pd.to_datetime(df["start_date"]).max()
    return (last + timedelta(days=1)).strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────────────────
#  Activities sync
# ─────────────────────────────────────────────────────────────────────────────

def fetch_and_merge_activities(start: str, df_existing: pd.DataFrame) -> pd.DataFrame:
    from src.ingestion.strava import fetch_strava_activities, build_activity_row

    # Lê FC repouso do garmin_insights.json se disponível (dado histórico)
    resting_hr, max_hr = 45, 195
    if INSIGHTS_FILE.exists():
        try:
            with open(INSIGHTS_FILE) as f:
                ins = json.load(f)
            resting_hr = ins.get("rhr_today") or resting_hr
        except Exception:
            pass

    after_ts  = _date_to_ts(start)
    before_ts = _date_to_ts(date.today()) + 86400  # inclui hoje

    print(f"\n📥 Buscando atividades Strava desde {start}...")
    raw = fetch_strava_activities(after_ts=after_ts, before_ts=before_ts)

    if not raw:
        print("  Nenhuma atividade nova encontrada.")
        return df_existing

    print(f"  ✅ {len(raw)} atividades encontradas.")

    new_rows = [build_activity_row(a, resting_hr=resting_hr, max_hr=max_hr) for a in raw]
    df_new   = pd.DataFrame(new_rows)
    df_new["start_date"] = pd.to_datetime(df_new["start_date"])

    if df_existing.empty:
        return df_new.sort_values("start_date").reset_index(drop=True)

    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=["id"], keep="last")
    df_combined["start_date"] = pd.to_datetime(df_combined["start_date"])
    return df_combined.sort_values("start_date").reset_index(drop=True)


def save_activities(df: pd.DataFrame) -> None:
    CONSOLIDATED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CONSOLIDATED, index=False, sep=";")
    print(f"\n💾 {len(df)} atividades salvas em {CONSOLIDATED.name}")

    upsert_act, _ = _try_import_db()
    if upsert_act:
        try:
            n = upsert_act(df)
            print(f"   SQLite: {n} atividades upserted")
        except Exception as e:
            print(f"   [SQLite] atividades: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  Laps sync
# ─────────────────────────────────────────────────────────────────────────────

def sync_laps(df_acts: pd.DataFrame, start: str) -> None:
    """
    Busca laps das atividades novas (desde `start`) e salva no CSV.
    1 requisição por atividade — controle de rate limit automático.
    """
    from src.ingestion.strava import fetch_laps_for_activity

    df_existing_laps = load_existing_laps()
    existing_ids = set(df_existing_laps["activity_id"].astype(str).unique()) \
                   if not df_existing_laps.empty else set()

    # Filtra só atividades Run/TrailRun novas
    new_acts = df_acts[
        (df_acts["start_date"] >= pd.Timestamp(start)) &
        (df_acts["sport_type"].isin(["Run", "TrailRun"])) &
        (~df_acts["id"].astype(str).isin(existing_ids))
    ].copy()

    if new_acts.empty:
        print("  Laps já atualizados.")
        return

    print(f"\n📥 Buscando laps para {len(new_acts)} atividades novas...")
    all_new_laps = []
    for i, (_, row) in enumerate(new_acts.iterrows(), 1):
        act_id    = int(row["id"])
        sport     = str(row["sport_type"])
        laps_rows = fetch_laps_for_activity(act_id, sport)
        all_new_laps.extend(laps_rows)
        if i % 10 == 0:
            print(f"  → {i}/{len(new_acts)} atividades")

    if not all_new_laps:
        print("  Nenhum lap encontrado.")
        return

    df_new_laps = pd.DataFrame(all_new_laps)
    df_all_laps = pd.concat([df_existing_laps, df_new_laps], ignore_index=True) \
                  if not df_existing_laps.empty else df_new_laps
    df_all_laps = df_all_laps.drop_duplicates(
        subset=["activity_id", "lap_index"], keep="last"
    )

    LAPS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_all_laps.to_csv(LAPS_CSV, index=False, sep=";")
    print(f"💾 {len(df_all_laps)} laps salvos em {LAPS_CSV.name}")

    _, upsert_laps = _try_import_db()
    if upsert_laps:
        try:
            n = upsert_laps(df_new_laps)
            print(f"   SQLite: {n} laps upserted")
        except Exception as e:
            print(f"   [SQLite] laps: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PerformanceRun sync (Strava / Amazfit T-Rex 3)")
    parser.add_argument("--full",       action="store_true", help="Backfill completo desde STRAVA_START")
    parser.add_argument("--date",       type=str,            help="Data de início (YYYY-MM-DD)")
    parser.add_argument("--laps-only",  action="store_true", help="Só atualiza laps das atividades existentes")
    args = parser.parse_args()

    df_existing = load_existing_activities()

    if args.laps_only:
        if df_existing.empty:
            print("Nenhuma atividade no CSV. Rode sem --laps-only primeiro.")
            return
        start = get_last_date(load_existing_laps()) if not load_existing_laps().empty else STRAVA_START
        sync_laps(df_existing, start)
        return

    # Determinar janela de sync
    if args.full:
        start = STRAVA_START
    elif args.date:
        start = args.date
    else:
        start = get_last_date(df_existing)

    print(f"🔄 PerformanceRun Sync — Strava (Amazfit T-Rex 3)")
    print(f"   Janela: {start} → {date.today()}")

    df = fetch_and_merge_activities(start, df_existing)
    save_activities(df)
    sync_laps(df, start)

    print("\n✅ Sync concluído!")
    print("👉 Próximo passo:")
    print("   git add data/processed/ && git commit -m 'data: sync' && git push")


if __name__ == "__main__":
    main()
