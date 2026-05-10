"""
PerformanceRun — Streamlit Dashboard  (versão otimizada)
"""
import re
import ast
import math
import datetime as _dt
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

try:
    import pydeck as pdk
    HAS_PYDECK = True
except ImportError:
    HAS_PYDECK = False
try:
    from streamlit_folium import st_folium
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

st.set_page_config(page_title="PerformanceRun 🏃", page_icon="🏃",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""<style>
@media (max-width: 768px) {
    .block-container { padding: .5rem .5rem 2rem !important; }
    [data-testid="column"] { min-width: 0 !important; }
    .stPlotlyChart > div { overflow-x: auto !important; }
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.2rem !important; }
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 1rem;
}
</style>""", unsafe_allow_html=True)

# ── Constantes ────────────────────────────────────────────────────────────────
INTENSITY_COLORS = {
    "Leve":           "#27AE60",
    "Moderado":       "#F1C40F",
    "Moderado Firme": "#E67E22",
    "Forte":          "#E74C3C",
    "Muito Forte":    "#922B21",
    "Skate":          "#95A5A6",
}
INTENSITY_ORDER = ["Leve","Moderado","Moderado Firme","Forte","Muito Forte","Skate"]

ZONA_COLORS = {
    "Sem FC":            "#BDC3C7",
    "Z1 - Regenerativo": "#2ECC71",
    "Z2 - Aeróbico":     "#3498DB",
    "Z3 - Tempo":        "#F39C12",
    "Z4 - Limiar":       "#E67E22",
    "Z5 - VO2max":       "#E74C3C",
}
ZONA_ORDER = ["Sem FC","Z1 - Regenerativo","Z2 - Aeróbico",
              "Z3 - Tempo","Z4 - Limiar","Z5 - VO2max"]

GREEN, BLUE, AMBER, RED, PURPLE, GRAY = \
    "#2ECC71","#3498DB","#F39C12","#E74C3C","#9B59B6","#95A5A6"

MESES_PT = {"Jan":"jan","Feb":"fev","Mar":"mar","Apr":"abr","May":"mai",
            "Jun":"jun","Jul":"jul","Aug":"ago","Sep":"set","Oct":"out",
            "Nov":"nov","Dec":"dez"}
DIAS_PT  = {"Monday":"Seg","Tuesday":"Ter","Wednesday":"Qua",
            "Thursday":"Qui","Friday":"Sex","Saturday":"Sáb","Sunday":"Dom"}
DIAS_ORDER_PT = ["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"]

# Zonas fáceis — usado em calc_intensidade_fc (módulo-level, não recriado por chamada)
_ZONAS_FACEIS_REL = {"Z1", "Z2", "Sem FC"}

# FC_MAX padrão — sobrescrito pelo sidebar
FC_MAX = 195

KEYWORDS_INTENSIDADE = {
    "Muito Forte": ["intervalad","tiro","interval","vo2","muito forte",
                    "repetição","repeticao","série","serie","prova","teste"],
    "Forte":       ["fartlek","forte","threshold","limiar"],
    "Moderado Firme": ["progressiv","ritmado","moderado firme","tempo run"],
    "Moderado":    ["longo","moderado","moder","base","contínuo","continuo",
                    "aeróbic","aerobic"],
    "Leve":        ["regenerat","fácil","facil","easy","recovery",
                    "leve","caminhad","walk","solto"],
}

# ── Regex pré-compilados (uma vez no startup, reutilizados em cada chamada) ───
_KW_COMPILED: dict[str, re.Pattern] = {
    intensity: re.compile(
        "|".join(re.escape(kw) for kw in kws), re.IGNORECASE
    )
    for intensity, kws in KEYWORDS_INTENSIDADE.items()
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def mesano_pt(dt_series):
    return dt_series.dt.strftime("%b %Y").apply(
        lambda x: f"{MESES_PT.get(x[:3], x[:3])} {x[4:]}")

def normalize_dt(col):
    return pd.to_datetime(col, dayfirst=True, errors="coerce", utc=True).dt.tz_convert(None)

def fmt_pace(sec):
    """Scalar: segundos → 'M:SS'. Use fmt_pace_vec() para Series."""
    if pd.isna(sec) or sec <= 0:
        return "—"
    s = int(sec)
    return f"{s // 60}:{s % 60:02d}"

def fmt_pace_vec(sec_series: pd.Series) -> pd.Series:
    """Vetorizado: ~5x mais rápido que .apply(fmt_pace) em Series grandes."""
    arr  = sec_series.to_numpy(dtype=float)
    inv  = np.isnan(arr) | (arr <= 0)
    mins = np.where(inv, 0, arr // 60).astype(int)
    secs = np.where(inv, 0, arr %  60).astype(int)
    # constrói strings via list comprehension (evita overhead de .apply por célula)
    out  = [f"{m}:{s:02d}" if not i else "—" for m, s, i in zip(mins, secs, inv)]
    return pd.Series(out, index=sec_series.index)

def set_pace_yaxis(fig, pace_sec_series, step_sec=30):
    mn = max(0, int(pace_sec_series.min()) - step_sec)
    mx = int(pace_sec_series.max()) + step_sec
    vals = list(range(mn - mn % step_sec, mx + step_sec, step_sec))
    fig.update_yaxes(
        autorange="reversed",
        tickvals=[v / 60 for v in vals],
        ticktext=[fmt_pace(v) for v in vals],
        title="Pace (min/km)",
    )
    return fig

def zona_fc(hr):
    """Escalar — mantido para compatibilidade pontual."""
    if pd.isna(hr): return "Sem FC"
    if hr < FC_MAX * 0.70: return "Z1 - Regenerativo"
    if hr < FC_MAX * 0.80: return "Z2 - Aeróbico"
    if hr < FC_MAX * 0.87: return "Z3 - Tempo"
    if hr < FC_MAX * 0.93: return "Z4 - Limiar"
    return "Z5 - VO2max"

def zona_fc_vec(hr_series: pd.Series) -> pd.Series:
    """Vetorizado com np.select — substitui .apply(zona_fc) em toda a Series."""
    fc  = hr_series.to_numpy(dtype=float, na_value=np.nan)
    nan = np.isnan(fc)
    out = np.select(
        [nan,
         (~nan) & (fc < FC_MAX * 0.70),
         (~nan) & (fc < FC_MAX * 0.80),
         (~nan) & (fc < FC_MAX * 0.87),
         (~nan) & (fc < FC_MAX * 0.93)],
        ["Sem FC",
         "Z1 - Regenerativo",
         "Z2 - Aeróbico",
         "Z3 - Tempo",
         "Z4 - Limiar"],
        default="Z5 - VO2max",
    )
    return pd.Series(out, index=hr_series.index, name="Zona FC")

def hex_to_rgba(hex_color, alpha=200):
    h = hex_color.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)] + [alpha]

def pace_to_rgba(pace_sec, min_pace=220, max_pace=420, alpha=220):
    t = min(1, max(0, (pace_sec - min_pace) / (max_pace - min_pace)))
    return [round(46 + t*185), round(204 - t*128), round(113 - t*53), alpha]

def fc_to_hex(fc_bpm):
    if pd.isna(fc_bpm) or float(fc_bpm) <= 0: return "#3498DB"
    fc = float(fc_bpm)
    if fc < FC_MAX * 0.70: return "#3498DB"
    if fc < FC_MAX * 0.80: return "#2ECC71"
    if fc < FC_MAX * 0.87: return "#F39C12"
    if fc < FC_MAX * 0.93: return "#E67E22"
    return "#E74C3C"

def elev_gain_to_hex(elev_m_per_km):
    if pd.isna(elev_m_per_km) or float(elev_m_per_km) < 0: return "#2ECC71"
    t = min(1.0, max(0.0, float(elev_m_per_km) / 45.0))
    return "#{:02X}{:02X}{:02X}".format(
        round(46 + t*185), round(204 - t*128), round(113 - t*53))

def pace_to_hex(pace_sec):
    t = min(1, max(0, (float(pace_sec or 300) - 220) / 200))
    r = round(231 - t * (231 - 46))
    g = round(76  + t * (204 - 76))
    b = round(60  + t * (113 - 60))
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def cat_intensity(df):
    if "Intensidade" not in df.columns:
        return df
    df = df.copy()
    df["Intensidade"] = pd.Categorical(df["Intensidade"],
                                        categories=INTENSITY_ORDER, ordered=True)
    return df


# ── decode_polyline cacheado ──────────────────────────────────────────────────
@st.cache_data(max_entries=2000, show_spinner=False)
def decode_polyline(encoded):
    """Decodifica Google Encoded Polyline → [(lat, lng)]. Cacheado por string."""
    if not encoded or pd.isna(encoded) or str(encoded) in ("nan", "None", ""):
        return []
    encoded = str(encoded)
    coords, idx, lat, lng = [], 0, 0, 0
    while idx < len(encoded):
        for is_lng in (False, True):
            shift = result = 0
            while True:
                if idx >= len(encoded):
                    break
                b = ord(encoded[idx]) - 63
                idx += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if (result & 1) else (result >> 1)
            if not is_lng: lat += delta
            else:          lng += delta
        coords.append((lat / 1e5, lng / 1e5))
    return coords

# ── _classify_by_name com regex pré-compilado ─────────────────────────────────
def _classify_by_name(name):
    if not name or pd.isna(name):
        return None
    s = str(name)
    for intensity, pattern in _KW_COMPILED.items():
        if pattern.search(s):
            return intensity
    return None

# ── calc_intensidade_fc com zona_rel vetorizado ───────────────────────────────
def calc_intensidade_fc(laps_df, act_names=None):
    if laps_df.empty or "average_heartrate" not in laps_df.columns:
        return pd.Series(dtype=str)

    hr_max = laps_df["average_heartrate"].dropna().max()
    hr_max = hr_max if (not pd.isna(hr_max) and hr_max > 150) else 195

    # ── Zona relativa VETORIZADA (substitui .apply(zona_rel)) ────────────────
    fc  = laps_df["average_heartrate"].to_numpy(dtype=float, na_value=np.nan)
    pct = fc / hr_max
    nan = np.isnan(pct)
    laps_df = laps_df.copy()
    laps_df["Zona_Rel"] = np.select(
        [nan,
         (~nan) & (pct < 0.65),
         (~nan) & (pct < 0.78),
         (~nan) & (pct < 0.85),
         (~nan) & (pct < 0.92)],
        ["Sem FC", "Z1", "Z2", "Z3", "Z4"],
        default="Z5",
    )

    results = {}
    for aid, grp in laps_df.groupby("activity_id"):
        # 1ª — nome da atividade tem prioridade
        if act_names is not None and aid in act_names.index:
            by_name = _classify_by_name(act_names.get(aid))
            if by_name:
                results[aid] = by_name
                continue

        grp = grp.sort_values("lap_index")
        if len(grp) > 3:
            if grp.iloc[0]["Zona_Rel"] in _ZONAS_FACEIS_REL:
                grp = grp.iloc[1:]
            if len(grp) > 1 and grp.iloc[-1]["Zona_Rel"] in _ZONAS_FACEIS_REL:
                grp = grp.iloc[:-1]

        total = grp["moving_time_sec"].sum()
        if total == 0:
            continue

        sem_fc = grp[grp["Zona_Rel"] == "Sem FC"]["moving_time_sec"].sum()
        if sem_fc / total > 0.5:
            continue

        def pct_z(z):
            return grp[grp["Zona_Rel"] == z]["moving_time_sec"].sum() / total

        z1, z2, z3 = pct_z("Z1"), pct_z("Z2"), pct_z("Z3")
        z4, z5     = pct_z("Z4"), pct_z("Z5")
        pct_z45    = z4 + z5
        pct_z12    = z1 + z2

        if   z5 >= 0.20 or pct_z45 >= 0.50: intensity = "Muito Forte"
        elif pct_z45 >= 0.30:                intensity = "Forte"
        elif pct_z45 >= 0.15 or z3 >= 0.25: intensity = "Moderado Firme"
        elif pct_z12 >= 0.75:                intensity = "Leve"
        else:                                intensity = "Moderado"
        results[aid] = intensity

    return pd.Series(results, name="Intensidade_FC")

# ── PMC cacheado ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def calc_pmc(df_run_all, ctl_days=42, atl_days=7):
    """EMA sobre suffer_score diário → CTL / ATL / TSB. Cacheado."""
    if "suffer_score" not in df_run_all.columns or df_run_all["suffer_score"].isna().all():
        return pd.DataFrame()
    ss = (df_run_all[df_run_all["suffer_score"].notna()]
          .set_index("start_date")["suffer_score"]
          .resample("D").sum())
    if ss.empty:
        return pd.DataFrame()
    full_idx = pd.date_range(ss.index.min(), ss.index.max(), freq="D")
    ss = ss.reindex(full_idx, fill_value=0)

    k_ctl = 2 / (ctl_days + 1)
    k_atl = 2 / (atl_days + 1)
    ctl_vals, atl_vals, ctl, atl = [], [], 0.0, 0.0
    for load in ss:
        ctl += k_ctl * (load - ctl)
        atl += k_atl * (load - atl)
        ctl_vals.append(round(ctl, 1))
        atl_vals.append(round(atl, 1))

    pmc = pd.DataFrame({"Data": ss.index, "CTL": ctl_vals, "ATL": atl_vals})
    pmc["TSB"] = (pmc["CTL"] - pmc["ATL"]).round(1)
    return pmc

def compute_main_laps_pace(laps_group):
    """Pace do bloco principal, removendo aquec/desaquec."""
    if laps_group.empty:
        return None
    laps = laps_group.sort_values("lap_index")
    laps = laps[laps["pace_sec_km"].notna() & (laps["pace_sec_km"] > 0) & (laps["pace_sec_km"] < 500)]
    if len(laps) == 0:
        return None
    if len(laps) <= 2:
        return float(laps["pace_sec_km"].mean())
    mediana   = float(laps["pace_sec_km"].median())
    threshold = mediana * 1.15
    if len(laps) > 3 and float(laps.iloc[0]["pace_sec_km"]) > threshold:
        laps = laps.iloc[1:]
    if len(laps) > 2 and float(laps.iloc[-1]["pace_sec_km"]) > threshold:
        laps = laps.iloc[:-1]
    if laps.empty:
        return None
    total_dist = float(laps["distance_km"].sum())
    if total_dist <= 0:
        return float(laps["pace_sec_km"].mean())
    return float((laps["pace_sec_km"] * laps["distance_km"]).sum() / total_dist)


# ══════════════════════════════════════════════════════════════════════════════
#  DADOS
# ══════════════════════════════════════════════════════════════════════════════
BASE = "data/processed"

@st.cache_data(ttl=300)
def load_all(base):
    import os
    def read(name):
        p = f"{base}/{name}"
        return pd.read_csv(p, sep=";", encoding="utf-8-sig") \
               if os.path.exists(p) else pd.DataFrame()

    act = read("activities_consolidated.csv")
    if not act.empty:
        act["start_date"] = normalize_dt(act["start_date"])
        act["MesAno"]     = mesano_pt(act["start_date"])
        act["MesAnoOrd"]  = act["start_date"].dt.to_period("M").apply(lambda x: x.ordinal)
        act["Semana"]     = act["start_date"].dt.to_period("W").apply(lambda x: x.ordinal)
        act["SemanaStr"]  = act["start_date"].dt.strftime("Sem %V/%Y")
        act["DiaSemana"]  = act["start_date"].dt.day_name().map(DIAS_PT)
        for _c in ["latitude", "longitude"]:
            if _c in act.columns:
                act[_c] = pd.to_numeric(act[_c], errors="coerce")
        rename_map = {c: "Intensidade" for c in act.columns if c.lower() == "intensidade"}
        act = act.rename(columns=rename_map)
        if "Intensidade" in act.columns:
            if act["Intensidade"].dropna().empty:
                act = act.drop(columns=["Intensidade"])
            else:
                act["Intensidade"] = act["Intensidade"].astype(str).str.strip().str.title()
                act["Intensidade"] = act["Intensidade"].replace("Nan", None)

    laps = read("activity_laps_consolidated.csv")
    if not laps.empty:
        laps["start_date"] = normalize_dt(laps["start_date"])
        laps["MesAno"]     = mesano_pt(laps["start_date"])
        laps["MesAnoOrd"]  = laps["start_date"].dt.to_period("M").apply(lambda x: x.ordinal)

    be = read("activity_best_efforts_consolidated.csv")
    if not be.empty:
        be["start_date"] = normalize_dt(be["start_date"])
        be["MesAno"]     = mesano_pt(be["start_date"])
        be["MesAnoOrd"]  = be["start_date"].dt.to_period("M").apply(lambda x: x.ordinal)

    # Classificação de intensidade por FC (dentro do cache — só roda quando dados mudam)
    if not laps.empty and not act.empty:
        act_names      = act.set_index("id")["name"] if "name" in act.columns else None
        intensidade_fc = calc_intensidade_fc(laps, act_names=act_names)
        if not intensidade_fc.empty:
            fc_df          = intensidade_fc.reset_index()
            fc_df.columns  = ["id", "Intensidade_FC"]
            act            = act.merge(fc_df, on="id", how="left")

    return act, laps, be

df_raw, laps_raw, be_raw = load_all(BASE)

# PMC all-time — cacheado via @st.cache_data em calc_pmc
_runs_raw = df_raw[df_raw["sport_type"].isin(["Run","TrailRun"])].copy() \
            if not df_raw.empty else df_raw
pmc_raw = calc_pmc(_runs_raw)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("🎛️ Filtros")
min_d = df_raw["start_date"].min().date()
max_d = df_raw["start_date"].max().date()

_col1, _col2 = st.sidebar.columns([3, 1])
with _col1:
    st.markdown("**Período**")
with _col2:
    if st.button("↺", help="Resetar para o período completo", key="reset_date"):
        st.session_state["date_range"] = (min_d, max_d)

_quick_cols = st.sidebar.columns(3)
_today = _dt.date.today()
_quick = {
    "7d":   (max(min_d, _today - _dt.timedelta(days=7)),   min(_today, max_d)),
    "30d":  (max(min_d, _today - _dt.timedelta(days=30)),  min(_today, max_d)),
    "3m":   (max(min_d, _today - _dt.timedelta(days=90)),  min(_today, max_d)),
    "6m":   (max(min_d, _today - _dt.timedelta(days=180)), min(_today, max_d)),
    "Ano":  (max(min_d, _dt.date(_today.year, 1, 1)),      min(_today, max_d)),
    "Tudo": (min_d, max_d),
}
for _i, (_lbl, _rng) in enumerate(_quick.items()):
    if _quick_cols[_i % 3].button(_lbl, key=f"qd_{_lbl}", use_container_width=True):
        st.session_state["date_range"] = _rng

date_range = st.sidebar.date_input(
    "Período", value=st.session_state.get("date_range", (min_d, max_d)),
    min_value=min_d, max_value=max_d, key="date_range",
    label_visibility="collapsed")

sports_all      = sorted(df_raw["sport_type"].dropna().unique())
_default_sports = [s for s in ["Run","TrailRun"] if s in sports_all] or sports_all
selected_sports = st.sidebar.multiselect("Modalidade", sports_all, default=_default_sports)

has_fc_class = ("Intensidade_FC" in df_raw.columns and df_raw["Intensidade_FC"].notna().any())
if has_fc_class:
    int_mode = st.sidebar.radio(
        "Classificação de Intensidade",
        ["Automática (FC)", "Manual"],
        help="Automática: % tempo em zona de FC por atividade. Manual: coluna Intensidade manual.")
else:
    int_mode = "Manual"

int_col  = "Intensidade_FC" if int_mode == "Automática (FC)" else "Intensidade"
int_opts = [i for i in INTENSITY_ORDER
            if int_col in df_raw.columns
            and i in df_raw[int_col].dropna().unique()]
sel_int  = st.sidebar.multiselect("Filtrar Intensidade", int_opts, default=int_opts)

st.sidebar.markdown("---")
FC_MAX = st.sidebar.number_input(
    "❤️ FC Máxima pessoal (bpm)",
    min_value=150, max_value=230,
    value=st.session_state.get("fc_max_val", 195),
    step=1, key="fc_max_val",
    help="Sua FC máxima registrada. Define os limites de cada zona cardíaca."
)
_z1 = round(FC_MAX * 0.70); _z2 = round(FC_MAX * 0.80)
_z3 = round(FC_MAX * 0.87); _z4 = round(FC_MAX * 0.93)
st.sidebar.caption(
    f"Z1 < {_z1} · Z2 {_z1}–{_z2} · Z3 {_z2}–{_z3} · Z4 {_z3}–{_z4} · Z5 ≥ {_z4}"
)

# Recalcula Zona FC com FCmax personalizado — VETORIZADO
if not laps_raw.empty:
    laps_raw = laps_raw.copy()
    laps_raw["Zona FC"] = zona_fc_vec(laps_raw["average_heartrate"])

# ── Filtros de data / modalidade / intensidade ────────────────────────────────
s_dt = pd.Timestamp(date_range[0]) if len(date_range) >= 1 else pd.Timestamp(min_d)
e_dt = (pd.Timestamp(date_range[1]) + pd.Timedelta(hours=23, minutes=59, seconds=59)) \
       if len(date_range) == 2 else (pd.Timestamp(max_d) + pd.Timedelta(hours=23, minutes=59, seconds=59))

def filt_act(d):
    mask = ((d["start_date"] >= s_dt) & (d["start_date"] <= e_dt)
            & d["sport_type"].isin(selected_sports))
    if sel_int and int_col in d.columns:
        mask &= d[int_col].isin(sel_int) | d[int_col].isna()
    return d[mask].copy()

def filt_laps(d):
    if d.empty: return d
    return d[(d["start_date"] >= s_dt) & (d["start_date"] <= e_dt)
             & d["activity_sport_type"].isin(selected_sports)].copy()

def filt_be(d):
    if d.empty: return d
    return d[(d["start_date"] >= s_dt) & (d["start_date"] <= e_dt)
             & d["activity_sport_type"].isin(selected_sports)].copy()

df      = filt_act(df_raw)
laps    = filt_laps(laps_raw)
be      = filt_be(be_raw)
df_run  = df[df["sport_type"].isin(["Run","TrailRun"])].copy()
lps_run = laps[laps["activity_sport_type"].isin(["Run","TrailRun"])].copy() \
          if not laps.empty else laps.copy()

for _df in [df, df_run]:
    if int_col in _df.columns:
        _df["Intensidade"] = _df[int_col]
    elif "Intensidade" not in _df.columns:
        _df["Intensidade"] = None


# ── Helpers cacheados (all-time, usam dados brutos) ───────────────────────────
@st.cache_data(show_spinner=False)
def _melhor_be_cached(_be_raw):
    """Retorna dict nome_lower → melhor pace (seg). Cacheado."""
    if _be_raw.empty:
        return {}
    return (_be_raw.groupby(_be_raw["name"].str.lower())["pace_sec_km"]
            .min().to_dict())

@st.cache_data(show_spinner=False)
def _melhor_3km_cached(_laps_raw):
    """Melhor pace em 3km consecutivos. Cacheado."""
    lps = _laps_raw[_laps_raw["activity_sport_type"].isin(["Run","TrailRun"])]
    if lps.empty:
        return "—"
    best = None
    for _, grp in lps.groupby("activity_id"):
        km = grp[(grp["distance_m"] >= 900) & (grp["distance_m"] <= 1100)]
        if len(km) >= 3:
            t = km.nsmallest(3, "lap_index")["moving_time_sec"].sum()
            if best is None or t < best:
                best = t
    return fmt_pace(best / 3) if best else "—"

_be_cache = _melhor_be_cached(be_raw)

def melhor_be(nome):
    v = _be_cache.get(nome.lower())
    return fmt_pace(v) if v else "—"

def melhor_3km():
    return _melhor_3km_cached(laps_raw)


def analyze_run(laps):
    """Gera insights textuais de uma atividade a partir dos seus laps."""
    insights = []
    if laps.empty:
        return insights
    if len(laps) >= 4 and "pace_sec_km" in laps.columns:
        first  = laps["pace_sec_km"].iloc[:len(laps)//2].mean()
        second = laps["pace_sec_km"].iloc[len(laps)//2:].mean()
        delta  = second - first
        if delta > 10:
            insights.append(f"⚠️ Queda de ritmo (+{delta:.0f}s/km)")
        elif delta < -10:
            insights.append(f"🚀 Negative split ({abs(delta):.0f}s/km)")
    if "average_heartrate" in laps.columns and laps["average_heartrate"].notna().any():
        first  = laps["average_heartrate"].iloc[:len(laps)//2].mean()
        second = laps["average_heartrate"].iloc[len(laps)//2:].mean()
        drift  = second - first
        if drift > 5:
            insights.append(f"❤️ Deriva cardíaca (+{drift:.0f} bpm)")
    if "pace_sec_km" in laps.columns and laps["pace_sec_km"].notna().any():
        var = laps["pace_sec_km"].std()
        if not pd.isna(var):
            if var < 8:
                insights.append("💪 Ritmo consistente")
            elif var > 20:
                insights.append("⚠️ Ritmo irregular")
    return insights


# ══════════════════════════════════════════════════════════════════════════════

tab_hoje, tab_desemp, tab_carga, tab_mapa, tab_hist = st.tabs([
    "🏠 Dashboard",
    "⚡ Desempenho",
    "💓 Carga & Zonas",
    "🗺️ Mapa",
    "📋 Histórico",
])

# ══════════════════════════════════════════════════════════════════════════════
#  1 · DASHBOARD  —  visão do dia + resumo rápido
# ══════════════════════════════════════════════════════════════════════════════
with tab_hoje:
    st.title("🏠 Dashboard")

    hoje = pd.Timestamp.now()
    # ACWR — últimos 7 dias reais (não semana-calendário)
    ult7d  = _runs_raw[_runs_raw["start_date"] >= hoje - timedelta(days=7)]
    ult28d = _runs_raw[_runs_raw["start_date"] >= hoje - timedelta(days=28)]

    carga_aguda   = float(ult7d["suffer_score"].sum())  if "suffer_score" in _runs_raw.columns else 0.0
    carga_cronica = float(ult28d["suffer_score"].sum()) / 4 if "suffer_score" in _runs_raw.columns else 0.0
    acwr          = round(carga_aguda / carga_cronica, 2) if carga_cronica > 0 else 0.0

    if   acwr < 0.8:  acwr_lbl, acwr_cor = "🟡 Subcarregado",   "normal"
    elif acwr <= 1.3: acwr_lbl, acwr_cor = "🟢 Zona Segura",    "normal"
    elif acwr <= 1.5: acwr_lbl, acwr_cor = "🟠 Atenção",        "inverse"
    else:             acwr_lbl, acwr_cor = "🔴 Alto Risco",      "inverse"

    # TSB atual (do PMC)
    tsb_at = float(pmc_raw["TSB"].iloc[-1]) if not pmc_raw.empty else 0.0
    ctl_at = float(pmc_raw["CTL"].iloc[-1]) if not pmc_raw.empty else 0.0
    if   tsb_at >  20: tsb_lbl = "😴 Descansado"
    elif tsb_at >= 5:  tsb_lbl = "✅ Forma ideal"
    elif tsb_at >= -10:tsb_lbl = "⚙️ Treinando"
    elif tsb_at >= -20:tsb_lbl = "⚠️ Fatigado"
    else:              tsb_lbl = "🛑 Overtraining"

    km_7d = float(ult7d["distance_km"].sum())

    # Mês corrente
    mes_run = df_run[df_run["start_date"].dt.to_period("M") == hoje.to_period("M")]
    km_mes  = float(mes_run["distance_km"].sum())
    tr_mes  = len(mes_run)

    # ── Cards linha 1: métricas do período filtrado ───────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("🏃 Atividades",  f"{len(df):,}")
    c2.metric("📏 Distância",   f"{df['distance_km'].sum():,.0f} km")
    c3.metric("⏱️ Tempo",       f"{df['moving_time_sec'].sum()/3600:,.1f} h")
    c4.metric("⚡ Pace Médio",  fmt_pace(df_run["pace_sec_km"].mean()))
    fc_med = df_run["average_heartrate"].mean()
    c5.metric("❤️ FC Média",   f"{fc_med:.0f} bpm" if not pd.isna(fc_med) else "—")
    c6.metric("🔥 Calorias",   f"{df['calories'].sum():,.0f}" if df["calories"].notna().any() else "—")

    st.markdown("---")

    # ── Cards linha 2: forma do dia (all-time, não filtrado) ─────────────────
    st.caption("**Forma atual** — baseada em todo o histórico (independente do filtro de período)")
    ca, cb, cc, cd = st.columns(4)
    ca.metric("⚡ ACWR (7d)",     f"{acwr:.2f}", acwr_lbl, delta_color=acwr_cor,
              help="Acute:Chronic Workload Ratio — razão entre a carga dos últimos 7 dias e a média das 4 semanas anteriores.\n\n"
     "Zona segura: 0,8–1,3.\n\n"
     "Abaixo de 0,8 = subcarregado (espaço para aumentar).\n\n"
     "Acima de 1,3 = atenção.\n\n"
     "Acima de 1,5 = alto risco de lesão por overload.")

    cb.metric("✨ TSB — Forma",    f"{tsb_at:+.0f}", tsb_lbl,
              delta_color="normal" if tsb_at >= 0 else "inverse",
              help="TSB = CTL − ATL (Fitness − Fadiga)  •  +5 a +20: janela de pico  •  0 a −10: treinando  •  abaixo de −20: overtraining")
    cc.metric("💪 CTL — Fitness",  f"{ctl_at:.0f}",
              help="Chronic Training Load — carga crônica calculada por média exponencial dos últimos 42 dias "
                   "sobre o suffer score. Representa sua base de condicionamento: quanto maior, maior o motor aeróbico.")
    cd.metric("📏 KM últimos 7d",  f"{km_7d:.0f} km")

    # Alert inteligente
    if acwr > 1.5:
        st.error("🛑 ACWR crítico — reduza o volume esta semana para evitar lesão.")
    elif acwr > 1.3:
        st.warning("⚠️ ACWR elevado — monitore sinais de fadiga.")
    elif tsb_at < -20:
        st.error("🛑 TSB muito negativo — risco de overtraining. Priorize recuperação.")
    elif tsb_at >= 5 and tsb_at <= 20:
        st.success("✅ Forma em dia — boa janela para treino de qualidade ou competição.")
    elif acwr < 0.8:
        st.info("📉 Carga baixa — espaço para aumentar volume com segurança.")

    st.markdown("---")

    # ── Linha visual 1: Mix Intensidade + Distância Mensal ───────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        if "Intensidade" in df.columns and df["Intensidade"].notna().any():
            df_i = cat_intensity(df)["Intensidade"].value_counts().reset_index()
            df_i.columns = ["Intensidade","Qtd"]
            df_i = df_i[df_i["Intensidade"].isin(INTENSITY_ORDER)]
            fig = px.pie(df_i, names="Intensidade", values="Qtd",
                         title="🎯 Mix de Intensidade", hole=0.45,
                         color="Intensidade", color_discrete_map=INTENSITY_COLORS,
                         category_orders={"Intensidade": INTENSITY_ORDER})
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(showlegend=False, margin=dict(t=40,b=0,l=0,r=0))
            st.plotly_chart(fig, width="stretch")
    with col_b:
        df_m = (df.groupby(["MesAnoOrd","MesAno"])
                  .agg(KM=("distance_km","sum")).reset_index()
                  .sort_values("MesAnoOrd").tail(12))
        fig = px.bar(df_m, x="MesAno", y="KM",
                     title="📏 Distância por Mês (km) — últimos 12 meses",
                     color_discrete_sequence=[BLUE],
                     labels={"KM":"km","MesAno":""}, text_auto=".0f")
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_tickangle=-45, showlegend=False,
                          margin=dict(t=40,b=0,l=0,r=0))
        st.plotly_chart(fig, width="stretch")

    # ── Linha visual 2: Pace mensal + dias da semana ──────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        df_p = (df_run[df_run["pace_sec_km"].notna()]
                .groupby(["MesAnoOrd","MesAno"])
                .agg(Pace=("pace_sec_km","mean")).reset_index()
                .sort_values("MesAnoOrd").tail(12))
        df_p["Pace_fmt"] = fmt_pace_vec(df_p["Pace"])
        df_p["Pace_min"] = df_p["Pace"] / 60
        df_p["Roll3M"]   = df_p["Pace_min"].rolling(3, min_periods=1).mean()
        fig = go.Figure()
        fig.add_bar(x=df_p["MesAno"], y=df_p["Pace_min"], name="Pace mensal",
                    marker_color=BLUE, opacity=0.5,
                    customdata=df_p[["Pace_fmt"]].values,
                    hovertemplate="%{x}<br>%{customdata[0]}/km<extra></extra>")
        fig.add_scatter(x=df_p["MesAno"], y=df_p["Roll3M"], name="Média 3M",
                        mode="lines+markers", line=dict(color=RED, width=2, dash="dash"))
        set_pace_yaxis(fig, df_p["Pace"])
        fig.update_layout(title="⚡ Evolução do Pace + Média 3M",
                          xaxis_tickangle=-45, margin=dict(t=40,b=0,l=0,r=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("📖 Eixo Y invertido: mais alto = mais rápido.")
    with col_b:
        df_dia = (df["DiaSemana"].value_counts()
                  .reindex(DIAS_ORDER_PT).fillna(0).reset_index()
                  .rename(columns={"DiaSemana":"Dia","count":"Qtd"}))
        fig = px.bar(df_dia, x="Dia", y="Qtd",
                     title="📅 Dias mais ativos",
                     color_discrete_sequence=[PURPLE], text_auto=True)
        fig.update_layout(showlegend=False, margin=dict(t=40,b=0,l=0,r=0))
        st.plotly_chart(fig, width="stretch")

    # ── Metas do mês (simples) ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Metas do Mês")
    mg1, mg2, mg3, mg4 = st.columns(4)
    mg1.metric("📏 KM este mês",  f"{km_mes:.0f} km",    f"{km_mes/150*100:.0f}% de 150 km")
    mg2.metric("🏃 Treinos",      f"{tr_mes}",            f"{tr_mes/16*100:.0f}% de 16")
    mg3.metric("🥇 Melhor 5K",    melhor_be("5k"))
    mg4.metric("🥇 Melhor 10K",   melhor_be("10k"))

    # ── 🔥 Streak de semanas ativas ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔥 Consistência — Semanas Ativas")

    _all_runs = _runs_raw.copy() if not _runs_raw.empty else pd.DataFrame()
    if not _all_runs.empty:
        _all_runs["_week"] = _all_runs["start_date"].dt.to_period("W")
        _wk_counts = (_all_runs.groupby("_week").size().reset_index(name="runs"))
        _wk_counts["active"] = _wk_counts["runs"] >= 3

        _streak_atual = 0
        for _a in reversed(_wk_counts["active"].tolist()):
            if _a: _streak_atual += 1
            else:  break

        _best, _cur = 0, 0
        for _a in _wk_counts["active"].tolist():
            if _a: _cur += 1; _best = max(_best, _cur)
            else:  _cur = 0

        _total_active = int(_wk_counts["active"].sum())
        _total_weeks  = len(_wk_counts)
        _pct_active   = _total_active / _total_weeks * 100 if _total_weeks else 0

        sk1, sk2, sk3, sk4 = st.columns(4)
        sk1.metric("🔥 Streak Atual",   f"{_streak_atual} sem.", "semanas ≥ 3 corridas consecutivas")
        sk2.metric("🏆 Melhor Streak",  f"{_best} sem.")
        sk3.metric("✅ Semanas Ativas",  f"{_total_active}/{_total_weeks}", f"{_pct_active:.0f}% de consistência")
        sk4.metric("📅 Total de Semanas", f"{_total_weeks}")

        _wk_plot = _wk_counts.tail(26).copy()
        _wk_plot["semana_str"] = _wk_plot["_week"].astype(str).str.replace(
            r"(\d{4})-W(\d+)", r"S\2/\1", regex=True)
        _wk_plot["cor"]   = _wk_plot["active"].map({True: "#2ECC71", False: "#E74C3C"})
        _wk_plot["alpha"] = _wk_plot["active"].map({True: 1.0, False: 0.45})

        fig_streak = go.Figure()
        fig_streak.add_bar(
            x=_wk_plot["semana_str"], y=_wk_plot["runs"],
            marker=dict(color=_wk_plot["cor"].tolist(),
                        opacity=_wk_plot["alpha"].tolist(),
                        line=dict(width=0)),
            text=_wk_plot["runs"], textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{y} corridas<extra></extra>",
        )
        fig_streak.add_hline(y=3, line_dash="dot", line_color="#F1C40F", line_width=2,
                             annotation_text="  mínimo saudável (3/sem)",
                             annotation_position="top left",
                             annotation_font=dict(size=11, color="#F1C40F"))
        if _streak_atual > 0:
            fig_streak.add_annotation(
                x=_wk_plot["semana_str"].iloc[-1],
                y=float(_wk_plot["runs"].iloc[-1]) + 0.3,
                text=f"🔥 {_streak_atual}", showarrow=False,
                font=dict(size=13, color="#2ECC71"), xanchor="center")
        fig_streak.update_layout(
            title=dict(text="Corridas por semana — últimas 26 semanas", font=dict(size=15), x=0),
            xaxis=dict(tickangle=-45, showgrid=False, tickfont=dict(size=9)),
            yaxis=dict(title="Corridas", gridcolor="rgba(128,128,128,0.15)"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=45, b=10, l=0, r=0), showlegend=False)
        st.plotly_chart(fig_streak, width="stretch")
        if   _streak_atual >= 8: st.success(f"🔥 {_streak_atual} semanas consecutivas! Consistência de elite.")
        elif _streak_atual >= 4: st.info(f"💪 {_streak_atual} semanas em sequência — hábito sólido.")
        elif _streak_atual == 0: st.warning("⚠️ Sequência interrompida. Que tal retomar esta semana?")

# ══════════════════════════════════════════════════════════════════════════════
#  2 · DESEMPENHO  —  PRs, pace, eficiência, condições
# ══════════════════════════════════════════════════════════════════════════════
with tab_desemp:
    st.title("⚡ Desempenho")

    # ── Cards de PRs ──────────────────────────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🥇 Melhor 3km",  melhor_3km())
    c2.metric("🥇 Melhor 5K",   melhor_be("5k"))
    c3.metric("🥇 Melhor 10K",  melhor_be("10k"))
    c4.metric("🥇 Melhor HM",   melhor_be("half-marathon"), help="Half Marathon 21,1 km")
    c5.metric("🏅 PRs Totais",  f"{int(df_run['pr_count'].sum()):,}"
                                  if df_run["pr_count"].notna().any() else "—")
    st.markdown("---")

    # ── 🏁 Preditor de Prova — Fórmula de Riegel ─────────────────────────────
    def _riegel(t1_sec, d1_m, d2_m): return t1_sec * (d2_m / d1_m) ** 1.06
    def _fmt_hms(sec):
        sec = int(sec); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
        return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}:{s:02d}"

    _DIST_M = {"1K":1000,"3K":3000,"5K":5000,"10K":10000,"HM 21K":21097,"Maratona":42195}
    _ref_sec, _ref_dist, _ref_nome = None, None, None
    if not be_raw.empty:
        for _nb, _db in [("10k",10000),("5k",5000),("1k",1000)]:
            _sub = be_raw[be_raw["name"].str.lower() == _nb]
            if not _sub.empty:
                _ref_sec  = float(_sub["pace_sec_km"].min() * _db / 1000)
                _ref_dist = _db; _ref_nome = _nb.upper(); break

    if _ref_sec and _ref_dist:
        st.subheader("🏁 Preditor de Provas")
        st.caption(f"Baseado no seu melhor **{_ref_nome}** ({_fmt_hms(_ref_sec)}) · "
                   "Fórmula de Riegel: T₂ = T₁ × (D₂/D₁)^1.06")
        _preds = {d: _riegel(_ref_sec, _ref_dist, m) for d,m in _DIST_M.items() if m != _ref_dist}
        _ICONS = {"1K":"🚀","3K":"💨","5K":"🏃","10K":"⚡","HM 21K":"🌟","Maratona":"🏅"}
        _cols_pred = st.columns(len(_preds))
        for _ci, (_d, _t) in zip(_cols_pred, _preds.items()):
            _ci.metric(f"{_ICONS.get(_d,'')} {_d}", _fmt_hms(_t),
                       f"pace {fmt_pace(_t/(_DIST_M[_d]/1000))}/km", delta_color="off")
        # Gráfico
        _all_preds = {**{_ref_nome: _ref_sec}, **_preds}
        _chart_data = sorted(
            [{"Distância": d,
              "Pace_sec":  _all_preds[d] / (_DIST_M.get(d, _ref_dist) / 1000),
              "Pace_min":  _all_preds[d] / (_DIST_M.get(d, _ref_dist) / 1000) / 60,
              "Pace_fmt":  fmt_pace(_all_preds[d] / (_DIST_M.get(d, _ref_dist) / 1000)),
              "Tempo":     _fmt_hms(_all_preds[d]),
              "is_ref":    d == _ref_nome}
             for d in _all_preds if _DIST_M.get(d, _ref_dist) > 0],
            key=lambda x: _DIST_M.get(x["Distância"], _ref_dist))
        _cd = pd.DataFrame(_chart_data)
        fig_pred = go.Figure()
        fig_pred.add_scatter(x=_cd["Distância"], y=_cd["Pace_min"],
                             fill="tozeroy", fillcolor="rgba(52,152,219,0.07)",
                             line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig_pred.add_scatter(
            x=_cd["Distância"], y=_cd["Pace_min"],
            mode="lines+markers+text",
            line=dict(color="#3498DB", width=2.5, shape="spline"),
            marker=dict(size=[14 if r else 9 for r in _cd["is_ref"]],
                        color=["#F1C40F" if r else "#3498DB" for r in _cd["is_ref"]],
                        line=dict(width=2, color="white")),
            text=_cd["Pace_fmt"], textposition="top center", textfont=dict(size=10),
            customdata=_cd[["Tempo","Distância"]].values,
            hovertemplate="<b>%{customdata[1]}</b><br>Tempo: <b>%{customdata[0]}</b><br>Pace: <b>%{text}/km</b><extra></extra>",
            showlegend=False)
        _ref_row = _cd[_cd["is_ref"]]
        if not _ref_row.empty:
            fig_pred.add_scatter(x=_ref_row["Distância"], y=_ref_row["Pace_min"],
                                 mode="markers+text",
                                 marker=dict(size=16, color="#F1C40F",
                                             line=dict(width=2, color="white"), symbol="star"),
                                 text=["⭐ referência"], textposition="bottom center",
                                 textfont=dict(size=10, color="#F1C40F"),
                                 showlegend=False, hoverinfo="skip")
        set_pace_yaxis(fig_pred, _cd["Pace_sec"])
        fig_pred.update_layout(
            title=dict(text=f"Pace predito por distância (ref: {_ref_nome})", font=dict(size=14), x=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=45, b=10, l=0, r=10))
        st.plotly_chart(fig_pred, width="stretch")
        st.caption("⭐ = tempo real · demais = estimativas Riegel. Margem maior para distâncias longas.")
    st.markdown("---")

    # ── Evolução melhor pace 5K / 10K ────────────────────────────────────────
    if not be.empty:
        be_sel = be[be["name"].str.lower().isin(["5k","10k"])].copy()
        if not be_sel.empty:
            be_best = (be_sel.groupby(["MesAnoOrd","MesAno","name"])
                             .agg(Pace=("pace_sec_km","min")).reset_index()
                             .sort_values("MesAnoOrd"))
            be_best["Pace_fmt"] = fmt_pace_vec(be_best["Pace"])
            be_best["Pace_min"] = be_best["Pace"] / 60
            fig = px.line(be_best, x="MesAno", y="Pace_min", color="name",
                          markers=True, custom_data=["Pace_fmt"],
                          title="📈 Evolução Melhor Pace — 5K e 10K",
                          color_discrete_map={"5k": RED, "10k": BLUE,
                                              "5K": RED, "10K": BLUE},
                          labels={"Pace_min":"Pace (min/km)","MesAno":"","name":"Distância"})
            fig.update_traces(
                hovertemplate="<b>%{x}</b><br>Melhor pace: %{customdata[0]}/km<extra></extra>")
            set_pace_yaxis(fig, be_best["Pace"])
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, width="stretch")

    col_a, col_b = st.columns(2)

    # ── Pace médio por intensidade (bloco principal) ──────────────────────────
    with col_a:
        if "Intensidade" in df_run.columns and not lps_run.empty:
            _mp = (lps_run.groupby("activity_id")
                   .apply(compute_main_laps_pace).dropna().reset_index())
            _mp.columns = ["id","pace_main"]
            df_run_p = df_run.merge(_mp, on="id", how="left")
            df_run_p["pace_plot"] = df_run_p["pace_main"].fillna(df_run_p["pace_sec_km"])
        else:
            df_run_p = df_run.copy()
            df_run_p["pace_plot"] = df_run_p["pace_sec_km"]

        df_b = cat_intensity(df_run_p[df_run_p["pace_plot"].notna()].copy())
        df_agg = (df_b.groupby("Intensidade", observed=True)["pace_plot"]
                     .agg(Media="mean", DP="std").reset_index().dropna())
        if not df_agg.empty:
            df_agg["Media_min"] = df_agg["Media"] / 60
            df_agg["DP_min"]    = df_agg["DP"] / 60
            df_agg["Label"]     = fmt_pace_vec(df_agg["Media"])
            df_agg["Cor"]       = df_agg["Intensidade"].map(INTENSITY_COLORS)
            fig = go.Figure()
            for _, row in df_agg.iterrows():
                fig.add_bar(x=[row["Intensidade"]], y=[row["Media_min"]],
                            error_y=dict(type="data", array=[row["DP_min"]], visible=True),
                            marker_color=row["Cor"], name=row["Intensidade"],
                            text=row["Label"], textposition="outside")
            set_pace_yaxis(fig, df_agg["Media"])
            fig.update_layout(title="🎯 Pace por Tipo de Treino", showlegend=False)
            st.plotly_chart(fig, width="stretch")
            st.caption("Pace do bloco principal (aquec/desaquec excluídos). Barra = desvio padrão.")

    # ── Consistência de pace (CV%) ────────────────────────────────────────────
    with col_b:
        if not lps_run.empty:
            def cv_pace(grp):
                p = grp["pace_sec_km"].dropna()
                return (p.std() / p.mean() * 100) if len(p) > 1 and p.mean() > 0 else None
            cv_df = lps_run.groupby("activity_id").apply(cv_pace).dropna().reset_index()
            cv_df.columns = ["activity_id","CV_Pace"]
            lps_cv = lps_run.merge(cv_df, on="activity_id")
            df_cv_m = (lps_cv.groupby(["MesAnoOrd","MesAno"])
                             .agg(CV=("CV_Pace","mean")).reset_index()
                             .sort_values("MesAnoOrd"))
            fig = px.line(df_cv_m, x="MesAno", y="CV", markers=True,
                          title="🎯 Consistência de Pace por Mês (CV%)",
                          color_discrete_sequence=[AMBER],
                          labels={"CV":"CV (%)","MesAno":""})
            fig.add_hline(y=10, line_dash="dash", line_color=GREEN,
                          annotation_text="< 10% — ritmo controlado")
            fig.add_hline(y=25, line_dash="dash", line_color=RED,
                          annotation_text="> 25% — treino intervalado")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, width="stretch")
            st.caption("CV < 10%: ritmo uniforme. CV > 25%: sessão de tiros/intervalos.")


    # ── 🦵 Cadência Média Mensal ──────────────────────────────────────────────
    if not lps_run.empty and "average_cadence" in lps_run.columns:
        _cad = lps_run[lps_run["average_cadence"].notna() & (lps_run["average_cadence"] > 50)].copy()
        _cad["spm"] = _cad["average_cadence"] * 2
        if not _cad.empty:
            _cad_m = (_cad.groupby(["MesAnoOrd","MesAno"])
                         .agg(Cadencia=("spm","mean"), DP=("spm","std")).reset_index()
                         .sort_values("MesAnoOrd"))
            _cad_m["Cor"] = _cad_m["Cadencia"].apply(
                lambda x: "#E74C3C" if x < 160 else "#F1C40F" if x < 170
                else "#2ECC71" if x <= 185 else "#3498DB")
            st.markdown("---")
            st.subheader("🦵 Cadência — Economia de Corrida")
            st.caption("Ideal: **175–185 spm** · < 170 = passada longa, maior risco de lesão.")
            _cad_atual = float(_cad_m["Cadencia"].iloc[-1])
            _cad_media = float(_cad_m["Cadencia"].mean())
            cd1, cd2, cd3 = st.columns(3)
            cd1.metric("🦵 Cadência Atual",  f"{_cad_atual:.0f} spm",
                       f"{_cad_atual-_cad_media:+.1f} vs média",
                       delta_color="normal" if 175<=_cad_atual<=185 else "inverse")
            cd2.metric("📊 Média Histórica", f"{_cad_media:.0f} spm")
            cd3.metric("📍 Zona", ("🔴 Baixa" if _cad_atual<160 else "🟡 Atenção"
                                   if _cad_atual<170 else "✅ Ideal" if _cad_atual<=185 else "🔵 Alta"))
            fig_cad = go.Figure()
            fig_cad.add_hrect(y0=175, y1=185, fillcolor="rgba(46,204,113,0.10)",
                              line=dict(width=0), annotation_text="zona ideal 175–185",
                              annotation_position="top right",
                              annotation_font=dict(size=10, color="#2ECC71"))
            fig_cad.add_scatter(x=_cad_m["MesAno"], y=_cad_m["Cadencia"],
                                fill="tozeroy", fillcolor="rgba(52,152,219,0.06)",
                                line=dict(width=0), showlegend=False, hoverinfo="skip")
            fig_cad.add_scatter(
                x=_cad_m["MesAno"], y=_cad_m["Cadencia"],
                mode="lines+markers+text",
                line=dict(color="#3498DB", width=2.5, shape="spline"),
                marker=dict(size=10, color=_cad_m["Cor"].tolist(),
                            line=dict(width=2, color="white")),
                text=_cad_m["Cadencia"].apply(lambda x: f"{x:.0f}"),
                textposition="top center", textfont=dict(size=9),
                error_y=dict(type="data", array=_cad_m["DP"].tolist(),
                             visible=True, color="rgba(52,152,219,0.3)", thickness=1.5),
                hovertemplate="<b>%{x}</b><br>Cadência: <b>%{y:.0f} spm</b><extra></extra>",
                showlegend=False)
            for _yref, _cor, _txt in [(160,"#E74C3C","160 — mín"),(170,"#F1C40F","170 — atenção"),(185,"#2ECC71","185 — máx ideal")]:
                fig_cad.add_hline(y=_yref, line_dash="dot", line_color=_cor, line_width=1,
                                  opacity=0.6, annotation_text=f"  {_txt}",
                                  annotation_position="top left",
                                  annotation_font=dict(size=9, color=_cor))
            fig_cad.update_layout(
                title=dict(text="Cadência média por mês (spm)", font=dict(size=14), x=0),
                xaxis=dict(tickangle=-45, showgrid=False),
                yaxis=dict(title="spm", range=[140, 200], gridcolor="rgba(128,128,128,0.12)"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=45, b=10, l=0, r=0))
            st.plotly_chart(fig_cad, width="stretch")
            if _cad_atual < 170:
                st.warning("⚠️ Cadência baixa — experimente aumentar a frequência sem mudar a velocidade. Metrônomo a 175 bpm ajuda.")
            elif 175 <= _cad_atual <= 185:
                st.success("✅ Cadência na faixa ideal. Boa economia de movimento.")

    # ── Eficiência Aeróbica Z2 ────────────────────────────────────────────────
    if not lps_run.empty and "Zona FC" in lps_run.columns:
        lps_z2 = lps_run[
            (lps_run["Zona FC"] == "Z2 - Aeróbico") &
            lps_run["pace_sec_km"].notna() &
            (lps_run["pace_sec_km"] > 0) &
            (lps_run["pace_sec_km"] < 480)].copy()
        if not lps_z2.empty and len(lps_z2["MesAno"].unique()) >= 2:
            st.markdown("---")
            st.subheader("🏃 Eficiência Aeróbica — Pace na Zona 2")
            st.caption(
                "**O KPI mais honesto de resistência:** o ritmo que você sustenta "
                "na mesma frequência cardíaca de Z2. Ficou mais rápido na mesma FC = motor maior.")
            ae_m = (lps_z2.groupby(["MesAnoOrd","MesAno"])
                          .agg(PaceZ2=("pace_sec_km","mean"), Amostras=("pace_sec_km","count"))
                          .reset_index().sort_values("MesAnoOrd"))
            ae_m = ae_m[ae_m["Amostras"] >= 5]
            if len(ae_m) >= 2:
                delta_ae = ae_m["PaceZ2"].iloc[0] - ae_m["PaceZ2"].iloc[-1]
                ae_m["PaceZ2_fmt"] = fmt_pace_vec(ae_m["PaceZ2"])
                ae_m["PaceZ2_min"] = ae_m["PaceZ2"] / 60
                ca1, ca2, ca3 = st.columns(3)
                ca1.metric("Pace Z2 — início",  ae_m["PaceZ2_fmt"].iloc[0]  + "/km")
                ca2.metric("Pace Z2 — atual",   ae_m["PaceZ2_fmt"].iloc[-1] + "/km")
                ca3.metric("Ganho aeróbico",    f"{abs(delta_ae):.0f}s/km",
                           f"+{delta_ae:.0f}s/km mais rápido na mesma FC" if delta_ae > 0
                           else "Sem melhora ainda",
                           delta_color="normal" if delta_ae > 0 else "inverse")
                fig_ae = go.Figure()
                fig_ae.add_scatter(
                    x=ae_m["MesAno"], y=ae_m["PaceZ2_min"],
                    mode="lines+markers+text", text=ae_m["PaceZ2_fmt"],
                    textposition="top center",
                    line=dict(color=PURPLE, width=2.5), marker=dict(size=8, color=PURPLE),
                    fill="tozeroy", fillcolor="rgba(155,89,182,0.07)",
                    customdata=ae_m["Amostras"].values,
                    hovertemplate="%{x}<br>Pace Z2: %{text}/km<br>Laps: %{customdata}<extra></extra>")
                set_pace_yaxis(fig_ae, ae_m["PaceZ2"])
                fig_ae.update_layout(title="Pace médio em Z2 por mês  (↓ = motor aeróbico maior)",
                                     xaxis_tickangle=-45, showlegend=False)
                st.plotly_chart(fig_ae, width="stretch")
                if   delta_ae >= 15: st.success(f"🚀 +{delta_ae:.0f}s/km mais rápido na mesma FC de Z2!")
                elif delta_ae > 0:   st.info(f"📈 Melhora de +{delta_ae:.0f}s/km. Continue o volume fácil.")
                else:                st.warning("⚠️ Eficiência Z2 estável ou em queda. Verifique excesso de Z3.")

    # ── Condições externas (expander) ─────────────────────────────────────────
    with st.expander("🌤️ Impacto do Clima no Pace"):
        df_w = df_run[df_run["weather_temp"].notna()].copy() \
               if "weather_temp" in df_run.columns else pd.DataFrame()
        if df_w.empty:
            st.info("Dados de clima não disponíveis.")
        else:
            col_cl1, col_cl2 = st.columns(2)
            with col_cl1:
                df_w["Pace_min"] = df_w["pace_sec_km"] / 60
                fig = px.scatter(df_w, x="weather_temp", y="Pace_min",
                                 title="Pace vs Temperatura",
                                 trendline="lowess", trendline_color_override=RED,
                                 opacity=0.6,
                                 labels={"weather_temp":"°C","Pace_min":"Pace (min/km)"})
                fig.update_yaxes(autorange="reversed")
                fig.add_vrect(x0=15, x1=26, fillcolor=GREEN, opacity=0.07,
                              annotation_text="Confortável")
                st.plotly_chart(fig, width="stretch")
            with col_cl2:
                if "weather_rain" in df_w.columns:
                    pace_chuva = df_w[df_w["weather_rain"] > 0]["pace_sec_km"].mean()
                    pace_seco  = df_w[df_w["weather_rain"] == 0]["pace_sec_km"].mean()
                    r1, r2, r3 = st.columns(3)
                    r1.metric("☀️ Sem chuva",   fmt_pace(pace_seco))
                    r2.metric("🌧️ Com chuva",   fmt_pace(pace_chuva))
                    if not (pd.isna(pace_chuva) or pd.isna(pace_seco)):
                        delta = pace_chuva - pace_seco
                        r3.metric("Δ impacto",
                                  f"+{delta:.0f}s/km" if delta > 0 else f"{delta:.0f}s/km")
                bins   = [0,14,18,24,28,50]
                labels = ["≤14°C","15–18°C","18–24°C","24–28°C","≥28°C"]
                df_w["Faixa"] = pd.cut(df_w["weather_temp"], bins=bins, labels=labels)
                df_f = (df_w.groupby("Faixa", observed=True)
                            .agg(Pace=("pace_sec_km","mean")).reset_index())
                df_f["Pace_fmt"] = fmt_pace_vec(df_f["Pace"])
                df_f["Pace_min"] = df_f["Pace"] / 60
                fig = px.bar(df_f, x="Faixa", y="Pace_min",
                             title="Pace por Temperatura",
                             color="Faixa", text="Pace_fmt",
                             labels={"Faixa":"","Pace_min":"Pace (min/km)"})
                set_pace_yaxis(fig, df_f["Pace"].dropna())
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width="stretch")

    with st.expander("⛰️ Análise de Elevação"):
        df_e = df_run[df_run["elevation_gain"].notna() & (df_run["elevation_gain"] > 0)].copy()
        if df_e.empty:
            st.info("Nenhum dado de elevação disponível.")
        else:
            df_e["elev_km"] = df_e["elevation_gain"] / df_e["distance_km"]
            ec1,ec2,ec3,ec4 = st.columns(4)
            ec1.metric("⛰️ Elevação Total",    f"{df_e['elevation_gain'].sum():,.0f} m")
            ec2.metric("📈 Maior Subida",       f"{df_e['elevation_gain'].max():.0f} m")
            ec3.metric("📐 Gradiente Médio",    f"{df_e['elev_km'].mean():.1f} m/km")
            ec4.metric("🏔️ Runs >300m",        f"{len(df_e[df_e['elevation_gain']>=300])}")
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                df_pe = df_e[df_e["pace_sec_km"].notna()].copy()
                df_pe["Pace_min"] = df_pe["pace_sec_km"] / 60
                fig = px.scatter(df_pe, x="elevation_gain", y="Pace_min",
                                 title="Pace vs Elevação",
                                 trendline="lowess", trendline_color_override=RED,
                                 opacity=0.65,
                                 labels={"elevation_gain":"Elevação (m)","Pace_min":"Pace (min/km)"})
                set_pace_yaxis(fig, df_pe["pace_sec_km"])
                st.plotly_chart(fig, width="stretch")
            with col_e2:
                top10 = df_e.nlargest(10,"elevation_gain")[
                    ["start_date","name","distance_km","elevation_gain","elev_km","pace_sec_km"]].copy()
                top10["Data"]     = top10["start_date"].dt.strftime("%d/%m/%Y")
                top10["Pace"]     = fmt_pace_vec(top10["pace_sec_km"])
                top10["Elev/km"]  = top10["elev_km"].apply(lambda x: f"{x:.1f}")
                top10["Distância"]= top10["distance_km"].apply(lambda x: f"{x:.1f} km")
                top10["Elevação"] = top10["elevation_gain"].apply(lambda x: f"{x:.0f} m")
                st.markdown("**Top 10 atividades com maior elevação**")
                st.dataframe(top10[["Data","name","Distância","Elevação","Elev/km","Pace"]]
                               .rename(columns={"name":"Atividade"}),
                             hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  3 · CARGA & ZONAS  —  gestão de carga + fisiologia cardíaca
# ══════════════════════════════════════════════════════════════════════════════
with tab_carga:
    st.title("💓 Carga & Zonas")

    hoje_c = pd.Timestamp.now()

    # ── PMC — CTL / ATL / TSB histórico ──────────────────────────────────────
    st.subheader("📊 Performance Management Chart")
    st.caption(
        "**CTL** (42d) = fitness acumulado · "
        "**ATL** (7d) = fadiga recente · "
        "**TSB** = CTL − ATL · Janela de pico: TSB entre +5 e +20")

    if not pmc_raw.empty:
        pmc_filt = pmc_raw[
            (pd.to_datetime(pmc_raw["Data"]) >= s_dt) &
            (pd.to_datetime(pmc_raw["Data"]) <= e_dt)].copy()
        if not pmc_filt.empty:
            pmc_filt["Semana"] = pd.to_datetime(pmc_filt["Data"]).dt.to_period("W").apply(
                lambda x: x.start_time)
            pmc_w  = pmc_filt.groupby("Semana").last().reset_index()
            ctl_at = pmc_w["CTL"].iloc[-1] if len(pmc_w) else 0
            atl_at = pmc_w["ATL"].iloc[-1] if len(pmc_w) else 0
            tsb_at2= pmc_w["TSB"].iloc[-1] if len(pmc_w) else 0
            cp1,cp2,cp3 = st.columns(3)
            cp1.metric("💪 CTL — Fitness", f"{ctl_at:.0f}",
                       help="Chronic Training Load — média exponencial 42 dias do suffer score. "
                            "Representa sua base de condicionamento. Cresce devagar (semanas) e cai devagar.")
            cp2.metric("😓 ATL — Fadiga",  f"{atl_at:.0f}",
                       help="Acute Training Load — média exponencial 7 dias do suffer score. "
                            "Representa a fadiga recente. Sobe rápido com treinos pesados, cai rápido com descanso.")
            cp3.metric("✨ TSB — Forma",   f"{tsb_at2:+.0f}",
                       delta=("✅ Pico" if 5<=tsb_at2<=20 else "⚠️ Fatigado" if tsb_at2<-15 else "OK"),
                       delta_color="normal" if tsb_at2>=0 else "inverse",
                       help="Training Stress Balance = CTL − ATL (Fitness − Fadiga). "
                            "+5 a +20: janela de pico — ideal para provas e treinos de qualidade. "
                            "0 a +5: neutro, treinando normalmente. "
                            "−10 a 0: alguma fadiga acumulada. "
                            "Abaixo de −20: risco de overtraining.")
            fig_pmc = go.Figure()
            fig_pmc.add_scatter(x=pmc_w["Semana"], y=pmc_w["CTL"], name="CTL — fitness",
                                mode="lines", line=dict(color=BLUE, width=2.5),
                                fill="tozeroy", fillcolor="rgba(52,152,219,0.08)")
            fig_pmc.add_scatter(x=pmc_w["Semana"], y=pmc_w["ATL"], name="ATL — fadiga",
                                mode="lines", line=dict(color=RED, width=2, dash="dot"))
            fig_pmc.add_scatter(x=pmc_w["Semana"], y=pmc_w["TSB"], name="TSB — forma",
                                mode="lines+markers", line=dict(color=GREEN, width=2),
                                marker=dict(size=4), yaxis="y2")
            fig_pmc.add_hline(y=0, line_color=GRAY, line_dash="dash", line_width=1, yref="y2")
            fig_pmc.add_hrect(y0=5, y1=20, fillcolor=GREEN, opacity=0.06, line_width=0,
                              yref="y2", annotation_text="Janela de forma",
                              annotation_position="top right")
            fig_pmc.update_layout(
                title="CTL · ATL · TSB — semana a semana",
                yaxis=dict(title="CTL / ATL"),
                yaxis2=dict(title="TSB", overlaying="y", side="right",
                            showgrid=False, zeroline=False),
                legend=dict(orientation="h", y=-0.18), hovermode="x unified")
            st.plotly_chart(fig_pmc, width="stretch")
            if   tsb_at2 > 20:    st.info("😴 Descansado — boa janela para qualidade ou competição.")
            elif tsb_at2 >= 5:    st.success("✅ Forma ideal (+5 a +20).")
            elif tsb_at2 >= -10:  st.info("⚙️ Treinando normalmente. Monitore a fadiga.")
            elif tsb_at2 >= -20:  st.warning("⚠️ Fadiga acumulada. Considere um dia leve.")
            else:                  st.error("🛑 Risco de overtraining. Reduza a carga.")
    else:
        st.info("PMC indisponível (coluna `suffer_score` ausente ou vazia).")

    st.markdown("---")

    # ── ACWR histórico + Variação de carga ───────────────────────────────────
    st.subheader("⚡ ACWR — Acute:Chronic Workload Ratio")
    st.caption(
        "Compara esforço dos últimos 7 dias com a média das 4 semanas anteriores. "
        "**Zona segura: 0,8 – 1,3.** Acima de 1,5 = alto risco de lesão.")

    col_ac1, col_ac2 = st.columns(2)
    with col_ac1:
        if df_run["suffer_score"].notna().any():
            weekly = (df_run[df_run["suffer_score"].notna()]
                      .set_index("start_date")["suffer_score"].resample("W").sum())
            acwr_df = pd.DataFrame({"Carga": weekly}).reset_index()
            acwr_df.columns = ["Semana","Carga"]
            # ACWR correto: rolling(1) / rolling(4).mean() por semana
            acwr_df["ATL_w"]  = acwr_df["Carga"].rolling(1).sum()
            acwr_df["CTL_w"]  = acwr_df["Carga"].rolling(4).mean()
            acwr_df["ACWR"]   = (acwr_df["ATL_w"] / acwr_df["CTL_w"]).replace([float("inf")], None)
            acwr_df = acwr_df.dropna(subset=["ACWR"])
            fig = px.line(acwr_df, x="Semana", y="ACWR", markers=True,
                          title="ACWR Histórico",
                          color_discrete_sequence=[BLUE], labels={"ACWR":"Ratio","Semana":""})
            fig.add_hrect(y0=0.8, y1=1.3, fillcolor=GREEN, opacity=0.08,
                          annotation_text="Zona Segura")
            fig.add_hline(y=1.5, line_dash="dash", line_color=RED,
                          annotation_text="Risco Alto")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Coluna `suffer_score` não disponível.")
    with col_ac2:
        if df_run["suffer_score"].notna().any():
            df_carga = (df_run[df_run["suffer_score"].notna()]
                        .groupby(["Semana","SemanaStr"])
                        .agg(Carga=("suffer_score","sum")).reset_index()
                        .sort_values("Semana").tail(16))
            df_carga["Δ%"] = df_carga["Carga"].pct_change() * 100
            df_carga["cor"] = df_carga["Δ%"].apply(
                lambda x: RED if not pd.isna(x) and abs(x) > 10 else GREEN)
            fig = go.Figure(go.Bar(
                x=df_carga["SemanaStr"], y=df_carga["Δ%"],
                marker_color=df_carga["cor"],
                text=df_carga["Δ%"].apply(lambda x: f"{x:+.0f}%" if not pd.isna(x) else ""),
                textposition="outside"))
            fig.add_hline(y=10,  line_dash="dash", line_color=AMBER,
                          annotation_text="±10% limite seguro")
            fig.add_hline(y=-10, line_dash="dash", line_color=AMBER)
            fig.update_layout(title="Variação de Carga Semanal (%)",
                              xaxis_tickangle=-45, yaxis_title="%")
            st.plotly_chart(fig, width="stretch")

    # ── Volume semanal e MoM ──────────────────────────────────────────────────
    st.markdown("---")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        df_sem = (df_run.groupby(["Semana","SemanaStr"])
                        .agg(KM=("distance_km","sum")).reset_index()
                        .sort_values("Semana").tail(24))
        fig = px.bar(df_sem, x="SemanaStr", y="KM",
                     title="📅 Volume Semanal — últimas 24 semanas",
                     color_discrete_sequence=[PURPLE], text_auto=".0f",
                     labels={"SemanaStr":"","KM":"km"})
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with col_v2:
        df_mom = (df_run.groupby(["MesAnoOrd","MesAno"])
                        .agg(KM=("distance_km","sum")).reset_index()
                        .sort_values("MesAnoOrd"))
        df_mom["Δ%"] = df_mom["KM"].pct_change() * 100
        df_mom["cor"] = df_mom["Δ%"].apply(
            lambda x: RED   if not pd.isna(x) and x < -10
            else AMBER if not pd.isna(x) and x > 30
            else GREEN if not pd.isna(x) else GRAY)
        fig = go.Figure(go.Bar(
            x=df_mom["MesAno"], y=df_mom["Δ%"], marker_color=df_mom["cor"],
            text=df_mom["Δ%"].apply(lambda x: f"{x:+.0f}%" if not pd.isna(x) else ""),
            textposition="outside"))
        fig.add_hline(y=0,  line_color=GRAY, line_dash="dash")
        fig.add_hline(y=30, line_color=AMBER, line_dash="dot",
                      annotation_text="+30% — risco overload")
        fig.update_layout(title="📊 Crescimento MoM (%)", yaxis_title="%",
                          xaxis_tickangle=-45)
        st.plotly_chart(fig, width="stretch")
        st.caption("MoM = Month over Month. >30% aumenta risco de lesão.")


    # ── 📅 Comparativo Ano a Ano ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📅 Comparativo Ano a Ano")
    st.caption("Volume mensal: ano atual vs ano anterior.")
    _ano_atual = pd.Timestamp.now().year
    _ano_ant   = _ano_atual - 1
    _yoy_base  = _runs_raw.copy() if not _runs_raw.empty else pd.DataFrame()
    if not _yoy_base.empty:
        _yoy_base["_ano"] = _yoy_base["start_date"].dt.year
        _yoy_base["_mes"] = _yoy_base["start_date"].dt.month
        _yoy_at  = (_yoy_base[_yoy_base["_ano"]==_ano_atual]
                    .groupby("_mes").agg(KM=("distance_km","sum")).reset_index())
        _yoy_ant = (_yoy_base[_yoy_base["_ano"]==_ano_ant]
                    .groupby("_mes").agg(KM=("distance_km","sum")).reset_index())
        _MO = list(range(1,13))
        _ML = [MESES_PT.get(pd.Timestamp(2000,m,1).strftime("%b").capitalize(),
               pd.Timestamp(2000,m,1).strftime("%b")) for m in _MO]
        _yoy_at  = pd.DataFrame({"_mes":_MO}).merge(_yoy_at,  on="_mes", how="left").fillna(0)
        _yoy_ant = pd.DataFrame({"_mes":_MO}).merge(_yoy_ant, on="_mes", how="left").fillna(0)
        _km_at  = float(_yoy_at["KM"].sum()); _km_ant = float(_yoy_ant["KM"].sum())
        _pct_d  = (_km_at-_km_ant)/_km_ant*100 if _km_ant>0 else 0
        yc1,yc2,yc3 = st.columns(3)
        yc1.metric(f"📏 KM {_ano_atual}", f"{_km_at:,.0f} km",
                   f"{_km_at-_km_ant:+.0f} km vs {_ano_ant}")
        yc2.metric(f"📏 KM {_ano_ant}",   f"{_km_ant:,.0f} km")
        yc3.metric("📈 Variação anual",   f"{_pct_d:+.1f}%",
                   delta_color="normal" if _pct_d>=0 else "inverse")
        fig_yoy = go.Figure()
        fig_yoy.add_scatter(x=_ML, y=_yoy_ant["KM"], name=str(_ano_ant),
                            mode="lines+markers", fill="tozeroy",
                            fillcolor="rgba(149,165,166,0.10)",
                            line=dict(color="#95A5A6", width=2, dash="dot", shape="spline"),
                            marker=dict(size=7, color="#95A5A6", line=dict(width=1.5,color="white")),
                            hovertemplate=f"<b>{_ano_ant} — %{{x}}</b><br>%{{y:.0f}} km<extra></extra>")
        fig_yoy.add_scatter(x=_ML, y=_yoy_at["KM"], name=str(_ano_atual),
                            mode="lines+markers+text", fill="tozeroy",
                            fillcolor="rgba(52,152,219,0.12)",
                            line=dict(color="#3498DB", width=3, shape="spline"),
                            marker=dict(size=9, color="#3498DB", line=dict(width=2,color="white")),
                            text=[f"{v:.0f}" if v>0 else "" for v in _yoy_at["KM"]],
                            textposition="top center", textfont=dict(size=9, color="#3498DB"),
                            hovertemplate=f"<b>{_ano_atual} — %{{x}}</b><br>%{{y:.0f}} km<extra></extra>")
        # Triângulos verdes nos meses em que superou o ano anterior
        _sup = _yoy_at[(_yoy_at["KM"]>_yoy_ant["KM"].values) & (_yoy_ant["KM"].values>0)]
        if not _sup.empty:
            _sup_labels = [_ML[i-1] for i in _sup["_mes"]]
            fig_yoy.add_scatter(x=_sup_labels, y=_sup["KM"], mode="markers",
                                marker=dict(size=13, color="#2ECC71", symbol="triangle-up",
                                            line=dict(width=1.5,color="white")),
                                name="Acima do ano ant.",
                                hovertemplate="<b>%{x}</b><br>+%{y:.0f} km<extra></extra>")
        fig_yoy.update_layout(
            title=dict(text=f"Volume mensal: {_ano_atual} vs {_ano_ant}", font=dict(size=15), x=0),
            xaxis=dict(showgrid=False, tickfont=dict(size=11)),
            yaxis=dict(title="km", gridcolor="rgba(128,128,128,0.12)", rangemode="tozero"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.15, x=0),
            margin=dict(t=50, b=10, l=0, r=0), hovermode="x unified")
        st.plotly_chart(fig_yoy, width="stretch")
        with st.expander("Ver tabela mês a mês"):
            _dtbl = [{"Mês":_ML[i-1].capitalize(),
                      str(_ano_ant): f"{_yoy_ant.loc[_yoy_ant['_mes']==i,'KM'].values[0]:.0f} km",
                      str(_ano_atual): f"{_yoy_at.loc[_yoy_at['_mes']==i,'KM'].values[0]:.0f} km",
                      "Δ": f"{_yoy_at.loc[_yoy_at['_mes']==i,'KM'].values[0]-_yoy_ant.loc[_yoy_ant['_mes']==i,'KM'].values[0]:+.0f} km"}
                     for i in _MO if (_yoy_at.loc[_yoy_at['_mes']==i,'KM'].values[0]>0 or
                                       _yoy_ant.loc[_yoy_ant['_mes']==i,'KM'].values[0]>0)]
            st.dataframe(pd.DataFrame(_dtbl), hide_index=True, use_container_width=True)
    else:
        st.info("Dados insuficientes para comparativo anual.")

    # ── Zonas FC ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("❤️ Zonas de Frequência Cardíaca")

    if lps_run.empty:
        st.info("Dados de laps não disponíveis para análise de zonas.")
    else:
        col_z1, col_z2 = st.columns(2)
        with col_z1:
            lps_run["tempo_min"] = lps_run["moving_time_sec"] / 60
            df_zona = (lps_run.groupby("Zona FC")
                               .agg(Tempo=("tempo_min","sum")).reset_index())
            df_zona["Zona FC"] = pd.Categorical(df_zona["Zona FC"],
                                                 categories=ZONA_ORDER, ordered=True)
            df_zona = df_zona.sort_values("Zona FC")
            total_t = df_zona["Tempo"].sum()
            df_zona["Pct"] = df_zona["Tempo"] / total_t * 100

            # Cálculo de polarização
            z12 = df_zona[df_zona["Zona FC"].isin(["Z1 - Regenerativo","Z2 - Aeróbico"])]["Pct"].sum()
            z3  = df_zona[df_zona["Zona FC"] == "Z3 - Tempo"]["Pct"].sum()
            z45 = df_zona[df_zona["Zona FC"].isin(["Z4 - Limiar","Z5 - VO2max"])]["Pct"].sum()

            fig = px.bar(df_zona, x="Zona FC", y="Pct",
                         title="⏱️ % Tempo em Zona FC",
                         color="Zona FC", color_discrete_map=ZONA_COLORS,
                         text=df_zona["Pct"].apply(lambda x: f"{x:.1f}%"),
                         labels={"Pct":"% Tempo","Zona FC":""})
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")

            # Métricas de polarização abaixo do gráfico
            pm1, pm2, pm3 = st.columns(3)
            pm1.metric("🟢 Z1+Z2 (fácil)", f"{z12:.0f}%",
                       f"{z12-80:+.0f}pp vs ideal 80%", delta_color="normal")
            pm2.metric("🟡 Z3 (cinza)",    f"{z3:.0f}%",
                       f"{z3-5:+.0f}pp vs ideal 5%",   delta_color="inverse")
            pm3.metric("🔴 Z4+Z5 (intenso)",f"{z45:.0f}%",
                       f"{z45-15:+.0f}pp vs ideal 15%",delta_color="normal")

            if z3 > 15:
                st.warning(f"⚠️ {z3:.0f}% em Z3 (zona cinza). "
                           "Reduza para Z2 ou suba para Z4 — evite o meio-termo.")
            elif z12 >= 75:
                st.success(f"✅ {z12:.0f}% em Z1+Z2 — boa base aeróbica (modelo Seiler).")

        with col_z2:
            # Pie ideal vs atual lado a lado
            IDEAL_ZONA = {"Z1 - Regenerativo":35,"Z2 - Aeróbico":45,
                          "Z3 - Tempo":5,"Z4 - Limiar":10,"Z5 - VO2max":5}
            df_z_ideal = pd.DataFrame([{"Zona FC":z,"Pct":p}
                                       for z,p in IDEAL_ZONA.items()])
            df_z_ideal["Zona FC"] = pd.Categorical(df_z_ideal["Zona FC"],
                                                   categories=ZONA_ORDER, ordered=True)
            fig2 = px.pie(df_z_ideal, names="Zona FC", values="Pct",
                          title="Modelo polarizado ideal (Seiler)",
                          hole=0.5, color="Zona FC", color_discrete_map=ZONA_COLORS,
                          category_orders={"Zona FC": ZONA_ORDER})
            fig2.update_traces(textinfo="percent+label", textposition="inside")
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, width="stretch")
            st.caption(
                "📖 **Modelo Seiler:** ~80% fácil (Z1+Z2), ~5% cinza (Z3 — evitar), "
                "~15% intenso (Z4+Z5). Z3 é fisiol. cara sem desenvolver base ou velocidade.")

        # Deriva cardíaca e FC mensal
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            lps_s    = lps_run.sort_values(["activity_id","lap_index"])
            first_fc = lps_s.groupby("activity_id")["average_heartrate"].first()
            last_fc  = lps_s.groupby("activity_id")["average_heartrate"].last()
            deriv    = (last_fc - first_fc).reset_index()
            deriv.columns = ["activity_id","drift"]
            deriv = deriv.merge(
                df_run[["id","MesAno","MesAnoOrd"]].rename(columns={"id":"activity_id"}),
                on="activity_id", how="left")
            df_deriv_m = (deriv.groupby(["MesAnoOrd","MesAno"])
                               .agg(Deriva=("drift","mean")).reset_index()
                               .sort_values("MesAnoOrd"))
            fig = px.line(df_deriv_m, x="MesAno", y="Deriva", markers=True,
                          title="📈 Deriva Cardíaca Média (bpm)",
                          color_discrete_sequence=[RED],
                          labels={"Deriva":"Δ bpm","MesAno":""})
            fig.add_hline(y=0,  line_dash="dash", line_color=GRAY)
            fig.add_hline(y=10, line_dash="dot",  line_color=AMBER,
                          annotation_text="+10 bpm — atenção")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, width="stretch")
            st.caption("Deriva alta (>10 bpm) = sinal de fadiga ou desidratação.")
        with col_d2:
            df_fc_m = (df_run[df_run["average_heartrate"].notna()]
                       .groupby(["MesAnoOrd","MesAno"])
                       .agg(FC=("average_heartrate","mean")).reset_index()
                       .sort_values("MesAnoOrd"))
            if not df_fc_m.empty:
                tend = ("📈 Subindo" if df_fc_m["FC"].iloc[-1] > df_fc_m["FC"].iloc[0]
                        else "📉 Caindo" if df_fc_m["FC"].iloc[-1] < df_fc_m["FC"].iloc[0]
                        else "➡️ Estável")
                fig = px.line(df_fc_m, x="MesAno", y="FC", markers=True,
                              title=f"❤️ FC Média por Mês — {tend}",
                              color_discrete_sequence=[RED],
                              labels={"FC":"bpm","MesAno":""})
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, width="stretch")
                st.caption("FC em queda na mesma intensidade = adaptação cardiovascular positiva.")


# ══════════════════════════════════════════════════════════════════════════════
#  4 · MAPA  — rotas com análise por lap (bloco mantido)
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def _haversine_km(c1, c2):
    la1, lo1 = math.radians(c1[0]), math.radians(c1[1])
    la2, lo2 = math.radians(c2[0]), math.radians(c2[1])
    d = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return 2 * 6371 * math.asin(math.sqrt(d))


@st.cache_data(show_spinner=False, max_entries=80)
def _build_route_map_html(
    act_data:  tuple,   # ((id, name, date_str, km, pace_sec, hr, elev, color, poly_str, insights_str), ...)
    laps_data: tuple,   # ((act_id, lap_idx, dist_km, pace_sec, hr, max_hr, elev_gain, time_sec), ...)
    lat_c: float,
    lng_c: float,
    tile: str,
    height: int,
) -> str:
    """
    Constrói o HTML completo do mapa Folium e devolve como string cacheada.
    Pan / zoom / cliques ficam 100% no browser — sem rerun Python.
    """
    from collections import defaultdict

    ESRI_SAT  = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                 "World_Imagery/MapServer/tile/{z}/{y}/{x}", "Tiles © Esri")
    ESRI_TOPO = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                 "World_Topo_Map/MapServer/tile/{z}/{y}/{x}", "Tiles © Esri")
    TILES = {
        "Claro":       lambda m: folium.TileLayer("CartoDB positron",    name="Claro").add_to(m),
        "Escuro":      lambda m: folium.TileLayer("CartoDB dark_matter", name="Escuro").add_to(m),
        "Satélite":    lambda m: folium.TileLayer(ESRI_SAT[0],  attr=ESRI_SAT[1],  name="Sat").add_to(m),
        "Topográfico": lambda m: folium.TileLayer("OpenTopoMap",          name="Topo").add_to(m),
        "Topo ESRI":   lambda m: folium.TileLayer(ESRI_TOPO[0], attr=ESRI_TOPO[1], name="TopoE").add_to(m),
    }

    m = folium.Map(location=[lat_c, lng_c], zoom_start=13, tiles=None, control_scale=True)
    TILES.get(tile, TILES["Claro"])(m)

    try:
        from folium.plugins import MiniMap
        MiniMap(toggle_display=True, position="bottomleft",
                tile_layer="CartoDB positron", zoom_level_offset=-5).add_to(m)
    except Exception:
        pass

    # Indexa laps por atividade
    laps_by_act = defaultdict(list)
    for lap in laps_data:
        laps_by_act[lap[0]].append(lap)

    def _fmt(sec):
        if not sec or sec <= 0: return "—"
        s = int(sec); return f"{s//60}:{s%60:02d}"

    def _km_markers(coords, color, fg):
        """Marcadores numéricos a cada 1 km ao longo da rota."""
        cum, last_km = 0.0, 0
        for i in range(1, len(coords)):
            cum += _haversine_km(coords[i-1], coords[i])
            km_int = int(cum)
            if km_int > last_km:
                last_km = km_int
                folium.Marker(
                    coords[i],
                    icon=folium.DivIcon(
                        html=(f"<div style='font-size:9px;font-weight:700;color:#fff;"
                              f"background:{color};padding:1px 5px;border-radius:10px;"
                              f"box-shadow:0 1px 3px rgba(0,0,0,.55);white-space:nowrap'>"
                              f"{km_int} km</div>"),
                        icon_size=(44, 16), icon_anchor=(22, 8),
                    ),
                ).add_to(fg)

    def _lap_popup_html(lap, name, date, color):
        lap_idx, dist_km, pace_sec, hr, max_hr, elev_gain, time_sec = lap[1:]
        rows = "".join(
            f"<tr><td style='color:#888;padding:1px 8px 1px 0'>{k}</td>"
            f"<td><b>{v}</b></td></tr>"
            for k, v in [
                ("Lap",       str(int(lap_idx))),
                ("Distância", f"{dist_km:.2f} km"),
                ("Pace",      f"{_fmt(pace_sec)}/km"),
                ("Tempo",     _fmt(time_sec)),
                ("FC média",  f"{int(hr)} bpm"     if hr     > 0 else "—"),
                ("FC máx",    f"{int(max_hr)} bpm" if max_hr > 0 else "—"),
                ("Elevação",  f"{elev_gain:.0f} m" if elev_gain  else "—"),
            ]
        )
        return (f"<div style='font-family:sans-serif;min-width:175px;padding:2px'>"
                f"<b style='font-size:12px'>{name[:28]}</b>"
                f"<span style='color:#888;font-size:10px'> · {date}</span><br>"
                f"<div style='margin:4px 0 6px;padding:2px 7px;border-radius:4px;"
                f"background:{color}22;border-left:3px solid {color};"
                f"font-size:11px;font-weight:600'>Lap {int(lap_idx)} · {dist_km:.2f} km</div>"
                f"<table style='font-size:11px;width:100%'>{rows}</table></div>")

    def _act_popup_html(a):
        _, name, date, km, pace_sec, hr, elev, color, _, insights_str = a
        rows = "".join(
            f"<tr><td style='color:#888;padding:2px 10px 2px 0'>{k}</td>"
            f"<td><b>{v}</b></td></tr>"
            for k, v in [
                ("Distância", f"{km:.1f} km"),
                ("Pace",      f"{_fmt(pace_sec)}/km"),
                ("FC média",  f"{int(hr)} bpm" if hr   > 0 else "—"),
                ("Elevação",  f"{elev:.0f} m"  if elev > 0 else "—"),
            ]
        )
        insights_block = (f"<div style='margin-top:6px;font-size:11px;color:#555'>"
                          + insights_str.replace("|", "<br>") + "</div>") if insights_str else ""
        return (f"<div style='font-family:sans-serif;min-width:190px;padding:2px'>"
                f"<b style='font-size:13px'>{name[:32]}</b><br>"
                f"<span style='color:#888;font-size:11px'>{date}</span>"
                f"<table style='font-size:12px;width:100%;margin-top:6px'>{rows}</table>"
                f"{insights_block}</div>")

    for a in act_data:
        act_id, name, date, km, pace_sec, hr, elev, color, poly_str, _ = a
        coords = decode_polyline(poly_str) if poly_str else []

        fg = folium.FeatureGroup(name=f"{date} — {name[:22]} ({km:.1f} km)", show=True)

        if coords:
            n_pts = len(coords) - 1

            # Rota animada (AntPath) com popup resumo
            try:
                from folium.plugins import AntPath
                AntPath(coords, color=color, weight=4.5, dash_array=[12, 20],
                        delay=800, opacity=0.92,
                        popup=folium.Popup(_act_popup_html(a), max_width=250)).add_to(fg)
            except Exception:
                folium.PolyLine(coords, color=color, weight=4.5, opacity=0.9,
                                popup=folium.Popup(_act_popup_html(a), max_width=250)).add_to(fg)

            # Marcadores de km
            _km_markers(coords, color, fg)

            # Overlays invisíveis por lap — clique mostra detalhes do lap
            act_laps = sorted(laps_by_act.get(act_id, []), key=lambda x: x[1])
            if act_laps:
                total_dist = sum(l[2] for l in act_laps) or 1
                cum_frac   = 0.0
                for lap in act_laps:
                    frac = lap[2] / total_dist
                    i0   = int(cum_frac * n_pts)
                    i1   = min(n_pts, int((cum_frac + frac) * n_pts) + 1)
                    seg  = coords[i0:i1 + 1]
                    if len(seg) >= 2:
                        folium.PolyLine(
                            seg, color=color, weight=14, opacity=0.001,
                            popup=folium.Popup(_lap_popup_html(lap, name, date, color), max_width=230),
                            tooltip=f"Lap {int(lap[1])} · {_fmt(lap[3])}/km",
                        ).add_to(fg)
                    cum_frac += frac
            else:
                # sem laps: overlay único com popup geral
                folium.PolyLine(coords, color=color, weight=14, opacity=0.001,
                                popup=folium.Popup(_act_popup_html(a), max_width=250)).add_to(fg)

            # Ponto de início
            folium.CircleMarker(coords[0], radius=6, color=color,
                                fill=True, fill_opacity=1.0, weight=2,
                                tooltip="Início").add_to(fg)

        else:
            # Fallback: apenas ponto lat/lng
            lat_f = hr   # reaproveitado — ver serialização abaixo; usar coluna correta
            lng_f = elev
            folium.CircleMarker(
                location=[lat_f, lng_f], radius=8, color=color,
                fill=True, fill_opacity=0.9,
                popup=folium.Popup(_act_popup_html(a), max_width=250),
                tooltip=name,
            ).add_to(fg)

        fg.add_to(m)

    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    return m._repr_html_()


# ══════════════════════════════════════════════════════════════════════════════
#  10 · MAPA DE ROTAS
# ══════════════════════════════════════════════════════════════════════════════
with tab_mapa:

    st.title("🗺️ Mapa de Rotas")

    # Detecta coluna de polyline
    POLY_CANDIDATES = ["map_polyline","polyline","map.polyline",
                       "summary_polyline","map_summary_polyline"]
    poly_col = next(
        (c for c in POLY_CANDIDATES if c in df_raw.columns and df_raw[c].notna().any()),
        None)
    has_ll = "latitude" in df_run.columns and "longitude" in df_run.columns

    if not has_ll and poly_col is None:
        st.error("Nenhum dado de GPS encontrado.")
        st.stop()

    # Filtra atividades com GPS válido
    df_map = df_run.copy()
    if poly_col:
        df_map = df_map[df_map[poly_col].notna() & (df_map[poly_col].astype(str).str.len() > 4)]
    elif has_ll:
        df_map = df_map.dropna(subset=["latitude","longitude"])
    df_map = df_map.sort_values("start_date", ascending=False)

    if df_map.empty:
        st.warning("Nenhuma atividade com GPS encontrada no período.")
        st.stop()

    # Labels de seleção
    def make_label(row):
        dt  = row["start_date"].strftime("%d/%m/%Y")
        km  = float(row.get("distance_km") or 0)
        tag = f" [{row['Intensidade']}]" \
              if "Intensidade" in row and str(row["Intensidade"]) not in ("","None","nan") else ""
        return f"{dt} — {row['name'][:35]} ({km:.1f} km){tag}"

    label_map      = {make_label(r): r["id"] for _, r in df_map.iterrows()}
    labels         = list(label_map.keys())
    selected_labels = st.multiselect(
        "Selecione atividades",
        options=labels,
        default=labels[:min(5, len(labels))],
        placeholder="Buscar atividade...",
    )

    if not selected_labels:
        st.info("Selecione ao menos uma atividade.")
        st.stop()

    selected_ids = [label_map[l] for l in selected_labels]
    df_map       = df_map[df_map["id"].isin(selected_ids)].copy()
    st.caption(f"**{len(df_map)}** atividade(s) selecionada(s) · {len(labels)} disponíveis no período")

    # Centro do mapa (via polyline ou lat/lng)
    if poly_col and not df_map.empty:
        _s = decode_polyline(df_map[poly_col].dropna().iloc[0])
        lat_c = _s[0][0] if _s else -23.55
        lng_c = _s[0][1] if _s else -46.63
    elif has_ll and df_map["latitude"].notna().any():
        lat_c = float(df_map["latitude"].dropna().mean())
        lng_c = float(df_map["longitude"].dropna().mean())
    else:
        lat_c, lng_c = -23.55, -46.63

    # Serializa dados para o cache (apenas tipos imutáveis/hashable)
    def _route_color(row):
        return INTENSITY_COLORS.get(str(row.get("Intensidade") or "Moderado"), BLUE)

    act_data = tuple(
        (
            row["id"],
            str(row.get("name") or ""),
            row["start_date"].strftime("%d/%m/%Y"),
            float(row.get("distance_km") or 0),
            float(row.get("pace_sec_km") or 0),
            float(row.get("average_heartrate") or 0),
            float(row.get("elevation_gain") or 0),
            _route_color(row),
            str(row[poly_col]) if poly_col and pd.notna(row.get(poly_col)) else "",
            "|".join(analyze_run(lps_run[lps_run["activity_id"] == row["id"]])),
        )
        for _, row in df_map.iterrows()
    )

    _laps_sel = lps_run[lps_run["activity_id"].isin(selected_ids)] \
                if not lps_run.empty else pd.DataFrame()
    laps_data = tuple(
        (
            row["activity_id"],
            int(row.get("lap_index") or 0),
            float(row.get("distance_km") or 0),
            float(row.get("pace_sec_km") or 0),
            float(row.get("average_heartrate") or 0),
            float(row.get("max_heartrate") or 0),
            float(row.get("total_elevation_gain") or 0),
            float(row.get("moving_time_sec") or 0),
        )
        for _, row in _laps_sel.iterrows()
    ) if not _laps_sel.empty else ()

    # Controles + render do mapa via st.fragment
    # (só este bloco re-executa ao trocar tile/expandir — não o script todo)
    try:
        _frag = st.fragment
    except AttributeError:
        def _frag(f): return f

    @_frag
    def _map_controls():
        col1, col2 = st.columns([4, 1])
        with col1:
            tile = st.radio("Mapa base",
                            ["Claro","Escuro","Satélite","Topográfico","Topo ESRI"],
                            horizontal=True, key="mapa_tile",
                            help="Satélite/Topo ESRI: ótimos para trail.")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            expand = st.checkbox("🔍 Ampliar", value=False, key="mapa_expand")

        height = 720 if expand else 550

        # Constrói (ou recupera do cache) o HTML — instantâneo na 2ª chamada
        with st.spinner("Preparando mapa…"):
            html_map = _build_route_map_html(
                act_data=act_data, laps_data=laps_data,
                lat_c=lat_c, lng_c=lng_c, tile=tile, height=height)

        # Renderiza dentro de iframe: pan/zoom/clique não disparam rerun Python
        components.html(html_map, height=height, scrolling=False)
        st.caption(
            "🖱️ **Clique na rota** para detalhes do lap · "
            "🔢 Números = km percorridos · "
            "🗂️ Controle de camadas no canto superior direito do mapa · "
            "⚡ Mapa cacheado — trocar de aba não recarrega"
        )

    _map_controls()

    # Comparação automática
    if len(df_map) >= 2:
        st.markdown("---")
        st.markdown("### 🔍 Comparação")

        valid_pace = df_map[df_map["pace_sec_km"].notna() & (df_map["pace_sec_km"] > 0)]

        if not valid_pace.empty:
            melhor = valid_pace.loc[valid_pace["pace_sec_km"].idxmin()]
            pior   = valid_pace.loc[valid_pace["pace_sec_km"].idxmax()]
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"🏆 Melhor pace\n\n**{fmt_pace(melhor['pace_sec_km'])}/km**\n\n{melhor['name']}")
            with c2:
                st.warning(f"📉 Pace mais lento\n\n**{fmt_pace(pior['pace_sec_km'])}/km**\n\n{pior['name']}")

# ══════════════════════════════════════════════════════════════════════════════
#  11 · HISTÓRICO
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  5 · HISTÓRICO  — log de atividades + detalhes por lap
# ══════════════════════════════════════════════════════════════════════════════
with tab_hist:
    st.title("📋 Histórico")

    if df_run.empty:
        st.info("Nenhuma atividade no período selecionado.")
    else:
        busca = st.text_input("🔍 Buscar por nome",
                              placeholder="ex: regenerativo, prova, longo...")
        df_hv = df_run.copy()
        if busca:
            df_hv = df_hv[df_hv["name"].str.contains(busca, case=False, na=False)]
        st.caption(f"{len(df_hv)} atividades")

        # Timeline
        df_sc = df_hv[df_hv["pace_sec_km"].notna()].copy()
        df_sc["Pace_fmt"]  = fmt_pace_vec(df_sc["pace_sec_km"])
        df_sc["Pace_min"]  = df_sc["pace_sec_km"] / 60
        df_sc["Tempo_fmt"] = (df_sc["moving_time_sec"] / 60).apply(
            lambda x: f"{int(x//60)}h{int(x%60):02d}m"
            if not pd.isna(x) and x >= 60 else (f"{int(x)}min" if not pd.isna(x) else "—"))
        df_sc["Data_str"]  = df_sc["start_date"].dt.strftime("%d/%m/%Y")
        fig = px.scatter(
            df_sc, x="start_date", y="Pace_min",
            size="distance_km", size_max=28,
            color="Intensidade" if "Intensidade" in df_sc.columns else None,
            color_discrete_map=INTENSITY_COLORS,
            category_orders={"Intensidade": INTENSITY_ORDER},
            custom_data=["name","Data_str","distance_km","Pace_fmt","Tempo_fmt","elevation_gain"],
            title="📍 Todas as atividades — tamanho = distância · cor = intensidade",
            labels={"start_date":"","Pace_min":"Pace (min/km)"}, opacity=0.85)
        fig.update_traces(hovertemplate=(
            "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
            "📏 %{customdata[2]:.1f} km · ⚡ %{customdata[3]}/km<br>"
            "⏱️ %{customdata[4]} · ⛰️ %{customdata[5]:.0f} m<extra></extra>"))
        set_pace_yaxis(fig, df_sc["pace_sec_km"])
        st.plotly_chart(fig, width="stretch")
        st.caption("📖 Eixo Y invertido — pontos mais altos = mais rápido.")

        # Tabela completa
        st.markdown("---")
        cols = {"start_date":"Data","name":"Atividade","distance_km":"km",
                "moving_time_sec":"Tempo","pace_sec_km":"Pace",
                "average_heartrate":"FC Média","elevation_gain":"Elev (m)",
                "calories":"Calorias"}
        if "Intensidade" in df_hv.columns:
            cols["Intensidade"] = "Intensidade"
        df_tab = df_hv[[c for c in cols if c in df_hv.columns]].copy()
        df_tab = df_tab.sort_values("start_date", ascending=False)
        df_tab["start_date"]         = df_tab["start_date"].dt.strftime("%d/%m/%Y %H:%M")
        df_tab["pace_sec_km"]        = fmt_pace_vec(df_tab["pace_sec_km"])
        df_tab["moving_time_sec"]    = (df_tab["moving_time_sec"] / 60).apply(
            lambda x: f"{int(x//60)}h{int(x%60):02d}m"
            if not pd.isna(x) and x >= 60 else (f"{int(x)}min" if not pd.isna(x) else "—"))
        df_tab["distance_km"]        = df_tab["distance_km"].apply(
            lambda x: f"{x:.2f}" if not pd.isna(x) else "—")
        df_tab["average_heartrate"]  = df_tab["average_heartrate"].apply(
            lambda x: f"{int(x)} bpm" if not pd.isna(x) else "—")
        df_tab["elevation_gain"]     = df_tab["elevation_gain"].apply(
            lambda x: f"{x:.0f}" if not pd.isna(x) else "—")
        df_tab["calories"]           = df_tab["calories"].apply(
            lambda x: f"{int(x)}" if not pd.isna(x) else "—")
        df_tab = df_tab.rename(columns=cols)
        st.dataframe(df_tab, hide_index=True, width="stretch")

        # Detalhes por lap
        st.markdown("---")
        st.subheader("🔎 Detalhes por Lap")
        if lps_run.empty:
            st.info("Dados de laps não disponíveis.")
        else:
            ids_com_laps = lps_run["activity_id"].unique()
            df_sel = df_hv[df_hv["id"].isin(ids_com_laps)].copy()
            df_sel = df_sel.sort_values("start_date", ascending=False)
            df_sel["label"] = (df_sel["start_date"].dt.strftime("%d/%m/%Y") +
                               " — " + df_sel["name"].fillna("Sem nome") +
                               " (" + df_sel["distance_km"].apply(lambda x: f"{x:.1f}km") + ")")
            ativ_selecionada = st.selectbox(
                "Selecione uma corrida:",
                options=df_sel["id"].tolist(),
                format_func=lambda x: df_sel.set_index("id").loc[x, "label"])

            if ativ_selecionada:
                laps_ativ = (lps_run[lps_run["activity_id"] == ativ_selecionada]
                             .sort_values("lap_index").copy())
                act_info  = df_hv[df_hv["id"] == ativ_selecionada].iloc[0]
                c1,c2,c3,c4,c5 = st.columns(5)
                c1.metric("📏 Distância",  f"{act_info['distance_km']:.2f} km")
                c2.metric("⚡ Pace Médio", fmt_pace(act_info["pace_sec_km"]))
                c3.metric("❤️ FC Média",
                          f"{act_info['average_heartrate']:.0f} bpm"
                          if not pd.isna(act_info.get("average_heartrate", float("nan"))) else "—")
                c4.metric("⛰️ Elevação",
                          f"{act_info['elevation_gain']:.0f} m"
                          if not pd.isna(act_info.get("elevation_gain", float("nan"))) else "—")
                c5.metric("🔢 Laps", f"{len(laps_ativ)}")
                st.markdown(f"**{act_info['name']}** — "
                            f"{act_info['start_date'].strftime('%d/%m/%Y %H:%M')}")

                col_a, col_b = st.columns(2)
                with col_a:
                    laps_pace = laps_ativ[
                        laps_ativ["pace_sec_km"].notna() &
                        (laps_ativ["distance_m"] >= 200) &
                        (laps_ativ["pace_sec_km"] <= 480)].copy()
                    n_ig = len(laps_ativ) - len(laps_pace)
                    if laps_pace.empty:
                        st.info("Nenhum lap com distância ≥ 200m.")
                    else:
                        laps_pace["Pace_fmt"] = fmt_pace_vec(laps_pace["pace_sec_km"])
                        laps_pace["Pace_min"] = laps_pace["pace_sec_km"] / 60
                        laps_pace["Lap"]      = laps_pace["lap_index"].astype(str)
                        p50 = laps_pace["pace_sec_km"].median()
                        laps_pace["cor"] = laps_pace["pace_sec_km"].apply(
                            lambda x: GREEN if x < p50*0.97
                            else RED if x > p50*1.03 else BLUE)
                        fig = go.Figure(go.Bar(
                            x=laps_pace["Lap"], y=laps_pace["Pace_min"],
                            text=laps_pace["Pace_fmt"], textposition="outside",
                            marker_color=laps_pace["cor"].tolist(),
                            customdata=laps_pace[["distance_m","Pace_fmt"]].values,
                            hovertemplate="Lap %{x}<br>Pace: %{customdata[1]}/km<br>"
                                          "Distância: %{customdata[0]:.0f}m<extra></extra>"))
                        set_pace_yaxis(fig, laps_pace["pace_sec_km"])
                        titulo = "⚡ Pace por Lap"
                        if n_ig > 0:
                            titulo += f" ({n_ig} micro-laps ocultados)"
                        fig.update_layout(title=titulo, xaxis_title="Lap")
                        st.plotly_chart(fig, width="stretch")
                with col_b:
                    laps_fc = laps_ativ[laps_ativ["average_heartrate"].notna()].copy()
                    if not laps_fc.empty:
                        fig = go.Figure()
                        fig.add_scatter(x=laps_fc["lap_index"].astype(str),
                                        y=laps_fc["average_heartrate"],
                                        mode="lines+markers+text",
                                        text=laps_fc["average_heartrate"].apply(
                                            lambda x: f"{x:.0f}"),
                                        textposition="top center",
                                        line=dict(color=RED, width=2),
                                        marker=dict(size=8), name="FC Média")
                        if laps_fc["max_heartrate"].notna().any():
                            fig.add_scatter(x=laps_fc["lap_index"].astype(str),
                                            y=laps_fc["max_heartrate"], mode="lines",
                                            line=dict(color=RED, width=1, dash="dot"),
                                            name="FC Máx", opacity=0.5)
                        fig.update_layout(title="❤️ FC por Lap",
                                          xaxis_title="Lap", yaxis_title="bpm")
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info("FC não disponível para esta atividade.")

                cols_lap = {
                    "lap_index":"Lap","distance_m":"Distância (m)",
                    "moving_time_sec":"Tempo","pace_sec_km":"Pace",
                    "average_heartrate":"FC Média","max_heartrate":"FC Máx",
                    "total_elevation_gain":"Elev (m)","average_cadence":"Cadência"}
                df_laps_tab = laps_ativ[[c for c in cols_lap if c in laps_ativ.columns]].copy()
                df_laps_tab["pace_sec_km"]         = fmt_pace_vec(df_laps_tab["pace_sec_km"])
                df_laps_tab["moving_time_sec"]      = df_laps_tab["moving_time_sec"].apply(
                    lambda x: f"{int(x//60)}:{int(x%60):02d}" if not pd.isna(x) else "—")
                df_laps_tab["distance_m"]           = df_laps_tab["distance_m"].apply(
                    lambda x: f"{x:.0f}" if not pd.isna(x) else "—")
                df_laps_tab["average_heartrate"]    = df_laps_tab["average_heartrate"].apply(
                    lambda x: f"{x:.0f}" if not pd.isna(x) else "—")
                df_laps_tab["max_heartrate"]        = df_laps_tab["max_heartrate"].apply(
                    lambda x: f"{x:.0f}" if not pd.isna(x) else "—")
                df_laps_tab["total_elevation_gain"] = df_laps_tab["total_elevation_gain"].apply(
                    lambda x: f"{x:.1f}" if not pd.isna(x) else "—")
                df_laps_tab["average_cadence"]      = df_laps_tab["average_cadence"].apply(
                    lambda x: f"{x*2:.0f} spm" if not pd.isna(x) else "—")
                df_laps_tab = df_laps_tab.rename(columns=cols_lap)
                st.dataframe(df_laps_tab, hide_index=True, width="stretch")
