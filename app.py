"""
PerformanceRun — Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

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

# ── Responsividade mobile ───────────────────────────────────────────────────
st.markdown("""<style>
/* Mobile: reduz padding e permite scroll horizontal em gráficos */
@media (max-width: 768px) {
    .block-container { padding: .5rem .5rem 2rem !important; }
    [data-testid="column"] { min-width: 0 !important; }
    .stPlotlyChart > div { overflow-x: auto !important; }
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.2rem !important; }
}
/* Sidebar compacta em qualquer tela */
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 1rem;
}
</style>""", unsafe_allow_html=True)

INTENSITY_COLORS = {
    "Leve":           "#27AE60",  # verde
    "Moderado":       "#F1C40F",  # amarelo
    "Moderado Firme": "#E67E22",  # laranja
    "Forte":          "#E74C3C",  # vermelho
    "Muito Forte":    "#922B21",  # vermelho escuro
    "Skate":          "#95A5A6",  # cinza
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
DIAS_PT   = {"Monday":"Seg","Tuesday":"Ter","Wednesday":"Qua",
             "Thursday":"Qui","Friday":"Sex","Saturday":"Sáb","Sunday":"Dom"}
DIAS_ORDER_PT = ["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"]

def mesano_pt(dt_series):
    return dt_series.dt.strftime("%b %Y").apply(
        lambda x: f"{MESES_PT.get(x[:3], x[:3])} {x[4:]}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_dt(col):
    return pd.to_datetime(col, dayfirst=True, errors="coerce", utc=True).dt.tz_convert(None)

def fmt_pace(sec):
    if pd.isna(sec) or sec <= 0: return "—"
    s = int(sec)
    return f"{s // 60}:{s % 60:02d}"

def set_pace_yaxis(fig, pace_sec_series, step_sec=30):
    mn = max(0, int(pace_sec_series.min()) - step_sec)
    mx = int(pace_sec_series.max()) + step_sec
    vals  = list(range(mn - mn % step_sec, mx + step_sec, step_sec))
    fig.update_yaxes(
        autorange="reversed",
        tickvals=[v / 60 for v in vals],
        ticktext=[fmt_pace(v) for v in vals],
        title="Pace (min/km)",
    )
    return fig

def zona_fc(hr):
    if pd.isna(hr): return "Sem FC"
    if hr < 137:    return "Z1 - Regenerativo"
    if hr < 165:    return "Z2 - Aeróbico"
    if hr < 175:    return "Z3 - Tempo"
    if hr < 185:    return "Z4 - Limiar"
    return "Z5 - VO2max"

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

def _classify_by_name(name):
    if not name or pd.isna(name):
        return None
    n = str(name).lower()
    for intensity, keywords in KEYWORDS_INTENSIDADE.items():
        if any(kw in n for kw in keywords):
            return intensity
    return None

def calc_intensidade_fc(laps_df, act_names=None):
    if laps_df.empty or "average_heartrate" not in laps_df.columns:
        return pd.Series(dtype=str)

    hr_max = laps_df["average_heartrate"].dropna().max()
    hr_max = hr_max if (not pd.isna(hr_max) and hr_max > 150) else 195

    def zona_rel(hr):
        if pd.isna(hr): return "Sem FC"
        p = hr / hr_max
        if p < 0.65: return "Z1"
        if p < 0.78: return "Z2"
        if p < 0.85: return "Z3"
        if p < 0.92: return "Z4"
        return "Z5"

    laps_df = laps_df.copy()
    laps_df["Zona_Rel"] = laps_df["average_heartrate"].apply(zona_rel)
    ZONAS_FACEIS_REL = {"Z1", "Z2", "Sem FC"}
    results = {}

    for aid, grp in laps_df.groupby("activity_id"):
        if act_names is not None and aid in act_names.index:
            by_name = _classify_by_name(act_names.get(aid))
            if by_name:
                results[aid] = by_name
                continue

        grp = grp.sort_values("lap_index").copy()
        if len(grp) > 3:
            if grp.iloc[0]["Zona_Rel"] in ZONAS_FACEIS_REL:
                grp = grp.iloc[1:]
            if len(grp) > 1 and grp.iloc[-1]["Zona_Rel"] in ZONAS_FACEIS_REL:
                grp = grp.iloc[:-1]

        total = grp["moving_time_sec"].sum()
        if total == 0:
            continue

        sem_fc = grp[grp["Zona_Rel"] == "Sem FC"]["moving_time_sec"].sum()
        if sem_fc / total > 0.5:
            continue

        def pct(z):
            return grp[grp["Zona_Rel"] == z]["moving_time_sec"].sum() / total

        z1, z2, z3 = pct("Z1"), pct("Z2"), pct("Z3")
        z4, z5     = pct("Z4"), pct("Z5")
        pct_z45    = z4 + z5
        pct_z12    = z1 + z2

        if z5 >= 0.20 or pct_z45 >= 0.50:
            intensity = "Muito Forte"
        elif pct_z45 >= 0.30:
            intensity = "Forte"
        elif pct_z45 >= 0.15 or z3 >= 0.25:
            intensity = "Moderado Firme"
        elif pct_z12 >= 0.75:
            intensity = "Leve"
        else:
            intensity = "Moderado"

        results[aid] = intensity

    return pd.Series(results, name="Intensidade_FC")

def cat_intensity(df):
    if "Intensidade" not in df.columns: return df
    df = df.copy()
    df["Intensidade"] = pd.Categorical(df["Intensidade"],
                                        categories=INTENSITY_ORDER, ordered=True)
    return df

# ── NOVO: PMC — CTL / ATL / TSB ──────────────────────────────────────────────
def calc_pmc(df_run_all, ctl_days=42, atl_days=7):
    """
    Calcula CTL (fitness acumulado), ATL (fadiga recente) e TSB (forma)
    via exponential moving average sobre o suffer_score diário.

    CTL decay = 42 dias  |  ATL decay = 7 dias  |  TSB = CTL − ATL
    """
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

# ── Decodificador de Google Encoded Polyline (sem dependências externas) ────
def decode_polyline(encoded):
    """Decodifica Google Encoded Polyline → lista de (lat, lng)."""
    if not encoded or pd.isna(encoded) or str(encoded) in ("nan","None",""):
        return []
    encoded = str(encoded)
    coords, idx, lat, lng = [], 0, 0, 0
    while idx < len(encoded):
        for is_lng in (False, True):
            shift = result = 0
            while True:
                if idx >= len(encoded): break
                b = ord(encoded[idx]) - 63
                idx += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20: break
            delta = ~(result >> 1) if (result & 1) else (result >> 1)
            if not is_lng: lat += delta
            else:          lng += delta
        coords.append((lat / 1e5, lng / 1e5))
    return coords

def hex_to_rgba(hex_color, alpha=200):
    """Converte #RRGGBB → [R, G, B, A] para pydeck."""
    h = hex_color.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)] + [alpha]

def pace_to_rgba(pace_sec, min_pace=220, max_pace=420, alpha=220):
    """Gradiente verde (rápido) → vermelho (lento) para heatmap."""
    t = min(1, max(0, (pace_sec - min_pace) / (max_pace - min_pace)))
    return [round(46 + t*185), round(204 - t*128), round(113 - t*53), alpha]

def compute_main_laps_pace(laps_group):
    """Pace da fase principal (remove aquec/desaquec — laps >15% mais lentos que mediana)."""
    if laps_group.empty:
        return None
    laps = laps_group.sort_values("lap_index").copy()
    laps = laps[laps["pace_sec_km"].notna() & (laps["pace_sec_km"] > 0) & (laps["pace_sec_km"] < 500)]
    if len(laps) == 0:
        return None
    if len(laps) <= 2:
        return float(laps["pace_sec_km"].mean())
    mediana  = float(laps["pace_sec_km"].median())
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

def fc_to_hex(fc_bpm):
    """FC em bpm → cor por zona cardíaca (azul Z1 → vermelho Z5)."""
    if pd.isna(fc_bpm) or float(fc_bpm) <= 0: return "#3498DB"
    fc = float(fc_bpm)
    if fc < 137: return "#3498DB"
    if fc < 165: return "#2ECC71"
    if fc < 175: return "#F39C12"
    if fc < 185: return "#E67E22"
    return "#E74C3C"

def elev_gain_to_hex(elev_m_per_km):
    """Elevação m/km → cor: verde (plano) → vermelho (morro, >45m/km)."""
    if pd.isna(elev_m_per_km) or float(elev_m_per_km) < 0: return "#2ECC71"
    t = min(1.0, max(0.0, float(elev_m_per_km) / 45.0))
    return "#{:02X}{:02X}{:02X}".format(
        round(46 + t*185), round(204 - t*128), round(113 - t*53))

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
        for _c in ["latitude","longitude"]:
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
        laps["Zona FC"]    = laps["average_heartrate"].apply(zona_fc)

    be = read("activity_best_efforts_consolidated.csv")
    if not be.empty:
        be["start_date"] = normalize_dt(be["start_date"])
        be["MesAno"]     = mesano_pt(be["start_date"])
        be["MesAnoOrd"]  = be["start_date"].dt.to_period("M").apply(lambda x: x.ordinal)

    if not laps.empty and not act.empty:
        act_names = act.set_index("id")["name"] if "name" in act.columns else None
        intensidade_fc = calc_intensidade_fc(laps, act_names=act_names)
        if not intensidade_fc.empty:
            fc_df = intensidade_fc.reset_index()
            fc_df.columns = ["id", "Intensidade_FC"]
            act = act.merge(fc_df, on="id", how="left")

    return act, laps, be

df_raw, laps_raw, be_raw = load_all(BASE)

# PMC all-time (só corridas) — computado aqui para manter contexto histórico completo
_runs_raw = df_raw[df_raw["sport_type"].isin(["Run","TrailRun"])].copy() \
            if not df_raw.empty else df_raw
pmc_raw = calc_pmc(_runs_raw)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("🎛️ Filtros")
min_d = df_raw["start_date"].min().date()
max_d = df_raw["start_date"].max().date()

import datetime as _dt
_col1, _col2 = st.sidebar.columns([3, 1])
with _col1:
    st.markdown("**Período**")
with _col2:
    if st.button("↺", help="Resetar para o período completo", key="reset_date"):
        st.session_state["date_range"] = (min_d, max_d)

# Atalhos de período
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
_default_sports = [s for s in ["Run","TrailRun"] if s in sports_all]
if not _default_sports:
    _default_sports = sports_all
selected_sports = st.sidebar.multiselect("Modalidade", sports_all,
                                          default=_default_sports)
has_fc_class = ("Intensidade_FC" in df_raw.columns
                and df_raw["Intensidade_FC"].notna().any())
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

s_dt = pd.Timestamp(date_range[0]) if len(date_range) >= 1 else pd.Timestamp(min_d)
e_dt = (pd.Timestamp(date_range[1]) + pd.Timedelta(hours=23, minutes=59, seconds=59)) \
       if len(date_range) == 2 else (pd.Timestamp(max_d) + pd.Timedelta(hours=23, minutes=59, seconds=59))

def filt_act(d):
    mask = ((d["start_date"] >= s_dt) & (d["start_date"] <= e_dt)
            & d["sport_type"].isin(selected_sports))
    if sel_int and int_col in d.columns:
        # Inclui atividades sem intensidade calculada (NaN = sem FC ou sem laps)
        # Só exclui quando a intensidade É conhecida e NÃO está selecionada
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
          if not laps.empty else laps

for _df in [df, df_run]:
    if int_col in _df.columns:
        _df["Intensidade"] = _df[int_col]
    elif "Intensidade" not in _df.columns:
        _df["Intensidade"] = None

def melhor_be(nome):
    if be_raw.empty: return "—"
    s = be_raw[be_raw["name"].str.lower() == nome.lower()]
    return fmt_pace(s["pace_sec_km"].min()) if not s.empty else "—"

def melhor_3km():
    lps_all = laps_raw[laps_raw["activity_sport_type"].isin(["Run","TrailRun"])].copy() \
              if not laps_raw.empty else laps_raw
    if lps_all.empty: return "—"
    best = None
    for _, grp in lps_all.groupby("activity_id"):
        km = grp[(grp["distance_m"] >= 900) & (grp["distance_m"] <= 1100)]
        if len(km) >= 3:
            t = km.nsmallest(3,"lap_index")["moving_time_sec"].sum()
            if best is None or t < best: best = t
    return fmt_pace(best / 3) if best else "—"

# ══════════════════════════════════════════════════════════════════════════════
#  ABAS
# ══════════════════════════════════════════════════════════════════════════════
tab_geral, tab_perf, tab_fc, tab_intel, tab_elev, tab_clima, tab_metas, tab_vol, tab_coach, tab_mapa, tab_hist = st.tabs([
    "📊 Visão Geral","⚡ Performance e Pace","❤️ Frequência Cardíaca",
    "🧠 Inteligência de Treino","⛰️ Elevação","🌤️ Clima",
    "🎯 Metas e Benchmarks","📈 Volume e Evolução","🧑‍🏫 Visão Treinador",
    "🗺️ Mapa de Rotas","📋 Histórico",
])

# ══════════════════════════════════════════════════════════════════════════════
#  1 · VISÃO GERAL
# ══════════════════════════════════════════════════════════════════════════════
with tab_geral:
    st.title("📊 Visão Geral")

    c1,c2,c3 = st.columns(3)
    c1.metric("🏃 Atividades",  f"{len(df):,}")
    c2.metric("📏 Distância",   f"{df['distance_km'].sum():,.0f} km")
    c3.metric("⏱️ Tempo Total", f"{df['moving_time_sec'].sum()/3600:,.1f} h")

    c4,c5,c6 = st.columns(3)
    c4.metric("⚡ Pace Médio",  fmt_pace(df_run["pace_sec_km"].mean()))
    fc_med = df_run["average_heartrate"].mean()
    c5.metric("❤️ FC Média",   f"{fc_med:.0f} bpm" if not pd.isna(fc_med) else "—")
    cal = df["calories"].sum()
    c6.metric("🔥 Calorias",   f"{cal:,.0f} kcal" if df["calories"].notna().any() else "—")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        df_m = (df.groupby(["MesAnoOrd","MesAno"])
                  .agg(KM=("distance_km","sum")).reset_index()
                  .sort_values("MesAnoOrd"))
        fig = px.bar(df_m, x="MesAno", y="KM",
                     title="📏 Distância por Mês (km)",
                     color_discrete_sequence=[BLUE],
                     labels={"KM":"km","MesAno":""}, text_auto=".0f")
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with col_b:
        if "Intensidade" in df.columns and df["Intensidade"].notna().any():
            df_i = cat_intensity(df)["Intensidade"].value_counts().reset_index()
            df_i.columns = ["Intensidade","Qtd"]
            df_i = df_i[df_i["Intensidade"].isin(INTENSITY_ORDER)]
            fig = px.pie(df_i, names="Intensidade", values="Qtd",
                         title="🎯 Mix de Intensidade", hole=0.42,
                         color="Intensidade", color_discrete_map=INTENSITY_COLORS,
                         category_orders={"Intensidade": INTENSITY_ORDER})
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, width="stretch")

            label = "🤖 Automática (FC)" if int_mode == "Automática (FC)" else "✍️ Manual"
            with st.expander(f"Como cada categoria é definida? — {label}"):
                if int_mode == "Automática (FC)":
                    st.markdown(
                        "**1ª — Nome da atividade:** palavras-chave têm prioridade.\n\n"
                        "**2ª — Zonas de FC relativas ao FCmax** (fallback)."
                    )
                    criterios = {
                        "🟢 Leve":           "Z1+Z2 ≥ 75%  *(< 78% FCmax)*",
                        "🔵 Moderado":       "Demais casos",
                        "🟡 Moderado Firme": "Z4+Z5 ≥ 15% **ou** Z3 ≥ 25%",
                        "🟠 Forte":          "Z4+Z5 ≥ 30%",
                        "🔴 Muito Forte":    "Z5 ≥ 20% **ou** Z4+Z5 ≥ 50%",
                    }
                    for cat, criterio in criterios.items():
                        st.markdown(f"**{cat}** → {criterio}")
                else:
                    st.markdown("Modo **Manual** — coluna `Intensidade` preenchida manualmente.")

    df_p = (df_run[df_run["pace_sec_km"].notna()]
            .groupby(["MesAnoOrd","MesAno"])
            .agg(Pace=("pace_sec_km","mean")).reset_index()
            .sort_values("MesAnoOrd"))
    df_p["Pace_fmt"] = df_p["Pace"].apply(fmt_pace)
    df_p["Pace_min"] = df_p["Pace"] / 60
    df_p["Roll3M"]   = df_p["Pace_min"].rolling(3, min_periods=1).mean()

    fig = go.Figure()
    fig.add_bar(x=df_p["MesAno"], y=df_p["Pace_min"],
                name="Pace mensal", marker_color=BLUE, opacity=0.5,
                customdata=df_p["Pace_fmt"],
                hovertemplate="%{x}<br>Pace: %{customdata}/km<extra></extra>")
    fig.add_scatter(x=df_p["MesAno"], y=df_p["Roll3M"], name="Média 3M",
                    mode="lines+markers", line=dict(color=RED, width=2, dash="dash"))
    set_pace_yaxis(fig, df_p["Pace"])
    fig.update_layout(title="⚡ Evolução do Pace Médio + Média Móvel 3M",
                      xaxis_tickangle=-45)
    st.plotly_chart(fig, width="stretch")
    st.caption("📖 Eixo Y invertido: quanto mais alto no gráfico, mais rápido.")

    if len(df_p) >= 2:
        last_pace = df_p["Pace_min"].iloc[-1]
        avg_pace  = df_p["Pace_min"].mean()
        if last_pace < avg_pace:
            st.success(f"📈 Pace do último mês ({df_p['Pace_fmt'].iloc[-1]}/km) "
                       f"abaixo da média histórica ({fmt_pace(avg_pace*60)}/km) — evolução positiva!")
        else:
            st.info(f"ℹ️ Pace do último mês ({df_p['Pace_fmt'].iloc[-1]}/km) "
                    f"acima da média ({fmt_pace(avg_pace*60)}/km).")

    df_dia = (df["DiaSemana"].value_counts()
              .reindex(DIAS_ORDER_PT).fillna(0)
              .reset_index().rename(columns={"DiaSemana":"Dia","count":"Qtd"}))
    fig = px.bar(df_dia, x="Dia", y="Qtd",
                 title="📅 Atividades por Dia da Semana",
                 color_discrete_sequence=[PURPLE], text_auto=True)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
#  2 · PERFORMANCE E PACE
# ══════════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.title("⚡ Performance e Pace")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🥇 Melhor 3km",   melhor_3km())
    c2.metric("🥇 Melhor 5K",    melhor_be("5k"))
    c3.metric("🥇 Melhor 10K",   melhor_be("10k"))
    c4.metric("🥇 Melhor HM",    melhor_be("half-marathon"), help="HM = Half Marathon · 21,1 km")
    c5.metric("🏅 PRs Totais",   f"{int(df_run['pr_count'].sum()):,}"
                                  if df_run["pr_count"].notna().any() else "—")

    st.markdown("---")

    if not be.empty:
        be_sel = be[be["name"].str.lower().isin(["5k","10k"])].copy()
        if not be_sel.empty:
            be_best = (be_sel.groupby(["MesAnoOrd","MesAno","name"])
                             .agg(Pace=("pace_sec_km","min")).reset_index()
                             .sort_values("MesAnoOrd"))
            be_best["Pace_fmt"] = be_best["Pace"].apply(fmt_pace)
            be_best["Pace_min"] = be_best["Pace"] / 60
            fig = px.line(be_best, x="MesAno", y="Pace_min", color="name",
                          markers=True, custom_data=["Pace_fmt"],
                          title="📈 Evolução Melhor Pace — 5K e 10K",
                          color_discrete_map={"5k": RED, "10k": BLUE, "5K": RED, "10K": BLUE},
                          labels={"Pace_min":"Pace (min/km)","MesAno":"","name":"Distância"})
            fig.update_traces(
                hovertemplate="<b>%{x}</b><br>Melhor pace: %{customdata[0]}/km<extra></extra>")
            set_pace_yaxis(fig, be_best["Pace"])
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, width="stretch")

    col_a, col_b = st.columns(2)

    with col_a:
        if "Intensidade" in df_run.columns:
            # Usa pace do bloco principal (exclui aquec/desaquec por atividade)
            if not lps_run.empty:
                _mp = (lps_run.groupby("activity_id")
                       .apply(compute_main_laps_pace).dropna().reset_index())
                _mp.columns = ["id", "pace_main"]
                df_run_p = df_run.merge(_mp, on="id", how="left")
                df_run_p["pace_plot"] = df_run_p["pace_main"].fillna(df_run_p["pace_sec_km"])
                _src = "bloco principal (aquec/desaquec excluídos)"
            else:
                df_run_p = df_run.copy()
                df_run_p["pace_plot"] = df_run_p["pace_sec_km"]
                _src = "corrida completa"
            df_b = cat_intensity(df_run_p[df_run_p["pace_plot"].notna()].copy())
            df_agg = (df_b.groupby("Intensidade", observed=True)["pace_plot"]
                         .agg(Media="mean", DP="std").reset_index().dropna())
            df_agg["Media_min"] = df_agg["Media"] / 60
            df_agg["DP_min"]    = df_agg["DP"] / 60
            df_agg["Label"]     = df_agg["Media"].apply(fmt_pace)
            df_agg["Cor"]       = df_agg["Intensidade"].map(INTENSITY_COLORS)
            fig = go.Figure()
            for _, row in df_agg.iterrows():
                fig.add_bar(
                    x=[row["Intensidade"]], y=[row["Media_min"]],
                    error_y=dict(type="data", array=[row["DP_min"]], visible=True),
                    marker_color=row["Cor"], name=row["Intensidade"],
                    text=row["Label"], textposition="outside",
                )
            set_pace_yaxis(fig, df_agg["Media"])
            fig.update_layout(title="🎯 Pace Médio por Intensidade", showlegend=False)
            st.plotly_chart(fig, width="stretch")
            st.caption(
                f"Pace do **{_src}**. Laps de aquecimento/desaquecimento "
                "(>15% mais lentos que a mediana da sessão) são excluídos.")

    with col_b:
        df_s = df_run[df_run["pace_sec_km"].notna() & df_run["distance_km"].notna()].copy()
        df_s["Pace_min"] = df_s["pace_sec_km"] / 60
        fig = px.scatter(df_s, x="distance_km", y="Pace_min",
                         title="📍 Pace vs Distância (LOWESS)",
                         color="Intensidade" if "Intensidade" in df_s.columns else None,
                         color_discrete_map=INTENSITY_COLORS,
                         category_orders={"Intensidade": INTENSITY_ORDER},
                         trendline="lowess", trendline_color_override=RED,
                         opacity=0.65,
                         labels={"distance_km":"Distância (km)","Pace_min":"Pace (min/km)"})
        set_pace_yaxis(fig, df_s["pace_sec_km"])
        st.plotly_chart(fig, width="stretch")

    df_ef = (df_run[df_run["efficiency_index"].notna()]
             .groupby(["MesAnoOrd","MesAno"])
             .agg(Ef=("efficiency_index","mean")).reset_index()
             .sort_values("MesAnoOrd"))
    if not df_ef.empty:
        fig = px.line(df_ef, x="MesAno", y="Ef", markers=True,
                      title="🧠 Índice de Eficiência Mensal (pace ÷ FC — maior = melhor)",
                      color_discrete_sequence=[GREEN],
                      labels={"Ef":"Eficiência","MesAno":""})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
#  3 · FREQUÊNCIA CARDÍACA
# ══════════════════════════════════════════════════════════════════════════════
with tab_fc:
    st.title("❤️ Frequência Cardíaca")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("❤️ FC Média",    f"{df_run['average_heartrate'].mean():.0f} bpm"
                                 if df_run["average_heartrate"].notna().any() else "—")
    c2.metric("🔴 FC Máx Reg.", f"{df_run['max_heartrate'].max():.0f} bpm"
                                 if df_run["max_heartrate"].notna().any() else "—")

    if "weather_temp" in df_run.columns:
        fc_calor = df_run[df_run["weather_temp"] >= 28]["average_heartrate"].mean()
        fc_frio  = df_run[df_run["weather_temp"] <= 14]["average_heartrate"].mean()
        c3.metric("🌡️ FC no Calor (≥28°C)", f"{fc_calor:.0f} bpm" if not pd.isna(fc_calor) else "—")
        c4.metric("❄️ FC no Frio (≤14°C)",  f"{fc_frio:.0f} bpm"  if not pd.isna(fc_frio)  else "—")

    st.markdown("---")

    if lps_run.empty:
        st.info("Dados de laps não disponíveis para este período.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            lps_run["tempo_min"] = lps_run["moving_time_sec"] / 60
            df_zona = (lps_run.groupby("Zona FC")
                               .agg(Tempo=("tempo_min","sum")).reset_index())
            df_zona["Zona FC"] = pd.Categorical(df_zona["Zona FC"],
                                                 categories=ZONA_ORDER, ordered=True)
            df_zona = df_zona.sort_values("Zona FC")
            total_t = df_zona["Tempo"].sum()
            df_zona["Pct"] = df_zona["Tempo"] / total_t * 100
            fig = px.bar(df_zona, x="Zona FC", y="Pct",
                         title="⏱️ % Tempo em Zona FC",
                         color="Zona FC", color_discrete_map=ZONA_COLORS,
                         text=df_zona["Pct"].apply(lambda x: f"{x:.1f}%"),
                         labels={"Pct":"% Tempo","Zona FC":""})
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")
            st.caption("📖 Z1 = regenerativo → Z5 = esforço máximo. "
                       "Ideal para base: >70% do tempo em Z1+Z2.")

        with col_b:
            lps_s = lps_run.sort_values(["activity_id","lap_index"])
            first_fc = lps_s.groupby("activity_id")["average_heartrate"].first()
            last_fc  = lps_s.groupby("activity_id")["average_heartrate"].last()
            deriv = (last_fc - first_fc).reset_index()
            deriv.columns = ["activity_id","drift"]
            deriv = deriv.merge(
                df_run[["id","MesAno","MesAnoOrd"]].rename(columns={"id":"activity_id"}),
                on="activity_id", how="left")
            df_deriv_m = (deriv.groupby(["MesAnoOrd","MesAno"])
                               .agg(Deriva=("drift","mean")).reset_index()
                               .sort_values("MesAnoOrd"))
            fig = px.line(df_deriv_m, x="MesAno", y="Deriva", markers=True,
                          title="📈 Deriva Cardíaca Média (bpm — menor = melhor)",
                          color_discrete_sequence=[RED],
                          labels={"Deriva":"Δ bpm","MesAno":""})
            fig.add_hline(y=0,  line_dash="dash", line_color=GRAY)
            fig.add_hline(y=10, line_dash="dot",  line_color=AMBER,
                          annotation_text="+10 bpm — atenção")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, width="stretch")

            if not df_deriv_m.empty:
                last_drift = df_deriv_m["Deriva"].iloc[-1]
                if last_drift > 10:
                    st.warning("⚠️ Deriva cardíaca alta — sinal de fadiga ou desidratação.")
                elif last_drift < 5:
                    st.success("✅ Deriva cardíaca baixa — boa estabilidade cardiovascular.")

        df_fc_m = (df_run[df_run["average_heartrate"].notna()]
                   .groupby(["MesAnoOrd","MesAno"])
                   .agg(FC=("average_heartrate","mean")).reset_index()
                   .sort_values("MesAnoOrd"))
        fig = px.line(df_fc_m, x="MesAno", y="FC", markers=True,
                      title="❤️ FC Média por Mês",
                      color_discrete_sequence=[RED],
                      labels={"FC":"bpm","MesAno":""})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, width="stretch")

        # ── NOVO: Distribuição de Zonas vs Modelo Polarizado ─────────────────
        st.markdown("---")
        st.subheader("🎯 Distribuição de Zonas vs Modelo Polarizado")

        lz = lps_run.copy()
        lz["tempo_min"] = lz["moving_time_sec"] / 60
        df_zona_atual = (lz.groupby("Zona FC").agg(Tempo=("tempo_min","sum")).reset_index())
        df_zona_atual["Zona FC"] = pd.Categorical(df_zona_atual["Zona FC"],
                                                   categories=ZONA_ORDER, ordered=True)
        df_zona_atual = df_zona_atual.sort_values("Zona FC")
        tot_z = df_zona_atual["Tempo"].sum()
        df_zona_atual["Pct"] = (df_zona_atual["Tempo"] / tot_z * 100).round(1)

        IDEAL_ZONA = {
            "Sem FC":            0.0,
            "Z1 - Regenerativo": 35.0,
            "Z2 - Aeróbico":     45.0,
            "Z3 - Tempo":        5.0,
            "Z4 - Limiar":       10.0,
            "Z5 - VO2max":       5.0,
        }
        df_zona_ideal = pd.DataFrame([
            {"Zona FC": z, "Pct": p} for z, p in IDEAL_ZONA.items() if p > 0
        ])
        df_zona_ideal["Zona FC"] = pd.Categorical(df_zona_ideal["Zona FC"],
                                                   categories=ZONA_ORDER, ordered=True)
        df_zona_ideal = df_zona_ideal.sort_values("Zona FC")

        z12_at = df_zona_atual[df_zona_atual["Zona FC"].isin(
            ["Z1 - Regenerativo","Z2 - Aeróbico"])]["Pct"].sum()
        z3_at  = df_zona_atual[df_zona_atual["Zona FC"] == "Z3 - Tempo"]["Pct"].sum()
        z45_at = df_zona_atual[df_zona_atual["Zona FC"].isin(
            ["Z4 - Limiar","Z5 - VO2max"])]["Pct"].sum()
        pol_idx = round((z12_at + z45_at - z3_at * 2) / 100, 2)

        cm1, cm2, cm3, cm4 = st.columns(4)
        cm1.metric("🟢 Z1+Z2 (fácil)", f"{z12_at:.0f}%",
                   f"{z12_at - 80:+.0f}pp vs ideal 80%",
                   delta_color="normal")
        cm2.metric("🟡 Z3 zona cinza",  f"{z3_at:.0f}%",
                   f"{z3_at - 5:+.0f}pp vs ideal 5%",
                   delta_color="inverse")
        cm3.metric("🔴 Z4+Z5 (intenso)", f"{z45_at:.0f}%",
                   f"{z45_at - 15:+.0f}pp vs ideal 15%",
                   delta_color="normal")
        cm4.metric("📊 Índice de Polarização", f"{pol_idx:.2f}",
                   help="(Z1+Z2 + Z4+Z5 − 2×Z3) / 100. Próximo de 1 = bem polarizado.")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig_at = px.pie(
                df_zona_atual[df_zona_atual["Pct"] > 0],
                names="Zona FC", values="Pct",
                title="Distribuição atual",
                hole=0.5,
                color="Zona FC", color_discrete_map=ZONA_COLORS,
                category_orders={"Zona FC": ZONA_ORDER})
            fig_at.update_traces(textinfo="percent+label", textposition="inside")
            fig_at.update_layout(showlegend=False)
            st.plotly_chart(fig_at, width="stretch")

        with col_p2:
            fig_id = px.pie(
                df_zona_ideal[df_zona_ideal["Pct"] > 0],
                names="Zona FC", values="Pct",
                title="Modelo polarizado ideal",
                hole=0.5,
                color="Zona FC", color_discrete_map=ZONA_COLORS,
                category_orders={"Zona FC": ZONA_ORDER})
            fig_id.update_traces(textinfo="percent+label", textposition="inside")
            fig_id.update_layout(showlegend=False)
            st.plotly_chart(fig_id, width="stretch")

        st.caption(
            "📖 **Modelo polarizado (Seiler):** ~80% em Z1+Z2 (conversa fácil), "
            "~5% em Z3 (zona cinza — evitar), ~15% em Z4+Z5 (alta intensidade). "
            "A zona cinza é cara fisiologicamente mas não desenvolve base aeróbica nem velocidade máxima."
        )

        if z3_at > 20:
            st.warning(f"⚠️ {z3_at:.0f}% em Z3 (zona cinza) — quatro vezes o ideal. "
                       "Reduza o esforço nos treinos médios para Z2 ou suba para Z4 nos tiros.")
        elif z3_at > 10:
            st.warning(f"⚠️ {z3_at:.0f}% em Z3. Tente redistribuir para Z1+Z2.")
        elif z12_at < 65:
            st.warning(f"⚠️ Apenas {z12_at:.0f}% em Z1+Z2. Adicione mais volume aeróbico fácil.")
        elif z12_at >= 75:
            st.success(f"✅ {z12_at:.0f}% em Z1+Z2 — boa base aeróbica.")


# ══════════════════════════════════════════════════════════════════════════════
#  4 · INTELIGÊNCIA DE TREINO
# ══════════════════════════════════════════════════════════════════════════════
with tab_intel:
    st.title("🧠 Inteligência de Treino")

    if lps_run.empty:
        st.info("Dados de laps não disponíveis.")
    else:
        def cv_pace(grp):
            p = grp["pace_sec_km"].dropna()
            return (p.std() / p.mean() * 100) if len(p) > 1 and p.mean() > 0 else None

        cv_df = lps_run.groupby("activity_id").apply(cv_pace).dropna().reset_index()
        cv_df.columns = ["activity_id","CV_Pace"]
        cv_medio = cv_df["CV_Pace"].mean()

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("🎯 Consistência (CV%)", f"{cv_medio:.1f}%"
                  if not pd.isna(cv_medio) else "—",
                  help="< 10% = ritmo controlado | > 25% = intervalado")
        c2.metric("📊 Laps / Atividade",
                  f"{lps_run.groupby('activity_id').size().mean():.1f}")
        elev = lps_run["total_elevation_gain"].mean()
        c3.metric("⛰️ Elev. Média / Lap",
                  f"{elev:.1f} m" if not pd.isna(elev) else "—")
        c4.metric("💓 FC Média nos Laps",
                  f"{lps_run['average_heartrate'].mean():.0f} bpm"
                  if lps_run["average_heartrate"].notna().any() else "—")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            recent_ids = (lps_run.sort_values("start_date", ascending=False)
                          ["activity_id"].unique()[:10])
            df_box = lps_run[lps_run["activity_id"].isin(recent_ids)].copy()
            # Filtra paces de caminhada/outliers antes de agregar
            df_box = df_box[df_box["pace_sec_km"].notna() &
                            (df_box["pace_sec_km"] > 0) &
                            (df_box["pace_sec_km"] <= 500)]
            # Agrupa por atividade (não por data — evita sobreposição)
            act_info = lps_run.drop_duplicates("activity_id")[["activity_id","start_date"]].copy()
            act_info["Label"] = act_info["start_date"].dt.strftime("%d/%m")
            df_box = df_box.merge(act_info[["activity_id","Label"]], on="activity_id", how="left")
            df_agg2 = (df_box.groupby(["activity_id","Label"])["pace_sec_km"]
                             .agg(Media="mean", DP="std").reset_index().dropna()
                             .sort_values("activity_id").tail(10))
            df_agg2["Media_min"] = df_agg2["Media"] / 60
            df_agg2["DP_min"]    = df_agg2["DP"] / 60
            df_agg2["PaceLabel"] = df_agg2["Media"].apply(fmt_pace)
            fig = go.Figure(go.Bar(
                x=df_agg2["Label"], y=df_agg2["Media_min"],
                error_y=dict(type="data", array=df_agg2["DP_min"].tolist(), visible=True),
                marker_color=BLUE, text=df_agg2["PaceLabel"], textposition="outside",
            ))
            set_pace_yaxis(fig, df_agg2["Media"])
            fig.update_layout(title="📅 Pace Médio — 10 Treinos Mais Recentes",
                              xaxis=dict(tickangle=-30))
            st.plotly_chart(fig, width="stretch")
            st.caption("Uma barra por treino. Traço = variação interna (desvio padrão).")

        with col_b:
            df_hist = lps_run[lps_run["pace_sec_km"].notna() &
                              (lps_run["pace_sec_km"] > 0) &
                              (lps_run["pace_sec_km"] <= 480)].copy()  # exclui caminhadas >8:00/km
            df_hist["Pace_min"] = df_hist["pace_sec_km"] / 60
            fig = px.histogram(df_hist, x="Pace_min", nbins=35,
                               title="📊 Distribuição de Pace — Todos os Laps (até 8:00/km)",
                               color_discrete_sequence=[PURPLE],
                               labels={"Pace_min":"Pace (min/km)"})
            fig.update_yaxes(title="Nº de Laps")
            st.plotly_chart(fig, width="stretch")
            st.caption("Laps > 8:00/km (caminhadas e recuperações muito lentas) excluídos.")

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


# ══════════════════════════════════════════════════════════════════════════════
#  5 · ELEVAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
with tab_elev:
    st.title("⛰️ Elevação")

    df_e = df_run[df_run["elevation_gain"].notna() & (df_run["elevation_gain"] > 0)].copy()

    if df_e.empty:
        st.info("Nenhum dado de elevação disponível para o período selecionado.")
    else:
        df_e["elev_km"] = df_e["elevation_gain"] / df_e["distance_km"]

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("⛰️ Elevação Total",       f"{df_e['elevation_gain'].sum():,.0f} m")
        c2.metric("📈 Maior Subida",          f"{df_e['elevation_gain'].max():.0f} m")
        c3.metric("📏 Elevação Média/Run",    f"{df_e['elevation_gain'].mean():.0f} m")
        c4.metric("📐 Gradiente Médio",       f"{df_e['elev_km'].mean():.1f} m/km", help="Metros de subida por km percorrido. 10 m/km ≈ 1% de inclinação média.")
        runs_montanha = len(df_e[df_e["elevation_gain"] >= 300])
        c5.metric("🏔️ Runs c/ >300m",        f"{runs_montanha}")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            df_em = (df_e.groupby(["MesAnoOrd","MesAno"])
                         .agg(Elev=("elevation_gain","sum")).reset_index()
                         .sort_values("MesAnoOrd"))
            fig = px.bar(df_em, x="MesAno", y="Elev",
                         title="📅 Elevação Total por Mês (m)",
                         color_discrete_sequence=[AMBER],
                         labels={"Elev":"metros","MesAno":""}, text_auto=".0f")
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig, width="stretch")

        with col_b:
            fig = px.histogram(df_e, x="elevation_gain", nbins=30,
                               title="📊 Distribuição de Elevação por Atividade",
                               color_discrete_sequence=[AMBER],
                               labels={"elevation_gain":"Elevação (m)"})
            fig.update_yaxes(title="Nº de Atividades")
            st.plotly_chart(fig, width="stretch")

        col_a, col_b = st.columns(2)

        with col_a:
            df_pe = df_e[df_e["pace_sec_km"].notna()].copy()
            df_pe["Pace_min"] = df_pe["pace_sec_km"] / 60
            fig = px.scatter(df_pe, x="elevation_gain", y="Pace_min",
                             title="📍 Pace vs Elevação",
                             trendline="lowess", trendline_color_override=RED,
                             opacity=0.65,
                             color="Intensidade" if "Intensidade" in df_pe.columns else None,
                             color_discrete_map=INTENSITY_COLORS,
                             category_orders={"Intensidade": INTENSITY_ORDER},
                             labels={"elevation_gain":"Elevação (m)",
                                     "Pace_min":"Pace (min/km)"})
            set_pace_yaxis(fig, df_pe["pace_sec_km"])
            st.plotly_chart(fig, width="stretch")

        with col_b:
            df_fe = df_e[df_e["average_heartrate"].notna()].copy()
            fig = px.scatter(df_fe, x="elevation_gain", y="average_heartrate",
                             title="❤️ FC Média vs Elevação",
                             trendline="lowess", trendline_color_override=RED,
                             opacity=0.65,
                             color="Intensidade" if "Intensidade" in df_fe.columns else None,
                             color_discrete_map=INTENSITY_COLORS,
                             category_orders={"Intensidade": INTENSITY_ORDER},
                             labels={"elevation_gain":"Elevação (m)",
                                     "average_heartrate":"FC Média (bpm)"})
            st.plotly_chart(fig, width="stretch")

        top10 = (df_e.nlargest(10, "elevation_gain")
                     [["start_date","name","distance_km","elevation_gain","elev_km","pace_sec_km"]]
                     .copy())
        top10["Data"]       = top10["start_date"].dt.strftime("%d/%m/%Y")
        top10["Pace"]       = top10["pace_sec_km"].apply(fmt_pace)
        top10["Elev/km"]    = top10["elev_km"].apply(lambda x: f"{x:.1f} m/km")
        top10["Distância"]  = top10["distance_km"].apply(lambda x: f"{x:.1f} km")
        top10["Elevação"]   = top10["elevation_gain"].apply(lambda x: f"{x:.0f} m")
        st.markdown("### 🏔️ Top 10 Atividades com Maior Elevação")
        st.dataframe(
            top10[["Data","name","Distância","Elevação","Elev/km","Pace"]]
                 .rename(columns={"name":"Atividade"}),
            hide_index=True, use_container_width=True
        )

        if "Intensidade" in df_e.columns:
            df_ei = (cat_intensity(df_e)
                     .groupby("Intensidade", observed=True)
                     .agg(ElevTotal=("elevation_gain","sum"),
                          ElevMedia=("elevation_gain","mean"),
                          Qtd=("id","count"))
                     .reset_index())
            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.bar(df_ei, x="Intensidade", y="ElevTotal",
                             title="⛰️ Elevação Total por Tipo de Treino",
                             color="Intensidade", color_discrete_map=INTENSITY_COLORS,
                             category_orders={"Intensidade": INTENSITY_ORDER},
                             text=df_ei["ElevTotal"].apply(lambda x: f"{x:.0f}m"),
                             labels={"ElevTotal":"metros","Intensidade":""})
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width="stretch")
            with col_b:
                fig = px.bar(df_ei, x="Intensidade", y="ElevMedia",
                             title="📐 Elevação Média por Atividade e Tipo",
                             color="Intensidade", color_discrete_map=INTENSITY_COLORS,
                             category_orders={"Intensidade": INTENSITY_ORDER},
                             text=df_ei["ElevMedia"].apply(lambda x: f"{x:.0f}m"),
                             labels={"ElevMedia":"metros","Intensidade":""})
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width="stretch")

        df_grad = (df_e.groupby(["MesAnoOrd","MesAno"])
                       .agg(Grad=("elev_km","mean")).reset_index()
                       .sort_values("MesAnoOrd"))
        fig = px.line(df_grad, x="MesAno", y="Grad", markers=True,
                      title="📈 Gradiente Médio por Mês (m/km)",
                      color_discrete_sequence=[AMBER],
                      labels={"Grad":"m/km","MesAno":""})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
#  6 · CLIMA
# ══════════════════════════════════════════════════════════════════════════════
with tab_clima:
    st.title("🌤️ Clima e Condições")

    df_w = df_run[df_run["weather_temp"].notna()].copy() \
           if "weather_temp" in df_run.columns else pd.DataFrame()

    if df_w.empty:
        st.warning("Dados de clima não disponíveis para o período selecionado.")
    else:
        n_chuva   = int((df_w["weather_rain"] > 0).sum()) if "weather_rain" in df_w.columns else 0
        pct_chuva = n_chuva / len(df_w) * 100
        n_ideal   = len(df_w[(df_w["weather_temp"] >= 18) & (df_w["weather_temp"] <= 24)])
        pct_ideal = n_ideal / len(df_w) * 100

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("🌡️ Temp Média",        f"{df_w['weather_temp'].mean():.1f} °C")
        c2.metric("🌧️ Treinos c/ Chuva",  f"{n_chuva} ({pct_chuva:.0f}%)")
        c3.metric("✅ Temp Ideal 18–24°C", f"{pct_ideal:.0f}%")
        c4.metric("💧 Umidade Média",      f"{df_w['weather_humidity'].mean():.0f}%"
                  if "weather_humidity" in df_w.columns else "—")
        c5.metric("💨 Vento Médio",        f"{df_w['weather_wind_speed'].mean():.1f} km/h"
                  if "weather_wind_speed" in df_w.columns else "—")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            df_w["Pace_min"] = df_w["pace_sec_km"] / 60
            fig = px.scatter(df_w, x="weather_temp", y="Pace_min",
                             title="🌡️ Pace vs Temperatura (LOWESS)",
                             trendline="lowess", trendline_color_override=RED,
                             opacity=0.6,
                             color="Intensidade" if "Intensidade" in df_w.columns else None,
                             color_discrete_map=INTENSITY_COLORS,
                             category_orders={"Intensidade": INTENSITY_ORDER},
                             labels={"weather_temp":"Temperatura (°C)",
                                     "Pace_min":"Pace (min/km)"})
            fig.update_yaxes(autorange="reversed")
            fig.add_vrect(x0=15, x1=26, fillcolor=GREEN, opacity=0.07,
                          annotation_text="Confortável")
            st.plotly_chart(fig, width="stretch")

        with col_b:
            bins   = [0,14,18,24,28,50]
            labels = ["≤14°C (frio)","15–18°C","18–24°C","24–28°C","≥28°C (calor)"]
            df_w["Faixa"] = pd.cut(df_w["weather_temp"], bins=bins, labels=labels)
            df_f = (df_w.groupby("Faixa", observed=True)
                        .agg(Pace=("pace_sec_km","mean"), Qtd=("id","count"))
                        .reset_index())
            df_f["Pace_fmt"] = df_f["Pace"].apply(fmt_pace)
            df_f["Pace_min"] = df_f["Pace"] / 60
            fig = px.bar(df_f, x="Faixa", y="Pace_min",
                         title="📊 Pace Médio por Faixa de Temperatura",
                         color="Faixa", text="Pace_fmt",
                         labels={"Faixa":"","Pace_min":"Pace (min/km)"})
            set_pace_yaxis(fig, df_f["Pace"].dropna())
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")

        if "weather_rain" in df_w.columns:
            pace_chuva = df_w[df_w["weather_rain"] > 0]["pace_sec_km"].mean()
            pace_seco  = df_w[df_w["weather_rain"] == 0]["pace_sec_km"].mean()
            c1,c2,c3 = st.columns(3)
            c1.metric("🌧️ Pace na Chuva",  fmt_pace(pace_chuva))
            c2.metric("☀️ Pace Sem Chuva", fmt_pace(pace_seco))
            if not (pd.isna(pace_chuva) or pd.isna(pace_seco)):
                delta = pace_chuva - pace_seco
                c3.metric("Δ Impacto da Chuva",
                          f"+{delta:.0f}s/km" if delta > 0 else f"{delta:.0f}s/km")
                if delta > 15:
                    st.warning(f"🌧️ Chuva impacta significativamente — +{delta:.0f}s/km.")
                elif delta <= 5:
                    st.success("💪 Ótima resiliência na chuva — impacto mínimo no pace.")


# ══════════════════════════════════════════════════════════════════════════════
#  7 · METAS E BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════
with tab_metas:
    st.title("🎯 Metas e Benchmarks")

    hoje     = pd.Timestamp.now()
    mes_run  = df_run[df_run["start_date"].dt.to_period("M") == hoje.to_period("M")]
    km_mes   = mes_run["distance_km"].sum()
    tr_mes   = len(mes_run)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("📏 KM este Mês",      f"{km_mes:.0f} km",
              f"{km_mes/150*100:.0f}% meta 150km")
    c2.metric("🏃 Treinos este Mês", f"{tr_mes}",
              f"{tr_mes/16*100:.0f}% meta 16 treinos")
    c3.metric("🥇 Melhor 5K",        melhor_be("5k"))
    c4.metric("🥇 Melhor 10K",       melhor_be("10k"))

    st.markdown("---")

    if not be.empty:
        col_a, col_b = st.columns(2)

        with col_a:
            be_ev = be[be["name"].str.lower().isin(["5k","10k","1k"])].copy()
            if not be_ev.empty:
                be_ev_m = (be_ev.groupby(["MesAnoOrd","MesAno","name"])
                                .agg(Pace=("pace_sec_km","min")).reset_index()
                                .sort_values("MesAnoOrd"))
                be_ev_m["Pace_fmt"] = be_ev_m["Pace"].apply(fmt_pace)
                be_ev_m["Pace_min"] = be_ev_m["Pace"] / 60
                fig = px.line(be_ev_m, x="MesAno", y="Pace_min", color="name",
                              markers=True, custom_data=["Pace_fmt"],
                              title="📈 Evolução dos Melhores Esforços",
                              labels={"Pace_min":"Pace (min/km)","MesAno":"","name":"Distância"})
                fig.update_traces(
                    hovertemplate="<b>%{x}</b><br>Melhor pace: %{customdata[0]}/km<extra></extra>")
                set_pace_yaxis(fig, be_ev_m["Pace"])
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True, key="fig_be_evolucao")

        with col_b:
            pr_df = be[be["pr_rank"].notna()]
            if not pr_df.empty:
                pr_count = pr_df["name"].value_counts().reset_index()
                pr_count.columns = ["Distância","PRs"]
                fig = px.bar(pr_count, x="Distância", y="PRs",
                             title="🏅 PRs Conquistados por Distância",
                             color_discrete_sequence=[AMBER], text_auto=True)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key="fig_prs")

    df_meta = (df_run.groupby(["MesAnoOrd","MesAno"])
                     .agg(KM=("distance_km","sum"), Treinos=("id","count"))
                     .reset_index().sort_values("MesAnoOrd").tail(12))
    df_meta["PctKM"]      = (df_meta["KM"] / 150 * 100).clip(upper=120)
    df_meta["PctTreinos"] = (df_meta["Treinos"] / 16 * 100).clip(upper=120)

    fig = go.Figure()
    fig.add_bar(x=df_meta["MesAno"], y=df_meta["PctKM"],
                name="% Meta 150km", marker_color=BLUE,
                text=df_meta["KM"].apply(lambda x: f"{x:.0f}km"),
                textposition="outside")
    fig.add_bar(x=df_meta["MesAno"], y=df_meta["PctTreinos"],
                name="% Meta 16 treinos", marker_color=GREEN,
                text=df_meta["Treinos"], textposition="outside")
    fig.add_hline(y=100, line_dash="dash", line_color=AMBER, annotation_text="Meta 100%")
    fig.update_layout(barmode="group", title="📊 % Metas Mensais — últimos 12 meses",
                      yaxis_title="%", xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True, key="fig_metas_mensais")

    # ── Previsões de Prova — Fórmula de Riegel ───────────────────────────────
    st.markdown("---")
    # ── Previsões de Prova — Fórmula de Riegel ───────────────────────────────
    st.markdown("---")
    st.subheader("🔮 Previsões de Prova")
    st.caption(
        "Fórmula de Riegel: **T₂ = T₁ × (D₂/D₁)^1.06** — mesma base usada pelo Strava. "
        "Base escolhida automaticamente pelo melhor esforço disponível mais longo."
    )

    def _riegel(t1, d1, d2):
        if not t1 or t1 <= 0 or d1 <= 0: return None
        return t1 * (d2 / d1) ** 1.06

    def _fmt_time(sec):
        if sec is None or sec <= 0: return "—"
        s = int(sec)
        h, rem = divmod(s, 3600)
        m, ss  = divmod(rem, 60)
        return f"{h}h{m:02d}m{ss:02d}s" if h > 0 else f"{m}:{ss:02d}"

    def _best_sec(dist_label):
        if be_raw.empty: return None
        s = be_raw[be_raw["name"].str.lower() == dist_label.lower()]
        if s.empty: return None
        best_pace = s["pace_sec_km"].min()
        if pd.isna(best_pace): return None
        dist_map = {"1k": 1.0, "5k": 5.0, "10k": 10.0,
                    "half-marathon": 21.097, "marathon": 42.195}
        km = dist_map.get(dist_label.lower())
        return best_pace * km if km else None

    _bases = [
        ("Maratona", _best_sec("marathon"),      42.195),
        ("Meia",     _best_sec("half-marathon"), 21.097),
        ("10K",      _best_sec("10k"),           10.0),
        ("5K",       _best_sec("5k"),             5.0),
        ("1K",       _best_sec("1k"),             1.0),
    ]
    # Usa a base mais longa disponível
    _base_auto = next(((n, t, d) for n, t, d in _bases if t), None)

    _alvos = [("5K", 5.0), ("10K", 10.0), ("Meia (21K)", 21.097), ("Maratona (42K)", 42.195)]
    _CONF  = {
        ("1K","5K"):"⚠️ Estimativa","1K10K":"⚠️ Estimativa",
        ("5K","5K"):"✅ Alta precisão",("5K","10K"):"✅ Alta precisão",
        ("5K","Meia (21K)"):"🟡 Boa estimativa",("5K","Maratona (42K)"):"⚠️ Estimativa",
        ("10K","5K"):"✅ Alta precisão",("10K","10K"):"✅ Alta precisão",
        ("10K","Meia (21K)"):"✅ Alta precisão",("10K","Maratona (42K)"):"🟡 Boa estimativa",
        ("Meia","5K"):"🟡 Boa estimativa",("Meia","10K"):"✅ Alta precisão",
        ("Meia","Meia (21K)"):"✅ Alta precisão",("Meia","Maratona (42K)"):"✅ Alta precisão",
        ("Maratona","5K"):"🟡 Boa estimativa",("Maratona","10K"):"✅ Alta precisão",
        ("Maratona","Meia (21K)"):"✅ Alta precisão",("Maratona","Maratona (42K)"):"✅ Alta precisão",
    }

    if not _base_auto:
        st.info("Nenhum best effort disponível. Rode com o Strava ativo para registrar "
                "esforços em distâncias padrão (1K, 5K, 10K…).")
    else:
        _bn, _t1, _d1 = _base_auto
        st.caption(f"Base de referência usada: **{_bn}** ({_fmt_time(_t1)} · {fmt_pace(_t1/_d1)}/km)")

        # 4 cards de previsão
        _cols = st.columns(4)
        for _i, (_label, _d2) in enumerate(_alvos):
            _t2        = _riegel(_t1, _d1, _d2)
            _conf      = _CONF.get((_bn, _label), "🟡 Estimativa")
            _pace_pred = (_t2 / _d2) if _t2 else None
            with _cols[_i]:
                st.metric(f"🏁 {_label}", _fmt_time(_t2),
                          f"{fmt_pace(_pace_pred)}/km" if _pace_pred else "")
                st.caption(_conf)

        # Gráfico comparativo: todas as bases × todas as distâncias alvo
        st.markdown("##### Comparativo entre bases de referência")
        _bases_disp = [(n, t, d) for n, t, d in _bases if t]

        if len(_bases_disp) >= 2:
            _traces_data = []
            for _label, _d2 in _alvos:
                _paces = []
                _times = []
                for _bname, _bt, _bd in _bases_disp:
                    _t2c = _riegel(_bt, _bd, _d2)
                    _paces.append((_t2c / _d2 / 60) if _t2c else None)
                    _times.append(fmt_pace(_t2c / _d2) if _t2c else "—")
                _traces_data.append((_label, _paces, _times))

            _base_names = [b[0] for b in _bases_disp]
            _DIST_COLORS = [BLUE, GREEN, AMBER, RED]
            fig_rie = go.Figure()
            for _idx, (_lbl, _pcs, _tms) in enumerate(_traces_data):
                fig_rie.add_bar(
                    name=_lbl,
                    x=_base_names,
                    y=_pcs,
                    text=_tms,
                    textposition="outside",
                    marker_color=_DIST_COLORS[_idx % len(_DIST_COLORS)],
                )

            _all_pace_sec = [
                _riegel(_bt, _bd, _d2) / _d2
                for _, _d2 in _alvos
                for _, _bt, _bd in _bases_disp
                if _riegel(_bt, _bd, _d2)
            ]
            set_pace_yaxis(fig_rie, pd.Series(_all_pace_sec))
            fig_rie.update_layout(
                barmode="group",
                title="Pace previsto por distância alvo × base de referência",
                xaxis_title="Base de referência",
                legend=dict(orientation="h", y=-0.18),
                height=340,
            )
            st.plotly_chart(fig_rie, use_container_width=True, key="fig_riegel_comp")
            st.caption(
                "Barras próximas entre si = consistência entre distâncias. "
                "Base curta muito mais otimista que base longa = perfil de velocista — "
                "invista em volume aeróbico para melhorar as provas longas."
            )


# ══════════════════════════════════════════════════════════════════════════════
#  8 · VOLUME E EVOLUÇÃO
# ══════════════════════════════════════════════════════════════════════════════
with tab_vol:
    st.title("📈 Volume e Evolução")

    col_a, col_b = st.columns(2)

    with col_a:
        df_acum = (df_run.sort_values("start_date")
                         .assign(KM_Acum=lambda x: x["distance_km"].cumsum()))
        fig = px.area(df_acum, x="start_date", y="KM_Acum",
                      title="📏 Distância Acumulada",
                      color_discrete_sequence=[BLUE],
                      labels={"start_date":"","KM_Acum":"km"})
        st.plotly_chart(fig, width="stretch")

    with col_b:
        df_sem = (df_run.groupby(["Semana","SemanaStr"])
                        .agg(KM=("distance_km","sum")).reset_index()
                        .sort_values("Semana").tail(24))
        fig = px.bar(df_sem, x="SemanaStr", y="KM",
                     title="📅 Volume Semanal — últimas 24 semanas",
                     color_discrete_sequence=[PURPLE], text_auto=".0f",
                     labels={"SemanaStr":"","KM":"km"})
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    if "Intensidade" in df_run.columns:
        df_int_m = cat_intensity(
            df_run.groupby(["MesAnoOrd","MesAno","Intensidade"])
                  .agg(KM=("distance_km","sum")).reset_index()
                  .sort_values("MesAnoOrd"))
        fig = px.bar(df_int_m, x="MesAno", y="KM", color="Intensidade",
                     title="🎯 Volume por Intensidade e Mês",
                     color_discrete_map=INTENSITY_COLORS,
                     category_orders={"Intensidade": INTENSITY_ORDER},
                     labels={"KM":"km","MesAno":"","Intensidade":"Intensidade"})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, width="stretch")

    df_mom = (df_run.groupby(["MesAnoOrd","MesAno"])
                    .agg(KM=("distance_km","sum")).reset_index()
                    .sort_values("MesAnoOrd"))
    df_mom["Δ%"] = df_mom["KM"].pct_change() * 100
    df_mom["cor"] = df_mom["Δ%"].apply(
        lambda x: RED if not pd.isna(x) and x < -10
        else AMBER if not pd.isna(x) and x > 30
        else GREEN if not pd.isna(x) else GRAY)
    fig = go.Figure(go.Bar(
        x=df_mom["MesAno"], y=df_mom["Δ%"], marker_color=df_mom["cor"],
        text=df_mom["Δ%"].apply(lambda x: f"{x:+.0f}%" if not pd.isna(x) else ""),
        textposition="outside"))
    fig.add_hline(y=0,   line_color=GRAY, line_dash="dash")
    fig.add_hline(y=30,  line_color=AMBER, line_dash="dot",
                  annotation_text="+30% — risco overload")
    fig.add_hline(y=-10, line_color=AMBER, line_dash="dot")
    fig.update_layout(title="📊 Crescimento de Volume MoM (%)",
                      yaxis_title="%", xaxis_tickangle=-45)
    st.plotly_chart(fig, width="stretch")
    st.caption("**MoM = Month over Month** — variação de quilometragem em relação ao mês anterior. "
               "Verde = crescimento moderado (saudável) · Vermelho = queda ou salto excessivo (>30% é risco de lesão).")


# ══════════════════════════════════════════════════════════════════════════════
#  9 · VISÃO TREINADOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_coach:
    st.title("🧑‍🏫 Visão Treinador")

    hoje  = pd.Timestamp.now()
    # Semana do calendário actual (segunda a domingo)
    _inicio_semana = (hoje - timedelta(days=hoje.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    ult7  = df_run[df_run["start_date"] >= _inicio_semana]
    ult28 = df_run[df_run["start_date"] >= hoje - timedelta(days=28)]

    carga_aguda   = ult7["suffer_score"].sum()  if "suffer_score" in df_run.columns else 0
    carga_cronica = ult28["suffer_score"].sum() / 4 if "suffer_score" in df_run.columns else 0
    acwr          = carga_aguda / carga_cronica if carga_cronica > 0 else 0

    if   acwr < 0.8:  zona_r, cor_r = "🟡 Subcarregado",   AMBER
    elif acwr <= 1.3: zona_r, cor_r = "🟢 Zona Segura",    GREEN
    elif acwr <= 1.5: zona_r, cor_r = "🟠 Risco Moderado", "#E67E22"
    else:             zona_r, cor_r = "🔴 Alto Risco",      RED

    if "Intensidade" in df_run.columns:
        leves    = df_run["Intensidade"].isin(["Leve","Moderado"]).sum()
        intensos = df_run["Intensidade"].isin(["Forte","Muito Forte"]).sum()
        ratio_li = f"{leves}:{intensos}"
    else:
        ratio_li = "—"

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("⚡ ACWR",               f"{acwr:.2f}", zona_r)
    c2.metric("🔥 Carga Aguda 7d",     f"{carga_aguda:.0f}")
    c3.metric("📊 Carga Crônica 28d",  f"{carga_cronica:.0f}")
    c4.metric("🎯 Ratio Leve:Intenso", ratio_li, help="Proporção de treinos fáceis/moderados vs. fortes/muito fortes. Ideal: 4:1 (80% fácil, 20% intenso — modelo polarizado de Seiler).")
    c5.metric("📏 KM esta semana",     f"{ult7['distance_km'].sum():.0f} km", help="Quilometragem de segunda-feira a hoje (semana actual do calendário).")

    st.caption("📖 **ACWR:** compara esforço dos últimos 7d com a média das 4 semanas. "
               "Entre 0,8 e 1,3 = equilíbrio seguro. Acima de 1,5 = risco elevado de lesão.")

    if   acwr > 1.5: st.warning("⚠️ ACWR acima de 1,5 — alto risco. Reduza o volume esta semana.")
    elif acwr > 1.3: st.warning("🟠 ACWR entre 1,3 e 1,5 — zona de atenção.")
    elif acwr < 0.8: st.info("📉 ACWR abaixo de 0,8 — espaço para aumentar volume com segurança.")
    else:            st.success("✅ ACWR dentro da zona segura (0,8–1,3).")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        if df_run["suffer_score"].notna().any():
            weekly = (df_run[df_run["suffer_score"].notna()]
                      .set_index("start_date")["suffer_score"]
                      .resample("W").sum())
            acwr_df = pd.DataFrame({"Carga": weekly}).reset_index()
            acwr_df.columns = ["Semana","Carga"]
            acwr_df["Aguda"]   = acwr_df["Carga"].rolling(1).sum()
            acwr_df["Cronica"] = acwr_df["Carga"].rolling(4).mean()
            acwr_df["ACWR"]    = (acwr_df["Aguda"] / acwr_df["Cronica"]).replace([float("inf")], None)
            acwr_df = acwr_df.dropna(subset=["ACWR"])
            fig = px.line(acwr_df, x="Semana", y="ACWR", markers=True,
                          title="⚡ ACWR Histórico por Semana",
                          color_discrete_sequence=[BLUE],
                          labels={"ACWR":"Ratio","Semana":""})
            fig.add_hrect(y0=0.8, y1=1.3, fillcolor=GREEN, opacity=0.08,
                          annotation_text="Zona Segura (0.8–1.3)")
            fig.add_hline(y=1.5, line_dash="dash", line_color=RED,
                          annotation_text="Risco Alto (1.5)")
            st.plotly_chart(fig, width="stretch")

    with col_b:
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
            fig.update_layout(title="📊 Variação de Carga Semanal (%)",
                              xaxis_tickangle=-45, yaxis_title="%")
            st.plotly_chart(fig, width="stretch")

    df_fc8 = (df_run[df_run["average_heartrate"].notna()]
              .groupby(["Semana","SemanaStr"])
              .agg(FC=("average_heartrate","mean")).reset_index()
              .sort_values("Semana").tail(8))
    if not df_fc8.empty:
        tend = ("📈 Subindo — possível fadiga acumulada"
                if df_fc8["FC"].iloc[-1] > df_fc8["FC"].iloc[0]
                else "📉 Caindo — boa adaptação cardiovascular"
                if df_fc8["FC"].iloc[-1] < df_fc8["FC"].iloc[0]
                else "➡️ Estável")
        st.info(f"**Tendência FC (últimas 8 semanas):** {tend}")
        fig = px.line(df_fc8, x="SemanaStr", y="FC", markers=True,
                      title="❤️ FC Média — Últimas 8 Semanas",
                      color_discrete_sequence=[RED],
                      labels={"FC":"bpm","SemanaStr":""})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, width="stretch")

    # ── NOVO: PMC — CTL / ATL / TSB ──────────────────────────────────────────
    if not pmc_raw.empty:
        st.markdown("---")
        st.subheader("📊 Performance Management Chart — CTL · ATL · TSB")
        st.caption(
            "**CTL** (42d): condicionamento acumulado — quanto maior, maior a base fitness. "
            "**ATL** (7d): fadiga recente. "
            "**TSB** = CTL − ATL. TSB +5 a +20 = janela de pico de forma para qualidade máxima ou competição."
        )

        pmc_filt = pmc_raw[
            (pd.to_datetime(pmc_raw["Data"]) >= s_dt) &
            (pd.to_datetime(pmc_raw["Data"]) <= e_dt)
        ].copy()

        if not pmc_filt.empty:
            pmc_filt["Semana"] = pd.to_datetime(pmc_filt["Data"]).dt.to_period("W").apply(
                lambda x: x.start_time)
            pmc_w = pmc_filt.groupby("Semana").last().reset_index()

            ctl_at = pmc_w["CTL"].iloc[-1] if len(pmc_w) else 0
            atl_at = pmc_w["ATL"].iloc[-1] if len(pmc_w) else 0
            tsb_at = pmc_w["TSB"].iloc[-1] if len(pmc_w) else 0

            cp1, cp2, cp3 = st.columns(3)
            cp1.metric("💪 CTL (fitness)", f"{ctl_at:.0f}",
                       help="Carga crônica 42 dias. Quanto maior, maior a base.")
            cp2.metric("😓 ATL (fadiga)",  f"{atl_at:.0f}",
                       help="Carga aguda 7 dias.")
            cp3.metric("✨ TSB (forma)",   f"{tsb_at:+.0f}",
                       delta=("Pico de forma ✅" if 5 <= tsb_at <= 20
                              else "Fatigado ⚠️" if tsb_at < -15
                              else "OK"),
                       delta_color="normal" if tsb_at >= 0 else "inverse")

            fig_pmc = go.Figure()
            fig_pmc.add_scatter(x=pmc_w["Semana"], y=pmc_w["CTL"],
                                name="CTL — fitness", mode="lines",
                                line=dict(color=BLUE, width=2.5),
                                fill="tozeroy", fillcolor="rgba(52,152,219,0.08)")
            fig_pmc.add_scatter(x=pmc_w["Semana"], y=pmc_w["ATL"],
                                name="ATL — fadiga", mode="lines",
                                line=dict(color=RED, width=2, dash="dot"))
            fig_pmc.add_scatter(x=pmc_w["Semana"], y=pmc_w["TSB"],
                                name="TSB — forma", mode="lines+markers",
                                line=dict(color=GREEN, width=2),
                                marker=dict(size=4), yaxis="y2")
            fig_pmc.add_hline(y=0, line_color=GRAY, line_dash="dash",
                              line_width=1, yref="y2")
            fig_pmc.add_hrect(y0=5, y1=20, fillcolor=GREEN, opacity=0.06,
                              line_width=0, yref="y2",
                              annotation_text="Janela de forma",
                              annotation_position="top right")
            fig_pmc.update_layout(
                title="CTL · ATL · TSB — semana a semana",
                yaxis=dict(title="CTL / ATL"),
                yaxis2=dict(title="TSB (forma)", overlaying="y", side="right",
                            showgrid=False, zeroline=False),
                legend=dict(orientation="h", y=-0.18),
                hovermode="x unified",
            )
            st.plotly_chart(fig_pmc, width="stretch")

            if   tsb_at > 20:   st.info("😴 TSB alto — atleta descansado. Boa janela para treino de qualidade ou competição.")
            elif tsb_at >= 5:   st.success("✅ TSB na janela ideal (+5 a +20). Forma em dia.")
            elif tsb_at >= -10: st.info("⚙️ TSB neutro — treinando normalmente. Monitorar fadiga.")
            elif tsb_at >= -20: st.warning("⚠️ TSB negativo — fadiga acumulada. Considere um dia de recuperação.")
            else:               st.error("🛑 TSB muito negativo — risco de overtraining. Reduza a carga.")

    # ── NOVO: Polarização — zonas vs ideal ───────────────────────────────────
    if not lps_run.empty and "Zona FC" in lps_run.columns:
        st.markdown("---")
        st.subheader("🎯 Perfil de Polarização — período selecionado")

        lz2 = lps_run.copy()
        lz2["tempo_min"] = lz2["moving_time_sec"] / 60
        dz2 = (lz2.groupby("Zona FC").agg(Tempo=("tempo_min","sum")).reset_index())
        dz2["Zona FC"] = pd.Categorical(dz2["Zona FC"], categories=ZONA_ORDER, ordered=True)
        dz2 = dz2.sort_values("Zona FC")
        tot2 = dz2["Tempo"].sum()
        dz2["Pct_atual"] = (dz2["Tempo"] / tot2 * 100).round(1)
        ideal_v = {"Sem FC":0,"Z1 - Regenerativo":35,"Z2 - Aeróbico":45,
                   "Z3 - Tempo":5,"Z4 - Limiar":10,"Z5 - VO2max":5}
        dz2["Pct_ideal"] = dz2["Zona FC"].map(ideal_v).fillna(0)

        fig_pol = go.Figure()
        fig_pol.add_bar(x=dz2["Zona FC"], y=dz2["Pct_atual"],
                        name="Atual",
                        marker_color=[ZONA_COLORS.get(str(z), GRAY) for z in dz2["Zona FC"]],
                        text=dz2["Pct_atual"].apply(lambda x: f"{x:.0f}%"),
                        textposition="outside", opacity=0.85)
        fig_pol.add_scatter(x=dz2["Zona FC"], y=dz2["Pct_ideal"],
                            name="Ideal (polarizado)", mode="markers+lines",
                            line=dict(color=GRAY, width=1.5, dash="dash"),
                            marker=dict(size=9, color=GRAY))
        fig_pol.update_layout(
            title="% Tempo em Zona — Atual vs Ideal Polarizado",
            yaxis_title="% do tempo",
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_pol, width="stretch")
        st.caption(
            "📖 Linha tracejada = modelo polarizado ideal (Seiler). "
            "Barras acima da linha = excesso nessa zona. O maior problema típico é excesso em Z3 e déficit em Z1."
        )

    # ── NOVO: Eficiência Aeróbica — pace em Z2 por mês ───────────────────────
    if not lps_run.empty and "Zona FC" in lps_run.columns:
        lps_z2 = lps_run[
            (lps_run["Zona FC"] == "Z2 - Aeróbico") &
            lps_run["pace_sec_km"].notna() &
            (lps_run["pace_sec_km"] > 0) &
            (lps_run["pace_sec_km"] < 480)
        ].copy()

        if not lps_z2.empty and len(lps_z2["MesAno"].unique()) >= 2:
            st.markdown("---")
            st.subheader("🏃 Eficiência Aeróbica — Pace na Zona 2")
            st.caption(
                "O **KPI mais honesto** para desenvolvimento de resistência: o ritmo sustentado "
                "na mesma frequência cardíaca de Z2. Quanto menor o valor (mais rápido na mesma FC), "
                "maior o motor aeróbico — sem nenhum custo extra de fadiga."
            )

            ae_m = (lps_z2.groupby(["MesAnoOrd","MesAno"])
                          .agg(PaceZ2=("pace_sec_km","mean"),
                               Amostras=("pace_sec_km","count"))
                          .reset_index()
                          .sort_values("MesAnoOrd"))
            ae_m = ae_m[ae_m["Amostras"] >= 5]

            if len(ae_m) >= 2:
                delta_ae = ae_m["PaceZ2"].iloc[0] - ae_m["PaceZ2"].iloc[-1]
                ae_m["PaceZ2_fmt"] = ae_m["PaceZ2"].apply(fmt_pace)
                ae_m["PaceZ2_min"] = ae_m["PaceZ2"] / 60

                ca1, ca2, ca3 = st.columns(3)
                ca1.metric("Pace Z2 — início",  ae_m["PaceZ2_fmt"].iloc[0]  + "/km")
                ca2.metric("Pace Z2 — atual",   ae_m["PaceZ2_fmt"].iloc[-1] + "/km")
                ca3.metric("Ganho aeróbico",    f"{abs(delta_ae):.0f}s/km",
                           f"+{delta_ae:.0f}s/km mais rápido na mesma FC" if delta_ae > 0
                           else "Ainda sem melhora",
                           delta_color="normal" if delta_ae > 0 else "inverse")

                fig_ae = go.Figure()
                fig_ae.add_scatter(
                    x=ae_m["MesAno"], y=ae_m["PaceZ2_min"],
                    mode="lines+markers+text",
                    text=ae_m["PaceZ2_fmt"],
                    textposition="top center",
                    line=dict(color=PURPLE, width=2.5),
                    marker=dict(size=8, color=PURPLE),
                    fill="tozeroy", fillcolor="rgba(155,89,182,0.07)",
                    customdata=ae_m["Amostras"].values,
                    hovertemplate="%{x}<br>Pace Z2: %{text}/km<br>Laps: %{customdata}<extra></extra>",
                )
                set_pace_yaxis(fig_ae, ae_m["PaceZ2"])
                fig_ae.update_layout(
                    title="Pace médio em Z2 por mês  (↓ = motor aeróbico maior)",
                    xaxis_tickangle=-45,
                    showlegend=False,
                )
                st.plotly_chart(fig_ae, width="stretch")

                if delta_ae >= 15:
                    st.success(
                        f"🚀 Ganho aeróbico real: +{delta_ae:.0f}s/km mais rápido na mesma FC de Z2. "
                        "O motor aeróbico está crescendo — continue priorizando o volume fácil."
                    )
                elif delta_ae > 0:
                    st.info(
                        f"📈 Melhora de +{delta_ae:.0f}s/km. Consistência e paciência — "
                        "base aeróbica é construída em meses, não semanas."
                    )
                else:
                    st.warning(
                        "⚠️ Eficiência aeróbica estável ou em queda. Verifique se há excesso "
                        "de Z3 consumindo a adaptação aeróbica, ou se o volume em Z2 está sendo mantido."
                    )


# ══════════════════════════════════════════════════════════════════════════════
#  10 · HISTÓRICO
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  10 · MAPA DE ROTAS — Folium animado + Comparativo de pace
# ══════════════════════════════════════════════════════════════════════════════
with tab_mapa:
    st.title("\U0001f5fa️ Mapa de Rotas")

    _POLY_CANDIDATES = ["map_polyline","polyline","map.polyline",
                        "summary_polyline","map_summary_polyline"]
    poly_col = next((c for c in _POLY_CANDIDATES
                     if c in df_raw.columns and df_raw[c].notna().any()), None)
    has_ll   = "latitude" in df_run.columns and "longitude" in df_run.columns

    # ── Cards de analise ────────────────────────────────────────────────────

    if not has_ll and poly_col is None:
        st.error("Nenhum dado de GPS encontrado.")
        st.stop()

    if poly_col is None:
        _msg = ("**Mostrando pontos de inicio** (colunas `latitude`/`longitude`). "
                "Para rotas completas, adicione ao extrator:\n\n"
                "```python\n'map_summary_polyline': activity.get('map', {}).get('summary_polyline', ''),\n```")
        st.info(_msg)

    # ── Preparacao dos dados ────────────────────────────────────────────────
    df_map = df_run.copy()
    if poly_col:
        df_map = df_map[df_map[poly_col].notna() & (df_map[poly_col].astype(str).str.len() > 4)]
    elif has_ll:
        df_map = df_map[df_map["latitude"].notna() & df_map["longitude"].notna()]
    df_map = df_map.sort_values("start_date", ascending=False)

    if df_map.empty:
        st.warning("Nenhuma atividade com dados de GPS no periodo selecionado.")
    else:
        # ── Seletor individual de atividades ──────────────────────────────────
        def _make_label(row):
            dt  = row["start_date"].strftime("%d/%m/%Y")
            nm  = str(row.get("name") or "")[:35]
            km  = round(float(row.get("distance_km") or 0), 1)
            int_v = str(row.get("Intensidade") or "")
            int_tag = f" [{int_v}]" if int_v and int_v != "None" else ""
            return f"{dt} — {nm} ({km}km){int_tag}"

        _label_to_id = {_make_label(row): row["id"] for _, row in df_map.iterrows()}
        _all_labels  = list(_label_to_id.keys())

        sel_labels = st.multiselect(
            "Selecione atividades para exibir e comparar",
            options=_all_labels,
            default=_all_labels[:min(10, len(_all_labels))],
            help="Digite para buscar por nome, data ou distância. Selecione várias para comparar no mapa e no gráfico de pace.",
            placeholder="Busque por nome, data ou distância...",
        )

        if not sel_labels:
            st.info("Selecione ao menos uma atividade acima para ver o mapa.")
            st.stop()

        _sel_ids = [_label_to_id[l] for l in sel_labels]
        df_map   = df_map[df_map["id"].isin(_sel_ids)].copy()
        st.caption(f"**{len(df_map)}** atividade(s) selecionada(s) · {len(_all_labels)} disponíveis no período")

        if poly_col and not df_map.empty:
            _fc = decode_polyline(df_map[poly_col].dropna().iloc[0])
            lat_c = _fc[0][0] if _fc else -23.55
            lng_c = _fc[0][1] if _fc else -46.63
        elif has_ll and df_map["latitude"].notna().any():
            lat_c = float(df_map["latitude"].dropna().mean())
            lng_c = float(df_map["longitude"].dropna().mean())
        else:
            lat_c, lng_c = -23.55, -46.63

        def get_color_hex(row):
            int_val = str(row.get("Intensidade","Moderado") or "Moderado")
            return INTENSITY_COLORS.get(int_val, BLUE)

        def pace_to_hex(pace_sec):
            # RED = rápido (baixo t), GREEN = lento (alto t)
            t = min(1, max(0, (float(pace_sec or 300) - 220) / 200))
            r = round(231 - t * (231 - 46))
            g = round(76  + t * (204 - 76))
            b = round(60  + t * (113 - 60))
            return "#{:02X}{:02X}{:02X}".format(r, g, b)

        # Monta lista de rotas
        routes = []
        for _, row in df_map.iterrows():
            coords = decode_polyline(row[poly_col]) if poly_col else []
            r_km   = float(row.get("distance_km") or 0)
            hr_v   = int(row["average_heartrate"]) if not pd.isna(row.get("average_heartrate", float("nan"))) else 0
            routes.append({
                "id":       row.get("id"),
                "name":     str(row.get("name","") or ""),
                "date":     row["start_date"].strftime("%d/%m/%Y"),
                "km":       round(r_km, 1),
                "pace":     fmt_pace(row.get("pace_sec_km", 0) or 0),
                "pace_sec": float(row.get("pace_sec_km") or 300),
                "hr":       hr_v,
                "intensity":str(row.get("Intensidade","Moderado") or "Moderado"),
                "color_hex":get_color_hex(row),
                "lat":      float(row["latitude"]) if has_ll and not pd.isna(row.get("latitude", float("nan"))) else lat_c,
                "lng":      float(row["longitude"]) if has_ll and not pd.isna(row.get("longitude", float("nan"))) else lng_c,
                "coords":   coords,
            })

        # ── SECAO 1: Mapa Folium (fragment isola do re-render externo) ───────────
        st.subheader("🗺️ Mapa — rotas animadas")

        if not HAS_FOLIUM:
            st.error("Instale: `pip install streamlit-folium folium`")
        else:
            _routes_snap = routes
            _lps_snap    = lps_run
            _poly_snap   = poly_col
            _lat2, _lng2 = lat_c, lng_c

            try:
                _frag = st.fragment
            except AttributeError:
                def _frag(f): return f
            _expandir = st.checkbox("🔍 Expandir mapa", value=False, key="expand_map")
            _map_h_outer = 720 if _expandir else 520
            @_frag
            def _render_folium():
                _MODES = [
                    "Intensidade",
                    "Pace (rapido/lento)",
                    "FC por zona",
                    "Elevacao por segmento",
                ]
                mode_map = st.radio("Colorir rotas por", _MODES, horizontal=True)
                tile_map = st.radio(
                    "Mapa base",
                    ["Claro", "Escuro", "Topográfico", "Satélite", "Topo ESRI", "Voyager"],
                    horizontal=True,
                    help="Satélite e Topo ESRI são os melhores para trail. Voyager é mais detalhado para asfalto urbano."
                )

                ESRI_SAT_URL   = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                ESRI_SAT_ATTR  = "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
                ESRI_TOPO_URL  = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
                ESRI_TOPO_ATTR = "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), and the GIS User Community"

                m = folium.Map(location=[_lat2, _lng2], zoom_start=13, tiles=None)

                if tile_map == "Claro":
                    folium.TileLayer("CartoDB positron", name="Claro").add_to(m)
                elif tile_map == "Escuro":
                    folium.TileLayer("CartoDB dark_matter", name="Escuro").add_to(m)
                elif tile_map == "Topográfico":
                    folium.TileLayer("OpenTopoMap", name="Topográfico").add_to(m)
                elif tile_map == "Satélite":
                    folium.TileLayer(ESRI_SAT_URL, attr=ESRI_SAT_ATTR, name="Satélite ESRI").add_to(m)
                elif tile_map == "Topo ESRI":
                    folium.TileLayer(ESRI_TOPO_URL, attr=ESRI_TOPO_ATTR, name="Topo ESRI").add_to(m)
                elif tile_map == "Voyager":
                    folium.TileLayer("CartoDB Voyager", name="Voyager").add_to(m)

                try:
                    from folium.plugins import AntPath, MiniMap
                    MiniMap(toggle_display=True, position="bottomleft",
                            tile_layer="CartoDB positron",
                            zoom_level_offset=-5).add_to(m)
                    _ant = True
                except Exception:
                    _ant = False

                def _seg_color(lap, r, mode):
                    if mode == "Pace (rapido/lento)":
                        return pace_to_hex(lap.get("pace_sec_km", r["pace_sec"]))
                    if mode == "FC por zona":
                        return fc_to_hex(lap.get("average_heartrate", r["hr"]))
                    if mode == "Elevacao por segmento":
                        d = float(lap.get("distance_km") or 1)
                        e = float(lap.get("total_elevation_gain") or 0)
                        return elev_gain_to_hex(e / d if d > 0 else 0)
                    return r["color_hex"]

                def _add_km_markers(coords, color, fg):
                    import math
                    def _hav(c1, c2):
                        la1,lo1=math.radians(c1[0]),math.radians(c1[1])
                        la2,lo2=math.radians(c2[0]),math.radians(c2[1])
                        d=math.sin((la2-la1)/2)**2+math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
                        return 2*6371*math.asin(math.sqrt(d))
                    cum, last = 0.0, 0
                    for i in range(1, len(coords)):
                        cum += _hav(coords[i-1], coords[i])
                        if int(cum) > last:
                            last = int(cum)
                            folium.Marker(coords[i], icon=folium.DivIcon(
                                html=f'<div style="font-size:9px;font-weight:700;color:#fff;'
                                     f'background:{color};padding:1px 5px;border-radius:10px;'
                                     f'box-shadow:0 1px 3px rgba(0,0,0,.5)">{last}</div>',
                                icon_size=(22,15), icon_anchor=(11,7)
                            )).add_to(fg)

                def _lap_popup_html(lap, r):
                    ln  = int(lap.get("lap_index", 0) or 0)
                    dk  = round(float(lap.get("distance_km") or 0), 2)
                    pc  = fmt_pace(lap.get("pace_sec_km") or 0)
                    hr  = int(lap.get("average_heartrate") or 0) or "—"
                    hrm = int(lap.get("max_heartrate") or 0) or "—"
                    elv = float(lap.get("total_elevation_gain") or 0)
                    cad = int(float(lap.get("average_cadence") or 0) * 2) or "—"
                    ci  = r["color_hex"]
                    rows = (
                        "<tr><td style='color:#888;padding:1px 8px 1px 0'>Pace</td>"
                        "<td><b>" + pc + "/km</b></td></tr>"
                        "<tr><td style='color:#888;padding:1px 8px 1px 0'>FC média</td>"
                        "<td><b>" + str(hr) + " bpm</b></td></tr>"
                        "<tr><td style='color:#888;padding:1px 8px 1px 0'>FC máx</td>"
                        "<td><b>" + str(hrm) + " bpm</b></td></tr>"
                        "<tr><td style='color:#888;padding:1px 8px 1px 0'>Elevação</td>"
                        "<td><b>" + f"{elv:.0f}" + " m</b></td></tr>"
                        + (f"<tr><td style='color:#888;padding:1px 8px 1px 0'>Cadência</td>"
                           f"<td><b>{cad} spm</b></td></tr>" if cad != "—" else "")
                    )
                    return (
                        "<div style='font-family:sans-serif;min-width:170px;padding:2px'>"
                        "<b style='font-size:12px'>" + r["name"][:25] + "</b>"
                        "<span style='color:#888;font-size:10px'> · " + r["date"] + "</span><br>"
                        "<div style='margin:4px 0 6px;padding:2px 7px;border-radius:4px;"
                        "background:" + ci + "20;border-left:3px solid " + ci + ";"
                        "font-size:11px;font-weight:600'>Lap " + str(ln) + " · " + str(dk) + " km</div>"
                        "<table style='font-size:11px;width:100%'>" + rows + "</table></div>"
                    )

                def _popup(r):
                    n = r["name"][:30]
                    dt = r["date"]
                    km = r["km"]
                    pc = r["pace"]
                    hr = r["hr"] or "—"
                    ci = r["color_hex"]
                    it = r["intensity"]
                    return (
                        "<div style='font-family:sans-serif;min-width:190px;padding:2px'>"
                        "<b style='font-size:13px'>" + n + "</b><br>"
                        "<span style='color:#888;font-size:11px'>" + dt + "</span>"
                        "<table style='font-size:12px;width:100%;margin-top:6px'>"
                        "<tr><td style='color:#888;padding:2px 10px 2px 0'>Dist</td>"
                        "<td><b>" + str(km) + " km</b></td></tr>"
                        "<tr><td style='color:#888;padding:2px 10px 2px 0'>Pace</td>"
                        "<td><b>" + pc + "/km</b></td></tr>"
                        "<tr><td style='color:#888;padding:2px 10px 2px 0'>FC</td>"
                        "<td><b>" + str(hr) + " bpm</b></td></tr>"
                        "</table>"
                        "<div style='margin-top:7px;padding:3px 8px;border-radius:4px;"
                        "background:" + ci + "20;border-left:3px solid " + ci + ";"
                        "font-size:11px;color:#444'>" + it + "</div></div>"
                    )

                for r in _routes_snap:
                    _fg_name = r["date"] + " — " + r["name"][:20] + " (" + str(r["km"]) + "km)"
                    fg = folium.FeatureGroup(name=_fg_name, show=True)

                    if _poly_snap and r["coords"]:
                        coords_all = r["coords"]
                        n_pts = len(coords_all) - 1
                        act_laps = (
                            _lps_snap[_lps_snap["activity_id"] == r["id"]]
                            .sort_values("lap_index")
                            if not _lps_snap.empty else pd.DataFrame())

                        # ── VISUAL: animação colorida por modo ──────────────
                        if mode_map == "Intensidade":
                            if _ant:
                                AntPath(r["coords"], color=r["color_hex"], weight=4.5,
                                        dash_array=[12, 20], delay=800, opacity=0.92,
                                        popup=folium.Popup(_popup(r), max_width=220)).add_to(fg)
                            else:
                                folium.PolyLine(r["coords"], color=r["color_hex"],
                                               weight=4.5, opacity=0.9,
                                               popup=folium.Popup(_popup(r), max_width=220)).add_to(fg)
                        else:
                            coords_all = r["coords"]
                            n_pts = len(coords_all) - 1
                            if not act_laps.empty and n_pts > 0:
                                total_km = act_laps["distance_km"].sum()
                                cum = 0.0
                                for _, lap in act_laps.iterrows():
                                    frac = (float(lap["distance_km"]) / total_km
                                            if total_km > 0 else 1 / len(act_laps))
                                    i0 = int(cum * n_pts)
                                    i1 = min(n_pts, int((cum + frac) * n_pts) + 1)
                                    seg = coords_all[i0:i1 + 1]
                                    if len(seg) >= 2:
                                        cseg = _seg_color(lap, r, mode_map)
                                        if _ant:
                                            AntPath(seg, color=cseg, weight=5,
                                                    dash_array=[12, 20], delay=800,
                                                    opacity=0.9).add_to(fg)
                                        else:
                                            folium.PolyLine(seg, color=cseg,
                                                           weight=5, opacity=0.9).add_to(fg)
                                    cum += frac
                            else:
                                cseg = _seg_color({}, r, mode_map)
                                if _ant:
                                    AntPath(coords_all, color=cseg, weight=5,
                                            dash_array=[12, 20], delay=800,
                                            opacity=0.9).add_to(fg)
                                else:
                                    folium.PolyLine(coords_all, color=cseg,
                                                   weight=5, opacity=0.9).add_to(fg)

                        # ── INTERATIVO: overlays invisíveis com popup por lap ──
                        if not act_laps.empty and n_pts > 0:
                            total_km2 = act_laps["distance_km"].sum()
                            cum2 = 0.0
                            for _, lap2 in act_laps.iterrows():
                                frac2 = (float(lap2["distance_km"]) / total_km2
                                         if total_km2 > 0 else 1 / len(act_laps))
                                i02 = int(cum2 * n_pts)
                                i12 = min(n_pts, int((cum2 + frac2) * n_pts) + 1)
                                seg2 = coords_all[i02:i12 + 1]
                                if len(seg2) >= 2:
                                    folium.PolyLine(
                                        seg2, color=r["color_hex"],
                                        weight=10, opacity=0.001,
                                        popup=folium.Popup(_lap_popup_html(lap2, r), max_width=220)
                                    ).add_to(fg)
                                cum2 += frac2
                        else:
                            folium.PolyLine(
                                coords_all, color=r["color_hex"],
                                weight=8, opacity=0.001,
                                popup=folium.Popup(_popup(r), max_width=220)
                            ).add_to(fg)

                        # ── KM MARKERS ──────────────────────────────────────
                        _add_km_markers(coords_all, r["color_hex"], fg)
                        folium.CircleMarker(
                            coords_all[0], radius=6, color=r["color_hex"],
                            fill=True, fill_opacity=1, weight=2,
                            tooltip="Início").add_to(fg)
                    else:
                        folium.CircleMarker(
                            [r["lat"], r["lng"]],
                            radius=max(5, min(16, r["km"] * 0.9)),
                            color=r["color_hex"], fill=True, fill_opacity=0.8,
                            popup=folium.Popup(_popup(r), max_width=220)).add_to(fg)

                    fg.add_to(m)

                folium.LayerControl(collapsed=True, position="topright").add_to(m)

                # Legenda transparente — construida sem backslash em f-string
                _li_int = "".join(
                    "<div style='display:flex;align-items:center;gap:6px;margin:3px 0;"
                    "color:rgba(255,255,255,.85)'>"
                    "<div style='width:16px;height:4px;background:" + c + ";"
                    "border-radius:2px;flex-shrink:0'></div>" + k + "</div>"
                    for k, c in INTENSITY_COLORS.items() if k != "Skate"
                )
                _li_pace = (
                    "<div style='display:flex;align-items:center;gap:5px;"
                    "color:rgba(255,255,255,.8)'><span>rápido</span>"
                    "<div style='width:50px;height:5px;border-radius:3px;"
                    "background:linear-gradient(to right,rgb(231,76,60),rgb(46,204,113))'>"
                    "</div><span>lento</span></div>"
                )
                _li_fc = "".join(
                    "<div style='display:flex;align-items:center;gap:6px;margin:3px 0;"
                    "color:rgba(255,255,255,.85)'>"
                    "<div style='width:16px;height:4px;background:" + c + ";"
                    "border-radius:2px'></div>" + z + "</div>"
                    for z, c in [("Z1 <137bpm","#3498DB"),("Z2 <165","#2ECC71"),
                                  ("Z3 <175","#F39C12"),("Z4 <185","#E67E22"),
                                  ("Z5 ≥185","#E74C3C")]
                )
                _li_elev = (
                    "<div style='display:flex;align-items:center;gap:5px;"
                    "color:rgba(255,255,255,.8)'><span>plano</span>"
                    "<div style='width:50px;height:5px;border-radius:3px;"
                    "background:linear-gradient(to right,rgb(46,204,113),rgb(231,76,60))'>"
                    "</div><span>morro</span></div>"
                )
                _LEG = {
                    "Intensidade":           ("Intensidade",          _li_int),
                    "Pace (rapido/lento)":   ("Pace",                 _li_pace),
                    "FC por zona":           ("FC",                   _li_fc),
                    "Elevacao por segmento": ("Elevação",       _li_elev),
                }
                leg_title, leg_items = _LEG.get(mode_map, ("", ""))
                legend_html = (
                    "<div style='position:absolute;bottom:36px;right:52px;z-index:9999;"
                    "background:rgba(10,10,10,0.72);padding:9px 13px;border-radius:10px;"
                    "font-size:12px;color:rgba(255,255,255,.9)'>"
                    "<div style='font-weight:600;margin-bottom:5px'>" + leg_title + "</div>"
                    + leg_items + "</div>"
                )
                m.get_root().html.add_child(folium.Element(legend_html))

                st_folium(m, use_container_width=True, height=_map_h_outer, returned_objects=[])
                _hint = "Clique numa rota para detalhes" if _poly_snap else "Tamanho = distância"
                st.caption(
                    _hint + " · Ative/desative rotas no controle (canto superior direito) · "
                    + ("Animação AntPath ativa" if _ant else "pip install folium para animação")
                )

            _render_folium()

        # ── SECAO 2: Comparativo de pace por lap ─────────────────────────────
        st.markdown("---")
        st.subheader("\U0001f3c3 Comparativo de pace por lap")
        st.caption("Substituicao do 3D — mais util: ve a evolucao do ritmo km a km dentro de cada corrida.")

        if lps_run.empty:
            st.info("Dados de laps nao disponíveis para analise de pace por segmento.")
        else:
            run_opts = {f"{r['date']} — {r['name'][:22]} ({r['km']}km)": r for r in routes}
            sel_runs = st.multiselect(
                "Selecione corridas para comparar (ate 5)",
                list(run_opts.keys()),
                default=list(run_opts.keys())[:min(3, len(run_opts))],
                max_selections=5,
            )

            fig_cmp = go.Figure()
            all_paces_cmp = []

            for label in sel_runs:
                r = run_opts[label]
                laps_r = (lps_run[lps_run["activity_id"] == r["id"]]
                          .sort_values("lap_index").copy())
                laps_r = laps_r[laps_r["pace_sec_km"].notna()
                                & (laps_r["pace_sec_km"] > 0)
                                & (laps_r["pace_sec_km"] < 600)]
                if laps_r.empty:
                    continue

                all_paces_cmp.extend(laps_r["pace_sec_km"].tolist())
                laps_r["pace_min"] = laps_r["pace_sec_km"] / 60
                laps_r["pace_fmt"] = laps_r["pace_sec_km"].apply(fmt_pace)

                _DIST_COLS = ["#E74C3C","#3498DB","#2ECC71","#9B59B6","#F39C12"]
                _ci = len([x for x in all_paces_cmp]) % len(_DIST_COLS) if all_paces_cmp else 0
                _col = _DIST_COLS[len([t for t in fig_cmp.data]) % len(_DIST_COLS)]
                fig_cmp.add_trace(go.Scatter(
                    x=laps_r["lap_index"],
                    y=laps_r["pace_min"],
                    name=label[:38],
                    mode="lines+markers",
                    line=dict(color=_col, width=2.5),
                    marker=dict(size=6, color=_col),
                    customdata=laps_r["pace_fmt"],
                    hovertemplate="Km %{x}<br>Pace: %{customdata}/km<extra></extra>",
                ))

            if fig_cmp.data and all_paces_cmp:
                pace_series = pd.Series([p for p in all_paces_cmp if 0 < p < 600])
                set_pace_yaxis(fig_cmp, pace_series)
                fig_cmp.update_layout(
                    title="Pace por lap — evolucao durante a corrida",
                    xaxis_title="Lap (km)",
                    legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_cmp, width="stretch")
                st.caption(
                    "Eixo Y invertido — mais alto = mais rapido. "
                    "Linha descendente = corrida progressiva (comeca lento, termina rapido). "
                    "Linha plana = ritmo uniforme. Picos = segmentos de recuperacao ou subidas."
                )
            elif sel_runs:
                st.info("Dados de laps nao encontrados para as corridas selecionadas. "
                        "Verifique se `activity_laps_consolidated.csv` esta atualizado.")

        # ── SECAO 3: Perfil de elevação — mapa satélite + gráfico ────────────
        if poly_col and not lps_run.empty:
            st.markdown("---")
            st.subheader("⛰️ Perfil de Elevação na Rota")

            _opts_elv = [r["date"] + " — " + r["name"] + " (" + str(r["km"]) + "km)"
                         for r in routes]
            sel_elv = st.selectbox("Selecione uma corrida para ver o perfil",
                                   options=_opts_elv, key="sel_elv") if _opts_elv else None

            if sel_elv:
                r_elv = next((r for r in routes
                              if r["date"] + " — " + r["name"] + " (" + str(r["km"]) + "km)" == sel_elv), None)

                if r_elv and r_elv["coords"]:
                    import ast
                    import numpy as np

                    coords_elv = r_elv["coords"]
                    n_elv      = len(coords_elv)

                    # ── Carrega altitude_stream ───────────────────────────────
                    _usou_stream = False
                    alt_stream_raw = df_map[df_map["id"] == r_elv["id"]]["altitude_stream"].values
                    alts_elv = []

                    if (len(alt_stream_raw) > 0
                            and str(alt_stream_raw[0]) not in ("nan", "None", "")
                            and len(str(alt_stream_raw[0])) > 4):
                        try:
                            alts_raw = ast.literal_eval(str(alt_stream_raw[0]))
                            alts_elv = np.interp(
                                np.linspace(0, 1, n_elv),
                                np.linspace(0, 1, len(alts_raw)),
                                alts_raw
                            ).tolist()
                            _usou_stream = True
                        except Exception:
                            pass

                    # ── Fallback: estimativa pelos laps ───────────────────────
                    if not _usou_stream:
                        laps_elv = lps_run[lps_run["activity_id"] == r_elv["id"]].sort_values("lap_index")
                        alts_elv = [0.0] * n_elv
                        if not laps_elv.empty:
                            n_pts_elv    = n_elv - 1
                            total_km_elv = laps_elv["distance_km"].sum()
                            cum_alt, cum_frac = 0.0, 0.0
                            for _, lap in laps_elv.iterrows():
                                frac = float(lap["distance_km"]) / total_km_elv if total_km_elv > 0 else 1 / len(laps_elv)
                                gain = float(lap.get("total_elevation_gain") or 0)
                                i0 = int(cum_frac * n_pts_elv)
                                i1 = min(n_pts_elv, int((cum_frac + frac) * n_pts_elv) + 1)
                                for i in range(i0, i1 + 1):
                                    progress = (i - i0) / max(1, i1 - i0)
                                    alts_elv[i] = cum_alt + gain * progress
                                cum_alt  += gain
                                cum_frac += frac

                    alt_min_v = min(alts_elv)
                    alt_max_v = max(alts_elv)

                    # ── Mapa Folium com satélite + rota colorida por altitude ──
                    if HAS_FOLIUM:
                        lat_c_elv = float(np.mean([c[0] for c in coords_elv]))
                        lng_c_elv = float(np.mean([c[1] for c in coords_elv]))

                        ESRI_SAT_URL  = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                        ESRI_SAT_ATTR = "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"

                        tile_elv = st.radio(
                            "Mapa base do perfil",
                            ["Satélite", "Topo ESRI", "Topográfico"],
                            horizontal=True,
                            key="tile_elv"
                        )

                        m_elv = folium.Map(location=[lat_c_elv, lng_c_elv],
                                           zoom_start=14, tiles=None)

                        if tile_elv == "Satélite":
                            folium.TileLayer(ESRI_SAT_URL, attr=ESRI_SAT_ATTR,
                                             name="Satélite").add_to(m_elv)
                        elif tile_elv == "Topo ESRI":
                            folium.TileLayer(
                                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
                                attr="Esri", name="Topo ESRI").add_to(m_elv)
                        else:
                            folium.TileLayer("OpenTopoMap", name="Topográfico").add_to(m_elv)

                        # Desenha segmentos coloridos por altitude
                        alt_range_v = (alt_max_v - alt_min_v) or 1
                        for i in range(len(coords_elv) - 1):
                            t = (alts_elv[i] - alt_min_v) / alt_range_v
                            r_c = round(46  + t * 185)
                            g_c = round(204 - t * 128)
                            b_c = round(113 - t * 53)
                            seg_color = "#{:02X}{:02X}{:02X}".format(r_c, g_c, b_c)
                            folium.PolyLine(
                                [coords_elv[i], coords_elv[i + 1]],
                                color=seg_color, weight=5, opacity=0.92,
                                tooltip=f"{alts_elv[i]:.0f}m"
                            ).add_to(m_elv)

                        # Marcadores de início e fim
                        folium.CircleMarker(
                            coords_elv[0], radius=7, color="#27AE60",
                            fill=True, fill_opacity=1,
                            tooltip=f"Início · {alts_elv[0]:.0f}m"
                        ).add_to(m_elv)
                        folium.CircleMarker(
                            coords_elv[-1], radius=7, color="#E74C3C",
                            fill=True, fill_opacity=1,
                            tooltip=f"Fim · {alts_elv[-1]:.0f}m"
                        ).add_to(m_elv)

                        # Marcador do ponto mais alto
                        idx_max = int(np.argmax(alts_elv))
                        folium.Marker(
                            coords_elv[idx_max],
                            icon=folium.DivIcon(
                                html=f'<div style="font-size:11px;font-weight:700;color:#fff;'
                                     f'background:#E74C3C;padding:2px 7px;border-radius:10px;'
                                     f'box-shadow:0 1px 4px rgba(0,0,0,.6)">▲ {alts_elv[idx_max]:.0f}m</div>',
                                icon_size=(80, 22), icon_anchor=(40, 11)
                            ),
                            tooltip=f"Ponto mais alto: {alts_elv[idx_max]:.0f}m"
                        ).add_to(m_elv)

                        # Legenda de altitude
                        legend_elv = (
                            "<div style='position:absolute;bottom:36px;right:52px;z-index:9999;"
                            "background:rgba(10,10,10,0.75);padding:9px 13px;border-radius:10px;"
                            "font-size:12px;color:#fff'>"
                            "<div style='font-weight:600;margin-bottom:6px'>Altitude</div>"
                            "<div style='display:flex;align-items:center;gap:6px'>"
                            "<div style='width:60px;height:6px;border-radius:3px;"
                            "background:linear-gradient(to right,#2ECC71,#F1C40F,#E74C3C)'></div></div>"
                            f"<div style='display:flex;justify-content:space-between;width:60px;"
                            f"font-size:10px;margin-top:2px'>"
                            f"<span>{alt_min_v:.0f}m</span><span>{alt_max_v:.0f}m</span></div></div>"
                        )
                        m_elv.get_root().html.add_child(folium.Element(legend_elv))

                        # Cards de métricas acima do mapa
                        gain_total = sum(max(0, alts_elv[i+1] - alts_elv[i])
                                         for i in range(len(alts_elv)-1))
                        loss_total = sum(max(0, alts_elv[i] - alts_elv[i+1])
                                         for i in range(len(alts_elv)-1))
                        ce1, ce2, ce3, ce4 = st.columns(4)
                        ce1.metric("▲ Ponto mais alto",  f"{alt_max_v:.0f} m")
                        ce2.metric("▼ Ponto mais baixo", f"{alt_min_v:.0f} m")
                        ce3.metric("↑ Subida acumulada", f"{gain_total:.0f} m")
                        ce4.metric("↓ Descida acumulada",f"{loss_total:.0f} m")

                        st_folium(m_elv, use_container_width=True, height=480,
                                  returned_objects=[])
                        st.caption(
                            ("✅ Altitude real do GPS · " if _usou_stream else "⚠️ Altitude estimada pelos laps · ")
                            + "Verde = baixo · Vermelho = alto · Hover para ver altitude de cada ponto"
                        )

                    # ── Perfil de elevação 2D (gráfico de linha) ─────────────
                    st.markdown("##### 📈 Perfil de elevação")

                    # Calcula distância acumulada em km
                    import math
                    def _hav_dist(c1, c2):
                        la1, lo1 = math.radians(c1[0]), math.radians(c1[1])
                        la2, lo2 = math.radians(c2[0]), math.radians(c2[1])
                        d = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
                        return 2 * 6371 * math.asin(math.sqrt(d))

                    dist_acum = [0.0]
                    for i in range(1, len(coords_elv)):
                        dist_acum.append(dist_acum[-1] + _hav_dist(coords_elv[i-1], coords_elv[i]))

                    fig_elv = go.Figure()
                    fig_elv.add_scatter(
                        x=dist_acum,
                        y=alts_elv,
                        mode="lines",
                        fill="tozeroy",
                        line=dict(width=0),
                        fillcolor="rgba(231,76,60,0.15)",
                        showlegend=False,
                        hoverinfo="skip",
                    )
                    
                    # Linha colorida por altitude
                    fig_elv.add_scatter(
                        x=dist_acum, y=alts_elv,
                        mode="lines+markers",
                        line=dict(color="#E74C3C", width=2),
                        marker=dict(
                            size=4, color=alts_elv,
                            colorscale=[[0,"#27AE60"],[0.5,"#F1C40F"],[1,"#E74C3C"]],
                            showscale=False,
                        ),
                        showlegend=False,
                        hovertemplate="Km %{x:.2f}<br>Altitude: %{y:.0f}m<extra></extra>",
                    )


                    # Marca o ponto mais alto
                    fig_elv.add_scatter(
                        x=[dist_acum[idx_max]], y=[alts_elv[idx_max]],
                        mode="markers+text",
                        marker=dict(size=10, color="#E74C3C"),
                        text=[f"▲ {alts_elv[idx_max]:.0f}m"],
                        textposition="top center",
                        showlegend=False,
                    )

                    fig_elv.update_layout(
                        xaxis_title="Distância (km)",
                        yaxis_title="Altitude (m)",
                        height=220,
                        margin=dict(l=0, r=0, t=10, b=40),
                        plot_bgcolor="rgba(0,0,0,0)")
                    
                    # 5) Renderiza → só uma vez, no final
                    st.plotly_chart(fig_elv, use_container_width=True)

                    # ── Tabela analítica por km ──────────────────────────────
                    laps_tbl = (
                        lps_run[lps_run["activity_id"] == r_elv["id"]]
                        .sort_values("lap_index")
                        if not lps_run.empty else pd.DataFrame()
                    )
                    _avg_ps_tbl = float(laps_tbl["pace_sec_km"].mean()) \
                        if not laps_tbl.empty and laps_tbl["pace_sec_km"].notna().any() else 0.0
                    _avg_fc_tbl = float(laps_tbl["average_heartrate"].mean()) \
                        if not laps_tbl.empty and laps_tbl["average_heartrate"].notna().any() else 0.0

                    def _fz(fc):
                        if fc<=0: return "—"
                        if fc<137: return "Z1"
                        if fc<165: return "Z2"
                        if fc<175: return "Z3"
                        if fc<185: return "Z4"
                        return "Z5"

                    def _fhex(fc):
                        if fc<=0: return "#888"
                        if fc<137: return "#3498DB"
                        if fc<165: return "#2ECC71"
                        if fc<175: return "#F39C12"
                        if fc<185: return "#E67E22"
                        return "#E74C3C"

                    def _ramp(t):
                        return "rgb({},{},{})".format(
                            round(46+t*185), round(204-t*128), round(113-t*53))

                    # Constrói linhas por km
                    _km_rows = []
                    _cum_g   = 0.0
                    _total_d = dist_acum[-1]

                    for _ki in range(1, int(_total_d) + 2):
                        _ks = float(_ki - 1)
                        _ke = min(float(_ki), _total_d)
                        if _ks >= _total_d - 0.05:
                            break
                        _i0 = next((i for i,d in enumerate(dist_acum) if d >= _ks), 0)
                        _i1 = next((i for i,d in enumerate(dist_acum) if d >= _ke), len(dist_acum)-1)
                        _seg = alts_elv[_i0:_i1+1]
                        if not _seg:
                            continue
                        _g = sum(max(0.0, _seg[j+1]-_seg[j]) for j in range(len(_seg)-1))
                        _l = sum(max(0.0, _seg[j]-_seg[j+1]) for j in range(len(_seg)-1))
                        _a = sum(_seg)/len(_seg)
                        _cum_g += _g

                        if not laps_tbl.empty and (_ki-1) < len(laps_tbl):
                            _lp = laps_tbl.iloc[_ki-1]
                            _ps = float(_lp.get("pace_sec_km") or 0)
                            _fc = float(_lp.get("average_heartrate") or 0)
                        else:
                            _ps, _fc = 0.0, 0.0

                        _km_rows.append({"km":_ki,"ps":_ps,"g":_g,"l":_l,
                                         "a":_a,"cg":_cum_g,"fc":_fc})

                    if _km_rows:
                        _ps_vals = [r["ps"] for r in _km_rows if r["ps"]>0]
                        _ps_min  = min(_ps_vals) if _ps_vals else 240
                        _ps_max  = max(_ps_vals) if _ps_vals else 420
                        _ps_avg  = sum(_ps_vals)/len(_ps_vals) if _ps_vals else 300
                        _gmax    = max(r["g"] for r in _km_rows) or 1.0

                        _rows_html = ""
                        for _r in _km_rows:
                            _t_p = min(1.0, max(0.0, (_r["ps"]-_ps_min)/max(1,_ps_max-_ps_min)))
                            _t_g = min(1.0, _r["g"]/_gmax)
                            _pc  = _ramp(_t_p) if _r["ps"]>0 else "#555"
                            _gc  = _ramp(_t_g)
                            _fcc = _fhex(_r["fc"])
                            _bar = round(min(60, _r["g"]/_gmax*60))
                            _pf  = fmt_pace(_r["ps"]) if _r["ps"]>0 else "—"
                            _fcs = f"{_r['fc']:.0f}" if _r["fc"]>0 else "—"
                            _dlt = int(_r["ps"]-_ps_avg) if _r["ps"]>0 else 0
                            _dc  = "#E74C3C" if _dlt>0 else "#2ECC71"
                            _ds  = "+" if _dlt>0 else ""
                            _rows_html += (
                                "<tr>"
                                f"<td>{_r['km']}</td>"
                                f"<td><span style='display:inline-block;padding:2px 9px;border-radius:100px;"
                                f"background:{_pc}22;color:{_pc};font-size:11px;font-weight:500'>{_pf}</span></td>"
                                f"<td><div style='display:flex;align-items:center;gap:5px'>"
                                f"<div style='width:{_bar}px;height:4px;min-width:2px;background:{_gc};"
                                f"border-radius:2px;flex-shrink:0'></div>"
                                f"<span style='color:{_gc};font-size:11px;font-weight:500'>+{_r['g']:.0f}m</span>"
                                f"</div></td>"
                                f"<td style='font-size:11px'>-{_r['l']:.0f}m</td>"
                                f"<td style='font-size:11px'>{_r['a']:.0f}m</td>"
                                f"<td style='font-size:11px'>{_r['cg']:.0f}m</td>"
                                f"<td><span style='display:inline-block;padding:1px 7px;border-radius:100px;"
                                f"background:{_fcc}22;color:{_fcc};font-size:11px;font-weight:500'>"
                                f"{_fcs} <span style='font-size:10px;opacity:.7'>{_fz(_r['fc'])}</span>"
                                f"</span></td>"
                                "</tr>"
                            )

                        # Linha de totais/médias
                        _tg2  = sum(r["g"] for r in _km_rows)
                        _tl2  = sum(r["l"] for r in _km_rows)
                        _fc2  = sum(r["fc"] for r in _km_rows if r["fc"]>0) / max(1, sum(1 for r in _km_rows if r["fc"]>0))
                        _fca  = _fhex(_fc2)
                        _rows_html += (
                            "<tr style='background:rgba(128,128,128,.06)'>"
                            "<td style='font-size:10px;font-weight:500'>Média/Total</td>"
                            f"<td><span style='display:inline-block;padding:2px 9px;border-radius:100px;"
                            f"border:0.5px solid rgba(128,128,128,.3);font-size:11px'>{fmt_pace(_ps_avg)}</span></td>"
                            f"<td style='font-size:11px;font-weight:500'>+{_tg2:.0f}m</td>"
                            f"<td style='font-size:11px'>-{_tl2:.0f}m</td>"
                            "<td></td><td></td>"
                            f"<td><span style='display:inline-block;padding:1px 7px;border-radius:100px;"
                            f"background:{_fca}22;color:{_fca};font-size:11px;font-weight:500'>"
                            f"{_fc2:.0f} <span style='font-size:10px;opacity:.7'>{_fz(_fc2)}</span>"
                            f"</span></td>"
                            "</tr>"
                        )

                        _legend = (
                            "<div style='display:flex;gap:14px;flex-wrap:wrap;font-size:11px;"
                            "color:var(--color-text-secondary,#888);margin-top:8px'>"
                            + "".join(
                                f"<span style='display:flex;align-items:center;gap:4px'>"
                                f"<span style='width:7px;height:7px;border-radius:50%;background:{c};"
                                f"display:inline-block'></span>{t}</span>"
                                for t,c in [
                                    ("Rápido","#2ECC71"),("Lento","#E74C3C"),
                                    ("Z1 <137bpm","#3498DB"),("Z2 <165","#2ECC71"),
                                    ("Z3 <175","#F39C12"),("Z4 <185","#E67E22"),
                                    ("Z5 ≥185","#E74C3C"),
                                ]
                            )
                            + "</div>"
                        )

                        _tbl_html = (
                            "<style>.kmt2{width:100%;border-collapse:collapse;font-size:12px;"
                            "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;table-layout:fixed}"
                            ".kmt2 th{font-size:10px;font-weight:500;color:var(--color-text-secondary,#888);"
                            "text-transform:uppercase;letter-spacing:.5px;padding:7px 8px;"
                            "border-bottom:0.5px solid rgba(128,128,128,.2);text-align:center;white-space:nowrap}"
                            ".kmt2 th:first-child{text-align:left}"
                            ".kmt2 td{padding:5px 8px;border-bottom:0.5px solid rgba(128,128,128,.08);"
                            "color:var(--color-text-secondary,#aaa);vertical-align:middle}"
                            ".kmt2 td:first-child{text-align:left;font-size:11px;font-weight:500}"
                            ".kmt2 td:not(:first-child){text-align:center}"
                            ".kmt2 tr:hover td{background:rgba(128,128,128,.04)}</style>"
                            "<table class='kmt2'><thead><tr>"
                            "<th style='width:36px'>Km</th>"
                            "<th style='width:78px'>Pace</th>"
                            "<th style='width:105px'>↑ subida</th>"
                            "<th style='width:62px'>↓ descida</th>"
                            "<th style='width:68px'>Alt. média</th>"
                            "<th style='width:62px'>↑ acum.</th>"
                            "<th style='width:78px'>FC</th>"
                            "</tr></thead>"
                            f"<tbody>{_rows_html}</tbody></table>"
                            + _legend
                        )

                        st.markdown("##### 📊 Análise por km")
                        st.markdown(_tbl_html, unsafe_allow_html=True)

                    if _usou_stream:
                        st.caption("")
                    else:
                        st.caption("⚠️ Altitude estimada pelos laps (stream não disponível)")
                    
                    
        
        
        
with tab_hist:
    st.title("📋 Histórico de Corridas")

    if df_run.empty:
        st.info("Nenhuma atividade encontrada para o período selecionado.")
    else:
        busca = st.text_input("🔍 Buscar por nome",
                              placeholder="ex: regenerativo, prova, longo...")
        df_hv = df_run.copy()
        if busca:
            df_hv = df_hv[df_hv["name"].str.contains(busca, case=False, na=False)]

        st.caption(f"{len(df_hv)} atividades no período")

        df_sc = df_hv[df_hv["pace_sec_km"].notna()].copy()
        df_sc["Pace_fmt"]  = df_sc["pace_sec_km"].apply(fmt_pace)
        df_sc["Pace_min"]  = df_sc["pace_sec_km"] / 60
        df_sc["Tempo_fmt"] = (df_sc["moving_time_sec"] / 60).apply(
            lambda x: f"{int(x//60)}h{int(x%60):02d}m" if not pd.isna(x) and x >= 60
            else f"{int(x)}min" if not pd.isna(x) else "—")
        df_sc["Data_str"]  = df_sc["start_date"].dt.strftime("%d/%m/%Y")

        fig = px.scatter(
            df_sc,
            x="start_date", y="Pace_min",
            size="distance_km", size_max=28,
            color="Intensidade" if "Intensidade" in df_sc.columns else None,
            color_discrete_map=INTENSITY_COLORS,
            category_orders={"Intensidade": INTENSITY_ORDER},
            custom_data=["name","Data_str","distance_km","Pace_fmt",
                         "Tempo_fmt","elevation_gain"],
            title="📍 Timeline — tamanho = distância · cor = intensidade",
            labels={"start_date":"","Pace_min":"Pace (min/km)"},
            opacity=0.85,
        )
        fig.update_traces(hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]}<br>"
            "📏 %{customdata[2]:.1f} km  ·  ⚡ %{customdata[3]}/km<br>"
            "⏱️ %{customdata[4]}  ·  ⛰️ %{customdata[5]:.0f} m"
            "<extra></extra>"))
        set_pace_yaxis(fig, df_sc["pace_sec_km"])
        fig.update_layout(xaxis_title="")
        st.plotly_chart(fig, width="stretch")

        st.markdown("---")

        cols = {"start_date":"Data","name":"Atividade","distance_km":"km",
                "moving_time_sec":"Tempo","pace_sec_km":"Pace",
                "average_heartrate":"FC Média","elevation_gain":"Elev (m)",
                "calories":"Calorias"}
        if "Intensidade" in df_hv.columns:
            cols["Intensidade"] = "Intensidade"

        df_tab = df_hv[[c for c in cols if c in df_hv.columns]].copy()
        df_tab = df_tab.sort_values("start_date", ascending=False)
        df_tab["start_date"]        = df_tab["start_date"].dt.strftime("%d/%m/%Y %H:%M")
        df_tab["pace_sec_km"]       = df_tab["pace_sec_km"].apply(fmt_pace)
        df_tab["moving_time_sec"]   = (df_tab["moving_time_sec"] / 60).apply(
            lambda x: f"{int(x//60)}h{int(x%60):02d}m" if not pd.isna(x) and x >= 60
            else f"{int(x)}min" if not pd.isna(x) else "—")
        df_tab["distance_km"]       = df_tab["distance_km"].apply(
            lambda x: f"{x:.2f}" if not pd.isna(x) else "—")
        df_tab["average_heartrate"] = df_tab["average_heartrate"].apply(
            lambda x: f"{int(x)} bpm" if not pd.isna(x) else "—")
        df_tab["elevation_gain"]    = df_tab["elevation_gain"].apply(
            lambda x: f"{x:.0f}" if not pd.isna(x) else "—")
        df_tab["calories"]          = df_tab["calories"].apply(
            lambda x: f"{int(x)}" if not pd.isna(x) else "—")
        df_tab = df_tab.rename(columns=cols)
        st.dataframe(df_tab, hide_index=True, width="stretch")

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
                "Selecione uma corrida para ver os laps:",
                options=df_sel["id"].tolist(),
                format_func=lambda x: df_sel.set_index("id").loc[x, "label"],
            )

            if ativ_selecionada:
                laps_ativ = (lps_run[lps_run["activity_id"] == ativ_selecionada]
                             .sort_values("lap_index").copy())

                act_info = df_hv[df_hv["id"] == ativ_selecionada].iloc[0]
                c1,c2,c3,c4,c5 = st.columns(5)
                c1.metric("📏 Distância",  f"{act_info['distance_km']:.2f} km")
                c2.metric("⚡ Pace Médio", fmt_pace(act_info["pace_sec_km"]))
                c3.metric("❤️ FC Média",
                          f"{act_info['average_heartrate']:.0f} bpm"
                          if not pd.isna(act_info.get("average_heartrate", float("nan"))) else "—")
                c4.metric("⛰️ Elevação",
                          f"{act_info['elevation_gain']:.0f} m"
                          if not pd.isna(act_info.get("elevation_gain", float("nan"))) else "—")
                c5.metric("🔢 Laps",       f"{len(laps_ativ)}")

                st.markdown(f"**{act_info['name']}** — "
                            f"{act_info['start_date'].strftime('%d/%m/%Y %H:%M')}")

                col_a, col_b = st.columns(2)

                with col_a:
                    laps_pace = laps_ativ[
                        laps_ativ["pace_sec_km"].notna() &
                        (laps_ativ["distance_m"] >= 200) &
                        (laps_ativ["pace_sec_km"] <= 480)
                    ].copy()

                    n_ignorados = len(laps_ativ) - len(laps_pace)

                    if laps_pace.empty:
                        st.info("Nenhum lap com distância ≥ 200m encontrado.")
                    else:
                        laps_pace["Pace_fmt"] = laps_pace["pace_sec_km"].apply(fmt_pace)
                        laps_pace["Pace_min"] = laps_pace["pace_sec_km"] / 60
                        laps_pace["Lap"]      = laps_pace["lap_index"].astype(str)
                        p50 = laps_pace["pace_sec_km"].median()
                        laps_pace["cor"] = laps_pace["pace_sec_km"].apply(
                            lambda x: GREEN if x < p50 * 0.97
                            else RED if x > p50 * 1.03 else BLUE)
                        fig = go.Figure(go.Bar(
                            x=laps_pace["Lap"], y=laps_pace["Pace_min"],
                            text=laps_pace["Pace_fmt"], textposition="outside",
                            marker_color=laps_pace["cor"].tolist(),
                            customdata=laps_pace[["distance_m","Pace_fmt"]].values,
                            hovertemplate="Lap %{x}<br>Pace: %{customdata[1]}/km<br>"
                                          "Distância: %{customdata[0]:.0f}m<extra></extra>",
                        ))
                        set_pace_yaxis(fig, laps_pace["pace_sec_km"])
                        titulo = "⚡ Pace por Lap"
                        if n_ignorados > 0:
                            titulo += f" ({n_ignorados} micro-laps ocultados)"
                        fig.update_layout(title=titulo, xaxis_title="Lap")
                        st.plotly_chart(fig, width="stretch")

                with col_b:
                    laps_fc = laps_ativ[laps_ativ["average_heartrate"].notna()].copy()
                    if not laps_fc.empty:
                        fig = go.Figure()
                        fig.add_scatter(
                            x=laps_fc["lap_index"].astype(str),
                            y=laps_fc["average_heartrate"],
                            mode="lines+markers+text",
                            text=laps_fc["average_heartrate"].apply(lambda x: f"{x:.0f}"),
                            textposition="top center",
                            line=dict(color=RED, width=2),
                            marker=dict(size=8),
                            name="FC Média",
                        )
                        if laps_fc["max_heartrate"].notna().any():
                            fig.add_scatter(
                                x=laps_fc["lap_index"].astype(str),
                                y=laps_fc["max_heartrate"],
                                mode="lines",
                                line=dict(color=RED, width=1, dash="dot"),
                                name="FC Máx",
                                opacity=0.5,
                            )
                        fig.update_layout(title="❤️ FC por Lap",
                                          xaxis_title="Lap", yaxis_title="bpm")
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info("FC não disponível para esta atividade.")

                cols_lap = {
                    "lap_index":          "Lap",
                    "distance_m":         "Distância (m)",
                    "moving_time_sec":    "Tempo",
                    "pace_sec_km":        "Pace",
                    "average_heartrate":  "FC Média",
                    "max_heartrate":      "FC Máx",
                    "total_elevation_gain": "Elev (m)",
                    "average_cadence":    "Cadência",
                }
                df_laps_tab = laps_ativ[[c for c in cols_lap if c in laps_ativ.columns]].copy()
                df_laps_tab["pace_sec_km"]     = df_laps_tab["pace_sec_km"].apply(fmt_pace)
                df_laps_tab["moving_time_sec"] = df_laps_tab["moving_time_sec"].apply(
                    lambda x: f"{int(x//60)}:{int(x%60):02d}" if not pd.isna(x) else "—")
                df_laps_tab["distance_m"]      = df_laps_tab["distance_m"].apply(
                    lambda x: f"{x:.0f}" if not pd.isna(x) else "—")
                df_laps_tab["average_heartrate"] = df_laps_tab["average_heartrate"].apply(
                    lambda x: f"{x:.0f}" if not pd.isna(x) else "—")
                df_laps_tab["max_heartrate"]   = df_laps_tab["max_heartrate"].apply(
                    lambda x: f"{x:.0f}" if not pd.isna(x) else "—")
                df_laps_tab["total_elevation_gain"] = df_laps_tab["total_elevation_gain"].apply(
                    lambda x: f"{x:.1f}" if not pd.isna(x) else "—")
                df_laps_tab["average_cadence"] = df_laps_tab["average_cadence"].apply(
                    lambda x: f"{x*2:.0f} spm" if not pd.isna(x) else "—")
                df_laps_tab = df_laps_tab.rename(columns=cols_lap)
                st.dataframe(df_laps_tab, hide_index=True, width="stretch")