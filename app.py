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

# ── Groq LLM assistant ────────────────────────────────────────────────────────
def _groq_build_system(context: str) -> str:
    """Monta o system prompt com perfil do atleta + dados do app."""
    _test_str = st.session_state.get("test_3k_str", "3:28")
    try:
        _tp = _test_str.strip().split(":")
        _test_sec = int(_tp[0]) * 60 + int(_tp[1])
    except Exception:
        _test_sec = 208
    def _s2m(s):
        s = int(s); return str(s // 60) + ":" + str(s % 60).zfill(2)
    _zonas = (
        "Trote/Regenerativo: acima de " + _s2m(_test_sec+85) + "/km\n"
        + "Muito Leve: " + _s2m(_test_sec+65) + " a " + _s2m(_test_sec+85) + "/km\n"
        + "Leve: " + _s2m(_test_sec+45) + " a " + _s2m(_test_sec+65) + "/km\n"
        + "Moderado: " + _s2m(_test_sec+32) + " a " + _s2m(_test_sec+45) + "/km\n"
        + "Moderado-Firme: " + _s2m(_test_sec+20) + " a " + _s2m(_test_sec+32) + "/km\n"
        + "Forte: " + _s2m(_test_sec+10) + " a " + _s2m(_test_sec+20) + "/km\n"
        + "Muito Forte Tiros Longos: " + _test_str + " a " + _s2m(_test_sec+10) + "/km\n"
        + "Muito Forte Tiros Curtos: abaixo de " + _test_str + "/km"
    )
    return (
        "Voce e o assistente de treino do Israel, corredor brasileiro de rua e trail running.\n\n"
        "PERFIL DO ATLETA:\n"
        "- Nome: Israel\n"
        "- Nivel: intermediario-avancado, treina ha ~18 meses com assessoria esportiva\n"
        "- Treinador: especialista em montanha na regiao\n"
        "- Proximo objetivo: Paulo Lopes Trail Run 21K em 01/08/2026\n"
        "  Percurso: 20,4 km / 1.354 m D+ / 5 blocos de subida\n"
        "  Maior subida: S3 (km 6,3-10,1) = 3,8 km / +463 m / 12% medio\n"
        "  Subida final: S5 (km 17-18) = +185 m / 17,7% medio\n\n"
        "TESTE DE 3KM: " + _test_str + "/km\n"
        "ZONAS DE PACE DO TREINADOR:\n" + _zonas + "\n\n"
        "METRICAS DO APP:\n"
        "- CTL: fitness acumulado 42 dias. Meta pre-prova: 55-70\n"
        "- ATL: fadiga 7 dias. TSB = CTL - ATL: forma atual\n"
        "- TSB +5 a +20 = janela de pico; abaixo de -15 = fatigado\n"
        "- ACWR zona segura 0,8-1,3. Acima de 1,5 = risco de lesao\n"
        "- Pace Vertical: metros de desnivel/hora. Quanto maior melhor para montanha\n"
        "- GAP: pace ajustado para terreno plano equivalente\n"
        "- Cadencia ideal 175-185 spm. Abaixo de 170 = passada longa, risco de lesao\n"
        "- Deriva cardiaca acima de 10 bpm = sinal de fadiga ou desidratacao\n\n"
        "PRINCIPIOS DE TREINO:\n"
        "- Regra 80/20: 80% volume em intensidade leve, 20% em alta intensidade\n"
        "- Em montanha: FC manda nas subidas, nao o pace\n"
        "- Caminhar subidas acima de 20% e tecnica, nao fraqueza\n"
        "- Progressao segura: maximo 10% de aumento de volume por semana\n\n"
        "DADOS ATUAIS DO APP (use para responder perguntas especificas):\n"
        + context + "\n\n"
        "ESTILO DE RESPOSTA:\n"
        "- SEMPRE em portugues brasileiro\n"
        "- Direto, pratico e motivador como um bom treinador\n"
        "- Use os dados acima para respostas especificas e personalizadas\n"
        "- Honesto: se algo estiver errado, diga claramente mas de forma construtiva\n"
        "- Maximo 4 paragrafos objetivos. Sem enrolacao.\n"
        "- Quando relevante, mencione a prova de 01/08 como referencia temporal."
    )

# Máximo de turnos enviados à API (controle de tokens no free tier)
_GROQ_MAX_TURNS = 10  # = 5 perguntas + 5 respostas → ~7.350 tokens por request

def _groq_ask(messages: list, context: str, api_key: str) -> str:
    """Envia historico de mensagens para o Groq e retorna a resposta.
    Limita a _GROQ_MAX_TURNS turnos enviados para controlar uso de tokens."""
    import requests as _req
    if not api_key or len(api_key) < 20:
        return "Configure a chave API do Groq no sidebar para usar o assistente."
    _system = _groq_build_system(context)
    # Mantém histórico completo na UI mas envia só os últimos N turnos à API
    _messages_to_send = messages[-_GROQ_MAX_TURNS:] if len(messages) > _GROQ_MAX_TURNS else messages
    try:
        _resp = _req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "system", "content": _system}] + _messages_to_send,
                "max_tokens": 700,
                "temperature": 0.65
            },
            timeout=30
        )
        _resp.raise_for_status()
        return _resp.json()["choices"][0]["message"]["content"]
    except _req.exceptions.Timeout:
        return "Timeout — tente novamente em alguns segundos."
    except Exception as _ex:
        return "Erro ao contatar Groq: " + str(_ex)

def _df_to_ctx_rows(df, cols, labels, max_rows=60):
    """Serializa um dataframe como linhas de texto compactas para contexto LLM."""
    import pandas as _pd
    if df is None or len(df) == 0:
        return "(sem dados)\n"
    rows = []
    header = " | ".join(labels)
    rows.append(header)
    rows.append("-" * len(header))
    for _, r in df.head(max_rows).iterrows():
        parts = []
        for col in cols:
            v = r.get(col, "")
            try:
                if _pd.isna(v):
                    parts.append("—")
                    continue
            except Exception:
                pass
            if isinstance(v, float):
                parts.append(f"{v:.1f}")
            else:
                parts.append(str(v))
        rows.append(" | ".join(parts))
    if len(df) > max_rows:
        rows.append(f"... (+{len(df)-max_rows} atividades não mostradas)")
    return "\n".join(rows) + "\n"

def _groq_widget(tab_name: str, context: str, key_suffix: str):
    _hist_key = "groq_hist_" + key_suffix
    _ctx_key  = "groq_ctx_"  + key_suffix

    # Inicializa histórico e atualiza contexto quando o app recarrega dados
    if _hist_key not in st.session_state:
        st.session_state[_hist_key] = []
    # Contexto dos dados pode mudar (novo período, novo sync) — sempre atualiza
    st.session_state[_ctx_key] = context

    with st.expander("🤖 Assistente de Treino — conversa com histórico", expanded=False):
        # ── Histórico visual ──────────────────────────────────────────────────
        _hist = st.session_state[_hist_key]
        if _hist:
            for _msg in _hist:
                if _msg["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(_msg["content"])
                else:
                    with st.chat_message("assistant", avatar="🏃"):
                        st.markdown(_msg["content"])
            # Botão limpar conversa
            if st.button("🗑️ Limpar conversa", key="groq_clear_" + key_suffix):
                st.session_state[_hist_key] = []
                st.rerun()
        else:
            st.caption("Nenhuma pergunta ainda. Comece a conversa abaixo!")

        # ── Input nova mensagem ───────────────────────────────────────────────
        st.divider()
        _col_input, _col_btn = st.columns([5, 1])
        with _col_input:
            _q = st.text_input(
                "Mensagem:",
                placeholder="Ex: Como estou indo? Qual foi meu treino mais pesado?",
                key="groq_q_" + key_suffix,
                label_visibility="collapsed"
            )
        with _col_btn:
            _send = st.button("Enviar ▶", key="groq_btn_" + key_suffix, use_container_width=True)

        if _send and _q.strip():
            # Adiciona pergunta ao histórico
            st.session_state[_hist_key].append({"role": "user", "content": _q.strip()})
            # Chama Groq com todo o histórico
            with st.spinner("Pensando..."):
                _ans = _groq_ask(
                    st.session_state[_hist_key],
                    st.session_state[_ctx_key],
                    GROQ_KEY
                )
            # Adiciona resposta ao histórico
            st.session_state[_hist_key].append({"role": "assistant", "content": _ans})
            st.rerun()

DIAS_PT  = {"Monday":"Seg","Tuesday":"Ter","Wednesday":"Qua",
            "Thursday":"Qui","Friday":"Sex","Saturday":"Sáb","Sunday":"Dom"}
DIAS_ORDER_PT = ["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"]

MESES_PT = {"Jan":"jan","Feb":"fev","Mar":"mar","Apr":"abr","May":"mai",
            "Jun":"jun","Jul":"jul","Aug":"ago","Sep":"set","Oct":"out",
            "Nov":"nov","Dec":"dez"}

# Zonas fáceis — usado em calc_intensidade_fc
_ZONAS_FACEIS_REL = {"Z1", "Z2", "Sem FC"}

KEYWORDS_INTENSIDADE = {
    "Muito Forte":    ["intervalad","tiro","interval","vo2","muito forte",
                       "repetição","repeticao","série","serie","prova","teste"],
    "Forte":          ["fartlek","forte","threshold","limiar"],
    "Moderado Firme": ["progressiv","ritmado","moderado firme","tempo run"],
    "Moderado":       ["longo","moderado","moder","base","contínuo","continuo",
                       "aeróbic","aerobic"],
    "Leve":           ["regenerat","fácil","facil","easy","recovery",
                       "leve","caminhad","walk","solto"],
}

# ── Regex pré-compilados (uma vez no startup) ─────────────────────────────────
_KW_COMPILED: dict[str, re.Pattern] = {
    intensity: re.compile(
        "|".join(re.escape(kw) for kw in kws), re.IGNORECASE
    )
    for intensity, kws in KEYWORDS_INTENSIDADE.items()
}

# ── Helpers ───────────────────────────────────────────────────────────────────
MESES_PT = {"Jan":"jan","Feb":"fev","Mar":"mar","Apr":"abr","May":"mai",
            "Jun":"jun","Jul":"jul","Aug":"ago","Sep":"set","Oct":"out",
            "Nov":"nov","Dec":"dez"}

def mesano_pt(dt_series):
    return dt_series.dt.strftime("%b %Y").apply(
        lambda x: f"{MESES_PT.get(x[:3], x[:3])} {x[4:]}" if isinstance(x, str) and len(x) >= 4 else "")

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
    """EMA sobre training_load diário (TRIMP) → CTL / ATL / TSB. Cacheado."""
    if "training_load" not in df_run_all.columns or df_run_all["training_load"].isna().all():
        return pd.DataFrame()
    ss = (df_run_all[df_run_all["training_load"].notna()]
          .set_index("start_date")["training_load"]
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
    """Pace do bloco principal.

    Para treinos contínuos: remove aquec/desaquec (primeiro/último lap lento).
    Para treinos intervalados (alta variância de pace): filtra laps de recuperação,
    mantendo apenas os laps rápidos (≤ 120% da mediana dos laps do bloco).
    """
    if laps_group.empty:
        return None
    laps = laps_group.sort_values("lap_index")
    laps = laps[laps["pace_sec_km"].notna() & (laps["pace_sec_km"] > 0) & (laps["pace_sec_km"] < 500)]
    if len(laps) == 0:
        return None
    if len(laps) <= 2:
        return float(laps["pace_sec_km"].mean())

    mediana = float(laps["pace_sec_km"].median())

    # Remove aquec/desaquec (só primeiro e último lap se >15% mais lentos)
    if len(laps) > 3 and float(laps.iloc[0]["pace_sec_km"]) > mediana * 1.15:
        laps = laps.iloc[1:]
    if len(laps) > 2 and float(laps.iloc[-1]["pace_sec_km"]) > mediana * 1.15:
        laps = laps.iloc[:-1]
    if laps.empty:
        return None

    # Detecta treino intervalado: coef. de variação alto → existem laps de recuperação
    mediana2 = float(laps["pace_sec_km"].median())
    cv = float(laps["pace_sec_km"].std() / mediana2) if mediana2 > 0 else 0
    if cv > 0.12 and len(laps) >= 4:
        # Intervalado: manter apenas laps rápidos (≤ 120% da mediana = sem recuperações)
        laps_rapidos = laps[laps["pace_sec_km"] <= mediana2 * 1.20]
        if len(laps_rapidos) >= 2:
            laps = laps_rapidos

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
        act["MesAnoOrd"]  = act["start_date"].dt.year * 12 + act["start_date"].dt.month
        act["Semana"]     = (act["start_date"] - pd.Timestamp("2020-01-01")).dt.days // 7
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
        laps["MesAnoOrd"]  = laps["start_date"].dt.year * 12 + laps["start_date"].dt.month

    be = pd.DataFrame()  # best_efforts removido — não usado com Garmin

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

# ── Sanitize cached date_range: clamp to [min_d, max_d] to avoid Streamlit
#    raising StreamlitAPIException when data changes between sessions ──────
def _clamp_date_range(key, _min, _max):
    v = st.session_state.get(key)
    if v is None:
        return
    try:
        if isinstance(v, (tuple, list)) and len(v) == 2:
            s = max(_min, min(_max, v[0]))
            e = max(_min, min(_max, v[1]))
            if s > e:
                s, e = _min, _max
            st.session_state[key] = (s, e)
        else:
            del st.session_state[key]
    except Exception:
        if key in st.session_state:
            del st.session_state[key]

_clamp_date_range("date_range", min_d, max_d)

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
        st.rerun()

# Use key= only; Streamlit uses session_state[key] as the current value.
# Do NOT pass value= when key= is set — avoids the dual-source conflict.
if "date_range" not in st.session_state:
    st.session_state["date_range"] = (min_d, max_d)

date_range = st.sidebar.date_input(
    "Período",
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
_test_3k_default = st.session_state.get("test_3k_str", "3:28")
TEST_3K_STR = st.sidebar.text_input(
    "🏁 Pace do Teste 3km (MM:SS)",
    value=_test_3k_default,
    key="test_3k_str",
    help="Pace médio do seu último teste de 3km. "
         "Define as zonas de intensidade do treinador no gráfico de laps."
)
def _parse_test_3k(s):
    """Converte MM:SS → segundos/km. Retorna None se inválido."""
    try:
        parts = s.strip().split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:      # HH:MM:SS
            return int(parts[1]) * 60 + int(parts[2])
    except Exception:
        pass
    return None

TEST_3K_SEC = _parse_test_3k(TEST_3K_STR)
if TEST_3K_SEC:
    st.sidebar.caption(
        f"Zonas ativas para teste {TEST_3K_STR}/km  ·  "
        f"Trote ≥ {TEST_3K_SEC+80}s · Leve {TEST_3K_SEC+50}–{TEST_3K_SEC+60}s · "
        f"Forte {TEST_3K_SEC+15}–{TEST_3K_SEC+20}s/km"
    )
else:
    st.sidebar.warning("Formato inválido — use MM:SS (ex: 3:28)")
    TEST_3K_SEC = None

def _pace_zone(pace_sec, test_sec):
    """Retorna (nome_zona, cor_hex) baseado no teste 3km do treinador.
    pace_sec = pace da atividade em seg/km
    test_sec = pace do teste 3km em seg/km
    Escala: verde (fácil) → amarelo → laranja → vermelho (muito forte)
    """
    if test_sec is None:
        return ("—", "#3498DB")
    d = pace_sec - test_sec   # positivo = mais lento que o teste (mais fácil)
    if   d <= 0:              return ("Muito Forte – Tiros Curtos",  "#C0392B")   # vermelho escuro
    elif d <= 10:             return ("Muito Forte – Tiros Longos",  "#E74C3C")   # vermelho
    elif d <= 20:             return ("Forte",                        "#E67E22")   # laranja
    elif d <= 32:             return ("Moderado–Firme",               "#F39C12")   # âmbar
    elif d <= 45:             return ("Moderado",                     "#F9E79F")   # amarelo claro
    elif d <= 65:             return ("Leve",                         "#ABEBC6")   # verde claro
    elif d <= 85:             return ("Muito Leve",                   "#2ECC71")   # verde
    else:                     return ("Trote/Regenerativo",           "#1ABC9C")   # verde-água

st.sidebar.markdown("---")
GROQ_KEY = st.sidebar.text_input(
    "🤖 Groq API Key",
    value=st.session_state.get("groq_key_val", ""),
    type="password",
    key="groq_key_val",
    help="Chave da API Groq (console.groq.com). Necessária para o Assistente de Treino em cada aba."
)

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
# best_efforts / melhor_be / melhor_3km removidos (não disponíveis via Garmin)


@st.cache_data(ttl=86400*30, show_spinner=False)
def _reverse_geocode(lat_r: float, lng_r: float) -> str:
    """
    Bairro, Cidade via Nominatim (OSM). Sem API key.
    Cacheado 30 dias — só chama a API na 1ª vez por localização.
    lat_r / lng_r: arredondados para 2 casas (~1 km de precisão).
    """
    try:
        import requests
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat_r, "lon": lng_r, "format": "json", "zoom": 15},
            headers={"User-Agent": "PerformanceRunDashboard/1.0"},
            timeout=5,
        )
        if r.status_code == 200:
            addr = r.json().get("address", {})
            bairro = (addr.get("suburb") or addr.get("neighbourhood")
                      or addr.get("quarter") or addr.get("city_district") or "")
            cidade = (addr.get("city") or addr.get("town")
                      or addr.get("village") or addr.get("municipality") or "")
            parts = [p for p in [bairro, cidade] if p]
            return ", ".join(parts)
    except Exception:
        pass
    return ""


def _start_coords(row, poly_col, has_ll):
    """Extrai lat/lng do ponto de início da atividade."""
    if poly_col and pd.notna(row.get(poly_col, None)):
        pts = decode_polyline(str(row[poly_col]))
        if pts:
            return pts[0][0], pts[0][1]
    if has_ll:
        lat = row.get("latitude")
        lng = row.get("longitude")
        if pd.notna(lat) and pd.notna(lng):
            return float(lat), float(lng)
    return None, None


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

tab_hoje, tab_desemp, tab_carga, tab_mapa, tab_hist, tab_sugerir, tab_plano = st.tabs([
    "🏠 Dashboard",
    "⚡ Desempenho",
    "💓 Carga & Zonas",
    "🗺️ Mapa",
    "📋 Histórico",
    "🎯 Sugerir Rota",
    "📅 Plano",
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

    carga_aguda   = float(ult7d["training_load"].sum())  if "training_load" in _runs_raw.columns else 0.0
    carga_cronica = float(ult28d["training_load"].sum()) / 4 if "training_load" in _runs_raw.columns else 0.0
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
    mg3.metric("🏔️ Elev. este mês", f"{int(df_run[df_run['start_date'].dt.to_period('M') == pd.Timestamp.now().to_period('M')]['elevation_gain'].sum())} m")
    mg4.metric("💓 FC Média",       f"{df_run['average_heartrate'].mean():.0f} bpm" if df_run["average_heartrate"].notna().any() else "—")

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

    # ── Assistente IA ─────────────────────────────────────────────────────────
    # ── contexto rico para o assistente de dashboard ──────────────────────────
    _df_dash_ctx = df_run.copy()
    _df_dash_ctx["Data"]      = _df_dash_ctx["start_date"].dt.strftime("%d/%m/%Y")
    _df_dash_ctx["Pace_fmt"]  = fmt_pace_vec(_df_dash_ctx["pace_sec_km"])
    _df_dash_ctx["FC"]        = (_df_dash_ctx["average_heartrate"].fillna(0).astype(int).apply(str)
                                  if "average_heartrate" in _df_dash_ctx.columns else "—")
    _df_dash_ctx["Elev"]      = (_df_dash_ctx["elevation_gain"].fillna(0).apply(lambda x: f"{int(x)}m")
                                  if "elevation_gain" in _df_dash_ctx.columns else "—")
    # volume por semana (últimas 16)
    _vol_sem = (df_run.set_index("start_date")["distance_km"]
                .resample("W").sum().tail(16).reset_index())
    _vol_sem_txt = "\n".join(
        f"  Semana {r['start_date'].strftime('%d/%m')}: {r['distance_km']:.1f} km"
        for _, r in _vol_sem.iterrows()
    )
    # volume por mês
    _vol_mes = (df_run.assign(mes=df_run["start_date"].dt.to_period("M"))
                .groupby("mes")["distance_km"].sum().tail(12))
    _vol_mes_txt = "\n".join(f"  {str(m)}: {v:.0f} km" for m, v in _vol_mes.items())

    _ctx_dash = (
        f"RESUMO DO PERÍODO: {s_dt.date()} a {e_dt.date()}\n"
        f"Total atividades: {len(df_run)} | Distância: {df_run['distance_km'].sum():.1f} km\n"
        f"Pace médio: {fmt_pace(df_run['pace_sec_km'].mean()) if df_run['pace_sec_km'].notna().any() else 'N/A'}/km\n"
        + (f"FC média: {df_run['average_heartrate'].mean():.0f} bpm\n" if "average_heartrate" in df_run.columns and df_run["average_heartrate"].notna().any() else "")
        + (f"ACWR (7d/4s): {ult7d['distance_km'].sum() / max(1, _runs_raw[(hoje-timedelta(days=35)<=_runs_raw['start_date']) & (_runs_raw['start_date']<hoje-timedelta(days=7))]['distance_km'].sum()/4):.2f}\n" if not ult7d.empty else "")
        + f"KM últimos 7 dias: {ult7d['distance_km'].sum():.1f} km\n"
        f"Mês atual (KM): {df_run[df_run['start_date'].dt.month == hoje.month]['distance_km'].sum():.1f} km\n"
        f"\nVOLUME POR SEMANA (últimas 16):\n{_vol_sem_txt}\n"
        f"\nVOLUME POR MÊS (últimos 12):\n{_vol_mes_txt}\n"
        f"\nÚLTIMAS 30 ATIVIDADES (Data | Nome | Dist | Pace | FC | Elevação):\n"
        + _df_to_ctx_rows(
            _df_dash_ctx.sort_values("start_date", ascending=False).head(30),
            ["Data", "name", "distance_km", "Pace_fmt", "FC", "Elev"],
            ["Data", "Nome", "Dist(km)", "Pace", "FC(bpm)", "Elev"],
            max_rows=30)
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  🏅 GARMIN INSIGHTS — dados exclusivos do relógio
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🏅 Garmin Insights")

    @st.cache_data(ttl=3600, show_spinner=False)
    def _load_garmin_insights():
        import json, os
        p = os.path.join(BASE, "garmin_insights.json")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)

    _gi = _load_garmin_insights()

    if _gi:
        _upd = _gi.get("updated_at", "")
        st.caption(f"Dados Garmin Connect · atualizado em {_upd} · rode `python sync.py` para atualizar")

        # ── Race Predictions ──────────────────────────────────────────────────
        st.markdown("**🎯 Previsões de Prova** *(baseado no VO₂max e treinos recentes)*")
        rp = _gi.get("race_predictions", {})
        _rp_cols = st.columns(4)
        for _col, (_dist, _label, _emoji) in zip(_rp_cols, [
            ("5K",            "5 km",       "⚡"),
            ("10K",           "10 km",      "🏃"),
            ("half_marathon", "Meia Marat.", "🏅"),
            ("marathon",      "Maratona",   "🏆"),
        ]):
            _pred = rp.get(_dist, {})
            _t    = _pred.get("time", "—")
            _pace_s = _pred.get("time_seconds", 0)
            # Compute pace (sec/km) for context
            _dist_km = {"5K": 5, "10K": 10, "half_marathon": 21.0975, "marathon": 42.195}[_dist]
            _pace_fmt = fmt_pace(_pace_s / _dist_km) if _pace_s else "—"
            _col.metric(f"{_emoji} {_label}", _t, f"⌛ {_pace_fmt}/km", delta_color="off")

        st.markdown("")

        # ── VO2max + RHR ──────────────────────────────────────────────────────
        _v2_col, _rhr_col, _bb_col = st.columns([1, 1, 2])

        with _v2_col:
            _vo2 = _gi.get("vo2max", {})
            _vo2_cur  = _vo2.get("current", None)
            _vo2_chg  = _vo2.get("change", 0)
            _vo2_delta = f"+{_vo2_chg:.1f}" if _vo2_chg >= 0 else f"{_vo2_chg:.1f}"
            if _vo2_cur:
                # VO2max fitness category
                if   _vo2_cur >= 60: _vo2_cat = "Elite"
                elif _vo2_cur >= 55: _vo2_cat = "Superior"
                elif _vo2_cur >= 49: _vo2_cat = "Excelente"
                elif _vo2_cur >= 43: _vo2_cat = "Bom"
                else:                _vo2_cat = "Médio"
                st.metric("🫁 VO₂max", f"{_vo2_cur:.0f} ml/kg/min",
                          f"{_vo2_delta} · {_vo2_cat}", delta_color="normal")
            else:
                st.metric("🫁 VO₂max", "—")

        with _rhr_col:
            _rhr = _gi.get("rhr_today")
            if _rhr:
                if   _rhr <= 45: _rhr_cat = "Atleta"
                elif _rhr <= 52: _rhr_cat = "Excelente"
                elif _rhr <= 59: _rhr_cat = "Bom"
                else:            _rhr_cat = "Normal"
                st.metric("❤️ FC Repouso", f"{_rhr} bpm", _rhr_cat, delta_color="off")
            else:
                st.metric("❤️ FC Repouso", "—")

        # ── Body Battery 30d ─────────────────────────────────────────────────
        with _bb_col:
            _bb_data = _gi.get("body_battery", [])
            if _bb_data:
                _bb_df = pd.DataFrame(_bb_data)
                _bb_df["date"] = pd.to_datetime(_bb_df["date"])
                _bb_df = _bb_df.tail(14)  # last 14 days

                # Saldo líquido: verde se carregou mais do que gastou, vermelho se não
                _bb_df["saldo"] = _bb_df["charged"].fillna(0) - _bb_df["drained"].fillna(0)
                _bb_colors = [GREEN if v >= 0 else RED for v in _bb_df["saldo"]]

                _fig_bb = go.Figure()
                _bb_df["hover"] = _bb_df.apply(
                    lambda r: f"Carregado: {int(r.get('charged',0) or 0)}<br>Gasto: {int(r.get('drained',0) or 0)}<br>Saldo: {int(r['saldo'])}",
                    axis=1
                )
                _fig_bb.add_bar(
                    x=_bb_df["date"].dt.strftime("%d/%m"),
                    y=_bb_df["saldo"],
                    marker_color=_bb_colors,
                    text=_bb_df["saldo"].apply(lambda v: f"+{int(v)}" if v >= 0 else str(int(v))),
                    textposition="outside",
                    customdata=_bb_df["hover"],
                    hovertemplate="<b>%{x}</b><br>%{customdata}<extra></extra>",
                    showlegend=False,
                )
                _fig_bb.add_hline(y=0, line_color="gray", line_width=1)
                _fig_bb.update_layout(
                    title="🔋 Body Battery — saldo diário (carregado − gasto)",
                    height=230,
                    margin=dict(t=40, b=10, l=0, r=0),
                    yaxis=dict(title="", zeroline=False),
                )
                st.plotly_chart(_fig_bb, use_container_width=True)
    else:
        st.info("📡 Dados Garmin não encontrados. Rode `python sync.py` localmente para gerar `garmin_insights.json`.")


    _groq_widget("Dashboard", _ctx_dash, "dash")

# ══════════════════════════════════════════════════════════════════════════════
#  2 · DESEMPENHO  —  PRs, pace, eficiência, condições
# ══════════════════════════════════════════════════════════════════════════════
with tab_desemp:
    st.title("⚡ Desempenho")

    # ── Cards de desempenho (Garmin) ──────────────────────────────────────────
    _best_pace = df_run["pace_sec_km"].min() if df_run["pace_sec_km"].notna().any() else None
    _avg_eff   = df_run["efficiency_index"].mean() if df_run["efficiency_index"].notna().any() else None
    _total_km  = df_run["distance_km"].sum()
    _avg_load  = df_run["training_load"].mean() if df_run["training_load"].notna().any() else None
    _avg_cad   = df_run["average_cadence"].mean() if df_run["average_cadence"].notna().any() else None
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("⚡ Melhor Pace",   fmt_pace(_best_pace) if _best_pace else "—")
    c2.metric("📊 Eficiência",    f"{_avg_eff:.4f}" if _avg_eff else "—", help="dist_km / (fc * min) × 1000")
    c3.metric("📏 Total KM",      f"{_total_km:,.0f} km")
    c4.metric("🔥 Carga Média",   f"{_avg_load:.1f}" if _avg_load else "—", help="TRIMP médio por treino")
    c5.metric("🦵 Cadência Média",f"{_avg_cad:.0f} spm" if _avg_cad else "—")
    st.markdown("---")

    col_a, col_b = st.columns(2)

    # ── Pace médio por intensidade (bloco principal) ──────────────────────────
    with col_a:
        if "Intensidade" in df_run.columns and not lps_run.empty:
            _mp = (lps_run.groupby("activity_id")
                   .apply(compute_main_laps_pace).dropna().reset_index())
            _mp.columns = ["id","pace_main"]
            # Somente atividades COM dados de lap — sem fallback para pace médio geral
            df_run_p = df_run.merge(_mp, on="id", how="inner")
            df_run_p["pace_plot"] = df_run_p["pace_main"]
            _n_lap_acts = len(df_run_p)
        else:
            df_run_p = pd.DataFrame()
            _n_lap_acts = 0

        if not df_run_p.empty:
            df_b = cat_intensity(df_run_p[df_run_p["pace_plot"].notna()].copy())
            df_agg = (df_b.groupby("Intensidade", observed=True)["pace_plot"]
                         .agg(Media="mean", DP="std").reset_index().dropna())
            _n_per_cat = df_b.groupby("Intensidade", observed=True).size().to_dict()
        else:
            df_agg = pd.DataFrame()
        if not df_agg.empty:
            df_agg["Media_min"] = df_agg["Media"] / 60
            df_agg["DP_min"]    = df_agg["DP"] / 60
            df_agg["Label"]     = fmt_pace_vec(df_agg["Media"])
            df_agg["Cor"]       = df_agg["Intensidade"].map(INTENSITY_COLORS)
            df_agg["n"]         = df_agg["Intensidade"].map(_n_per_cat).fillna(0).astype(int)
            fig = go.Figure()
            for _, row in df_agg.iterrows():
                fig.add_bar(x=[row["Intensidade"]], y=[row["Media_min"]],
                            error_y=dict(type="data", array=[row["DP_min"]], visible=True),
                            marker_color=row["Cor"], name=row["Intensidade"],
                            text=row["Label"], textposition="outside",
                            hovertemplate=f"<b>{row['Intensidade']}</b><br>Pace: {row['Label']}/km<br>n={row['n']} atividades<extra></extra>")
            set_pace_yaxis(fig, df_agg["Media"])
            fig.update_layout(
                title="🎯 Pace por Tipo de Treino",
                showlegend=False,
                annotations=[dict(
                    text=f"ℹ️ Calculado com {_n_lap_acts} atividades que possuem dados de lap",
                    xref="paper", yref="paper", x=0, y=-0.18,
                    showarrow=False, font=dict(size=11, color="gray"), xanchor="left"
                )]
            )
            st.plotly_chart(fig, width="stretch")
            st.caption("Pace do bloco principal (laps de recuperação excluídos). Barra = desvio padrão. Hover = nº de atividades por categoria.")
        else:
            st.info("Sem dados de lap disponíveis para calcular pace por tipo de treino.")

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
            # ── métricas derivadas ──────────────────────────────────────────────────
            df_e["elev_km"]   = df_e["elevation_gain"] / df_e["distance_km"]
            # Pace Vertical: metros de desnível positivo por hora de movimento
            _mt_col = "moving_time_sec" if "moving_time_sec" in df_e.columns else                       "moving_time_s"   if "moving_time_s"   in df_e.columns else None
            if _mt_col:
                df_e["vert_pace"] = df_e.apply(
                    lambda r: r["elevation_gain"] / (r[_mt_col] / 3600)
                    if pd.notna(r[_mt_col]) and r[_mt_col] > 0 else np.nan, axis=1)
            else:
                df_e["vert_pace"] = np.nan
            # GAP — Grade Adjusted Pace: normaliza o pace para equivalente plano
            # fórmula: gap = pace_sec / (1 + 0.033·g + 0.00012·g³)  onde g = grade %
            df_e["grade_pct"] = df_e["elev_km"] / 10          # m/km → % (÷ 10)
            df_e["gap_sec"]   = df_e.apply(
                lambda r: r["pace_sec_km"] / max(0.5, 1 + 0.033 * r["grade_pct"]
                                                     + 0.00012 * r["grade_pct"] ** 3)
                if pd.notna(r["pace_sec_km"]) else np.nan, axis=1)

            # ── KPI cards ──────────────────────────────────────────────────────────
            ec1, ec2, ec3, ec4 = st.columns(4)
            ec1.metric("⛰️ Elevação Total", f"{df_e['elevation_gain'].sum():,.0f} m")
            ec2.metric("📈 Maior Subida",   f"{df_e['elevation_gain'].max():.0f} m")
            vp_med = df_e["vert_pace"].median()
            ec3.metric("⚡ Pace Vertical",
                       f"{vp_med:.0f} m/h" if pd.notna(vp_med) else "—",
                       help="Mediana de metros de desnível positivo por hora — "
                            "quanto maior, mais eficiente nas subidas.")
            valid_gap = df_e[df_e["gap_sec"].notna() & df_e["pace_sec_km"].notna()]
            if not valid_gap.empty:
                custo = (valid_gap["pace_sec_km"] - valid_gap["gap_sec"]).mean()
                ec4.metric("🎯 Custo de Subida", f"+{custo:.0f} s/km",
                           help="Quanto a elevação adiciona ao pace real vs GAP (pace plano equivalente). "
                                "Maior = subidas mais custosas.")
            else:
                ec4.metric("🏔️ Runs >300 m", f"{len(df_e[df_e['elevation_gain'] >= 300])}")

            # ── linha 1: GAP scatter  |  acumulado por período ─────────────────────
            col_e1, col_e2 = st.columns(2)

            with col_e1:
                df_gap = df_e[df_e["gap_sec"].notna() & df_e["pace_sec_km"].notna()].copy()
                if not df_gap.empty:
                    df_gap["Pace_min"] = df_gap["pace_sec_km"] / 60
                    df_gap["GAP_min"]  = df_gap["gap_sec"] / 60
                    fig_gap = go.Figure()
                    fig_gap.add_scatter(
                        x=df_gap["Pace_min"], y=df_gap["GAP_min"],
                        mode="markers",
                        marker=dict(color=df_gap["elevation_gain"], colorscale="RdYlGn_r",
                                    size=8, opacity=0.75,
                                    colorbar=dict(title="Elevação<br>(m)", x=1.02)),
                        text=df_gap.apply(
                            lambda r: f"{r['name']}<br>{r['start_date'].strftime('%d/%m/%y')}", axis=1),
                        hovertemplate="Pace: %{x:.2f} min/km<br>GAP: %{y:.2f} min/km<br>%{text}<extra></extra>")
                    # diagonal: pace == GAP  (terreno plano)
                    mn = min(df_gap["Pace_min"].min(), df_gap["GAP_min"].min())
                    mx = max(df_gap["Pace_min"].max(), df_gap["GAP_min"].max())
                    fig_gap.add_scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                        line=dict(color=GRAY, dash="dash", width=1),
                                        showlegend=False)
                    fig_gap.update_layout(title="Pace Real vs GAP",
                                          xaxis_title="Pace Real (min/km)",
                                          yaxis_title="GAP — Plano Equivalente (min/km)",
                                          showlegend=False)
                    set_pace_yaxis(fig_gap, df_gap["pace_sec_km"])
                    fig_gap.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig_gap, width="stretch")
                    st.caption("Pontos abaixo da diagonal = subidas com alto custo de pace. "
                               "GAP normaliza para terreno plano equivalente.")

            with col_e2:
                df_e["_week"]  = df_e["start_date"].dt.to_period("W").dt.start_time
                df_e["_month"] = df_e["start_date"].dt.to_period("M").dt.start_time
                _period_opts   = {"Semana": "_week", "Mês": "_month"}
                _period_sel    = st.radio("Acumulado por", list(_period_opts.keys()),
                                          horizontal=True, key="elev_period")
                _pcol          = _period_opts[_period_sel]
                df_trend = df_e.groupby(_pcol).agg(
                    elev_total=("elevation_gain", "sum"),
                    n_runs=("elevation_gain", "count")
                ).reset_index().rename(columns={_pcol: "periodo"})
                fig_trend = go.Figure()
                fig_trend.add_bar(x=df_trend["periodo"], y=df_trend["elev_total"],
                                  name="Elevação total", marker_color=BLUE, opacity=0.75,
                                  hovertemplate="<b>%{x|%d/%m/%y}</b><br>%{y:.0f} m<extra></extra>")
                if len(df_trend) >= 4:
                    df_trend["rolling"] = df_trend["elev_total"].rolling(4, min_periods=1).mean()
                    fig_trend.add_scatter(x=df_trend["periodo"], y=df_trend["rolling"],
                                          mode="lines", name="Média móvel (4×)",
                                          line=dict(color=RED, width=2))
                fig_trend.update_layout(title=f"Acumulado de Elevação por {_period_sel}",
                                        yaxis_title="Elevação (m)",
                                        legend=dict(orientation="h", y=-0.22))
                st.plotly_chart(fig_trend, width="stretch")

            # ── linha 2: eficiência ao longo do tempo  |  Pace×FC×Elevação ─────────
            col_e3, col_e4 = st.columns(2)

            with col_e3:
                df_vp = df_e[df_e["vert_pace"].notna()].copy()
                if not df_vp.empty:
                    df_vp["_month"] = df_vp["start_date"].dt.to_period("M").dt.start_time
                    df_eff = df_vp.groupby("_month").agg(
                        vp_med=("vert_pace", "median"),
                        n=("vert_pace", "count")
                    ).reset_index()
                    fig_eff = go.Figure()
                    fig_eff.add_scatter(
                        x=df_eff["_month"], y=df_eff["vp_med"],
                        mode="lines+markers", name="Pace Vertical (m/h)",
                        line=dict(color=GREEN, width=2.5),
                        marker=dict(size=(df_eff["n"].clip(3, 10) * 1.5).astype(float)),
                        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:.0f} m/h<extra></extra>")
                    # tendência linear
                    if len(df_eff) >= 3:
                        import numpy as _np
                        x_num = (_np.arange(len(df_eff))).astype(float)
                        coef  = _np.polyfit(x_num, df_eff["vp_med"].values, 1)
                        trend = _np.polyval(coef, x_num)
                        fig_eff.add_scatter(x=df_eff["_month"], y=trend,
                                            mode="lines", name="Tendência",
                                            line=dict(color=AMBER if coef[0] < 0 else GREEN,
                                                      dash="dot", width=1.5))
                    fig_eff.update_layout(
                        title="Eficiência em Subidas — evolução",
                        yaxis_title="Pace Vertical (m/h)",
                        legend=dict(orientation="h", y=-0.22))
                    st.plotly_chart(fig_eff, width="stretch")
                    st.caption("Subindo = ficando mais eficiente nas subidas ao longo do tempo.")
                else:
                    st.info("Dados de tempo de movimento insuficientes para calcular pace vertical.")

            with col_e4:
                _hr_col = "average_heartrate" if "average_heartrate" in df_e.columns else                           "avg_hr" if "avg_hr" in df_e.columns else None
                _has_hr = _hr_col is not None
                df_pfc = df_e[df_e["pace_sec_km"].notna() & df_e[_hr_col].notna()].copy() if _has_hr else pd.DataFrame()
                if _has_hr and _hr_col != "avg_hr":
                    df_pfc = df_pfc.rename(columns={_hr_col: "avg_hr"})
                if not df_pfc.empty:
                    df_pfc["Pace_min"] = df_pfc["pace_sec_km"] / 60
                    _zone_map = {
                        "Z1 — Recuperação":  "#2ecc71",
                        "Z2 — Base aeróbica": "#3498db",
                        "Z3 — Tempo":         "#f39c12",
                        "Z4 — Limiar":        "#e67e22",
                        "Z5 — VO2max":        "#e74c3c",
                    }
                    def _hr_zone(hr):
                        if hr < 120:   return "Z1 — Recuperação"
                        elif hr < 140: return "Z2 — Base aeróbica"
                        elif hr < 155: return "Z3 — Tempo"
                        elif hr < 170: return "Z4 — Limiar"
                        else:          return "Z5 — VO2max"
                    df_pfc["Zona FC"] = df_pfc["avg_hr"].apply(_hr_zone)
                    fig_pfc = px.scatter(
                        df_pfc, x="elevation_gain", y="Pace_min",
                        color="Zona FC", size="distance_km",
                        color_discrete_map=_zone_map,
                        title="Pace × FC × Elevação",
                        labels={"elevation_gain": "Elevação (m)",
                                "Pace_min": "Pace (min/km)",
                                "distance_km": "Distância (km)"},
                        hover_data={"name": True, "avg_hr": ":.0f"},
                        opacity=0.82)
                    set_pace_yaxis(fig_pfc, df_pfc["pace_sec_km"])
                    fig_pfc.update_layout(legend=dict(orientation="h", y=-0.32, font_size=10))
                    st.plotly_chart(fig_pfc, width="stretch")
                else:
                    st.info("Sem dados de FC disponíveis para análise combinada.")

            # ── tabela top 10 ──────────────────────────────────────────────────────
            st.markdown("**Top 10 atividades com maior elevação**")
            top10 = df_e.nlargest(10, "elevation_gain")[
                ["start_date","name","distance_km","elevation_gain",
                 "elev_km","pace_sec_km","gap_sec","vert_pace"]].copy()
            top10["Data"]      = top10["start_date"].dt.strftime("%d/%m/%Y")
            top10["Pace"]      = fmt_pace_vec(top10["pace_sec_km"])
            top10["GAP"]       = fmt_pace_vec(top10["gap_sec"])
            top10["Elev/km"]   = top10["elev_km"].apply(lambda x: f"{x:.1f}")
            top10["Distância"] = top10["distance_km"].apply(lambda x: f"{x:.1f} km")
            top10["Elevação"]  = top10["elevation_gain"].apply(lambda x: f"{x:.0f} m")
            top10["Pace Vert"] = top10["vert_pace"].apply(
                lambda x: f"{x:.0f} m/h" if pd.notna(x) else "—")
            st.dataframe(
                top10[["Data","name","Distância","Elevação","Elev/km","Pace","GAP","Pace Vert"]]
                    .rename(columns={"name": "Atividade"}),
                hide_index=True, use_container_width=True)

    # ── Assistente IA ─────────────────────────────────────────────────────────
    # ── contexto rico para o assistente de desempenho ─────────────────────────
    # Top 10 elevação (já calculado acima como top10)
    _top10_txt = ""
    try:
        _t10 = df_e.nlargest(10, "elevation_gain")[
            ["start_date","name","distance_km","elevation_gain","elev_km",
             "pace_sec_km","gap_sec","vert_pace"]].copy()
        _t10["Data"]      = _t10["start_date"].dt.strftime("%d/%m/%Y")
        _t10["Pace_f"]    = fmt_pace_vec(_t10["pace_sec_km"])
        _t10["GAP_f"]     = fmt_pace_vec(_t10["gap_sec"])
        _t10["VP_f"]      = _t10["vert_pace"].apply(lambda x: f"{x:.0f} m/h" if pd.notna(x) else "—")
        _top10_txt = _df_to_ctx_rows(
            _t10, ["Data","name","distance_km","elevation_gain","Pace_f","GAP_f","VP_f"],
            ["Data","Atividade","Dist(km)","Elev(m)","Pace","GAP","PaceVert"],
            max_rows=10)
    except Exception:
        pass
    # Todas atividades do período (para perguntas genéricas)
    _df_d = df_run.copy()
    _df_d["Data"]    = _df_d["start_date"].dt.strftime("%d/%m/%Y")
    _df_d["Pace_f"]  = fmt_pace_vec(_df_d["pace_sec_km"])
    _df_d["FC_f"]    = (_df_d["average_heartrate"].fillna(0).astype(int).apply(str)
                        if "average_heartrate" in _df_d.columns else "—")
    _df_d["Elev_f"]  = (_df_d["elevation_gain"].fillna(0).apply(lambda x: f"{int(x)}m")
                        if "elevation_gain" in _df_d.columns else "—")
    _df_d["Cad_f"]   = (_df_d["average_cadence"].apply(lambda x: f"{x*2:.0f} spm" if pd.notna(x) else "—")
                        if "average_cadence" in _df_d.columns else "—")

    _ctx_desemp = (
        f"PERÍODO: {s_dt.date()} a {e_dt.date()}\n"
        f"Melhor pace período: {fmt_pace(df_run['pace_sec_km'].min()) if df_run['pace_sec_km'].notna().any() else '—'} | "
        f"Carga média: {df_run['training_load'].mean():.1f} TRIMP\n" if df_run['training_load'].notna().any() else "Carga: —\n"
        f"Total atividades: {len(df_run)} | Distância total: {df_run['distance_km'].sum():.1f} km\n"
        f"Pace médio: {fmt_pace(df_run['pace_sec_km'].mean()) if df_run['pace_sec_km'].notna().any() else 'N/A'}/km\n"
        + (f"Cadência média: {df_run['average_cadence'].mean()*2:.0f} spm\n" if "average_cadence" in df_run.columns and df_run["average_cadence"].notna().any() else "")
        + (f"FC média: {df_run['average_heartrate'].mean():.0f} bpm\n" if "average_heartrate" in df_run.columns and df_run["average_heartrate"].notna().any() else "")
        + (f"\nTOP 10 ATIVIDADES COM MAIOR ELEVAÇÃO:\n{_top10_txt}" if _top10_txt else "")
        + f"\nTODAS AS ATIVIDADES DO PERÍODO (Data | Nome | Dist | Pace | FC | Elev | Cadência):\n"
        + _df_to_ctx_rows(
            _df_d.sort_values("start_date", ascending=False),
            ["Data","name","distance_km","Pace_f","FC_f","Elev_f","Cad_f"],
            ["Data","Nome","Dist(km)","Pace","FC","Elev","Cadência"],
            max_rows=60)
    )
    _groq_widget("Desempenho", _ctx_desemp, "desemp")

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
            pmc_filt["Semana"] = pd.to_datetime(pmc_filt["Data"]).dt.to_period("W").dt.start_time
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
        st.info("PMC indisponível (training_load ausente ou vazio).")

    st.markdown("---")

    # ── ACWR histórico + Variação de carga ───────────────────────────────────
    st.subheader("⚡ ACWR — Acute:Chronic Workload Ratio")
    st.caption(
        "Compara esforço dos últimos 7 dias com a média das 4 semanas anteriores. "
        "**Zona segura: 0,8 – 1,3.** Acima de 1,5 = alto risco de lesão.")

    col_ac1, col_ac2 = st.columns(2)
    with col_ac1:
        if df_run["training_load"].notna().any():
            weekly = (df_run[df_run["training_load"].notna()]
                      .set_index("start_date")["training_load"].resample("W").sum())
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
            st.info("Dados de training_load não disponíveis.")
    with col_ac2:
        if df_run["training_load"].notna().any():
            df_carga = (df_run[df_run["training_load"].notna()]
                        .groupby(["Semana","SemanaStr"])
                        .agg(Carga=("training_load","sum")).reset_index()
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


    # ── Assistente IA ─────────────────────────────────────────────────────────
    if not pmc_raw.empty:
        _pmc_last = pmc_raw.iloc[-1]
        _ctl_v = float(_pmc_last.get("CTL", 0))
        _atl_v = float(_pmc_last.get("ATL", 0))
        _tsb_v = float(_pmc_last.get("TSB", 0))
    else:
        _ctl_v = _atl_v = _tsb_v = 0
    # ── contexto rico para o assistente de carga ──────────────────────────────
    # Volume por semana das últimas 16 semanas
    _carga_sem = (df_run.set_index("start_date")["distance_km"]
                  .resample("W").agg(["sum","count"]).tail(16).reset_index())
    _carga_sem_txt = "\n".join(
        f"  {r['start_date'].strftime('%d/%m')}: {r['sum']:.1f} km ({int(r['count'])} corridas)"
        for _, r in _carga_sem.iterrows()
    )
    # Distribuição de zonas FC se disponível
    _zonas_txt = ""
    if "average_heartrate" in df_run.columns and df_run["average_heartrate"].notna().any():
        _fc_max_ctx = st.session_state.get("fc_max", 195)
        _z_breaks = [0, 0.6, 0.7, 0.8, 0.9, 1.0, 999]
        _z_labels  = ["Z1(<60%)", "Z2(60-70%)", "Z3(70-80%)", "Z4(80-90%)", "Z5(90-100%)", "Z6(>100%)"]
        _hr_vals = df_run["average_heartrate"].dropna() / _fc_max_ctx
        _z_counts = pd.cut(_hr_vals, bins=_z_breaks, labels=_z_labels).value_counts().sort_index()
        _zonas_txt = " | ".join(f"{z}: {c} ativ." for z, c in _z_counts.items() if c > 0)

    _ctx_carga = (
        f"PERÍODO: {s_dt.date()} a {e_dt.date()}\n"
        f"CTL (fitness cumulativo 42d): {_ctl_v:.0f}\n"
        f"ATL (fadiga 7d): {_atl_v:.0f}\n"
        f"TSB (forma = CTL-ATL): {_tsb_v:.0f} "
        + ("(Forma — pronto pra prova!)" if 5<=_tsb_v<=20 else "(Fatigado)" if _tsb_v < -15 else "(Neutro)") + "\n"
        + f"Volume últimas 4 semanas: {df_run[df_run['start_date'] >= pd.Timestamp.now()-timedelta(days=28)]['distance_km'].sum():.0f} km\n"
        f"Total atividades no período: {len(df_run)}\n"
        + (f"FC média: {df_run['average_heartrate'].mean():.0f} bpm\n" if "average_heartrate" in df_run.columns and df_run["average_heartrate"].notna().any() else "")
        + (f"Distribuição zonas FC: {_zonas_txt}\n" if _zonas_txt else "")
        + f"\nVOLUME SEMANAL (últimas 16 semanas):\n{_carga_sem_txt}\n"
        + f"\nATIVIDADES RECENTES (últimas 30) — para análise de carga:\n"
        + _df_to_ctx_rows(
            df_run.sort_values("start_date", ascending=False).assign(
                Data=lambda d: d["start_date"].dt.strftime("%d/%m/%Y"),
                Pace_f=lambda d: fmt_pace_vec(d["pace_sec_km"]),
                FC_f=lambda d: (d["average_heartrate"].fillna(0).astype(int).astype(str)
                                if "average_heartrate" in d.columns else "—"),
                Elev_f=lambda d: (d["elevation_gain"].fillna(0).astype(int).astype(str) + "m"
                                  if "elevation_gain" in d.columns else "—")
            ).head(30),
            ["Data","name","distance_km","Pace_f","FC_f","Elev_f"],
            ["Data","Nome","Dist(km)","Pace","FC","Elev"],
            max_rows=30)
    )
    _groq_widget("Carga & Zonas", _ctx_carga, "carga")

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
@st.cache_data(ttl=1800, show_spinner=False, max_entries=12)
def _build_route_map_html(
    act_data:  tuple,   # ((id, name, date_str, km, pace_sec, hr, elev, color, poly_str, insights_str), ...)
    laps_data: tuple,   # ((act_id, lap_idx, dist_km, pace_sec, hr, max_hr, elev_gain, time_sec), ...)
    lat_c: float,
    lng_c: float,
    tile: str,
    height: int,
    color_by: str = "Padrão",  # "Padrão" | "Pace" | "FC" | "Elevação"
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
        "Claro":       lambda m: folium.TileLayer("CartoDB positron", name="Claro").add_to(m),
        "Satélite":    lambda m: folium.TileLayer(ESRI_SAT[0], attr=ESRI_SAT[1], name="Sat").add_to(m),
        "Topográfico": lambda m: folium.TileLayer("OpenTopoMap",      name="Topo").add_to(m),
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

    # ── Gradiente de cor por métrica ────────────────────────────────────────
    def _metric_hex(value, vmin, vmax, high_is_red=True):
        """Verde → Amarelo → Vermelho conforme intensidade da métrica."""
        if vmax <= vmin or value is None or value == 0:
            return "#f1c40f"
        t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
        if not high_is_red:
            t = 1.0 - t
        if t < 0.5:
            s = t * 2
            r = int(46  + (241 - 46)  * s)
            g = int(204 + (196 - 204) * s)
            b = int(113 + (15  - 113) * s)
        else:
            s = (t - 0.5) * 2
            r = int(241 + (231 - 241) * s)
            g = int(196 + (76  - 196) * s)
            b = int(15  + (60  - 15)  * s)
        return f"#{r:02x}{g:02x}{b:02x}"

    # pré-computa min/max da métrica escolhida em todos os laps
    _METRIC_IDX   = {"Pace": 3, "FC": 4, "Elevação": 6}
    _HIGH_IS_RED  = {"Pace": False, "FC": True, "Elevação": True}  # Pace: rápido=red=baixo valor
    _m_idx        = _METRIC_IDX.get(color_by)
    _m_vals       = [lap[_m_idx] for lap in laps_data
                     if _m_idx is not None and lap[_m_idx] and lap[_m_idx] > 0]
    _m_min, _m_max = (min(_m_vals), max(_m_vals)) if _m_vals else (0, 1)

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
        _, name, date, km, pace_sec, hr, elev, color, _, insights_str, location = a
        loc_block = (f"<div style='font-size:11px;color:#aaa;margin-bottom:4px'>"
                     f"📍 {location}</div>") if location else ""
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
                f"<span style='color:#888;font-size:11px'>{date}</span><br>"
                f"{loc_block}"
                f"<table style='font-size:12px;width:100%;margin-top:6px'>{rows}</table>"
                f"{insights_block}</div>")

    for a in act_data:
        act_id, name, date, km, pace_sec, hr, elev, color, poly_str, _, location = a
        coords = decode_polyline(poly_str) if poly_str else []

        fg = folium.FeatureGroup(name=f"{date} — {name[:22]} ({km:.1f} km)", show=True)

        if coords:
            n_pts = len(coords) - 1

            # ── Rota principal ──────────────────────────────────────────────
            if color_by == "Padrão":
                try:
                    from folium.plugins import AntPath
                    AntPath(coords, color=color, weight=4.5, dash_array=[12, 20],
                            delay=800, opacity=0.92,
                            popup=folium.Popup(_act_popup_html(a), max_width=250)).add_to(fg)
                except Exception:
                    folium.PolyLine(coords, color=color, weight=4.5, opacity=0.9,
                                    popup=folium.Popup(_act_popup_html(a), max_width=250)).add_to(fg)
            else:
                # traço guia cinza fino — os laps coloridos ficam por cima
                folium.PolyLine(coords, color="#888888", weight=2,
                                opacity=0.25).add_to(fg)

            # Marcadores de km
            _km_markers(coords, color, fg)

            # ── Segmentos por lap ────────────────────────────────────────────
            # Calculados primeiro; adicionados em ordem reversa para que
            # laps anteriores fiquem no topo e ganhem o hover nos limites.
            act_laps = sorted(laps_by_act.get(act_id, []), key=lambda x: x[1])
            if act_laps:
                total_dist = sum(l[2] for l in act_laps) or 1
                cum_frac, prev_i = 0.0, 0
                lap_segs = []
                for lap in act_laps:
                    cum_frac += lap[2] / total_dist
                    next_i    = min(n_pts, round(cum_frac * n_pts))
                    seg       = coords[prev_i:next_i + 1]
                    lap_segs.append((lap, seg))
                    prev_i    = next_i
                for lap, seg in reversed(lap_segs):
                    if len(seg) < 2:
                        continue
                    if color_by != "Padrão" and _m_idx is not None:
                        seg_color = _metric_hex(
                            lap[_m_idx], _m_min, _m_max,
                            high_is_red=_HIGH_IS_RED.get(color_by, True)
                        )
                        # segmento animado colorido por métrica (AntPath por lap)
                        try:
                            from folium.plugins import AntPath
                            AntPath(
                                seg, color=seg_color, weight=5,
                                dash_array=[12, 20], delay=800, opacity=0.92,
                            ).add_to(fg)
                        except Exception:
                            folium.PolyLine(seg, color=seg_color, weight=5, opacity=0.92).add_to(fg)
                        # overlay invisível para tooltip/popup
                        folium.PolyLine(
                            seg, color=seg_color, weight=14, opacity=0.001,
                            popup=folium.Popup(_lap_popup_html(lap, name, date, seg_color), max_width=230),
                            tooltip=f"Lap {int(lap[1])} · {_fmt(lap[3])}/km",
                        ).add_to(fg)
                    else:
                        # modo padrão: só overlay invisível
                        folium.PolyLine(
                            seg, color=color, weight=14, opacity=0.001,
                            popup=folium.Popup(_lap_popup_html(lap, name, date, color), max_width=230),
                            tooltip=f"Lap {int(lap[1])} · {_fmt(lap[3])}/km",
                        ).add_to(fg)
            else:
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

    folium.LayerControl(collapsed=True, position="topright").add_to(m)
    try:
        from folium.plugins import MousePosition
        MousePosition(
            position="bottomleft",
            separator=" | ",
            prefix="📍",
            lat_formatter="function(num){return num.toFixed(6);}",
            lng_formatter="function(num){return num.toFixed(6);}",
        ).add_to(m)
    except Exception:
        pass
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

    # Pré-geocodifica todas as atividades visíveis (usa cache — só 1 chamada/local/30d)
    _geo_cache: dict = {}
    for _, _gr in df_map.iterrows():
        _lat, _lng = _start_coords(_gr, poly_col, has_ll)
        if _lat and _lng:
            _key = (round(_lat, 2), round(_lng, 2))
            if _key not in _geo_cache:
                _geo_cache[_key] = _reverse_geocode(_key[0], _key[1])

    # Labels de seleção (agora com 📍 bairro)
    def make_label(row):
        dt  = row["start_date"].strftime("%d/%m/%Y")
        km  = float(row.get("distance_km") or 0)
        tag = f" [{row['Intensidade']}]" \
              if "Intensidade" in row and str(row["Intensidade"]) not in ("","None","nan") else ""
        _lat, _lng = _start_coords(row, poly_col, has_ll)
        loc = ""
        if _lat and _lng:
            loc_str = _geo_cache.get((round(_lat, 2), round(_lng, 2)), "")
            loc = f" · 📍 {loc_str}" if loc_str else ""
        return f"{dt} — {row['name'][:28]} ({km:.1f} km){tag}{loc}"

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
            # bairro/cidade — já pré-geocodificado no _geo_cache
            _geo_cache.get(
                (round(_slat, 2), round(_slng, 2)), ""
            ) if (_slat := _start_coords(row, poly_col, has_ll)[0])
              and (_slng := _start_coords(row, poly_col, has_ll)[1])
            else "",
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
                            ["Claro", "Satélite", "Topográfico"],
                            horizontal=True, key="mapa_tile",
                            help="Satélite / Topográfico: ótimos para trail.")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            expand = st.checkbox("🔍 Ampliar", value=False, key="mapa_expand")

        color_by = st.radio(
            "Colorir rota por",
            ["Padrão", "Pace", "FC", "Elevação"],
            horizontal=True, key="mapa_color_by",
            help=(
                "**Padrão**: cor da atividade  |  "
                "**Pace**: 🟢 lento → 🔴 rápido  |  "
                "**FC**: 🟢 baixa → 🔴 alta  |  "
                "**Elevação**: 🟢 plano → 🔴 muita subida"
            ),
        )

        height = 720 if expand else 550

        with st.spinner("Preparando mapa…"):
            html_map = _build_route_map_html(
                act_data=act_data, laps_data=laps_data,
                lat_c=lat_c, lng_c=lng_c, tile=tile, height=height,
                color_by=color_by)

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

    # ── 🔍 Comparar Trecho ─────────────────────────────────────────────────────
    st.markdown("---")

    # ── helpers GPS (definidos uma vez, usados no bloco abaixo) ──────────────
    @st.cache_data(show_spinner=False)
    def _polyline_with_cum(poly_str: str):
        """Decodifica polyline e calcula distância cumulativa em metros."""
        pts = decode_polyline(poly_str)
        if not pts:
            return [], []
        cum = [0.0]
        for i in range(1, len(pts)):
            cum.append(cum[-1] + _haversine_km(pts[i-1], pts[i]) * 1000)
        return pts, cum

    def _extract_seg_pts(pts, cum, km_start, km_end):
        """Retorna pontos GPS entre km_start e km_end (em metros)."""
        m_start, m_end = km_start * 1000, km_end * 1000
        return [p for p, c in zip(pts, cum) if m_start <= c <= m_end]

    def _closest_dist_m(point, pts):
        """Distância mínima (metros) de um ponto para qualquer ponto da polyline."""
        if not pts:
            return float("inf")
        return min(_haversine_km(point, p) * 1000 for p in pts)

    def _match_segment_on_run(seg_pts, cand_pts, cand_cum,
                               corridor_m: float = 80, coverage: float = 0.65,
                               ref_seg_km: float = 0.0):
        """
        Verifica se cand_pts passa pelo corredor geográfico de seg_pts.
        Retorna (km_start_cand, km_end_cand) no run candidato, ou None se não bate.

        corridor_m  : distância máxima em metros para considerar "mesmo lugar"
        coverage    : fração mínima dos pontos do segmento que precisa ter match

        Lógica de sequência: os índices do candidato devem ser monotonicamente
        crescentes (a corrida passa pelo trecho em ordem, sem loop/ida-e-volta).
        O span do candidato não pode ser > 2.5× o comprimento do trecho referência.
        """
        if not seg_pts or not cand_pts:
            return None

        # Para cada ponto do segmento de referência, acha o ponto mais próximo
        # no candidato — mas só avança no candidato (evita loops/retornos)
        matched_cand_idx = []
        min_allowed_j = 0   # só aceita índices crescentes
        for sp in seg_pts:
            best_d = float("inf")
            best_j = -1
            for j in range(min_allowed_j, len(cand_pts)):
                d = _haversine_km(sp, cand_pts[j]) * 1000
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_d <= corridor_m and best_j >= 0:
                matched_cand_idx.append(best_j)
                min_allowed_j = best_j  # próximos pontos só depois deste

        if len(matched_cand_idx) / max(len(seg_pts), 1) < coverage:
            return None   # candidato não cobre o trecho suficientemente

        min_j = matched_cand_idx[0]
        max_j = matched_cand_idx[-1]
        km_s  = cand_cum[min_j] / 1000
        km_e  = cand_cum[max_j] / 1000

        # Rejeita se o span do candidato for > 2.5× o tamanho do trecho referência
        ref_span_km = (cand_cum[-1] / 1000) if not cand_cum else 1.0
        # Comprimento real do trecho (parâmetro) ou fallback para reta entre pontos
        seg_ref_km  = ref_seg_km if ref_seg_km > 0 else \
                      (_haversine_km(seg_pts[0], seg_pts[-1]) if len(seg_pts) >= 2 else 1.0)
        cand_span   = km_e - km_s
        if cand_span > max(seg_ref_km * 2.5, 0.5):
            return None   # loop/ida-e-volta detectado — descarta

        return km_s, km_e  # km no candidato

    def _pace_in_range(laps_df, km_start_cand, km_end_cand):
        """
        Extrai pace medio dos laps que cobrem [km_start_cand, km_end_cand].
        Usa distancia cumulativa real por lap - funciona com laps manuais ou auto-lap.
        """
        if laps_df.empty:
            return None, None
        idx_col = ("lap_index" if "lap_index" in laps_df.columns
                   else "split" if "split" in laps_df.columns
                   else None)
        laps_sorted = (laps_df.sort_values(idx_col) if idx_col
                       else laps_df).reset_index(drop=True)
        cum, matching = 0.0, []
        for _, lap in laps_sorted.iterrows():
            lap_dist = float(lap.get("distance_km") or 0)
            lap_end  = cum + lap_dist
            if lap_end > km_start_cand and cum < km_end_cand:
                matching.append(lap)
            cum = lap_end
        if not matching:
            return None, None
        sel = pd.DataFrame(matching)
        sel = sel[
            sel["pace_sec_km"].notna() &
            (sel["pace_sec_km"] > 0) &
            (sel["pace_sec_km"] < 600)
        ]
        if sel.empty:
            return None, None
        if "distance_km" in sel.columns and sel["distance_km"].sum() > 0:
            pace_avg = float((sel["pace_sec_km"] * sel["distance_km"]).sum()
                             / sel["distance_km"].sum())
        else:
            pace_avg = float(sel["pace_sec_km"].mean())
        hr_avg = (float(sel["average_heartrate"].mean())
                  if "average_heartrate" in sel.columns
                  and sel["average_heartrate"].notna().any()
                  else float("nan"))
        return pace_avg, hr_avg
    with st.expander("🔍 Comparar trecho com outras corridas", expanded=False):
        st.caption(
            "Escolha uma corrida de referência e arraste o slider para definir o trecho. "
            "A busca usa **coordenadas GPS reais** — só aparecem corridas que "
            "efetivamente passaram pelo mesmo lugar, independente do número do km. "
            "💡 **Dica:** passe o mouse no mapa para ver as coordenadas no canto inferior esquerdo "
            "— útil para identificar o km exato de uma subida ou trecho específico.")

        # ── Corrida de referência ─────────────────────────────────────────────
        if len(df_map) > 1:
            _cmp_ref_opts = {
                f"{r['start_date'].strftime('%d/%m/%Y')} — {str(r['name'])[:30]} ({r['distance_km']:.1f} km)": int(r["id"])
                for _, r in df_map.iterrows()
            }
            _cmp_ref_lbl = st.selectbox(
                "Corrida de referência:", list(_cmp_ref_opts.keys()), key="cmp_ref_run")
            _cmp_ref_id = _cmp_ref_opts[_cmp_ref_lbl]
        else:
            _cmp_ref_id = int(df_map["id"].iloc[0])

        # Polyline da referência
        _cmp_ref_info = df_raw[df_raw["id"] == _cmp_ref_id]
        if _cmp_ref_info.empty:
            st.info("Corrida de referência não encontrada.")
            st.stop()
        _cmp_ref_info = _cmp_ref_info.iloc[0]

        _ref_poly_str = str(_cmp_ref_info.get(poly_col, "") or "") if poly_col else ""
        _ref_pts, _ref_cum = _polyline_with_cum(_ref_poly_str) if _ref_poly_str else ([], [])

        if len(_ref_pts) < 10:
            st.info("Esta corrida não tem dados GPS (polyline) suficientes para comparar trechos.")
        else:
            _ref_total_km = _ref_cum[-1] / 1000
            _max_km = max(1, int(_ref_total_km))

            # ── Slider: trecho em km ──────────────────────────────────────────
            _trecho = st.slider(
                "Trecho a comparar (km):",
                min_value=0.0, max_value=float(_max_km),
                value=(0.0, min(3.0, float(_max_km))),
                step=0.5, key="cmp_trecho_slider")
            _km_ini, _km_fim = _trecho

            if _km_fim <= _km_ini:
                st.warning("Ajuste o slider para selecionar um trecho com pelo menos 0.5 km.")
            else:
                # Pontos GPS do trecho de referência
                _seg_pts = _extract_seg_pts(_ref_pts, _ref_cum, _km_ini, _km_fim)

                if len(_seg_pts) < 3:
                    st.info("Trecho muito curto ou sem pontos GPS suficientes. Aumente o intervalo.")
                else:
                    # Amostrar pontos do segmento (máx 30) para acelerar busca
                    _step = max(1, len(_seg_pts) // 30)
                    _seg_sample = _seg_pts[::_step]

                    st.caption(
                        f"📍 Trecho de referência: km {_km_ini:.1f}–{_km_fim:.1f} "
                        f"({(_km_fim-_km_ini)*1000:.0f} m) · {len(_seg_sample)} pontos GPS de amostragem. "
                        "Buscando corridas que passam pelo mesmo corredor geográfico...")

                    # ── Busca GPS: todas as corridas com polyline ─────────────
                    _all_with_poly = df_raw[
                        df_raw["sport_type"].isin(["Run","TrailRun"]) &
                        df_raw[poly_col].notna() &
                        (df_raw[poly_col].astype(str).str.len() > 10) &
                        (df_raw["id"] != _cmp_ref_id)
                    ].copy() if poly_col else pd.DataFrame()

                    if _all_with_poly.empty:
                        st.info("Nenhuma outra corrida com dados GPS encontrada.")
                    else:
                        _results = []
                        for _, _cand in _all_with_poly.iterrows():
                            _cand_poly = str(_cand[poly_col])
                            _cand_pts, _cand_cum = _polyline_with_cum(_cand_poly)
                            if len(_cand_pts) < 5:
                                continue

                            _match = _match_segment_on_run(
                                _seg_sample, _cand_pts, _cand_cum,
                                corridor_m=80, coverage=0.65,
                                ref_seg_km=float(_km_fim - _km_ini))
                            if _match is None:
                                continue

                            _km_s_cand, _km_e_cand = _match
                            # Laps do candidato
                            _cand_laps = laps_raw[
                                laps_raw["activity_id"] == int(_cand["id"])
                            ]
                            _pace_cand, _hr_cand = _pace_in_range(
                                _cand_laps, _km_s_cand, _km_e_cand)
                            if _pace_cand is None:
                                continue
                            _results.append({
                                "activity_id": int(_cand["id"]),
                                "name":        str(_cand["name"]),
                                "start_date":  pd.to_datetime(_cand["start_date"], dayfirst=True, errors="coerce"),
                                "distance_km": float(_cand["distance_km"]),
                                "pace_avg":    _pace_cand,
                                "hr_avg":      _hr_cand,
                                "km_s":        _km_s_cand,
                                "km_e":        _km_e_cand,
                                "is_ref":      False,
                            })

                        # Adiciona corrida de referência
                        _ref_laps = laps_raw[
                            laps_raw["activity_id"] == _cmp_ref_id
                        ]
                        _ref_pace, _ref_hr = _pace_in_range(_ref_laps, _km_ini, _km_fim)
                        if _ref_pace:
                            _results.append({
                                "activity_id": _cmp_ref_id,
                                "name":        str(_cmp_ref_info["name"]) + " ⭐",
                                "start_date":  pd.to_datetime(_cmp_ref_info["start_date"], dayfirst=True, errors="coerce"),
                                "distance_km": float(_cmp_ref_info["distance_km"]),
                                "pace_avg":    _ref_pace,
                                "hr_avg":      _ref_hr,
                                "km_s":        _km_ini,
                                "km_e":        _km_fim,
                                "is_ref":      True,
                            })

                        if not _results:
                            st.info(
                                f"Nenhuma outra corrida passou pelo mesmo trecho geográfico "
                                f"(corredor de 80 m, cobertura ≥ 65%). "
                                "Tente ampliar o trecho ou verifique se outras corridas cobrem essa área.")
                        else:
                            _cmp_df = pd.DataFrame(_results).sort_values("pace_avg").reset_index(drop=True)
                            _cmp_df["dt_str"]    = _cmp_df["start_date"].dt.strftime("%d/%m/%Y")
                            _cmp_df["label"]     = _cmp_df["dt_str"] + " — " + _cmp_df["name"].str[:26]
                            _cmp_df["pace_fmt"]  = fmt_pace_vec(_cmp_df["pace_avg"])
                            _cmp_df["pace_min"]  = _cmp_df["pace_avg"] / 60

                            _n_res = len(_cmp_df)
                            st.caption(
                                f"✅ **{_n_res} corridas** passam pelo mesmo trecho geográfico. "
                                "⭐ = corrida de referência.")

                            # ── Bar chart ──────────────────────────────────────
                            _bar_colors = []
                            for _ci, _crow in _cmp_df.iterrows():
                                if _crow["is_ref"]:
                                    _bar_colors.append("#F1C40F")
                                elif _ci == 0:
                                    _bar_colors.append("#2ECC71")
                                elif _ci == _n_res - 1:
                                    _bar_colors.append("#E74C3C")
                                else:
                                    _bar_colors.append("#3498DB")

                            fig_cmp = go.Figure()
                            fig_cmp.add_bar(
                                x=_cmp_df["label"],
                                y=_cmp_df["pace_min"],
                                marker_color=_bar_colors,
                                text=_cmp_df["pace_fmt"],
                                textposition="outside",
                                customdata=_cmp_df[["dt_str","name","distance_km","hr_avg"]].values,
                                hovertemplate=(
                                    "<b>%{customdata[0]}</b> — %{customdata[1]}<br>"
                                    "Dist total: %{customdata[2]:.1f} km<br>"
                                    "FC média no trecho: %{customdata[3]:.0f} bpm<br>"
                                    f"Pace trecho: <b>%{{text}}/km</b><extra></extra>"))
                            set_pace_yaxis(fig_cmp, _cmp_df["pace_avg"])
                            fig_cmp.update_layout(
                                title=dict(
                                    text=f"Pace no trecho km {_km_ini:.1f}–{_km_fim:.1f} · match por GPS",
                                    font=dict(size=13), x=0),
                                xaxis=dict(tickangle=-35, showgrid=False),
                                yaxis=dict(gridcolor="rgba(128,128,128,0.12)"),
                                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                showlegend=False,
                                margin=dict(t=50, b=10, l=0, r=0))
                            st.plotly_chart(fig_cmp, use_container_width=True)
                            st.caption(
                                "🟡 Corrida de referência · 🟢 Mais rápida no trecho · "
                                "🔴 Mais lenta · 🔵 Demais  |  "
                                "Match: corridas que passaram no mesmo corredor geográfico (±80 m)")

                            # ── Evolução ao longo do tempo ──────────────────────
                            _evo = _cmp_df.sort_values("start_date")
                            if len(_evo) >= 3:
                                with st.expander("📈 Evolução deste trecho ao longo do tempo"):
                                    fig_evo = go.Figure()
                                    _evo_colors = [
                                        "#F1C40F" if r["is_ref"] else "#3498DB"
                                        for _, r in _evo.iterrows()
                                    ]
                                    fig_evo.add_scatter(
                                        x=_evo["dt_str"], y=_evo["pace_min"],
                                        mode="lines+markers",
                                        line=dict(color="#3498DB", width=2),
                                        marker=dict(size=10, color=_evo_colors,
                                                    line=dict(width=1.5, color="white")),
                                        text=_evo["pace_fmt"],
                                        hovertemplate=(
                                            "<b>%{x}</b><br>"
                                            "Pace: <b>%{text}/km</b><extra></extra>"))
                                    # Trend
                                    _xt = np.arange(len(_evo), dtype=float)
                                    _yt = _evo["pace_avg"].to_numpy(dtype=float)
                                    _ct = np.polyfit(_xt, _yt, 1)
                                    fig_evo.add_scatter(
                                        x=_evo["dt_str"],
                                        y=np.polyval(_ct, _xt) / 60,
                                        mode="lines", name="Tendência",
                                        line=dict(
                                            color="#2ECC71" if _ct[0] < 0 else "#E74C3C",
                                            width=2, dash="dash"),
                                        hoverinfo="skip")
                                    set_pace_yaxis(fig_evo, _evo["pace_avg"])
                                    fig_evo.update_layout(
                                        title=dict(
                                            text=f"Evolução histórica neste trecho",
                                            font=dict(size=13), x=0),
                                        xaxis=dict(tickangle=-45, showgrid=False),
                                        yaxis=dict(gridcolor="rgba(128,128,128,0.12)"),
                                        plot_bgcolor="rgba(0,0,0,0)",
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        showlegend=True,
                                        legend=dict(orientation="h", y=-0.25),
                                        margin=dict(t=45, b=10, l=0, r=0))
                                    st.plotly_chart(fig_evo, use_container_width=True)

                            # ── Tabela detalhada ─────────────────────────────────
                            with st.expander("📋 Detalhes de cada corrida no trecho"):
                                _tbl_cmp = _cmp_df[
                                    ["dt_str","name","distance_km","pace_fmt","hr_avg"]
                                ].copy()
                                _tbl_cmp["FC Média"] = _tbl_cmp["hr_avg"].apply(
                                    lambda x: f"{x:.0f} bpm" if pd.notna(x) and x > 0 else "—")
                                st.dataframe(
                                    _tbl_cmp[["dt_str","name","distance_km","pace_fmt","FC Média"]]
                                    .rename(columns={
                                        "dt_str":"Data", "name":"Atividade",
                                        "distance_km":"Dist total (km)", "pace_fmt":"Pace no trecho"}),
                                    hide_index=True, use_container_width=True)

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
            df_sel["label"] = df_sel.apply(
                lambda r: f"{r['start_date'].strftime('%d/%m/%Y')} — {r['name'] if pd.notna(r['name']) else 'Sem nome'} ({r['distance_km']:.1f}km)",
                axis=1)
            ativ_selecionada = st.selectbox(
                "Selecione uma corrida:",
                options=df_sel["id"].tolist(),
                format_func=lambda x: df_sel.set_index("id").loc[x, "label"])

            if ativ_selecionada:
                laps_ativ = (lps_run[lps_run["activity_id"] == ativ_selecionada]
                             .sort_values("lap_index").copy())
                act_info  = df_hv[df_hv["id"] == ativ_selecionada].iloc[0]
                c1, c2, c3, c4, c5 = st.columns(5)
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
                        if TEST_3K_SEC:
                            _zones = laps_pace["pace_sec_km"].apply(
                                lambda x: _pace_zone(x, TEST_3K_SEC))
                            laps_pace["cor"]       = _zones.apply(lambda z: z[1])
                            laps_pace["zona_nome"] = _zones.apply(lambda z: z[0])
                        else:
                            p50 = laps_pace["pace_sec_km"].median()
                            laps_pace["cor"]       = laps_pace["pace_sec_km"].apply(
                                lambda x: GREEN if x < p50*0.97
                                else RED if x > p50*1.03 else BLUE)
                            laps_pace["zona_nome"] = "—"
                        fig = go.Figure(go.Bar(
                            x=laps_pace["Lap"], y=laps_pace["Pace_min"],
                            text=laps_pace["Pace_fmt"], textposition="outside",
                            marker_color=laps_pace["cor"].tolist(),
                            customdata=laps_pace[["distance_m","Pace_fmt","zona_nome"]].values,
                            hovertemplate="Lap %{x}<br>Pace: %{customdata[1]}/km<br>"
                                          "Zona: %{customdata[2]}<br>"
                                          "Distância: %{customdata[0]:.0f}m<extra></extra>"))
                        set_pace_yaxis(fig, laps_pace["pace_sec_km"])
                        titulo = "⚡ Pace por Lap"
                        if n_ig > 0:
                            titulo += f" ({n_ig} micro-laps ocultados)"
                        if TEST_3K_SEC:
                            titulo += f"  ·  ref. teste {TEST_3K_STR}/km"
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
                    lambda x: f"{x:.0f} m" if not pd.isna(x) else "—")
                df_laps_tab["average_cadence"]      = df_laps_tab["average_cadence"].apply(
                    lambda x: f"{int(x*2)} spm" if not pd.isna(x) else "—")
                df_laps_tab["average_heartrate"]    = df_laps_tab["average_heartrate"].apply(
                    lambda x: f"{x:.0f}" if not pd.isna(x) else "—")
                df_laps_tab["max_heartrate"]        = df_laps_tab["max_heartrate"].apply(
                    lambda x: f"{x:.0f}" if not pd.isna(x) else "—")
                df_laps_tab["total_elevation_gain"] = df_laps_tab["total_elevation_gain"].apply(
                    lambda x: f"{x:.0f} m" if not pd.isna(x) else "—")
                st.dataframe(df_laps_tab.rename(columns=cols_lap),
                             hide_index=True, width="stretch")

                # ── Assistente de Atividade ───────────────────────────────────
                st.markdown("---")
                st.markdown("#### 🤖 Pergunte sobre este treino")

                # Monta contexto rico com TODOS os dados disponíveis do treino
                def _build_activity_context(act_info, laps_ativ, df_run):
                    lines = []
                    lines.append(f"TREINO: {act_info['name']}")
                    lines.append(f"Data: {act_info['start_date'].strftime('%d/%m/%Y %H:%M')}")
                    lines.append(f"Distância: {act_info['distance_km']:.2f} km")
                    lines.append(f"Pace médio: {fmt_pace(act_info['pace_sec_km'])}/km")

                    if pd.notna(act_info.get("moving_time_sec")):
                        t = int(act_info["moving_time_sec"])
                        lines.append(f"Tempo total: {t//3600}h{(t%3600)//60:02d}min")

                    if pd.notna(act_info.get("average_heartrate")):
                        lines.append(f"FC média: {act_info['average_heartrate']:.0f} bpm")
                    if pd.notna(act_info.get("max_heartrate")):
                        lines.append(f"FC máxima: {act_info['max_heartrate']:.0f} bpm")
                    if pd.notna(act_info.get("elevation_gain")):
                        lines.append(f"Elevação total: {act_info['elevation_gain']:.0f} m")
                    if pd.notna(act_info.get("average_cadence")):
                        lines.append(f"Cadência média: {act_info['average_cadence']*2:.0f} spm")
                    if pd.notna(act_info.get("calories")):
                        lines.append(f"Calorias: {act_info['calories']:.0f} kcal")
                    if pd.notna(act_info.get("training_load")):
                        lines.append(f"Training Load (TRIMP): {act_info['training_load']:.1f}")

                    # Clima
                    if pd.notna(act_info.get("weather_temp")):
                        lines.append(f"Temperatura: {act_info['weather_temp']:.1f}°C"
                                     + (f" (sensação {act_info['weather_feels_like']:.1f}°C)"
                                        if pd.notna(act_info.get("weather_feels_like")) else ""))
                    if pd.notna(act_info.get("weather_humidity")):
                        lines.append(f"Umidade: {act_info['weather_humidity']:.0f}%")
                    if pd.notna(act_info.get("weather_condition")):
                        lines.append(f"Condição: {act_info['weather_condition']}")

                    # Laps detalhados
                    if not laps_ativ.empty:
                        lines.append("\nDETALHE POR LAP (" + str(len(laps_ativ)) + " laps):")
                        for _, lap in laps_ativ.iterrows():
                            lap_line = f"  Lap {int(lap['lap_index'])}: "
                            parts = []
                            if pd.notna(lap.get("distance_m")):
                                parts.append(f"{lap['distance_m']:.0f}m")
                            if pd.notna(lap.get("pace_sec_km")):
                                zona = _pace_zone(lap["pace_sec_km"], TEST_3K_SEC)[0] if TEST_3K_SEC else ""
                                parts.append(f"pace {fmt_pace(lap['pace_sec_km'])}/km ({zona})")
                            if pd.notna(lap.get("average_heartrate")):
                                parts.append(f"FC {lap['average_heartrate']:.0f} bpm")
                            if pd.notna(lap.get("max_heartrate")):
                                parts.append(f"FCmax {lap['max_heartrate']:.0f}")
                            if pd.notna(lap.get("total_elevation_gain")):
                                parts.append(f"elev +{lap['total_elevation_gain']:.0f}m")
                            if pd.notna(lap.get("average_cadence")):
                                parts.append(f"cad {lap['average_cadence']*2:.0f}spm")
                            lap_line += " | ".join(parts)
                            lines.append(lap_line)

                    # Comparação com treinos similares (mesma distância ±20%)
                    dist = float(act_info["distance_km"])
                    similares = df_run[
                        (df_run["id"] != act_info["id"]) &
                        (df_run["distance_km"].between(dist * 0.8, dist * 1.2)) &
                        df_run["pace_sec_km"].notna()
                    ].sort_values("start_date", ascending=False).head(5)

                    if not similares.empty:
                        lines.append("\nCOMPARAÇÃO — últimos treinos similares ("
                                     + f"{dist*0.8:.1f}"
                                     + "–"
                                     + f"{dist*1.2:.1f}"
                                     + " km):")
                        for _, s in similares.iterrows():
                            s_line = (f"  {s['start_date'].strftime('%d/%m/%y')}: "
                                      f"{s['distance_km']:.1f}km · {fmt_pace(s['pace_sec_km'])}/km")
                            if pd.notna(s.get("average_heartrate")):
                                s_line += f" · FC {s['average_heartrate']:.0f}"
                            if pd.notna(s.get("elevation_gain")):
                                s_line += f" · elev {s['elevation_gain']:.0f}m"
                            lines.append(s_line)

                        # Evolução de pace vs média dos similares
                        pace_atual = float(act_info["pace_sec_km"])
                        pace_medio_sim = similares["pace_sec_km"].mean()
                        diff = pace_medio_sim - pace_atual
                        sinal = "mais rápido" if diff > 0 else "mais lento"
                        lines.append(f"  → Este treino foi {abs(diff):.0f}s/km {sinal} que a média dos similares")

                    return "\n".join(lines)

                _ctx_ativ = _build_activity_context(act_info, laps_ativ, df_run)

                _q_ativ = st.text_input(
                    "Pergunta sobre este treino:",
                    placeholder="Ex: Como foi minha FC neste treino? Os laps foram consistentes? Melhorei em relação a treinos parecidos?",
                    key=f"groq_q_ativ_{ativ_selecionada}"
                )
                if st.button("Perguntar sobre este treino ▶", key=f"groq_btn_ativ_{ativ_selecionada}"):
                    if _q_ativ.strip():
                        with st.spinner("Analisando seu treino..."):
                            _ans_ativ = _groq_ask([{"role": "user", "content": _q_ativ.strip()}], _ctx_ativ, GROQ_KEY)
                        st.session_state[f"groq_ans_ativ_{ativ_selecionada}"] = _ans_ativ
                if f"groq_ans_ativ_{ativ_selecionada}" in st.session_state:
                    st.markdown(st.session_state[f"groq_ans_ativ_{ativ_selecionada}"])


# ══════════════════════════════════════════════════════════════════════════════
#  6 · SUGERIR ROTA  — recomendação baseada em histórico + ORS (novo)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _build_route_clusters(df):
    """
    Agrupa corridas em rotas únicas pela posição de largada (±500m).
    Retorna lista de dicts com info agregada por rota.
    """
    df2 = df[df["latitude"].notna() & df["longitude"].notna()].copy()
    if df2.empty:
        return []
    # Arredonda para ~500m de precisão
    df2["_clat"] = (df2["latitude"]  * 200).round() / 200
    df2["_clng"] = (df2["longitude"] * 200).round() / 200
    routes = []
    for (clat, clng), grp in df2.groupby(["_clat", "_clng"]):
        grp = grp.sort_values("start_date", ascending=False)
        is_trail = (
            (grp["sport_type"] == "TrailRun").any() or
            grp["name"].str.lower().str.contains(
                r"trilha|trail|morro|pico|serra|monte|subida", regex=True, na=False
            ).any()
        )
        routes.append({
            "clat":        float(clat),
            "clng":        float(clng),
            "n_runs":      len(grp),
            "last_date":   grp["start_date"].iloc[0],
            "last_name":   str(grp["name"].iloc[0]),
            "avg_km":      float(grp["distance_km"].mean()),
            "avg_elev":    float(grp["elevation_gain"].mean()) if grp["elevation_gain"].notna().any() else 0.0,
            "avg_pace":    float(grp["pace_sec_km"].mean())    if grp["pace_sec_km"].notna().any() else 0.0,
            "best_pace":   float(grp["pace_sec_km"].min())     if grp["pace_sec_km"].notna().any() else 0.0,
            "is_trail":    bool(is_trail),
            "polylines":   grp["map_summary_polyline"].dropna().tolist()[:3],
            "ids":         grp["id"].tolist(),
        })
    return routes


def _score_route_match(route, target_km, elev_per_10km_min, elev_per_10km_max, surface_pref):
    """
    Score 0–100 de compatibilidade entre rota e critérios.
    Distância 50% | Altimetria 35% | Superfície 15%
    """
    # Distância
    ratio = abs(route["avg_km"] - target_km) / max(target_km, 0.1)
    dist_score = max(0.0, 1.0 - ratio * 2.0)  # 0 em ±50%

    # Altimetria normalizada por 10km
    elev_10 = route["avg_elev"] / max(route["avg_km"], 0.1) * 10
    if elev_per_10km_max == float("inf"):
        # montanhoso: quanto mais alto melhor, referência 200 m/10km
        elev_score = min(1.0, elev_10 / max(elev_per_10km_min, 10.0))
    elif elev_per_10km_min <= elev_10 <= elev_per_10km_max:
        elev_score = 1.0
    else:
        margin = max(elev_per_10km_max - elev_per_10km_min, 20)
        dist_from_range = min(
            abs(elev_10 - elev_per_10km_min),
            abs(elev_10 - elev_per_10km_max)
        )
        elev_score = max(0.0, 1.0 - dist_from_range / margin)

    # Superfície
    if surface_pref == "Tanto faz":
        surf_score = 1.0
    elif surface_pref == "Trilha":
        surf_score = 1.0 if route["is_trail"] else 0.1
    else:  # Asfalto
        surf_score = 0.1 if route["is_trail"] else 1.0

    return round(dist_score * 50 + elev_score * 35 + surf_score * 15, 1)



# ── Rota Inteligente: Grade SRTM + ORS ────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def _srtm_grid_hillpoints(lat: float, lng: float, radius_m: int, grid_m: int = 250):
    """Amostra elevacao SRTM em grade regular e retorna pontos ordenados por altitude."""
    import srtm, math

    def _hav_m(la1, lo1, la2, lo2):
        dlat = math.radians(la2 - la1); dlon = math.radians(lo2 - lo1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(la1))*math.cos(math.radians(la2))*math.sin(dlon/2)**2
        return 6_371_000 * 2 * math.asin(math.sqrt(a))

    elev_data = srtm.get_data()
    home_elev = elev_data.get_elevation(lat, lng) or 0
    dlat_per_m = 1.0 / 111_320
    dlng_per_m = 1.0 / (111_320 * math.cos(math.radians(lat)))
    steps = max(int(radius_m / grid_m), 1)
    points = []
    for i in range(-steps, steps + 1):
        for j in range(-steps, steps + 1):
            plat = lat + i * grid_m * dlat_per_m
            plng = lng + j * grid_m * dlng_per_m
            dist_m = _hav_m(lat, lng, plat, plng)
            if dist_m > radius_m or dist_m < 200:
                continue
            e = elev_data.get_elevation(plat, plng)
            if e is None:
                continue
            points.append({
                "lat": plat, "lng": plng, "elev": e,
                "gain": e - home_elev, "dist_m": dist_m,
            })
    return sorted(points, key=lambda x: x["elev"], reverse=True)


def _select_waypoints(hillpoints, target_elev_m, target_km,
                       min_spacing_m: int = 400, max_candidates: int = 12):
    """Retorna ate max_candidates pontos altos bem espalhados para o builder tentar."""
    import math

    def _hav_m(la1, lo1, la2, lo2):
        dlat = math.radians(la2 - la1); dlon = math.radians(lo2 - lo1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(la1))*math.cos(math.radians(la2))*math.sin(dlon/2)**2
        return 6_371_000 * 2 * math.asin(math.sqrt(a))

    candidates = [p for p in hillpoints if p["gain"] > 5]
    if not candidates:
        candidates = list(hillpoints)

    budget_m = target_km * 0.70 * 1000
    selected = []
    for p in candidates:
        if len(selected) >= max_candidates:
            break
        too_close = any(
            _hav_m(p["lat"], p["lng"], s["lat"], s["lng"]) < min_spacing_m
            for s in selected
        )
        if too_close:
            continue
        if p["dist_m"] * 2 > budget_m:
            continue
        selected.append(p)

    selected.sort(key=lambda x: x["dist_m"])
    return selected


def _ors_leg(a_lat, a_lng, b_lat, b_lng, profile, ors_key):
    """Rota ORS de A ate B (2 pontos - free tier compativel)."""
    import requests
    url = f"https://api.openrouteservice.org/v2/directions/{profile}/geojson"
    body = {
        "coordinates": [[a_lng, a_lat], [b_lng, b_lat]],
        "elevation": True,
    }
    try:
        r = requests.post(
            url, json=body,
            headers={"Authorization": ors_key, "Content-Type": "application/json"},
            timeout=12,
        )
        if r.status_code == 200:
            feat  = r.json()["features"][0]
            props = feat["properties"]
            coords = [(c[1], c[0]) for c in feat["geometry"]["coordinates"]]
            ascent = (props.get("ascent")
                      or props.get("summary", {}).get("ascent", 0)
                      or 0)
            dist   = props.get("summary", {}).get("distance", 0)
            return {"coords": coords, "distance_m": dist, "elevation_m": float(ascent)}
        try:
            err = r.json().get("error", {})
            msg = err.get("message", str(r.status_code)) if isinstance(err, dict) else str(err)
        except Exception:
            msg = f"HTTP {r.status_code}"
        return {"error": f"ORS {r.status_code}: {msg}"}
    except Exception as exc:
        return {"error": f"Conexao: {exc}"}


def _build_route_srtm_ors(start_lat, start_lng, target_km, target_elev_m,
                           profile, ors_key):
    """Constroi rota: grade SRTM -> waypoints altos -> pernas ORS com skip automatico."""
    radius_m = int(target_km / 4.0 * 1000)
    with st.spinner("Calculando elevacao SRTM da grade..."):
        hillpoints = _srtm_grid_hillpoints(start_lat, start_lng, radius_m)
    if not hillpoints:
        return {"error": "SRTM nao retornou dados para a sua regiao."}

    # get many candidates so we can skip unroutable ones
    candidates = _select_waypoints(hillpoints, target_elev_m, target_km)
    if not candidates:
        return {"error": f"Nenhum ponto com elevacao positiva no raio de {radius_m/1000:.1f} km."}

    # build route greedily: try each candidate; skip if ORS cannot reach it
    all_coords, total_dist, total_elev = [], 0.0, 0.0
    cur_lat, cur_lng = start_lat, start_lng
    used_wps, skipped = [], []
    target_wps = 4

    with st.spinner("Conectando waypoints via ORS..."):
        for wp in candidates:
            if len(used_wps) >= target_wps:
                break
            leg = _ors_leg(cur_lat, cur_lng, wp["lat"], wp["lng"], profile, ors_key)
            if "error" in leg:
                # 404 = no routable road near this grid point -> silently skip
                skipped.append({**wp, "reason": leg["error"]})
                continue
            all_coords += leg["coords"]
            total_dist  += leg["distance_m"]
            total_elev  += leg["elevation_m"]
            cur_lat, cur_lng = wp["lat"], wp["lng"]
            used_wps.append(wp)

        if not used_wps:
            reasons = "; ".join(s["reason"] for s in skipped[:3])
            return {"error": f"ORS nao encontrou rota para nenhum dos {len(skipped)} candidatos SRTM. {reasons}"}

        home_leg = _ors_leg(cur_lat, cur_lng, start_lat, start_lng, profile, ors_key)
        if "error" in home_leg:
            return {"error": f'ORS (volta): {home_leg["error"]}'}
        all_coords += home_leg["coords"]
        total_dist  += home_leg["distance_m"]
        total_elev  += home_leg["elevation_m"]

    return {
        "coords":         all_coords,
        "distance_m":     total_dist,
        "elevation_m":    total_elev,
        "radius_m":       radius_m,
        "top_elev":       hillpoints[0]["elev"] if hillpoints else 0,
        "waypoints_used": len(used_wps),
        "skipped":        len(skipped),
        "waypoints_info": [
            {
                "lat":     w["lat"],
                "lng":     w["lng"],
                "elev":    w["elev"],
                "gain_m":  round(w["gain"], 0),
                "dist_km": round(w["dist_m"] / 1000, 1),
            }
            for w in used_wps
        ],
    }


def _ors_single(lat, lng, target_m, profile, seed, ors_key, steepness_level):
    """Uma única chamada ORS round_trip com elevation=True."""
    import requests
    url = f"https://api.openrouteservice.org/v2/directions/{profile}/geojson"
    body = {
        "coordinates": [[lng, lat]],
        "elevation": True,          # ← ESSENCIAL para obter ascent/descent reais
        "options": {
            "round_trip": {
                "length": target_m,
                "points": 5,
                "seed":   seed,
            }
        }
    }
    if steepness_level > 0:
        body["options"]["profile_params"] = {
            "weightings": {"steepness_difficulty": {"level": steepness_level}}
        }
    try:
        r = requests.post(
            url,
            json=body,
            headers={"Authorization": ors_key, "Content-Type": "application/json"},
            timeout=12,
        )
        if r.status_code == 200:
            feat  = r.json()["features"][0]
            props = feat["properties"]
            # Coordenadas: ORS retorna [lng, lat, alt] quando elevation=True
            coords = [(c[1], c[0]) for c in feat["geometry"]["coordinates"]]
            ascent = props.get("ascent") or props.get("summary", {}).get("ascent", 0) or 0
            dist   = props.get("summary", {}).get("distance", 0)
            return {"coords": coords, "distance_m": dist, "elevation_m": float(ascent)}
    except Exception:
        pass
    return None


def _ors_round_trip(lat, lng, target_m, profile, seeds, ors_key, steepness_level=0):
    """
    Gera múltiplos traçados (um por seed) e devolve o melhor:
    - Se steepness_level == 0 (quer subidas): retorna o de MAIOR ganho de elevação
    - Se steepness_level == 3 (quer plano):   retorna o de MENOR ganho de elevação
    - Caso contrário: retorna o mais próximo do alvo de elevação (passado como seeds[-1])
    seeds: lista de ints (seeds a testar) + último elemento = target_elev_m (int)
    """
    target_elev = seeds[-1]
    seed_list   = seeds[:-1]
    results = []
    for s in seed_list:
        r = _ors_single(lat, lng, target_m, profile, s, ors_key, steepness_level)
        if r:
            results.append(r)
    if not results:
        return None
    if steepness_level == 0:
        # Quer montanha: pega o com mais elevação
        return max(results, key=lambda x: x["elevation_m"])
    elif steepness_level == 3:
        # Quer plano: pega o com menos elevação
        return min(results, key=lambda x: x["elevation_m"])
    else:
        # Ondulado: pega o mais próximo do alvo
        return min(results, key=lambda x: abs(x["elevation_m"] - target_elev))


with tab_sugerir:
    st.title("🎯 Sugerir Rota")
    st.caption("Encontra rotas do seu histórico que batem com seus critérios — ou gera um circuito novo via OpenRouteService.")

    # ── Inputs ──────────────────────────────────────────────────────────────
    with st.expander("🎛️ Parâmetros da busca", expanded=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            sug_km = st.slider(
                "📏 Distância alvo (km)", 3.0, 60.0, 10.0, 0.5, key="sug_dist"
            )
            sug_tol = st.slider(
                "Tolerância (%)", 10, 50, 25, 5, key="sug_tol",
                help="Faixa aceita: ex. 10km ±25% = 7.5–12.5 km"
            )

        with c2:
            sug_terreno = st.selectbox(
                "⛰️ Terreno",
                [
                    "Plano  (< 50 m/10km)",
                    "Ondulado  (50–150 m/10km)",
                    "Montanhoso  (> 150 m/10km)",
                    "Definir ganho manualmente",
                ],
                key="sug_terreno",
            )
            if sug_terreno.startswith("Definir"):
                sug_elev_target = st.slider(
                    "Ganho alvo (m)", 0, 2000, 200, 25, key="sug_elev"
                )
                _e_min = sug_elev_target * 0.6 / max(sug_km, 1) * 10
                _e_max = sug_elev_target * 1.4 / max(sug_km, 1) * 10
            elif "Plano" in sug_terreno:
                _e_min, _e_max = 0.0, 50.0
            elif "Ondulado" in sug_terreno:
                _e_min, _e_max = 50.0, 150.0
            else:
                _e_min, _e_max = 150.0, float("inf")

        with c3:
            sug_surf = st.selectbox(
                "🏃 Superfície", ["Tanto faz", "Asfalto / Estradão", "Trilha"],
                key="sug_surf"
            )
            _surf_pref = (
                "Trilha"  if sug_surf == "Trilha"
                else "Asfalto" if "Asfalto" in sug_surf
                else "Tanto faz"
            )
            sug_top_n = st.slider("Nº de sugestões", 3, 10, 5, 1, key="sug_n")

    # ── Histórico ────────────────────────────────────────────────────────────
    st.subheader("📚 Do seu histórico")

    _runs_sug = _runs_raw.copy() if not _runs_raw.empty else df_raw.copy()
    _route_clusters = _build_route_clusters(_runs_sug)

    if not _route_clusters:
        st.info("Nenhuma corrida com coordenadas disponíveis.")
    else:
        # Calcula score para cada rota
        _km_min = sug_km * (1 - sug_tol / 100)
        _km_max = sug_km * (1 + sug_tol / 100)

        for rt in _route_clusters:
            rt["score"]  = _score_route_match(rt, sug_km, _e_min, _e_max, _surf_pref)
            rt["in_range"] = _km_min <= rt["avg_km"] <= _km_max

        # Ordena: em_range primeiro, depois por score
        _scored = sorted(
            _route_clusters,
            key=lambda r: (not r["in_range"], -r["score"])
        )

        _top = _scored[:sug_top_n]

        if not _top:
            st.info("Nenhuma rota encontrada com esses critérios. Tente aumentar a tolerância.")
        else:
            # Mapa com as rotas sugeridas
            _sug_colors = [
                "#2ecc71","#3498db","#e67e22","#9b59b6","#e74c3c",
                "#1abc9c","#f39c12","#2980b9","#8e44ad","#c0392b",
            ]

            _map_center_lat = float(sum(r["clat"] for r in _top) / len(_top))
            _map_center_lng = float(sum(r["clng"] for r in _top) / len(_top))
            _sug_map = folium.Map(
                location=[_map_center_lat, _map_center_lng],
                zoom_start=13, tiles=None, control_scale=True
            )
            folium.TileLayer(
                "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                attr="CartoDB",
                name="Dark",
                max_zoom=19,
            ).add_to(_sug_map)

            for _ri, _rt in enumerate(_top):
                _col = _sug_colors[_ri % len(_sug_colors)]
                # Marcador de largada
                _pace_str = (
                    f"{int(_rt['avg_pace']//60)}:{int(_rt['avg_pace']%60):02d}/km"
                    if _rt["avg_pace"] > 0 else "—"
                )
                _popup_txt = (
                    f"<b>#{_ri+1} — {_rt['last_name'][:35]}</b><br>"
                    f"📏 {_rt['avg_km']:.1f} km &nbsp;|&nbsp; "
                    f"⛰️ {_rt['avg_elev']:.0f} m &nbsp;|&nbsp; "
                    f"⏱ {_pace_str}<br>"
                    f"🔁 {_rt['n_runs']} vez(es) &nbsp;|&nbsp; "
                    f"Score: {_rt['score']:.0f}/100"
                )
                folium.CircleMarker(
                    [_rt["clat"], _rt["clng"]],
                    radius=10,
                    color=_col, fill=True, fill_color=_col, fill_opacity=0.9,
                    popup=folium.Popup(_popup_txt, max_width=280),
                    tooltip=f"#{_ri+1} — {_rt['avg_km']:.1f}km · Score {_rt['score']:.0f}",
                ).add_to(_sug_map)

                # Polylines (até 3 corridas desta rota)
                for _poly_str in _rt["polylines"]:
                    if _poly_str and len(str(_poly_str)) > 5:
                        _pts = decode_polyline(str(_poly_str))
                        if _pts:
                            folium.PolyLine(
                                _pts, color=_col, weight=3.5,
                                opacity=0.75, smooth_factor=1,
                            ).add_to(_sug_map)

            # Fit bounds
            all_lats = [r["clat"] for r in _top]
            all_lngs = [r["clng"] for r in _top]
            _sug_map.fit_bounds([
                [min(all_lats) - 0.02, min(all_lngs) - 0.02],
                [max(all_lats) + 0.02, max(all_lngs) + 0.02],
            ])

            components.html(_sug_map._repr_html_(), height=420)

            # Cards das sugestões
            st.markdown("---")
            _card_cols = st.columns(min(len(_top), 3))
            for _ri, _rt in enumerate(_top):
                with _card_cols[_ri % 3]:
                    _pace_str2 = (
                        f"{int(_rt['avg_pace']//60)}:{int(_rt['avg_pace']%60):02d}/km"
                        if _rt["avg_pace"] > 0 else "—"
                    )
                    _best_str = (
                        f"{int(_rt['best_pace']//60)}:{int(_rt['best_pace']%60):02d}/km"
                        if _rt["best_pace"] > 0 else "—"
                    )
                    _surf_icon = "🏔️" if _rt["is_trail"] else "🛣️"
                    _score_bar = "█" * int(_rt["score"] / 10) + "░" * (10 - int(_rt["score"] / 10))
                    _in_rng_badge = "✅" if _rt["in_range"] else "⚠️"

                    st.markdown(
                        f"""
<div style="border:1px solid #444;border-radius:8px;padding:10px 12px;margin-bottom:8px;background:#1e1e2e">
  <div style="font-size:1.05em;font-weight:700;margin-bottom:4px">
    {_sug_colors[_ri % len(_sug_colors)] and "●"} #{_ri+1} {_surf_icon} {_rt["last_name"][:30]}
  </div>
  <div style="font-size:0.85em;color:#aaa;margin-bottom:6px">
    Última: {_rt["last_date"].strftime("%d/%m/%Y") if hasattr(_rt["last_date"],"strftime") else str(_rt["last_date"])[:10]}
    &nbsp;·&nbsp; {_rt["n_runs"]} corrida(s)
  </div>
  <div style="display:flex;gap:12px;flex-wrap:wrap;font-size:0.88em">
    <span>📏 <b>{_rt["avg_km"]:.1f} km</b> {_in_rng_badge}</span>
    <span>⛰️ <b>{_rt["avg_elev"]:.0f} m</b></span>
    <span>⏱ <b>{_pace_str2}</b></span>
    <span>🏆 <b>{_best_str}</b></span>
  </div>
  <div style="margin-top:6px;font-size:0.8em;color:#888">
    Score: <code style="color:#7eb8f7">{_rt["score"]:.0f}/100</code>
    &nbsp; <span style="font-family:monospace;color:#7eb8f7">{_score_bar}</span>
  </div>
</div>""",
                        unsafe_allow_html=True,
                    )

    # Lat/lng padrão: mediana das corridas com GPS (fallback: centro de Florianópolis)
    _has_gps = (
        not _runs_sug.empty
        and "latitude" in _runs_sug.columns
        and _runs_sug["latitude"].notna().any()
    )
    _default_lat = float(_runs_sug["latitude"].dropna().median())  if _has_gps else -27.5954
    _default_lng = float(_runs_sug["longitude"].dropna().median()) if _has_gps else -48.5480

    # ── Rota Inteligente: Grade SRTM + ORS ─────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Rota Inteligente  (Grade SRTM NASA + Traçado ORS)")

    with st.expander("ℹ️ Como funciona?", expanded=False):
        st.markdown(
            "**Metodologia em 3 etapas:**\n\n"
            "1. **SRTM NASA** — amostra elevação real em grade de 250 m dentro do raio de busca "
            "(`raio = distância / 4`). Dados locais do satélite — sem dep. de API externa.\n"
            "2. **Seleção de morros** — identifica pontos de maior altitude com espaçamento "
            "mínimo entre eles, dentro do orçamento de distância.\n"
            "3. **ORS (OpenRouteService)** — conecta os pontos em circuito fechado "
            "saindo e voltando ao ponto de largada.\n\n"
            "> Apenas a chave ORS é necessária. Sem Strava, sem Overpass."
        )

    col_ri1, col_ri2 = st.columns(2)
    with col_ri1:
        _ors_key_ri = st.text_input(
            "Chave ORS (OpenRouteService)",
            type="password",
            help="Chave gratuita em openrouteservice.org ou heigit.org",
            key="ors_key_ri",
        )
        _ri_dist = st.number_input(
            "Distância alvo (km)", min_value=3.0, max_value=60.0,
            value=10.0, step=0.5, key="ri_dist",
        )
        _ri_elev = st.number_input(
            "Ganho de elevação alvo (m)", min_value=50, max_value=3000,
            value=300, step=50, key="ri_elev",
        )
    with col_ri2:
        _ri_lat = st.number_input(
            "Latitude de largada", value=_default_lat,
            format="%.6f", key="ri_lat",
        )
        _ri_lng = st.number_input(
            "Longitude de largada", value=_default_lng,
            format="%.6f", key="ri_lng",
        )
        _ri_profile = st.selectbox(
            "Perfil de rota", ["foot-walking", "foot-hiking", "cycling-road"],
            key="ri_profile",
        )

    _ri_radius_preview = int(_ri_dist / 4.0 * 1000)
    st.caption(
        f"Raio de busca: **{_ri_radius_preview / 1000:.1f} km** "
        f"(= {_ri_dist:.0f} km ÷ 4)"
    )

    if st.button("🚀 Gerar Rota Inteligente", key="btn_ri"):
        if not _ors_key_ri:
            st.warning("Informe a chave ORS para continuar.")
        else:
            _ri_result = _build_route_srtm_ors(
                    start_lat=_ri_lat,
                    start_lng=_ri_lng,
                    target_km=_ri_dist,
                    target_elev_m=_ri_elev,
                    profile=_ri_profile,
                    ors_key=_ors_key_ri,
                )

            if "error" in _ri_result:
                st.error(_ri_result["error"])
            else:
                _ri_km   = _ri_result["distance_m"] / 1000
                _ri_gain = _ri_result["elevation_m"]
                _ri_wps  = _ri_result["waypoints_used"]
                _ri_rad  = _ri_result["radius_m"] / 1000
                _ri_top  = _ri_result["top_elev"]

                st.success(
                    f"Rota gerada: **{_ri_km:.1f} km** | "
                    f"**{_ri_gain:.0f} m** de ganho | "
                    f"{_ri_wps} waypoints usados"
                    + (f" ({_ri_result['skipped']} pulados)" if _ri_result.get("skipped") else "")
                    + f" | pico {_ri_top:.0f} m | raio {_ri_rad:.1f} km"
                )

                import folium
                from streamlit_folium import folium_static
                _ri_map = folium.Map(
                    location=[_ri_lat, _ri_lng],
                    zoom_start=13,
                    tiles="CartoDB dark_matter",
                )
                folium.PolyLine(
                    _ri_result["coords"], color="#FF4B4B", weight=3.5, opacity=0.9
                ).add_to(_ri_map)
                folium.Marker(
                    [_ri_lat, _ri_lng],
                    tooltip="Largada / Chegada",
                    icon=folium.Icon(color="green", icon="flag"),
                ).add_to(_ri_map)
                folium_static(_ri_map, width=700, height=420)

                if _ri_result.get("waypoints_info"):
                    st.markdown("**Waypoints selecionados (morros SRTM):**")
                    _ri_df = pd.DataFrame(_ri_result["waypoints_info"]).rename(columns={
                        "lat":      "Latitude",
                        "lng":      "Longitude",
                        "elev":     "Altitude (m)",
                        "gain_m":   "Ganho vs largada (m)",
                        "dist_km":  "Dist. da largada (km)",
                    })
                    st.dataframe(_ri_df, use_container_width=True, hide_index=True)

                _ri_gpx_pts = "\n".join(
                    f'    <trkpt lat="{la:.6f}" lon="{lo:.6f}"></trkpt>'
                    for la, lo in _ri_result["coords"]
                )
                _ri_gpx = (
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<gpx version="1.1" creator="PerformanceRun">\n'
                    "  <trk><name>Rota Inteligente</name><trkseg>\n"
                    + _ri_gpx_pts + "\n"
                    "  </trkseg></trk>\n</gpx>"
                )
                st.download_button(
                    label="⬇️ Baixar GPX",
                    data=_ri_gpx,
                    file_name="rota_inteligente.gpx",
                    mime="application/gpx+xml",
                    key="dl_ri_gpx",
                )


    # ── Rotas Novas via ORS ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🆕 Gerar rota nova")

    with st.expander("ℹ️ Como funciona?"):
        st.markdown(
            "**OpenRouteService (ORS)** é uma API gratuita baseada em OpenStreetMap que gera "
            "circuitos a partir de um ponto de partida, respeitando distância e perfil de terreno.\n\n"
            "Para usar:\n"
            "1. Cadastre-se gratuitamente em **[openrouteservice.org](https://openrouteservice.org/dev/#/signup)**\n"
            "2. Copie sua API Key no painel\n"
            "3. Cole abaixo\n\n"
            "O plano gratuito oferece 2.000 requisições/dia — mais do que suficiente para uso pessoal."
        )

    _ors_col1, _ors_col2 = st.columns([2, 1])
    with _ors_col1:
        ors_key = st.text_input(
            "🔑 API Key OpenRouteService (opcional)",
            type="password",
            placeholder="ey...",
            key="sug_ors_key",
        )
    with _ors_col2:
        ors_profile = st.selectbox(
            "Perfil", ["foot-hiking", "foot-walking"], key="sug_ors_profile",
            help="foot-hiking para trilhas e terrenos irregulares"
        )

    if ors_key:
        _ors_c1, _ors_c2, _ors_c3 = st.columns(3)
        with _ors_c1:
            ors_lat = st.number_input("Latitude largada", value=_default_lat,
                                       format="%.5f", key="sug_ors_lat")
            ors_lng = st.number_input("Longitude largada", value=_default_lng,
                                       format="%.5f", key="sug_ors_lng")
        with _ors_c2:
            ors_km   = st.slider("Distância (km)", 3.0, 60.0, float(sug_km), 0.5,
                                  key="sug_ors_km")
            ors_elev = st.slider(
                "⛰️ Ganho de elevação alvo (m)",
                0, 2000,
                min(400, int(sug_km * 30)),
                step=25, key="sug_ors_elev",
            )
            _ors_elev_per_10 = ors_elev / max(ors_km, 0.1) * 10
            if _ors_elev_per_10 < 40:
                _ors_steepness = 3
                _ors_perfil_lbl = "🛣️ Plano"
            elif _ors_elev_per_10 < 100:
                _ors_steepness = 2
                _ors_perfil_lbl = "〰️ Levemente ondulado"
            elif _ors_elev_per_10 < 180:
                _ors_steepness = 1
                _ors_perfil_lbl = "🌊 Ondulado"
            else:
                _ors_steepness = 0
                _ors_perfil_lbl = "⛰️ Montanhoso"
            st.caption(f"Perfil derivado: **{_ors_perfil_lbl}** ({_ors_elev_per_10:.0f} m/10km)")

        with _ors_c3:
            ors_go = st.button("🗺️ Gerar rota", key="sug_ors_go", use_container_width=True)

        if ors_go:
            _seeds_to_try = list(range(1, 7)) + [int(ors_elev)]
            with st.spinner(f"Testando 6 traçados e escolhendo o melhor para {ors_elev} m de ganho..."):
                _ors_result = _ors_round_trip(
                    float(ors_lat), float(ors_lng),
                    int(ors_km * 1000),
                    ors_profile,
                    _seeds_to_try,
                    ors_key,
                    steepness_level=_ors_steepness,
                )
            if _ors_result is None:
                st.error("Não foi possível gerar a rota. Verifique sua API Key e tente novamente.")
            else:
                _ors_dist = _ors_result["distance_m"] / 1000
                _ors_elev_real = _ors_result["elevation_m"]
                _elev_diff = _ors_elev_real - ors_elev
                _elev_pct  = int(_ors_elev_real / max(ors_elev, 1) * 100)
                _elev_badge = (
                    f"✅ {_ors_elev_real:.0f} m ({_elev_pct}% do alvo)"
                    if abs(_elev_diff) / max(ors_elev, 1) < 0.30
                    else f"⚠️ {_ors_elev_real:.0f} m ({_elev_pct}% do alvo — terreno disponível pode ser limitado)"
                )
                st.success(
                    f"🗺️ Rota gerada: **{_ors_dist:.1f} km** · ⛰️ {_elev_badge}"
                )
                _ors_pts = _ors_result["coords"]
                _ors_map = folium.Map(location=_ors_pts[0], zoom_start=14,
                                       tiles=None, control_scale=True)
                folium.TileLayer(
                    "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                    attr="CartoDB", name="Dark", max_zoom=19,
                ).add_to(_ors_map)
                folium.PolyLine(_ors_pts, color="#2ecc71", weight=4.5,
                                 opacity=0.9).add_to(_ors_map)
                folium.CircleMarker(
                    _ors_pts[0], radius=10, color="#f1c40f",
                    fill=True, fill_color="#f1c40f", fill_opacity=1.0,
                    tooltip="Largada / Chegada",
                ).add_to(_ors_map)
                _ors_map.fit_bounds([
                    [min(p[0] for p in _ors_pts), min(p[1] for p in _ors_pts)],
                    [max(p[0] for p in _ors_pts), max(p[1] for p in _ors_pts)],
                ])
                components.html(_ors_map._repr_html_(), height=450)

                _gpx_lines = [
                    '<?xml version="1.0" encoding="UTF-8"?>',
                    '<gpx version="1.1" creator="PerformanceRun">',
                    '  <trk><name>Rota Sugerida</name><trkseg>',
                ]
                for _p in _ors_pts:
                    _gpx_lines.append(f'    <trkpt lat="{_p[0]:.6f}" lon="{_p[1]:.6f}"/>')
                _gpx_lines += ["  </trkseg></trk>", "</gpx>"]
                _gpx_str = "\n".join(_gpx_lines)
                st.download_button(
                    "⬇️ Baixar GPX",
                    data=_gpx_str,
                    file_name="rota_sugerida.gpx",
                    mime="application/gpx+xml",
                    key="sug_gpx_dl",
                )
    else:
        st.caption("💡 Sem API Key, só as rotas do histórico ficam disponíveis. Funciona muito bem para quem já tem um bom volume de corridas!")

# ══════════════════════════════════════════════════════════════════════════════
#  7 · PLANO DE TREINO  — importação via screenshot + calendário + PMC projetado
# ══════════════════════════════════════════════════════════════════════════════
with tab_plano:
    import os as _os, base64, json as _json
    from datetime import date as _date, timedelta as _td

    PLAN_FILE   = _os.path.join(BASE, "training_plan.json")
    RACE_DATE   = pd.Timestamp("2026-08-01")
    DAYS_LEFT   = (RACE_DATE - pd.Timestamp.now()).days

    # ── Intensity → estimated TRIMP per km ───────────────────────────────────
    _INT_HR = {
        "Muito Leve":    0.60, "Leve":          0.68,
        "Moderado":      0.73, "Moderado Firme": 0.79,
        "Moderado-firme":0.79, "Forte":          0.84,
        "Muito Forte":   0.90, "Trote":          0.58,
    }
    _INT_PACE = {   # sec/km estimate for pace projection
        "Muito Leve": 360, "Leve": 330, "Moderado": 310,
        "Moderado Firme": 295, "Moderado-firme": 295,
        "Forte": 280, "Muito Forte": 255, "Trote": 390,
    }
    def _est_load(dist_km: float, intensity: str) -> float:
        hr_pct = _INT_HR.get(intensity, 0.73)
        pace_s = _INT_PACE.get(intensity, 310)
        dur_min = dist_km * pace_s / 60
        hr_abs  = hr_pct * 195
        return round(dur_min * (hr_abs - 45) / (195 - 45), 2)

    # ── Load / save plan ──────────────────────────────────────────────────────
    def _load_plan() -> list:
        if _os.path.exists(PLAN_FILE):
            try:
                with open(PLAN_FILE) as _f:
                    return _json.load(_f)
            except Exception:
                pass
        return []

    def _save_plan(plan: list):
        _os.makedirs(_os.path.dirname(PLAN_FILE), exist_ok=True)
        with open(PLAN_FILE, "w") as _f:
            _json.dump(plan, _f, ensure_ascii=False, indent=2)

    # ── Groq vision extraction ────────────────────────────────────────────────
    def _extract_from_screenshot(img_bytes: bytes, api_key: str) -> dict | None:
        import requests as _req
        b64 = base64.b64encode(img_bytes).decode()
        prompt = """Analise esta screenshot de um app de prescrição de treinos e extraia os dados em JSON.
Retorne SOMENTE o JSON, sem texto extra, no formato:
{
  "date": "YYYY-MM-DD",
  "training_type": "Intervalado|Treino de Ritmo|Longo|Regenerativo|Corrida|etc",
  "course": "Plano|Montanha|Trail",
  "distance_km": 10.0,
  "intensity": "Muito Leve|Leve|Moderado|Moderado Firme|Forte|Muito Forte",
  "description": "resumo curto do treino (1 linha)",
  "blocks": [
    {"distance_km": 2.0, "intensity": "Moderado", "note": "aquecimento"},
    {"reps": 3, "distance_km": 1.0, "intensity": "Muito Forte", "rest": "1:30min"},
    {"distance_km": 1.0, "intensity": "Leve", "note": "soltura"}
  ]
}
Se a data estiver no formato DD/MM/AAAA, converta para YYYY-MM-DD.
Se não conseguir extrair algum campo, use null."""
        try:
            _r = _req.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }],
                    "max_tokens": 800, "temperature": 0.1,
                },
                timeout=30
            )
            _r.raise_for_status()
            raw = _r.json()["choices"][0]["message"]["content"].strip()
            # strip markdown code fences if present
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
            if raw.endswith("```"):
                raw = "\n".join(raw.split("\n")[:-1])
            parsed = _json.loads(raw)
            # Auto-compute training_load if missing
            if parsed.get("distance_km") and parsed.get("intensity"):
                parsed["training_load"] = _est_load(
                    float(parsed["distance_km"]), parsed["intensity"]
                )
            return parsed
        except Exception as _e:
            return {"error": str(_e)}

    # ─────────────────────────────────────────────────────────────────────────
    st.title("📅 Plano de Treino")
    st.caption(f"🏁 Paulo Lopes Trail 21K · **{RACE_DATE.strftime('%d/%m/%Y')}** · "
               f"{'hoje!' if DAYS_LEFT == 0 else f'{DAYS_LEFT} dias'}")
    st.markdown("---")

    plan_data = _load_plan()

    # ── Upload screenshots ────────────────────────────────────────────────────
    with st.expander("📸 Importar treinos via screenshot", expanded=not bool(plan_data)):
        st.markdown(
            "Faça upload das screenshots do app do seu treinador. "
            "O assistente extrai data, tipo, distância e intensidade automaticamente."
        )
        if not GROQ_KEY or len(GROQ_KEY) < 20:
            st.warning("⚠️ Configure a chave Groq no sidebar para usar a extração automática.")
        else:
            uploaded = st.file_uploader(
                "Screenshots do plano",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key="plan_upload"
            )
            if uploaded:
                if st.button(f"🔍 Extrair {len(uploaded)} treino(s)", key="plan_extract_btn"):
                    existing_dates = {s["date"] for s in plan_data if "date" in s}
                    added = 0
                    errors = []
                    prog = st.progress(0)
                    for i, f in enumerate(uploaded):
                        with st.spinner(f"Lendo {f.name}..."):
                            result = _extract_from_screenshot(f.read(), GROQ_KEY)
                        if result and "error" not in result and result.get("date"):
                            if result["date"] not in existing_dates:
                                plan_data.append(result)
                                existing_dates.add(result["date"])
                                added += 1
                            else:
                                st.info(f"📅 {result['date']} já existe no plano — pulando.")
                        else:
                            errors.append(f"{f.name}: {result.get('error','sem data') if result else 'falha'}")
                        prog.progress((i + 1) / len(uploaded))
                    if added:
                        plan_data.sort(key=lambda x: x.get("date", ""))
                        _save_plan(plan_data)
                        st.success(f"✅ {added} treino(s) importado(s) com sucesso!")
                        st.rerun()
                    if errors:
                        for e in errors:
                            st.error(f"❌ {e}")

    if not plan_data:
        st.info("📭 Nenhum treino planejado ainda. Importe screenshots acima.")
        st.stop()

    # ── Filter: only future + today ───────────────────────────────────────────
    _today_str = _date.today().isoformat()
    plan_df = pd.DataFrame(plan_data)
    plan_df["date"] = pd.to_datetime(plan_df["date"], errors="coerce")
    plan_df = plan_df.dropna(subset=["date"]).sort_values("date")

    plan_future = plan_df[plan_df["date"] >= pd.Timestamp(_today_str)].copy()
    plan_past   = plan_df[plan_df["date"] <  pd.Timestamp(_today_str)].copy()

    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    col_stat1.metric("📋 Treinos planejados", len(plan_df))
    col_stat2.metric("⏳ Ainda por fazer", len(plan_future))
    col_stat3.metric("✅ Já realizados (plano)", len(plan_past))
    _total_plan_km = plan_df["distance_km"].sum() if "distance_km" in plan_df.columns else 0
    col_stat4.metric("📏 KM totais planejados", f"{_total_plan_km:.0f} km")

    st.markdown("---")

    # ── PMC PROJETADO ─────────────────────────────────────────────────────────
    st.subheader("📈 PMC Projetado até a prova")

    _pmc_real = calc_pmc(_runs_raw)   # _runs_raw = all Run/TrailRun activities
    if not _pmc_real.empty and "training_load" in plan_df.columns:
        # Build daily load series: real history + planned future
        _real_end = _pmc_real.index.max()
        _proj_start = max(_real_end + pd.Timedelta(days=1), pd.Timestamp(_today_str))

        # Future planned loads
        _fut_loads = {}
        for _, row in plan_future.iterrows():
            d = row["date"]
            tl = row.get("training_load") or (
                _est_load(float(row["distance_km"]), str(row.get("intensity","Moderado")))
                if pd.notna(row.get("distance_km")) else 0
            )
            _fut_loads[d.normalize()] = _fut_loads.get(d.normalize(), 0) + float(tl or 0)

        # Extend PMC from last real value
        _last = _pmc_real.iloc[-1]
        _ctl, _atl = float(_last["CTL"]), float(_last["ATL"])
        k_ctl, k_atl = 2/(42+1), 2/(7+1)

        _proj_rows = []
        _cur_day = _proj_start.normalize()
        _end_day  = RACE_DATE + pd.Timedelta(days=1)
        while _cur_day <= _end_day:
            _tl = _fut_loads.get(_cur_day, 0.0)
            _ctl = _ctl + k_ctl * (_tl - _ctl)
            _atl = _atl + k_atl * (_tl - _atl)
            _proj_rows.append({"date": _cur_day, "CTL": _ctl, "ATL": _atl,
                                "TSB": _ctl - _atl, "load": _tl, "projected": True})
            _cur_day += pd.Timedelta(days=1)

        _pmc_proj = pd.DataFrame(_proj_rows).set_index("date")

        # Combine real + projected for chart
        _pmc_plot = _pmc_real.copy()
        _pmc_plot["projected"] = False
        _comb = pd.concat([_pmc_plot, _pmc_proj])
        _comb = _comb[_comb.index >= pd.Timestamp(_today_str) - pd.Timedelta(days=30)]

        _real_part = _comb[~_comb["projected"]]
        _proj_part = _comb[_comb["projected"]]

        fig_pmc = go.Figure()
        # Real CTL
        fig_pmc.add_scatter(x=_real_part.index, y=_real_part["CTL"],
                            name="CTL (real)", line=dict(color=BLUE, width=2.5))
        # Projected CTL (dashed)
        fig_pmc.add_scatter(x=_proj_part.index, y=_proj_part["CTL"],
                            name="CTL (projetado)", line=dict(color=BLUE, width=2, dash="dash"))
        # Real TSB
        fig_pmc.add_scatter(x=_real_part.index, y=_real_part["TSB"],
                            name="TSB (real)", line=dict(color=GREEN, width=1.5))
        # Projected TSB (dashed)
        fig_pmc.add_scatter(x=_proj_part.index, y=_proj_part["TSB"],
                            name="TSB (projetado)", line=dict(color=GREEN, width=1.5, dash="dash"))
        # Race day marker
        fig_pmc.add_vline(x=RACE_DATE, line_dash="dot", line_color=RED,
                          annotation_text="🏁 Paulo Lopes 01/08",
                          annotation_position="top right")
        # TSB target zone
        fig_pmc.add_hrect(y0=5, y1=20, fillcolor="rgba(46,204,113,0.08)",
                          line_width=0, annotation_text="TSB ideal prova",
                          annotation_position="right")
        fig_pmc.update_layout(
            title="CTL e TSB: histórico real + projeção com plano de treino",
            height=350, margin=dict(t=45, b=10, l=0, r=0),
            legend=dict(orientation="h", y=-0.15),
            xaxis=dict(showgrid=False),
        )
        # Show projected CTL and TSB on race day
        if not _proj_part.empty:
            _race_row = _proj_part[_proj_part.index.normalize() == RACE_DATE.normalize()]
            if not _race_row.empty:
                _ctl_race = _race_row["CTL"].iloc[0]
                _tsb_race = _race_row["TSB"].iloc[0]
                _tsb_status = (
                    "✅ Na janela ideal (+5 a +20)" if 5 <= _tsb_race <= 20
                    else ("⚠️ Muito descansado (>+20)" if _tsb_race > 20
                          else "🔴 Estressado (<+5)" if _tsb_race < -15
                          else "⚠️ Abaixo do ideal (<+5)")
                )
                _r1, _r2, _r3 = st.columns(3)
                _r1.metric("🏋️ CTL na prova (proj.)", f"{_ctl_race:.1f}")
                _r2.metric("🎯 TSB na prova (proj.)", f"{_tsb_race:.1f}", _tsb_status,
                           delta_color="normal" if 5 <= _tsb_race <= 20 else "inverse")
                _r3.metric("📅 Dias até prova", str(DAYS_LEFT))
        st.plotly_chart(fig_pmc, use_container_width=True)
    elif _pmc_real.empty:
        st.info("PMC real não disponível — rode um treino primeiro.")
    else:
        st.info("Treinos sem training_load estimado — verifique a importação.")

    st.markdown("---")

    # ── CALENDÁRIO DE TREINOS FUTUROS ─────────────────────────────────────────
    st.subheader("🗓️ Próximos treinos")

    _INT_COLOR_PLAN = {
        "Muito Leve": "#27AE60", "Leve": "#2ECC71",
        "Moderado": "#F1C40F", "Moderado Firme": "#E67E22", "Moderado-firme": "#E67E22",
        "Forte": "#E74C3C", "Muito Forte": "#922B21", "Trote": "#1ABC9C",
    }

    # Group by week
    plan_future["week"] = plan_future["date"].dt.to_period("W")
    for _wk, _wk_df in plan_future.groupby("week"):
        _wk_start = _wk.start_time.strftime("%d/%m")
        _wk_end   = _wk.end_time.strftime("%d/%m")
        _wk_km    = _wk_df["distance_km"].sum() if "distance_km" in _wk_df.columns else 0
        _wk_load  = _wk_df["training_load"].sum() if "training_load" in _wk_df.columns else 0
        with st.expander(
            f"📅 Semana {_wk_start}–{_wk_end} · {_wk_km:.0f} km · carga estimada {_wk_load:.0f}",
            expanded=(_wk == plan_future["week"].iloc[0])
        ):
            for _, row in _wk_df.sort_values("date").iterrows():
                _dt   = row["date"]
                _dow  = ["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"][_dt.weekday()]
                _int  = row.get("intensity") or "Moderado"
                _col  = _INT_COLOR_PLAN.get(str(_int), "#7F8C8D")
                _km   = row.get("distance_km")
                _type = row.get("training_type") or row.get("type") or "Corrida"
                _desc = row.get("description") or ""
                _tl   = row.get("training_load")

                # Check if completed (matched against real activities)
                _completed = False
                if not df_run.empty:
                    _same_day = df_run[df_run["start_date"].dt.date == _dt.date()]
                    _completed = len(_same_day) > 0

                _done_icon = "✅" if _completed else "⏳"
                st.markdown(
                    f"{_done_icon} **{_dow} {_dt.strftime('%d/%m')}** "
                    f"— <span style='color:{_col};font-weight:bold'>{_int}</span> "
                    f"· {_type}"
                    + (f" · **{_km:.0f} km**" if _km else "")
                    + (f" · carga ~{_tl:.0f}" if _tl else ""),
                    unsafe_allow_html=True
                )
                if _desc:
                    st.caption(f"   {_desc}")

                # Blocks detail (se disponível)
                _blocks = row.get("blocks")
                if _blocks and isinstance(_blocks, list) and len(_blocks) > 0:
                    _block_lines = []
                    for b in _blocks:
                        if not isinstance(b, dict):
                            continue
                        _bi = str(b.get("intensity",""))
                        _bc = _INT_COLOR_PLAN.get(_bi, "#aaa")
                        if b.get("reps"):
                            _block_lines.append(
                                f"<span style='color:{_bc}'>● {b['reps']}×{b.get('distance_km','')}km {_bi}"
                                + (f" (rec {b['rest']})" if b.get("rest") else "") + "</span>"
                            )
                        elif b.get("distance_km"):
                            _note = b.get("note","")
                            _block_lines.append(
                                f"<span style='color:{_bc}'>● {b['distance_km']}km {_bi}"
                                + (f" — {_note}" if _note else "") + "</span>"
                            )
                    if _block_lines:
                        st.markdown("&nbsp;&nbsp;&nbsp;" + " &nbsp; ".join(_block_lines), unsafe_allow_html=True)
                st.divider()

    # ── MANAGE PLAN ───────────────────────────────────────────────────────────
    with st.expander("🗑️ Gerenciar plano (remover treinos)"):
        if plan_data:
            _del_opts = {
                f"{s.get('date','')} — {s.get('training_type','?')} {s.get('distance_km','?')}km": i
                for i, s in enumerate(plan_data)
            }
            _to_del = st.multiselect("Selecione para remover:", list(_del_opts.keys()),
                                     key="plan_del_select")
            if _to_del and st.button("🗑️ Remover selecionados", key="plan_del_btn"):
                _idxs = sorted({_del_opts[k] for k in _to_del}, reverse=True)
                for _i in _idxs:
                    plan_data.pop(_i)
                _save_plan(plan_data)
                st.success("Removido(s).")
                st.rerun()
