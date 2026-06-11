import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

# ── Paleta y estilo global ────────────────────────────────────────────────────
PRIMARY   = "#1B4FFF"   # azul eléctrico
ACCENT    = "#FF4C4C"   # rojo alerta
SURFACE   = "#0D0D1A"   # fondo oscuro
CARD      = "#13132B"   # tarjeta
MUTED     = "#6B7280"   # texto secundario
TEXT      = "#E8E8F0"   # texto principal
SUCCESS   = "#22D3A0"   # verde éxito

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor":   CARD,
    "axes.edgecolor":   "#2A2A4A",
    "axes.labelcolor":  TEXT,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "text.color":       TEXT,
    "grid.color":       "#1E1E3A",
    "grid.linewidth":   0.6,
    "font.family":      "sans-serif",
})

st.set_page_config(
    page_title="Churn Survival · EncomiExpress",
    page_icon="📦",
    layout="wide",
)

st.markdown(f"""
<style>
  html, body, [data-testid="stAppViewContainer"] {{
      background-color: {SURFACE};
      color: {TEXT};
  }}
  [data-testid="stSidebar"] {{
      background-color: {CARD};
      border-right: 1px solid #1E1E3A;
  }}
  .metric-card {{
      background: {CARD};
      border: 1px solid #1E1E3A;
      border-radius: 12px;
      padding: 20px 24px;
      text-align: center;
  }}
  .metric-label {{
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: {MUTED};
      margin-bottom: 6px;
  }}
  .metric-value {{
      font-size: 36px;
      font-weight: 700;
      color: {TEXT};
      line-height: 1;
  }}
  .metric-value.accent {{ color: {ACCENT}; }}
  .metric-value.success {{ color: {SUCCESS}; }}
  .metric-value.primary {{ color: {PRIMARY}; }}
  .risk-badge {{
      display: inline-block;
      padding: 6px 18px;
      border-radius: 999px;
      font-size: 14px;
      font-weight: 600;
      letter-spacing: 0.05em;
  }}
  .risk-high  {{ background:#FF4C4C22; color:{ACCENT};  border:1px solid {ACCENT}; }}
  .risk-med   {{ background:#FFB84422; color:#FFB844;   border:1px solid #FFB844; }}
  .risk-low   {{ background:#22D3A022; color:{SUCCESS}; border:1px solid {SUCCESS}; }}
  h1,h2,h3 {{ color: {TEXT} !important; }}
  .stSlider > label {{ color: {MUTED} !important; font-size: 12px; }}
  div[data-baseweb="select"] > div {{ background:{CARD} !important; border-color:#2A2A4A !important; }}
  .stButton button {{
      background: {PRIMARY};
      color: white;
      border: none;
      border-radius: 8px;
      font-weight: 600;
      padding: 10px 24px;
      width: 100%;
  }}
  .stButton button:hover {{ opacity: 0.88; }}
</style>
""", unsafe_allow_html=True)


# ── Datos y modelo (cacheados) ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Entrenando modelo…")
def entrenar():
    np.random.seed(42)
    n = 80
    datos = pd.DataFrame({
        "edad":            np.random.randint(22, 65, n),
        "antiguedad":      np.random.randint(1, 48, n),
        "plan":            np.random.choice(["Basico", "Premium"], n),
        "incidencias":     np.random.randint(0, 6, n),
        "uso_horas":       np.random.randint(5, 100, n),
        "pagos_atrasados": np.random.randint(0, 5, n),
    })
    datos["evento"] = (datos["incidencias"] + datos["pagos_atrasados"] > 3)
    datos["tiempo"] = np.random.randint(3, 48, n)

    X = pd.get_dummies(datos.drop(columns=["evento", "tiempo"]))
    y = Surv.from_arrays(event=datos["evento"], time=datos["tiempo"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    modelo = RandomSurvivalForest(
        n_estimators=100, min_samples_leaf=5, random_state=42
    )
    modelo.fit(X_train, y_train)

    scores = modelo.predict(X_test)
    eventos = np.array([e[0] for e in y_test])
    tiempos = np.array([e[1] for e in y_test])
    c_idx, *_ = concordance_index_censored(eventos, tiempos, scores)

    # Importancia de variables
    importancias = pd.Series(
        modelo.feature_importances_, index=X_train.columns
    ).sort_values(ascending=True)

    return modelo, X_train, X_test, y_train, y_test, c_idx, importancias, datos


modelo, X_train, X_test, y_train, y_test, c_idx, importancias, datos = entrenar()


# ── Sidebar · Cliente nuevo ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧍 Perfil del cliente")
    edad            = st.slider("Edad",             22, 65, 33)
    antiguedad      = st.slider("Antigüedad (meses)", 1, 48, 10)
    uso_horas       = st.slider("Uso (horas/mes)",   5, 100, 30)
    incidencias     = st.slider("Incidencias",        0, 5, 2)
    pagos_atrasados = st.slider("Pagos atrasados",    0, 4, 1)
    plan            = st.selectbox("Plan", ["Premium", "Basico"])

    st.markdown("---")
    predecir = st.button("Analizar cliente →")


# ── Cabecera ──────────────────────────────────────────────────────────────────
st.markdown("## 📦 Análisis de Churn · Random Survival Forest")
st.markdown(
    f"<span style='color:{MUTED};font-size:13px'>"
    "Modelo entrenado con datos sintéticos de clientes · "
    f"C-index test: <b style='color:{SUCCESS}'>{c_idx:.4f}</b>"
    "</span>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ── Métricas del modelo ───────────────────────────────────────────────────────
churn_rate = datos["evento"].mean()
m1, m2, m3, m4 = st.columns(4)

def metric_card(col, label, value, css_class=""):
    col.markdown(
        f"<div class='metric-card'>"
        f"  <div class='metric-label'>{label}</div>"
        f"  <div class='metric-value {css_class}'>{value}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

metric_card(m1, "Clientes entrenamiento", X_train.shape[0], "primary")
metric_card(m2, "Clientes prueba",        X_test.shape[0],  "primary")
metric_card(m3, "C-index (test)",         f"{c_idx:.4f}",   "success")
metric_card(m4, "Tasa de churn",          f"{churn_rate:.0%}", "accent")

st.markdown("<br>", unsafe_allow_html=True)


# ── Gráficos del modelo ───────────────────────────────────────────────────────
col_imp, col_dist = st.columns([1, 1])

# — Importancia de variables —
with col_imp:
    st.markdown("#### Importancia de variables")
    fig, ax = plt.subplots(figsize=(5, 3.4))
    colors = [PRIMARY if v == importancias.max() else "#2A2A6A" for v in importancias.values]
    ax.barh(importancias.index, importancias.values, color=colors, height=0.6)
    ax.set_xlabel("Importancia")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="x", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

# — Distribución de tiempos —
with col_dist:
    st.markdown("#### Distribución de tiempos al evento")
    fig2, ax2 = plt.subplots(figsize=(5, 3.4))
    tiempos_all = np.array([e[1] for e in Surv.from_arrays(
        event=datos["evento"], time=datos["tiempo"]
    )])
    ax2.hist(tiempos_all, bins=16, color=PRIMARY, alpha=0.85, edgecolor=SURFACE)
    ax2.set_xlabel("Tiempo (meses)")
    ax2.set_ylabel("Clientes")
    ax2.grid(axis="y", alpha=0.4)
    ax2.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)


# ── Predicción cliente ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Predicción para cliente nuevo")

nuevo_cliente = pd.DataFrame({
    "edad":            [edad],
    "antiguedad":      [antiguedad],
    "incidencias":     [incidencias],
    "uso_horas":       [uso_horas],
    "pagos_atrasados": [pagos_atrasados],
    "plan_Basico":     [1 if plan == "Basico" else 0],
    "plan_Premium":    [1 if plan == "Premium" else 0],
})

curva   = modelo.predict_survival_function(nuevo_cliente)
riesgo  = float(modelo.predict(nuevo_cliente)[0])

# Umbral de riesgo relativo al conjunto de entrenamiento
riesgo_train = modelo.predict(X_train)
p25 = np.percentile(riesgo_train, 33)
p75 = np.percentile(riesgo_train, 66)

if riesgo > p75:
    nivel, badge_cls, consejo = "ALTO", "risk-high", "Considera una llamada de retención proactiva."
elif riesgo > p25:
    nivel, badge_cls, consejo = "MEDIO", "risk-med", "Monitorear incidencias y pagos en las próximas semanas."
else:
    nivel, badge_cls, consejo = "BAJO", "risk-low", "Cliente estable. Priorizar otros perfiles."

col_curva, col_info = st.columns([2, 1])

with col_curva:
    fig3, ax3 = plt.subplots(figsize=(6, 3.6))
    ax3.plot(curva[0].x, curva[0].y, color=PRIMARY, lw=2.5, label="Prob. supervivencia")
    ax3.fill_between(curva[0].x, curva[0].y, alpha=0.12, color=PRIMARY)
    ax3.axhline(0.5, color=ACCENT, lw=1.2, ls="--", label="Umbral 50 %")
    ax3.set_xlabel("Tiempo (meses)")
    ax3.set_ylabel("P(no churn)")
    ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.35)
    ax3.spines[["top", "right"]].set_visible(False)
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)

with col_info:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='metric-card'>"
        f"  <div class='metric-label'>Puntaje de riesgo</div>"
        f"  <div class='metric-value accent'>{riesgo:.2f}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align:center'>"
        f"  <span class='risk-badge {badge_cls}'>Riesgo {nivel}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<br><p style='color:{MUTED};font-size:13px;text-align:center'>{consejo}</p>", unsafe_allow_html=True)

if not predecir:
    st.info("Ajusta los sliders del sidebar y pulsa **Analizar cliente →** para ver el resultado.")