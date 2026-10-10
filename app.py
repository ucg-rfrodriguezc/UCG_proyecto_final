import streamlit as st
from src.carga_de_datos import cargar_datos, limpiar_datos, reporte_calidad

st.set_page_config(page_title="Riesgo Crediticio", layout="wide")

@st.cache_data
def obtener_datos():
    crudo = cargar_datos()
    reporte = reporte_calidad(crudo)
    limpio, bitacora = limpiar_datos(crudo)
    return crudo, limpio, reporte, bitacora

crudo, limpio, reporte, bitacora = obtener_datos()

st.title("Análisis de Riesgo Crediticio")
st.subheader("Reporte de calidad (datos crudos)")
st.json(reporte)
st.subheader("Bitácora de limpieza")
st.json(bitacora)
st.dataframe(limpio.head(20))

from src.analisis_exploratorio import AnalizadorRiesgo

analizador = AnalizadorRiesgo(limpio)

st.header("Resumen de riesgo")

# --- Indicadores principales ---
p = analizador.perfil_general()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes", f"{p['clientes']:,}")
c2.metric("Tasa de impago", f"{p['tasa_impago_pct']}%")
c3.metric("Monto promedio", f"${p['monto_promedio']:,.0f}")
c4.metric("Ingreso mediano", f"${p['ingreso_mediano']:,.0f}")

# --- Segmentos de mayor y menor riesgo ---
ranking = analizador.ranking_riesgo().rename(
    columns={
        "variable": "Variable",
        "segmento": "Segmento",
        "clientes": "Clientes",
        "tasa_impago": "% impago",
    }
)
config_impago = st.column_config.ProgressColumn(
    "% impago", format="%.1f%%", min_value=0, max_value=100
)

izq, der = st.columns(2)
with izq:
    st.subheader("Mayor riesgo")
    st.dataframe(
        ranking.head(5),
        hide_index=True,
        column_config={"% impago": config_impago},
    )
with der:
    st.subheader("Menor riesgo")
    st.dataframe(
        ranking.tail(3).iloc[::-1],
        hide_index=True,
        column_config={"% impago": config_impago},
    )

# --- Correlaciones ---
st.subheader("Correlación con el impago (Pearson)")
corr = analizador.correlacion_con_objetivo().rename("Correlación").reset_index()
corr.columns = ["Variable", "Correlación"]
st.dataframe(
    corr,
    hide_index=True,
    column_config={
        "Correlación": st.column_config.NumberColumn(format="%.3f")
    },
)

# --- Contexto que recibirá el agente (útil para el video) ---
with st.expander("Ver contexto que se enviará al agente de IA"):
    st.code(analizador.contexto_para_agente(), language="text")


from src.carga_de_datos import COLUMNAS_NUMERICAS
from src.visualizador_de_graficos import (
    boxplot_por_estado,
    dispersion_riesgo,
    grafico_distribucion_objetivo,
    grafico_tasa_impago,
    heatmap_correlacion,
    heatmap_cruzado_seaborn,
    histograma_numerico,
)

st.header("Prueba de visualizaciones")
tasa_global = analizador.perfil_general()["tasa_impago_pct"]

t1, t2, t3, t4 = st.tabs(["Objetivo", "Riesgo por grupo", "Numéricas", "Relaciones"])

with t1:
    st.plotly_chart(grafico_distribucion_objetivo(limpio))

with t2:
    tipo = st.radio("Tipo de variable", ["Categórica", "Numérica segmentada"], horizontal=True)
    if tipo == "Categórica":
        col = st.selectbox("Variable", ["loan_grade", "loan_intent", "person_home_ownership", "cb_person_default_on_file"])
        tabla = analizador.tasa_por_categoria(col)
    else:
        col = st.selectbox("Variable", ["loan_percent_income", "loan_int_rate", "person_income", "person_age", "loan_amnt"])
        tabla = analizador.tasa_por_segmento(col)
    st.plotly_chart(grafico_tasa_impago(tabla, col, tasa_global, f"% de impago por {col}"))

with t3:
    col_num = st.selectbox("Variable numérica", COLUMNAS_NUMERICAS)
    log = st.checkbox("Escala logarítmica", value=(col_num == "person_income"))
    st.plotly_chart(histograma_numerico(limpio, col_num, escala_log=log))
    st.plotly_chart(boxplot_por_estado(limpio, col_num))

with t4:
    st.plotly_chart(heatmap_correlacion(analizador.matriz_correlacion()))
    st.plotly_chart(dispersion_riesgo(limpio, "loan_int_rate", "loan_percent_income"))
    st.pyplot(heatmap_cruzado_seaborn(limpio, "loan_grade", "loan_intent"))