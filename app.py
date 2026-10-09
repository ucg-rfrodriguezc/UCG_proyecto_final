import streamlit as st
from src.data_loader import cargar_datos, limpiar_datos, reporte_calidad

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

from src.eda import AnalizadorRiesgo

analizador = AnalizadorRiesgo(limpio)

st.header("Prueba del módulo EDA")
st.json(analizador.perfil_general())
st.subheader("Tasa de impago por grado")
st.dataframe(analizador.tasa_por_categoria("loan_grade"))
st.subheader("Correlación con impago")
st.dataframe(analizador.correlacion_con_objetivo())
st.text(analizador.contexto_para_agente())