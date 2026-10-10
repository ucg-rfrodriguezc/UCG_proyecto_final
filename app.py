# Punto de entrada de la aplicación.
# Permite coordinar: 
#   1. Carga los datos
#   2. Crea los objetos de análisis
#   3. Crea agente
#   4. Delega cada pestaña a src/interfaz.py.
import streamlit as st

from src import interfaz
from src.agente import AgenteRiesgo
from src.analisis_exploratorio import AnalizadorRiesgo
from src.carga_de_datos import cargar_datos, limpiar_datos, reporte_calidad

st.set_page_config(page_title="Riesgo crediticio", page_icon="📊", layout="wide")

@st.cache_data
def obtener_datos():
    crudo = cargar_datos()
    reporte = reporte_calidad(crudo)
    limpio, bitacora = limpiar_datos(crudo)
    return limpio, reporte, bitacora

limpio, reporte, bitacora = obtener_datos()
analizador = AnalizadorRiesgo(limpio)

interfaz.aplicar_estilos()
cliente, estado = interfaz.barra_lateral(limpio, reporte)
agente = AgenteRiesgo(analizador, cliente, bitacora)

st.title("Análisis de riesgo crediticio")
st.write(
    "¿Qué características del solicitante y del préstamo se asocian con el impago? "
    "Explora los datos y consulta al agente de IA."
)

pestanas = st.tabs(["Resumen", 
                    "Calidad de datos", 
                    "Exploración", 
                    "Agente de IA", 
                    "Proyecto"])
with pestanas[0]:
    interfaz.seccion_resumen(analizador)
with pestanas[1]:
    interfaz.seccion_calidad(reporte, bitacora, analizador)
with pestanas[2]:
    interfaz.seccion_exploracion(limpio, analizador)
with pestanas[3]:
    interfaz.seccion_agente(agente, cliente, estado, analizador)
with pestanas[4]:
    interfaz.seccion_proyecto()
