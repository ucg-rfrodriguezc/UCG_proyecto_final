# Interfaz de la aplicación (Streamlit).
# Este módulo contiene SOLO presentación: cada función dibuja una sección
# con datos que ya calcularon los demás módulos (análisis, gráficos y agente).
import time
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.agente import PROMPT_SISTEMA
from src.analisis_exploratorio import SEGMENTOS
from src.carga_de_datos import COLUMNAS_CATEGORICAS, COLUMNAS_NUMERICAS
from src.cliente_llm import MODELO_POR_DEFECTO, ClienteOllama
from src.visualizador_de_graficos import (
    boxplot_por_estado,
    dispersion_riesgo,
    grafico_correlacion_objetivo,
    grafico_distribucion_objetivo,
    grafico_tasa_impago,
    heatmap_correlacion,
    heatmap_cruzado_seaborn,
    histograma_numerico,
)

URL_DATASET = "https://www.kaggle.com/datasets/tahmidmir/credit-risk-dataset"

ETIQUETAS = {
    "person_age": "Edad",
    "person_income": "Ingreso anual",
    "person_home_ownership": "Tipo de vivienda",
    "person_emp_length": "Antigüedad laboral",
    "loan_intent": "Propósito del préstamo",
    "loan_grade": "Grado del préstamo",
    "loan_amnt": "Monto del préstamo",
    "loan_int_rate": "Tasa de interés",
    "loan_percent_income": "Préstamo sobre ingreso",
    "cb_person_default_on_file": "Incumplimiento previo",
    "cb_person_cred_hist_length": "Historial crediticio (años)",
    "loan_status": "Impago",
}

SUGERENCIAS = [
    "¿Cómo influye el grado del préstamo en el impago?",
    "¿Qué pasa con quienes piden más del 30% de su ingreso?",
    "¿Qué variables se correlacionan más con el impago?",
    "¿Para qué propósito de préstamo hay más riesgo?",
]

ESTILOS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1300px;}
[data-testid="stMetricValue"] {font-size: 1.75rem; font-weight: 700;}
[data-testid="stMetricLabel"] p {font-size: 0.85rem; opacity: 0.75;}
.stTabs [data-baseweb="tab-list"] {gap: 0.4rem;}
.stTabs [data-baseweb="tab"] {font-size: 1.02rem; font-weight: 600; padding: 0.6rem 1.1rem;}
</style>
"""

CONFIG_IMPAGO = st.column_config.ProgressColumn(
    "% impago", format="%.1f%%", min_value=0, max_value=100)

# ---------------------------------------------------------------- utilidades
# Nombre legible de una variable.
def _et(columna: str) -> str:
    return ETIQUETAS.get(columna, columna)

# Evita que Markdown interprete símbolos ($, <, >) como formato.
def _seguro(texto) -> str:
    return str(texto).replace("$", "\\$").replace("<", "&lt;").replace(">", "&gt;")

# Bloque con borde: título pequeño, dato principal y detalle.
def _tarjeta(titulo: str, destacado: str, detalle: str) -> None:
    with st.container(border=True):
        st.caption(titulo)
        st.markdown(f"**{destacado}**")
        st.write(detalle)

# Consulta el estado de Ollama (se guarda 15s para no repetir la consulta)."""
@st.cache_data(ttl=15, show_spinner=False)
def _estado_ollama(modelo: str) -> dict:
    return ClienteOllama(modelo).estado()

def aplicar_estilos() -> None:
    st.markdown(ESTILOS, unsafe_allow_html=True)

# ------------------------------------------------------------- barra lateral
# Crea la barra lateral y devuelve (cliente, estado) con el modelo elegido.
def barra_lateral(df: pd.DataFrame, reporte: dict):
    with st.sidebar:
        st.title("Riesgo crediticio")
        st.caption("Proyecto Final Integrador. Maestría en Inteligencia Artificial y Ciencia de Datos, UCG.")
        st.divider()

        st.subheader("Datos")
        st.write(f"**{len(df):,}** clientes analizados")
        st.caption(
            f"{reporte['filas']:,} registros originales y {reporte['columnas']} variables."
        )
        st.link_button("Ver dataset en Kaggle", URL_DATASET)
        st.divider()

        st.subheader("Modelo de IA local")
        base = _estado_ollama(MODELO_POR_DEFECTO)
        instalados = base["instalados"]
        if instalados:
            indice = instalados.index(MODELO_POR_DEFECTO) if MODELO_POR_DEFECTO in instalados else 0
            modelo = st.selectbox("Modelo de Ollama", instalados, index=indice)
        else:
            modelo = MODELO_POR_DEFECTO

        estado = _estado_ollama(modelo)
        if estado["conectado"] and estado["modelo_disponible"]:
            st.success(f"Conectado a Ollama con {modelo}.")
        elif estado["conectado"]:
            st.warning(estado["mensaje"])
        else:
            st.error(estado["mensaje"])
        if st.button("Volver a comprobar"):
            _estado_ollama.clear()
            st.rerun()

    return ClienteOllama(modelo), estado

# ------------------------------------------------------------------- resumen
def seccion_resumen(analizador) -> None:
    p = analizador.perfil_general()
    ranking = analizador.ranking_riesgo()
    corr = analizador.correlacion_con_objetivo()

    indicadores = [
        ("Clientes", f"{p['clientes']:,}"),
        ("Impagos", f"{p['impagos']:,}"),
        ("Tasa de impago", f"{p['tasa_impago_pct']}%"),
        ("Monto promedio", f"${p['monto_promedio']:,.0f}"),
        ("Ingreso mediano", f"${p['ingreso_mediano']:,.0f}"),
    ]
    for col, (etiqueta, valor) in zip(st.columns(5), indicadores):
        with col:
            with st.container(border=True):
                st.metric(etiqueta, valor)

    izq, der = st.columns([1, 1.3])
    with izq:
        st.plotly_chart(grafico_distribucion_objetivo(analizador.df), key="res_dona")
        st.caption(
            "Solo 1 de cada 5 préstamos termina en impago: los datos están desbalanceados."
        )
    with der:
        st.subheader("Hallazgos clave")
        mayor, menor = ranking.iloc[0], ranking.iloc[-1]
        veces = mayor["tasa_impago"] / p["tasa_impago_pct"]
        _tarjeta(
            "Segmento de mayor riesgo",
            f"{_et(mayor['variable'])}: {_seguro(mayor['segmento'])}",
            f"{mayor['tasa_impago']:.1f}% de impago, {veces:.1f} veces el promedio "
            f"({mayor['clientes']:,} clientes).",
        )
        _tarjeta(
            "Segmento de menor riesgo",
            f"{_et(menor['variable'])}: {_seguro(menor['segmento'])}",
            f"{menor['tasa_impago']:.1f}% de impago ({menor['clientes']:,} clientes).",
        )
        _tarjeta(
            "Variable numérica más asociada al impago",
            _et(corr.index[0]),
            f"Correlación de {corr.iloc[0]:+.2f} con el impago.",
        )

    st.subheader("Segmentos según su nivel de riesgo")
    vista = ranking.assign(Variable=ranking["variable"].map(_et))[
        ["Variable", "segmento", "clientes", "tasa_impago"]
    ].rename(columns={"segmento": "Segmento", "clientes": "Clientes", "tasa_impago": "% impago"})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Mayor riesgo**")
        st.dataframe(vista.head(5), hide_index=True, column_config={"% impago": CONFIG_IMPAGO})
    with c2:
        st.markdown("**Menor riesgo**")
        st.dataframe(
            vista.tail(3).iloc[::-1], hide_index=True, column_config={"% impago": CONFIG_IMPAGO}
        )
    st.caption(
        "Solo se listan segmentos con al menos 100 clientes, para evitar conclusiones "
        "con muestras pequeñas."
    )

    st.subheader("Qué variables numéricas se asocian con el impago")
    st.plotly_chart(
        grafico_correlacion_objetivo(corr.rename(index=_et)), key="res_correlacion"
    )
    st.caption("La correlación mide una relación lineal. No demuestra que una variable cause el impago.")

# ------------------------------------------------------------------- calidad
def seccion_calidad(reporte: dict, bitacora: dict, analizador) -> None:
    quitadas = bitacora["filas_iniciales"] - bitacora["filas_finales"]
    imputados = bitacora["emp_length_imputados"] + bitacora["int_rate_imputados"]
    metricas = [
        ("Filas originales", f"{bitacora['filas_iniciales']:,}", None),
        ("Filas tras la limpieza", f"{bitacora['filas_finales']:,}", f"-{quitadas:,}"),
        ("Duplicados eliminados", f"{bitacora['duplicados_eliminados']:,}", None),
        ("Valores imputados", f"{imputados:,}", None),
    ]
    for col, (etiqueta, valor, delta) in zip(st.columns(4), metricas):
        with col:
            with st.container(border=True):
                st.metric(etiqueta, valor, delta=delta, delta_color="off")

    st.subheader("Problemas encontrados y cómo se resolvieron")
    acciones = pd.DataFrame(
        [
            ("Registros duplicados", "Se eliminan", bitacora["duplicados_eliminados"]),
            ("Edad mayor a 100 años", "Se elimina el registro (dato imposible)",
             bitacora["filas_edad_invalida_eliminadas"]),
            ("Antigüedad laboral mayor a 60 años", "Se marca como nulo y se imputa",
             bitacora["antiguedad_invalida_a_nulo"]),
            ("Antigüedad laboral nula", "Se imputa con la mediana global",
             bitacora["emp_length_imputados"]),
            ("Tasa de interés nula", "Se imputa con la mediana de su grado de préstamo",
             bitacora["int_rate_imputados"]),
        ],
        columns=["Problema", "Regla aplicada", "Registros afectados"],
    )
    st.dataframe(acciones, hide_index=True)
    st.caption(
        "La antigüedad laboral nula incluye los registros marcados como inválidos arriba. "
        "La tasa de interés se imputa por grado porque ambas variables están muy ligadas."
    )

    izq, der = st.columns(2)
    with izq:
        st.subheader("Valores nulos en los datos originales")
        nulos = pd.DataFrame(
            [
                (_et(c), n, 100 * n / reporte["filas"])
                for c, n in reporte["nulos_por_columna"].items()
            ],
            columns=["Variable", "Nulos", "% del total"],
        )
        st.dataframe(
            nulos,
            hide_index=True,
            column_config={
                "% del total": st.column_config.ProgressColumn(
                    "% del total", format="%.1f%%", min_value=0, max_value=100
                )
            },
        )
    with der:
        st.subheader("Valores extremos que se conservan")
        maximo = analizador.df["person_income"].max()
        st.info(
            f"El ingreso anual llega a {maximo:,.0f}. Es un valor posible, así que no se "
            "elimina. Para no distorsionar los gráficos se usa escala logarítmica."
        )

    with st.expander("Ver reporte técnico completo"):
        st.json(reporte)
        st.json(bitacora)

# ---------------------------------------------------------------- exploración
def _explorar_grupos(analizador) -> None:
    opciones = COLUMNAS_CATEGORICAS + list(SEGMENTOS)
    col = st.selectbox("Variable a analizar", opciones, format_func=_et, key="expl_grupo")
    if col in COLUMNAS_CATEGORICAS:
        tabla = analizador.tasa_por_categoria(col)
    else:
        tabla = analizador.tasa_por_segmento(col)
    tasa_global = analizador.perfil_general()["tasa_impago_pct"]

    izq, der = st.columns([2, 1])
    with izq:
        titulo = f"% de impago según {_et(col).lower()}"
        st.plotly_chart(
            grafico_tasa_impago(tabla, col, tasa_global, titulo), key=f"expl_barras_{col}"
        )
    with der:
        alto = tabla.loc[tabla["tasa_impago"].idxmax()]
        bajo = tabla.loc[tabla["tasa_impago"].idxmin()]
        st.markdown("**Cómo leerlo**")
        st.write(
            f"La tasa más alta es {alto['tasa_impago']:.1f}% ({_seguro(alto[col])}), "
            f"{alto['tasa_impago'] / tasa_global:.1f} veces el promedio de {tasa_global}%. "
            f"La más baja es {bajo['tasa_impago']:.1f}% ({_seguro(bajo[col])})."
        )
        pequenos = tabla[tabla["clientes"] < 100]
        if not pequenos.empty:
            grupos = ", ".join(str(g) for g in pequenos[col])
            st.warning(f"Grupos con menos de 100 clientes ({grupos}): interpretar con cautela.")
        vista = tabla.rename(
            columns={col: _et(col), "clientes": "Clientes", "impagos": "Impagos",
                     "tasa_impago": "% impago"}
        )
        st.dataframe(vista, hide_index=True, column_config={"% impago": CONFIG_IMPAGO})

def _explorar_distribuciones(df: pd.DataFrame) -> None:
    col = st.selectbox("Variable numérica", COLUMNAS_NUMERICAS, format_func=_et, key="expl_num")
    log = st.checkbox("Escala logarítmica", value=(col == "person_income"), key=f"log_{col}")
    izq, der = st.columns(2)
    with izq:
        st.plotly_chart(histograma_numerico(df, col, escala_log=log), key=f"expl_hist_{col}")
    with der:
        st.plotly_chart(boxplot_por_estado(df, col), key=f"expl_box_{col}")
    st.caption(
        "En el boxplot, la caja contiene el 50% central de los clientes y los puntos "
        "aislados son valores atípicos. Si las cajas de Pagó e Impago se separan, la "
        "variable distingue entre ambos grupos."
    )

def _explorar_relaciones(df: pd.DataFrame, analizador) -> None:
    st.subheader("Correlación entre variables numéricas")
    st.plotly_chart(heatmap_correlacion(analizador.matriz_correlacion()), key="expl_heat")

    st.subheader("Dispersión de dos variables")
    c1, c2 = st.columns(2)
    with c1:
        x = st.selectbox("Eje X", COLUMNAS_NUMERICAS, index=4, format_func=_et, key="disp_x")
    with c2:
        y = st.selectbox("Eje Y", COLUMNAS_NUMERICAS, index=5, format_func=_et, key="disp_y")
    if x == y:
        st.info("Elige dos variables distintas.")
    else:
        st.plotly_chart(dispersion_riesgo(df, x, y), key=f"expl_disp_{x}_{y}")

    st.subheader("Cruce de dos variables categóricas")
    c3, c4 = st.columns(2)
    with c3:
        fila = st.selectbox("Filas", COLUMNAS_CATEGORICAS, index=2, format_func=_et, key="cruce_f")
    with c4:
        columna = st.selectbox("Columnas", COLUMNAS_CATEGORICAS, index=1, format_func=_et, key="cruce_c")
    if fila == columna:
        st.info("Elige dos variables distintas.")
    else:
        figura = heatmap_cruzado_seaborn(df, fila, columna)
        st.pyplot(figura)
        plt.close(figura)
        st.caption("Cada celda muestra el % de impago de esa combinación. Rojo oscuro indica mayor riesgo.")

def seccion_exploracion(df: pd.DataFrame, analizador) -> None:
    sub1, sub2, sub3 = st.tabs(["Riesgo por grupo", "Distribuciones", "Relaciones"])
    with sub1:
        _explorar_grupos(analizador)
    with sub2:
        _explorar_distribuciones(df)
    with sub3:
        _explorar_relaciones(df, analizador)

# --------------------------------------------------------------------- agente
# Gráfico asociado a una herramienta del agente, o None si no aplica.
def _grafico_evidencia(resultado, analizador):
    if resultado.tabla is None:
        return None
    if resultado.nombre.startswith("riesgo_"):
        col = resultado.nombre.replace("riesgo_", "", 1)
        tasa_global = analizador.perfil_general()["tasa_impago_pct"]
        return grafico_tasa_impago(
            resultado.tabla, col, tasa_global, f"% de impago según {_et(col).lower()}"
        )
    if resultado.nombre == "correlaciones":
        serie = resultado.tabla.set_index("variable")["correlacion"].rename(index=_et)
        return grafico_correlacion_objetivo(serie)
    return None

def _mostrar_turno(turno: dict, analizador) -> None:
    with st.chat_message("user"):
        st.markdown(_seguro(turno["pregunta"]))
    with st.chat_message("assistant"):
        if turno["error"]:
            st.error(turno["error"])
            st.caption("Los datos calculados siguen disponibles en la evidencia.")
        else:
            st.markdown(_seguro(turno["respuesta"]))
        herramientas = ", ".join(f"`{h}`" for h in turno["herramientas"])
        st.caption(
            f"Respondió en {turno['segundos']:.1f} s con {turno['modelo']}. "
            f"Herramientas usadas: {herramientas}."
        )
        with st.expander("Ver evidencia: datos calculados, gráfico y prompt"):
            t_datos, t_graf, t_prompt = st.tabs(
                ["Datos calculados", "Gráfico", "Prompt enviado al modelo"]
            )
            with t_datos:
                for r in turno["resultados"]:
                    st.markdown(f"**{r.descripcion}** (`{r.nombre}`)")
                    if r.tabla is not None:
                        st.dataframe(r.tabla, hide_index=True)
                    else:
                        st.write(r.texto)
            with t_graf:
                hubo_grafico = False
                for r in turno["resultados"]:
                    figura = _grafico_evidencia(r, analizador)
                    if figura is not None:
                        hubo_grafico = True
                        st.plotly_chart(figura, key=f"ev_{turno['id']}_{r.nombre}")
                if not hubo_grafico:
                    st.caption("Esta consulta no tiene un gráfico asociado.")
            with t_prompt:
                st.markdown("**Instrucciones del sistema**")
                st.code(PROMPT_SISTEMA, language="text")
                st.markdown("**Mensaje con los datos y la pregunta**")
                st.code(turno["prompt"], language="text")


def seccion_agente(agente, cliente, estado: dict, analizador) -> None:
    if "historial" not in st.session_state:
        st.session_state["historial"] = []
        st.session_state["n_turnos"] = 0

    st.write(
        "Haz una pregunta sobre el dataset. Python calcula las cifras y el modelo "
        "local solo las explica."
    )
    pasos = [
        ("1. Pregunta", "Python identifica qué análisis necesita tu pregunta."),
        ("2. Cálculo", "Python ejecuta las funciones y obtiene las cifras reales."),
        ("3. Interpretación", "El modelo local explica los resultados sin inventar datos."),
    ]
    for col, (titulo, detalle) in zip(st.columns(3), pasos):
        with col:
            with st.container(border=True):
                st.markdown(f"**{titulo}**")
                st.caption(detalle)

    if not (estado["conectado"] and estado["modelo_disponible"]):
        st.warning(f"{estado['mensaje']} Puedes preguntar igual: verás los datos calculados.")

    st.markdown("**Preguntas sugeridas**")
    pendiente = None
    columnas = st.columns(2)
    for i, texto in enumerate(SUGERENCIAS):
        with columnas[i % 2]:
            if st.button(texto, key=f"sugerencia_{i}"):
                pendiente = texto

    with st.form("form_pregunta", clear_on_submit=True):
        escrita = st.text_input(
            "O escribe tu pregunta",
            placeholder="Ejemplo: ¿Tener casa propia cambia el riesgo?",
        )
        enviar = st.form_submit_button("Preguntar")
    pregunta = pendiente or (escrita.strip() if enviar else "")

    if pregunta:
        with st.spinner("Calculando y consultando al modelo..."):
            inicio = time.perf_counter()
            turno = agente.responder(pregunta)
            turno["segundos"] = time.perf_counter() - inicio
        turno["modelo"] = cliente.modelo
        st.session_state["n_turnos"] += 1
        turno["id"] = st.session_state["n_turnos"]
        st.session_state["historial"].insert(0, turno)

    st.subheader("Conversación")
    if st.session_state["historial"]:
        if st.button("Limpiar conversación"):
            st.session_state["historial"] = []
    if not st.session_state["historial"]:
        st.info("Aún no hay preguntas. Elige una sugerida o escribe la tuya.")
    for turno in st.session_state["historial"]:
        _mostrar_turno(turno, analizador)

# ------------------------------------------------------------------- proyecto
def seccion_proyecto() -> None:
    st.subheader("Problema analítico")
    st.write(
        "¿Qué características del solicitante y del préstamo se asocian con un mayor "
        "riesgo de impago, y cómo se puede segmentar a los clientes según ese riesgo?"
    )

    st.subheader("Dataset")
    st.write(
        f"Credit Risk Dataset, publicado en [Kaggle]({URL_DATASET}). Contiene 32,581 "
        "solicitudes de préstamo con 12 variables. La variable objetivo es `loan_status`."
    )
    diccionario = pd.DataFrame(
        [
            ("person_age", "Edad del solicitante"),
            ("person_income", "Ingreso anual"),
            ("person_home_ownership", "Tipo de vivienda"),
            ("person_emp_length", "Años de antigüedad laboral"),
            ("loan_intent", "Propósito del préstamo"),
            ("loan_grade", "Grado del préstamo, de A (mejor) a G (peor)"),
            ("loan_amnt", "Monto del préstamo"),
            ("loan_int_rate", "Tasa de interés"),
            ("loan_status", "Variable objetivo: 0 = pagó, 1 = impago"),
            ("loan_percent_income", "Monto del préstamo como proporción del ingreso"),
            ("cb_person_default_on_file", "Incumplimiento previo registrado (Y/N)"),
            ("cb_person_cred_hist_length", "Años de historial crediticio"),
        ],
        columns=["Variable", "Descripción"],
    )
    st.dataframe(diccionario, hide_index=True)

    st.subheader("Arquitectura")
    arquitectura = pd.DataFrame(
        [
            ("app.py", "Punto de entrada: carga los datos y arma las pestañas"),
            ("src/interfaz.py", "Presentación: secciones, tarjetas y formularios"),
            ("src/carga_de_datos.py", "Lectura, validación y limpieza del CSV"),
            ("src/analisis_exploratorio.py", "Clase AnalizadorRiesgo: perfilado, segmentación y correlaciones"),
            ("src/visualizador_de_graficos.py", "Gráficos con Plotly y Seaborn"),
            ("src/cliente_llm.py", "Clase ClienteOllama: comunicación con el modelo local"),
            ("src/agente.py", "Clase AgenteRiesgo: herramientas autorizadas e interpretación"),
        ],
        columns=["Módulo", "Responsabilidad"],
    )
    st.dataframe(arquitectura, hide_index=True)

    st.subheader("Cómo se usa el modelo de lenguaje")
    st.write(
        "El modelo corre en tu equipo con Ollama (llama3.2:3b). No calcula nada: recibe "
        "resultados que Python ya obtuvo y los interpreta. Cada respuesta del agente "
        "muestra en su evidencia las herramientas usadas, los datos y el prompt enviado."
    )

    st.subheader("Alcance y limitaciones")
    st.markdown(
        "- Las correlaciones y tasas describen asociaciones, no causas.\n"
        "- Los grupos con pocos clientes (por ejemplo, el grado G) tienen cifras poco estables.\n"
        "- El agente elige herramientas por palabras clave: una pregunta con términos no "
        "previstos recibe un análisis general.\n"
        "- Un modelo pequeño puede equivocarse al redactar. Por eso cada respuesta se "
        "acompaña de las cifras calculadas para poder verificarla."
    )
