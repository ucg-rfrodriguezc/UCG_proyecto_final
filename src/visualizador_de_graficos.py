# Gráficos del proyecto (Plotly como principal, Seaborn como complemento).
# Todas las funciones reciben datos ya calculados y devuelven una figura;
# no usan Streamlit, así que app.py decide dónde mostrarlas.

import matplotlib

matplotlib.use("Agg")  # backend sin ventana, compatible con Streamlit
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from src.carga_de_datos import TARGET

COLOR_PAGO = "#2E86AB"
COLOR_IMPAGO = "#E4572E"
ETIQUETAS_ESTADO = {0: "Pagó", 1: "Impago"}
MAPA_COLORES = {"Pagó": COLOR_PAGO, "Impago": COLOR_IMPAGO}
PLANTILLA = "plotly_white"

# Copia del DataFrame con columna 'estado' legible (Pagó / Impago).
def _con_etiqueta(df: pd.DataFrame) -> pd.DataFrame:
    copia = df.copy()
    copia["estado"] = copia[TARGET].map(ETIQUETAS_ESTADO)
    return copia

# Dona con la proporción de pagos e impagos (variable binaria).
def grafico_distribucion_objetivo(df: pd.DataFrame) -> go.Figure:
    conteo = _con_etiqueta(df)["estado"].value_counts().reset_index()
    conteo.columns = ["estado", "clientes"]
    fig = px.pie(
        conteo, names="estado", values="clientes", hole=0.55,
        color="estado", color_discrete_map=MAPA_COLORES,
        title="Distribución de la variable objetivo",
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(template=PLANTILLA, showlegend=False)
    return fig

# Barras de % de impago por grupo, con línea de la tasa global.
# Sirve para variables categóricas y para numéricas segmentadas 
# (usa la tabla de AnalizadorRiesgo).
def grafico_tasa_impago(tabla: pd.DataFrame, columna: str,
                        tasa_global: float, titulo: str) -> go.Figure:
    fig = px.bar(
        tabla, x=columna, y="tasa_impago", text="tasa_impago",
        hover_data={"clientes": ":,", "impagos": ":,"},
        title=titulo, labels={"tasa_impago": "% de impago", columna: ""},
    )
    fig.update_traces(marker_color=COLOR_IMPAGO, texttemplate="%{text:.1f}%",
                      textposition="outside")
    fig.add_hline(
        y=tasa_global, line_dash="dash", line_color="gray",
        annotation_text=f"Promedio global: {tasa_global:.1f}%",
        annotation_position="top left",
    )
    fig.update_layout(template=PLANTILLA, yaxis_range=[0, 110])
    return fig

# Histograma superpuesto por estado, con boxplot marginal.
# escala_log: útil para ingreso, que tiene valores extremos.
def histograma_numerico(df: pd.DataFrame, columna: str,
                        escala_log: bool = False) -> go.Figure:
    fig = px.histogram(
        _con_etiqueta(df), x=columna, color="estado", nbins=50,
        barmode="overlay", opacity=0.65, marginal="box",
        color_discrete_map=MAPA_COLORES, log_x=escala_log,
        title=f"Distribución de {columna} según estado del préstamo",
    )
    fig.update_layout(template=PLANTILLA, legend_title_text="")
    return fig

# Boxplot de una variable numérica comparando Pagó vs Impago.
def boxplot_por_estado(df: pd.DataFrame, columna: str) -> go.Figure:
    fig = px.box(
        _con_etiqueta(df), x="estado", y=columna, color="estado",
        color_discrete_map=MAPA_COLORES,
        title=f"{columna} por estado del préstamo",
    )
    fig.update_layout(template=PLANTILLA, showlegend=False, xaxis_title="")
    return fig

# Mapa de calor de correlaciones de Pearson (rango -1 a 1).
def heatmap_correlacion(matriz: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        matriz, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Matriz de correlación (Pearson)",
    )
    fig.update_layout(template=PLANTILLA)
    return fig

# Dispersión de dos numéricas coloreada por estado (con muestreo).
def dispersion_riesgo(df: pd.DataFrame, x: str, y: str,
                      muestra: int = 3000) -> go.Figure:
    datos = _con_etiqueta(df)
    if len(datos) > muestra:  # evita sobrecargar el navegador
        datos = datos.sample(muestra, random_state=42)
    fig = px.scatter(
        datos, x=x, y=y, color="estado", opacity=0.55,
        color_discrete_map=MAPA_COLORES,
        title=f"{y} vs {x} (muestra de {len(datos):,} clientes)",
    )
    fig.update_layout(template=PLANTILLA, legend_title_text="")
    return fig

# Heatmap Seaborn: % de impago al cruzar dos variables categóricas.
def heatmap_cruzado_seaborn(df: pd.DataFrame, fila: str, columna: str):
    tabla = df.pivot_table(index=fila, columns=columna, values=TARGET,
                           aggfunc="mean") * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(tabla, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "% de impago"})
    ax.set_title(f"% de impago: {fila} vs {columna}")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig
