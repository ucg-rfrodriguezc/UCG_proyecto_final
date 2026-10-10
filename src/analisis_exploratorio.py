# Análisis exploratorio del riesgo crediticio.
# Contiene la clase AnalizadorRiesgo, que concentra todos los cálculos.
# El agente de IA recibirá unicamente resultados calculados de aquí.

import numpy as np
import pandas as pd

from src.carga_de_datos import COLUMNAS_CATEGORICAS, COLUMNAS_NUMERICAS, TARGET

# Segmentos para variables numéricas: (límites, etiquetas)
SEGMENTOS = {
    "person_age": ([0, 25, 35, 45, 60, 100], ["<=25", "26-35", "36-45", "46-60", ">60"]),
    "person_income": (
        [0, 30000, 60000, 100000, np.inf],
        ["<30k", "30k-60k", "60k-100k", ">100k"],
    ),
    "loan_percent_income": (
        [-0.01, 0.1, 0.2, 0.3, 1],
        ["<=10%", "10-20%", "20-30%", ">30%"],
    ),
    "loan_int_rate": ([0, 8, 11, 14, 30], ["<8%", "8-11%", "11-14%", ">14%"]),
    "loan_amnt": ([0, 5000, 10000, 15000, np.inf], ["<5k", "5k-10k", "10k-15k", ">15k"]),
}

# Clase Analizador de Riesgo.
# Calcula perfilado, segmentación y correlaciones sobre el dataset.
class AnalizadorRiesgo:
    def __init__(self, df: pd.DataFrame, target: str = TARGET):
        if target not in df.columns:
            raise ValueError(f"La columna objetivo '{target}' no existe.")
        self.df = df
        self.target = target

    # ---------- Perfilado ----------
    # Indicadores globales del dataset.
    def perfil_general(self) -> dict:

        total = len(self.df)
        impagos = int(self.df[self.target].sum())
        return {
            "clientes": total,
            "impagos": impagos,
            "tasa_impago_pct": round(100 * impagos / total, 2),
            "monto_promedio": round(float(self.df["loan_amnt"].mean()), 2),
            "ingreso_mediano": round(float(self.df["person_income"].median()), 2),
            "tasa_interes_promedio": round(float(self.df["loan_int_rate"].mean()), 2),
        }

    # Tabla de nulos por columna (cantidad y porcentaje).
    def resumen_nulos(self) -> pd.DataFrame:
        nulos = self.df.isna().sum()
        tabla = pd.DataFrame(
            {"nulos": nulos, "porcentaje": (100 * nulos / len(self.df)).round(2)}
        )
        return tabla.sort_values("nulos", ascending=False)

    # Estadísticos descriptivos de las variables numéricas.
    def resumen_numerico(self) -> pd.DataFrame:
        return self.df[COLUMNAS_NUMERICAS].describe().T.round(2)

    # ---------- Segmentación ----------
    # Tabla estándar: clientes, impagos y tasa de impago por grupo.
    def _tabla_riesgo(self, grupo: pd.Series, nombre: str) -> pd.DataFrame:
        tabla = (
            self.df.groupby(grupo, observed=True)[self.target]
            .agg(clientes="count", impagos="sum", tasa_impago="mean")
            .reset_index()
            .rename(columns={grupo.name or "index": nombre})
        )
        tabla["tasa_impago"] = (100 * tabla["tasa_impago"]).round(2)
        return tabla

    # Tasa de impago (%) por cada valor de una variable categórica.
    def tasa_por_categoria(self, columna: str) -> pd.DataFrame:
        if columna not in COLUMNAS_CATEGORICAS:
            raise ValueError(f"'{columna}' no es una variable categórica válida.")
        tabla = self._tabla_riesgo(self.df[columna], columna)
        return tabla.sort_values("tasa_impago", ascending=False).reset_index(drop=True)

    # Tasa de impago (%) por tramos de una variable numérica.
    def tasa_por_segmento(self, columna: str) -> pd.DataFrame:
        if columna not in SEGMENTOS:
            raise ValueError(
                f"Sin segmentación definida para '{columna}'. "
                f"Opciones: {list(SEGMENTOS)}"
            )
        limites, etiquetas = SEGMENTOS[columna]
        grupo = pd.cut(self.df[columna], bins=limites, labels=etiquetas)
        grupo.name = columna
        return self._tabla_riesgo(grupo, columna)

    # ---------- Correlaciones ----------
    # Correlación de Pearson entre numéricas y el objetivo.
    def matriz_correlacion(self) -> pd.DataFrame:
        columnas = COLUMNAS_NUMERICAS + [self.target]
        return self.df[columnas].corr().round(3)

    # Correlación de cada numérica con el impago, ordenada por magnitud.
    def correlacion_con_objetivo(self) -> pd.Series:
        corr = self.matriz_correlacion()[self.target].drop(self.target)
        return corr.reindex(corr.abs().sort_values(ascending=False).index)

    # ---------- Ranking y contexto para el agente ----------
    # Los segmentos con mayor tasa de impago, entre todas las variables.
    def ranking_riesgo(self, minimo_clientes: int = 100) -> pd.DataFrame:
        filas = []
        for col in COLUMNAS_CATEGORICAS:
            for _, r in self.tasa_por_categoria(col).iterrows():
                filas.append((col, str(r[col]), r["clientes"], r["tasa_impago"]))
        for col in SEGMENTOS:
            for _, r in self.tasa_por_segmento(col).iterrows():
                filas.append((col, str(r[col]), r["clientes"], r["tasa_impago"]))
        ranking = pd.DataFrame(
            filas, columns=["variable", "segmento", "clientes", "tasa_impago"]
        )
        ranking = ranking[ranking["clientes"] >= minimo_clientes]
        return ranking.sort_values("tasa_impago", ascending=False).reset_index(drop=True)

    # Texto con resultados YA calculados, para pasarlo al LLM.
    def contexto_para_agente(self) -> str:
        p = self.perfil_general()
        top = self.ranking_riesgo().head(5)
        bajo = self.ranking_riesgo().tail(3)
        corr = self.correlacion_con_objetivo().head(4)

        lineas = [
            f"Clientes: {p['clientes']}. Tasa global de impago: {p['tasa_impago_pct']}%.",
            "Segmentos de MAYOR riesgo (variable | segmento | clientes | % impago):",
        ]
        for _, r in top.iterrows():
            lineas.append(f"- {r['variable']} | {r['segmento']} | {r['clientes']} | {r['tasa_impago']}%")
        lineas.append("Segmentos de MENOR riesgo:")
        for _, r in bajo.iterrows():
            lineas.append(f"- {r['variable']} | {r['segmento']} | {r['clientes']} | {r['tasa_impago']}%")
        lineas.append("Correlación con impago (Pearson):")
        for nombre, valor in corr.items():
            lineas.append(f"- {nombre}: {valor}")
        return "\n".join(lineas)
