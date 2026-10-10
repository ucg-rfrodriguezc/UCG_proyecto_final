# Agente de IA para interpretar resultados de riesgo crediticio.
# Flujo:  pregunta -> selección de herramientas (Python) -> cálculo (Python) -> interpretación (LLM local vía Ollama).
# El modelo NUNCA calcula: solo recibe resultados ya calculados.
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.analisis_exploratorio import SEGMENTOS, AnalizadorRiesgo
from src.carga_de_datos import COLUMNAS_CATEGORICAS
from src.cliente_llm import ClienteOllama, ErrorOllama

MAX_HERRAMIENTAS = 3

PROMPT_SISTEMA = """Eres un analista de riesgo crediticio. Respondes en español.
Reglas estrictas:
1. Usa ÚNICAMENTE las cifras del bloque DATOS CALCULADOS. No inventes números.
2. Si los datos no permiten responder, dilo claramente.
3. Cita las cifras relevantes (porcentajes y cantidades de clientes).
4. Compara siempre con la tasa global de impago cuando sea útil.
5. Habla de asociación, no de causalidad.
6. Sé conciso: máximo 150 palabras."""

# Reglas de selección: (herramienta, patrón). Se evalúan en orden sobre la
# pregunta normalizada (minúsculas, sin tildes).
REGLAS = [
    ("riesgo_loan_percent_income", r"porcentaje|proporcion|carga financiera|cuota|percent|%|(de|del) (su |sus )?ingreso"),
    ("riesgo_person_income", r"ingreso|salario|sueldo|income"),
    ("riesgo_loan_grade", r"grado|grade|calificacion"),
    ("riesgo_loan_intent", r"proposito|motivo|finalidad|destino|intent|para que se pide"),
    ("riesgo_person_home_ownership", r"vivienda|casa propia|propiedad|alquil|hipoteca|home|rent"),
    ("riesgo_cb_person_default_on_file", r"historial|default|mora|morosidad|antecedente|incumplimiento previo"),
    ("riesgo_person_age", r"edad|joven|adulto|anos"),
    ("riesgo_loan_int_rate", r"\binteres\b|int_rate"),
    ("riesgo_loan_amnt", r"monto|importe|cantidad prestada|amnt"),
    ("correlaciones", r"correlacion|asocia|relacion entre|pesa|determina|factor"),
    ("calidad_datos", r"limpieza|nulo|calidad|duplicad|imputa|outlier|atipic"),
    ("ranking_riesgo", r"peor|segmento|perfil|critico|ranking|vulnerable|mayor riesgo|mas riesgo|alto riesgo"),
]
HERRAMIENTAS_POR_DEFECTO = ["resumen_general", "ranking_riesgo", "correlaciones"]


@dataclass
# Salida de una herramienta: texto para el LLM y tabla para la interfaz.
class ResultadoHerramienta:
    nombre: str
    descripcion: str
    texto: str
    tabla: Optional[pd.DataFrame] = None

# Minúsculas y sin tildes, para comparar con las reglas.
def _normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in sin_tildes if unicodedata.category(c) != "Mn")

# Agente que combina herramientas autorizadas con un modelo local.
class AgenteRiesgo:
    def __init__(self, analizador: AnalizadorRiesgo, cliente: ClienteOllama,
                 bitacora: Optional[dict] = None):
        self.analizador = analizador
        self.cliente = cliente
        self.bitacora = bitacora or {}
        self._herramientas = self._registrar_herramientas()

    # ---------- Registro de herramientas autorizadas ----------
    # Diccionario nombre -> (descripción, función). Solo estas se pueden usar.
    def _registrar_herramientas(self) -> dict:
        h = {
            "resumen_general": ("Indicadores globales del dataset", self._h_resumen),
            "ranking_riesgo": ("Segmentos de mayor y menor riesgo", self._h_ranking),
            "correlaciones": ("Correlación de variables numéricas con el impago", self._h_correlaciones),
            "calidad_datos": ("Bitácora de limpieza de datos", self._h_calidad),
        }
        for col in COLUMNAS_CATEGORICAS + list(SEGMENTOS):
            h[f"riesgo_{col}"] = (
                f"Tasa de impago según {col}",
                lambda c=col: self._h_riesgo_variable(c),
            )
        return h

    # Nombre -> descripción (para mostrar en la interfaz).
    def herramientas_disponibles(self) -> dict:
        return {n: d for n, (d, _) in self._herramientas.items()}

    # ---------- Implementación de herramientas ----------
    def _tasa_global(self) -> float:
        return self.analizador.perfil_general()["tasa_impago_pct"]

    def _h_resumen(self) -> ResultadoHerramienta:
        p = self.analizador.perfil_general()
        texto = (
            f"Clientes: {p['clientes']}. Impagos: {p['impagos']}. "
            f"Tasa global de impago: {p['tasa_impago_pct']}%. "
            f"Monto promedio: {p['monto_promedio']}. Ingreso mediano: {p['ingreso_mediano']}. "
            f"Tasa de interés promedio: {p['tasa_interes_promedio']}%."
        )
        return ResultadoHerramienta("resumen_general", "Indicadores globales", texto,
                                    pd.DataFrame([p]))

    def _h_riesgo_variable(self, col: str) -> ResultadoHerramienta:
        if col in COLUMNAS_CATEGORICAS:
            tabla = self.analizador.tasa_por_categoria(col)
        else:
            tabla = self.analizador.tasa_por_segmento(col)
        lineas = [f"Tasa global de impago: {self._tasa_global()}%. Impago por {col}:"]
        for _, r in tabla.iterrows():
            lineas.append(f"- {r[col]}: {r['clientes']} clientes, {r['impagos']} impagos, {r['tasa_impago']}% de impago")
        return ResultadoHerramienta(f"riesgo_{col}", f"Impago por {col}", "\n".join(lineas), tabla)

    def _h_ranking(self) -> ResultadoHerramienta:
        ranking = self.analizador.ranking_riesgo()
        lineas = ["Segmentos de MAYOR riesgo (variable | segmento | clientes | % impago):"]
        for _, r in ranking.head(5).iterrows():
            lineas.append(f"- {r['variable']} | {r['segmento']} | {r['clientes']} | {r['tasa_impago']}%")
        lineas.append("Segmentos de MENOR riesgo:")
        for _, r in ranking.tail(3).iterrows():
            lineas.append(f"- {r['variable']} | {r['segmento']} | {r['clientes']} | {r['tasa_impago']}%")
        return ResultadoHerramienta("ranking_riesgo", "Ranking de segmentos",
                                    "\n".join(lineas), ranking.head(10))

    def _h_correlaciones(self) -> ResultadoHerramienta:
        corr = self.analizador.correlacion_con_objetivo()
        lineas = ["Correlación de Pearson con el impago (-1 a 1):"]
        lineas += [f"- {n}: {v}" for n, v in corr.items()]
        tabla = corr.rename("correlacion").reset_index()
        tabla.columns = ["variable", "correlacion"]
        return ResultadoHerramienta("correlaciones", "Correlaciones con el impago",
                                    "\n".join(lineas), tabla)

    def _h_calidad(self) -> ResultadoHerramienta:
        if not self.bitacora:
            return ResultadoHerramienta("calidad_datos", "Bitácora de limpieza",
                                        "No hay bitácora de limpieza disponible.")
        lineas = ["Acciones de limpieza realizadas:"]
        lineas += [f"- {k}: {v}" for k, v in self.bitacora.items()]
        tabla = pd.DataFrame(list(self.bitacora.items()), columns=["accion", "cantidad"])
        return ResultadoHerramienta("calidad_datos", "Bitácora de limpieza",
                                    "\n".join(lineas), tabla)

    # ---------- Flujo del agente ----------
    # Elige herramientas autorizadas según la pregunta (máximo 3).
    def seleccionar_herramientas(self, pregunta: str) -> list:
        texto = _normalizar(pregunta)
        elegidas = []
        for nombre, patron in REGLAS:
            if not re.search(patron, texto):
                continue
            # Porcentaje del ingreso" ya cubre loan_percent_income: no repetir ingreso
            if nombre == "riesgo_person_income" and "riesgo_loan_percent_income" in elegidas:
                continue
            elegidas.append(nombre)
            if len(elegidas) == MAX_HERRAMIENTAS:
                break
        return elegidas or list(HERRAMIENTAS_POR_DEFECTO)

    # Ejecuta solo herramientas registradas; ignora cualquier otro nombre.
    def ejecutar_herramientas(self, nombres: list) -> list:
        resultados = []
        for nombre in nombres:
            if nombre not in self._herramientas:
                continue
            resultados.append(self._herramientas[nombre][1]())
        return resultados

    # Arma el mensaje con la pregunta y los datos calculados.
    def construir_prompt(self, pregunta: str, resultados: list) -> str:
        bloques = [f"[{r.nombre}]\n{r.texto}" for r in resultados]
        return (
            "DATOS CALCULADOS:\n" + "\n\n".join(bloques)
            + f"\n\nPREGUNTA DEL USUARIO: {pregunta}\n\n"
            "Responde usando solo los datos anteriores."
        )

    # Ejecuta el flujo completo y devuelve respuesta + evidencia.
    # Si Ollama falla, devuelve igualmente los datos calculados.
    def responder(self, pregunta: str) -> dict:

        nombres = self.seleccionar_herramientas(pregunta)
        resultados = self.ejecutar_herramientas(nombres)
        prompt = self.construir_prompt(pregunta, resultados)
        try:
            respuesta = self.cliente.generar(prompt, sistema=PROMPT_SISTEMA)
            error = None
        except ErrorOllama as e:
            respuesta, error = None, str(e)
        return {
            "pregunta": pregunta,
            "herramientas": nombres,
            "resultados": resultados,
            "prompt": prompt,
            "respuesta": respuesta,
            "error": error,
        }
