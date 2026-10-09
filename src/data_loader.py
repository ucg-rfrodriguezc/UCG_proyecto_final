#Módulo de carga, validación y limpieza de datos.
from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "credit_risk_dataset.csv"

COLUMNAS_NUMERICAS = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
]
COLUMNAS_CATEGORICAS = [
    "person_home_ownership", "loan_intent", "loan_grade",
    "cb_person_default_on_file",
]
TARGET = "loan_status"

# Reglas de validación
EDAD_MAXIMA = 100
ANTIGUEDAD_MAXIMA = 60

# Carga de datos
def cargar_datos(ruta=DATA_PATH) -> pd.DataFrame:
    """Lee el CSV y valida que existan las columnas esperadas."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {ruta}")
    df = pd.read_csv(ruta)
    esperadas = set(COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS + [TARGET])
    faltantes = esperadas - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en el dataset: {sorted(faltantes)}")
    return df

# Reporte general de calidad de datos
def reporte_calidad(df: pd.DataFrame) -> dict:
    """Calcula problemas de calidad sin modificar el DataFrame."""
    nulos = df.isna().sum()
    return {
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "duplicados": int(df.duplicated().sum()),
        "nulos_por_columna": {c: int(n) for c, n in nulos.items() if n > 0},
        "edades_invalidas": int((df["person_age"] > EDAD_MAXIMA).sum()),
        "antiguedad_invalida": int((df["person_emp_length"] > ANTIGUEDAD_MAXIMA).sum()),
        "tasa_incumplimiento": round(float(df[TARGET].mean()), 4),
    }

# Limpieza de datos
def limpiar_datos(df: pd.DataFrame):
    """Limpia el dataset y devuelve (df_limpio, bitacora).

    Decisiones:
    - Duplicados: se eliminan.
    - Edad > 100: se eliminan las filas (dato imposible).
    - Antigüedad laboral > 60: se marca como nulo y se imputa.
    - emp_length nulo: mediana global.
    - loan_int_rate nulo: mediana por loan_grade (la tasa depende del grado).
    """
    bitacora = {"filas_iniciales": len(df)}
    df = df.copy()

    antes = len(df)
    df = df.drop_duplicates()
    bitacora["duplicados_eliminados"] = antes - len(df)

    antes = len(df)
    df = df[df["person_age"] <= EDAD_MAXIMA]
    bitacora["filas_edad_invalida_eliminadas"] = antes - len(df)

    mask = df["person_emp_length"] > ANTIGUEDAD_MAXIMA
    bitacora["antiguedad_invalida_a_nulo"] = int(mask.sum())
    df.loc[mask, "person_emp_length"] = pd.NA

    bitacora["emp_length_imputados"] = int(df["person_emp_length"].isna().sum())
    df["person_emp_length"] = df["person_emp_length"].fillna(
        df["person_emp_length"].median()
    )

    bitacora["int_rate_imputados"] = int(df["loan_int_rate"].isna().sum())
    df["loan_int_rate"] = df["loan_int_rate"].fillna(
        df.groupby("loan_grade")["loan_int_rate"].transform("median")
    )

    df = df.reset_index(drop=True)
    bitacora["filas_finales"] = len(df)
    return df, bitacora