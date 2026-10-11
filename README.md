# Análisis de riesgo crediticio con agente de IA local

Aplicación en Streamlit que explora un dataset de préstamos, segmenta el riesgo de impago y permite hacer preguntas a un agente de IA que corre **en el propio equipo** con Ollama.

Proyecto Final Integrador de *Paradigmas de Programación para Inteligencia Artificial y Análisis de Datos*, Maestría en Inteligencia Artificial y Ciencia de Datos, Universidad Casa Grande.

- **Autor:** Ronald Fabrizio Rodriguez Castro
- **Periodo:** C2026P1
- **Video demostrativo (Google Drive):** https://drive.google.com/file/d/19-RqUtLmYa16o6PIX5KyII_PZxUpP-he/view?usp=drive_link

## Qué hace la aplicación

| Pestaña | Contenido |
|---|---|
| Resumen | Indicadores generales, hallazgos clave, segmentos de mayor y menor riesgo y correlaciones |
| Calidad de datos | Problemas detectados en el CSV y reglas de limpieza aplicadas |
| Exploración | Riesgo por grupo, distribuciones y relaciones entre variables |
| Agente de IA | Preguntas en lenguaje natural respondidas con cifras calculadas por Python |
| Proyecto | Problema analítico, diccionario de datos, arquitectura y limitaciones |

El agente **no calcula**: Python elige las herramientas de análisis y obtiene las cifras, y el modelo local solo las interpreta. Cada respuesta incluye su evidencia (datos, gráfico y prompt enviado) para poder verificarla.

## Dataset

- **Nombre:** Credit Risk Dataset
- **Fuente:** [Kaggle, tahmidmir/credit-risk-dataset](https://www.kaggle.com/datasets/tahmidmir/credit-risk-dataset)
- **Tamaño:** 32,581 solicitudes de préstamo y 12 variables (1.8 MB).
- **Problema analítico:** ¿qué características del solicitante y del préstamo se asocian con un mayor riesgo de impago, y cómo se puede segmentar a los clientes según ese riesgo?
- **Variable objetivo:** `loan_status` (0 = pagó, 1 = impago). Solo el 21.87 % de los préstamos termina en impago, así que las clases están desbalanceadas.

| Variable | Descripción |
|---|---|
| `person_age` | Edad del solicitante |
| `person_income` | Ingreso anual |
| `person_home_ownership` | Tipo de vivienda |
| `person_emp_length` | Años de antigüedad laboral |
| `loan_intent` | Propósito del préstamo |
| `loan_grade` | Grado del préstamo, de A (mejor) a G (peor) |
| `loan_amnt` | Monto del préstamo |
| `loan_int_rate` | Tasa de interés |
| `loan_status` | Variable objetivo |
| `loan_percent_income` | Monto del préstamo como proporción del ingreso |
| `cb_person_default_on_file` | Incumplimiento previo registrado (Y/N) |
| `cb_person_cred_hist_length` | Años de historial crediticio |

**Si el archivo no está en `data/`:** descárgalo desde la página de Kaggle (requiere una cuenta gratuita), descomprímelo y guárdalo como `data/credit_risk_dataset.csv`. Consulta la licencia en esa misma página.

### Limpieza aplicada

| Problema | Regla | Registros |
|---|---|---|
| Duplicados | Se eliminan | 165 |
| Edad mayor a 100 años | Se elimina el registro | 5 |
| Antigüedad laboral mayor a 60 años | Se marca como nulo y se imputa | 2 |
| Antigüedad laboral nula | Mediana global | 889 |
| Tasa de interés nula | Mediana por grado del préstamo | 3,094 |

El dataset final tiene 32,411 filas. El mayor ingreso original (6,000,000) pertenecía a un registro con edad de 144 años, que se elimina; el mayor ingreso válido (2,039,784) se conserva porque es un valor posible, y los ingresos se visualizan con escala logarítmica.

### Algunos resultados

| Segmento | % de impago |
|---|---|
| Promedio global | 21.87 % |
| Grado A | 9.96 % |
| Grado D | 59.06 % |
| Grado F | 70.54 % |
| Préstamo mayor al 30 % del ingreso | 70.44 % |

Las variables numéricas más correlacionadas con el impago son `loan_percent_income` (0.38) y `loan_int_rate` (0.334). Son asociaciones, no relaciones causales.

## Estructura del proyecto

```
UCG_proyecto_final/
├── app.py                          # Punto de entrada: coordina y arma las pestañas
├── README.md
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml                 # Tema de la interfaz
├── data/
│   └── credit_risk_dataset.csv
├── assets/                         # Capturas de pantalla
└── src/
    ├── __init__.py
    ├── carga_de_datos.py           # Lectura, validación y limpieza
    ├── analisis_exploratorio.py    # Clase AnalizadorRiesgo
    ├── visualizador_de_graficos.py # Gráficos con Plotly y Seaborn
    ├── cliente_llm.py              # Clase ClienteOllama
    ├── agente.py                   # Clase AgenteRiesgo
    └── interfaz.py                 # Secciones de la interfaz
```

| Módulo | Equivale en la guía a | Responsabilidad |
|---|---|---|
| `carga_de_datos.py` | `data_loader.py` | Cargar el CSV, validar columnas y limpiar |
| `analisis_exploratorio.py` | `eda.py` | Perfilado, segmentación y correlaciones |
| `visualizador_de_graficos.py` | `visualizations.py` | Gráficos |
| `cliente_llm.py` | `llm_client.py` | Comunicación HTTP con Ollama |
| `agente.py` | `agent.py` | Herramientas autorizadas e interpretación |
| `interfaz.py` | (adicional) | Presentación en Streamlit, sin lógica de análisis |

### Cómo funciona el agente

```
Pregunta  ->  Python elige herramientas  ->  Python calcula  ->  Ollama interpreta
```

El agente tiene 13 herramientas autorizadas (resumen general, ranking de riesgo, correlaciones, calidad de datos y una por cada variable de riesgo). Cualquier otro nombre se ignora. La selección se hace con reglas sobre la pregunta, lo que la hace reproducible y requiere una sola llamada al modelo.

## Requisitos

- Python 3.10 o superior
- Git
- [Ollama](https://ollama.com)
- Unos 2 GB libres de disco para el modelo y al menos 8 GB de RAM recomendados

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/ucg-rfrodriguezc/UCG_proyecto_final.git
cd UCG_proyecto_final
```

### 2. Crear y activar el entorno virtual

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS o Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Cuando está activo, el prompt muestra `(.venv)` al inicio.

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 4. Instalar Ollama

Descarga el instalador para tu sistema desde https://ollama.com y ejecútalo. Después abre la aplicación Ollama o ejecuta `ollama serve` para que el servicio quede activo en `http://localhost:11434`.

### 5. Descargar el modelo

```bash
ollama pull llama3.2:3b
```

Comprueba que quedó instalado:

```bash
ollama list
```

Para equipos con pocos recursos puede usarse el modelo más ligero `llama3.2:1b`:

```bash
ollama pull llama3.2:1b
```

La aplicación lista los modelos instalados en la barra lateral y permite elegir cuál usar. Para cambiar el modelo por defecto, edita `MODELO_POR_DEFECTO` en `src/cliente_llm.py`. **Versión usada en el desarrollo y las pruebas: `llama3.2:3b`.**

Los modelos de Ollama no se guardan en este repositorio.

## Ejecución

Desde la carpeta del proyecto, con el entorno virtual activo:

```bash
streamlit run app.py
```

Se abre en `http://localhost:8501`. Para probar el agente, ve a la pestaña **Agente de IA** y elige una pregunta sugerida, por ejemplo:

- ¿Cómo influye el grado del préstamo en el impago?
- ¿Qué pasa con quienes piden más del 30% de su ingreso?
- ¿Qué variables se correlacionan más con el impago?

La primera respuesta puede tardar hasta un minuto mientras el modelo se carga en memoria.

## Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| La barra lateral dice que no se pudo conectar con Ollama | El servicio no está activo | Abre la aplicación Ollama o ejecuta `ollama serve` |
| Aviso de que falta el modelo | No se descargó | Ejecuta `ollama pull llama3.2:3b` |
| `ModuleNotFoundError` | Entorno sin activar o dependencias sin instalar | Activa `.venv` y ejecuta `pip install -r requirements.txt` |
| PowerShell bloquea la activación del entorno | Política de ejecución | Ejecuta `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y vuelve a activar |
| `FileNotFoundError` con el dataset | Falta el CSV | Colócalo en `data/credit_risk_dataset.csv` (ver sección Dataset) |
| El modelo tarda demasiado | Equipo con pocos recursos | Usa `llama3.2:1b` |

## Limitaciones

- Las correlaciones y tasas describen asociaciones, no causas.
- Los grupos con pocos clientes (por ejemplo, el grado G con 64) producen cifras poco estables. La app los señala.
- El agente elige herramientas por palabras clave: una pregunta con términos no previstos recibe un análisis general.
- Un modelo pequeño puede equivocarse al redactar. Por eso cada respuesta muestra las cifras calculadas para poder verificarla.

## Historial de desarrollo

| Paso | Contenido |
|---|---|
| 1 | Entorno virtual, estructura del proyecto e instalación de Ollama |
| 2 | Carga y limpieza de datos (`carga_de_datos.py`) |
| 3 | Análisis exploratorio con la clase `AnalizadorRiesgo` |
| 4 | Visualizaciones con Plotly y Seaborn |
| 5 | Conexión con Ollama (`cliente_llm.py`) |
| 6 | Agente de IA con herramientas autorizadas (`agente.py`) |
| 7 | Interfaz final con pestañas (`interfaz.py`) |
| 8 | Documentación y reproducibilidad |

## Tecnologías

Python, Streamlit, pandas, NumPy, Plotly, Matplotlib, Seaborn, Requests y Ollama con `llama3.2:3b`.
