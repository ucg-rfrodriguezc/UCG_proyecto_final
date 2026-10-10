# Cliente para comunicarse con Ollama (modelo local).
# Solo se encarga de la comunicación HTTP: no conoce el dataset ni la lógica del agente.

import requests

MODELO_POR_DEFECTO = "llama3.2:3b"  # cambiar a "llama3.2:1b" si el equipo es limitado
URL_POR_DEFECTO = "http://localhost:11434"


class ErrorOllama(Exception):
    "Error de comunicación con Ollama, con un mensaje claro para el usuario."

# Envía mensajes a un modelo local de Ollama y devuelve su respuesta.
class ClienteOllama:
    def __init__(self, modelo: str = MODELO_POR_DEFECTO,
                 url: str = URL_POR_DEFECTO, timeout: int = 180):
        self.modelo = modelo
        self.url = url.rstrip("/")
        self.timeout = timeout

    # ---------- Diagnóstico ----------
    # Lista los modelos descargados. Lanza ErrorOllama si no hay servicio.
    def modelos_instalados(self) -> list:
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=5)
            r.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ErrorOllama(
                "No se pudo conectar con Ollama. Verifica que esté instalado "
                "y en ejecución (abre la aplicación Ollama o ejecuta 'ollama serve')."
            )
        except requests.exceptions.RequestException as e:
            raise ErrorOllama(f"Error consultando Ollama: {e}")
        return [m["name"] for m in r.json().get("models", [])]

    # Resumen de conexión para mostrar en la interfaz.
    def estado(self) -> dict:
        try:
            instalados = self.modelos_instalados()
        except ErrorOllama as e:
            return {"conectado": False, "modelo_disponible": False,
                    "instalados": [], "mensaje": str(e)}
        disponible = self.modelo in instalados
        mensaje = "Listo." if disponible else (
            f"Ollama funciona, pero falta el modelo. Ejecuta: ollama pull {self.modelo}"
        )
        return {"conectado": True, "modelo_disponible": disponible,
                "instalados": instalados, "mensaje": mensaje}

    # ---------- Generación ----------
    # Envía una pregunta (con instrucciones de sistema opcionales).
    # temperatura baja = respuestas más estables y menos inventadas.
    def generar(self, pregunta: str, sistema: str = "",
                temperatura: float = 0.2) -> str:
        mensajes = []
        if sistema:
            mensajes.append({"role": "system", "content": sistema})
        mensajes.append({"role": "user", "content": pregunta})

        cuerpo = {
            "model": self.modelo,
            "messages": mensajes,
            "stream": False,
            "options": {"temperature": temperatura, "num_ctx": 4096},
        }
        try:
            r = requests.post(f"{self.url}/api/chat", json=cuerpo,
                              timeout=self.timeout)
        except requests.exceptions.ConnectionError:
            raise ErrorOllama("No se pudo conectar con Ollama. ¿Está en ejecución?")
        except requests.exceptions.Timeout:
            raise ErrorOllama(
                "El modelo tardó demasiado en responder. Prueba con un modelo "
                "más ligero (llama3.2:1b) o una pregunta más corta."
            )

        if r.status_code == 404:
            raise ErrorOllama(
                f"El modelo '{self.modelo}' no está instalado. "
                f"Ejecuta: ollama pull {self.modelo}"
            )
        if r.status_code != 200:
            raise ErrorOllama(f"Ollama respondió con error {r.status_code}: {r.text[:200]}")

        texto = r.json().get("message", {}).get("content", "").strip()
        if not texto:
            raise ErrorOllama("El modelo devolvió una respuesta vacía.")
        return texto
