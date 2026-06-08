# Servidor Parakeet (API compatible OpenAI) en contenedor.
# El "cerebro" de IA queda autocontenido: ni venv, ni versión de Python, ni
# librerías del sistema que tunear. Solo necesita el modelo, que se monta como
# volumen (no se hornea en la imagen: pesa 640 MB y no debe versionarse).
FROM python:3.12-slim

# ffmpeg: server.py lo usa para decodificar audio (wav/webm/...) a 16 kHz mono.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala deps primero (capa cacheada; solo se reconstruye si cambian).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

# Dentro del contenedor escucha en todas las interfaces; la exposición real al
# host se acota al arrancar con  -p 127.0.0.1:8000:8000.
ENV PARAKEET_HOST=0.0.0.0 \
    PARAKEET_PORT=8000 \
    PARAKEET_MODEL_DIR=/models/parakeet-tdt-0.6b-v3

EXPOSE 8000

# Healthcheck: el endpoint /health ya existe en server.py.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["python", "server.py"]
