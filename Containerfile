# Servidor Parakeet (API compatible OpenAI) en contenedor.
#
# Soporta dos motores (PARAKEET_ENGINE, ver server.py):
#   - sherpa       : ONNX int8 vía sherpa-onnx (CPU).
#   - parakeet-cpp : GGUF q8_0 vía libparakeet.so (parakeet.cpp/ggml) con
#                    aceleración Vulkan en la iGPU.
#
# Por eso la imagen hornea libparakeet.so + los backends de ggml (compilados
# aquí desde fuente, etapa cppbuild) y el runtime de Vulkan (mesa). Los modelos
# NO van en la imagen: se montan como volumen (/models).
#
# Build base: ubuntu:24.04 en AMBAS etapas a propósito:
#   - El runtime necesita mesa >= 24 para manejar la iGPU RDNA3.5 (gfx1150 /
#     "STRIX1"); Debian (python:3.12-slim) trae mesa 22, demasiado vieja.
#   - Usar la misma base en build y runtime garantiza ABI (glibc/libstdc++)
#     compatible para libparakeet.so.

# Commit de parakeet.cpp fijado para reproducibilidad (sobrescribible).
ARG PARAKEET_CPP_COMMIT=b8012f11e5269126eddb7f4fd02f891a2ccc29b0

# ---------------------------------------------------------------------------
# cppbuild: compila libparakeet.so + backends ggml (CPU + Vulkan).
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS cppbuild
ARG PARAKEET_CPP_COMMIT
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
        glslc glslang-tools libvulkan-dev spirv-headers \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone https://github.com/mudler/parakeet.cpp . \
    && git checkout "${PARAKEET_CPP_COMMIT}" \
    && git submodule update --init --recursive

# PARAKEET_SHARED=ON -> libparakeet.so (la API C que carga server.py por ctypes).
# GGML_NATIVE=OFF    -> binario portable (sin ISA específica del host de build).
# PARAKEET_GGML_VULKAN=ON -> backend Vulkan para la iGPU.
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DPARAKEET_SHARED=ON \
        -DPARAKEET_BUILD_CLI=OFF \
        -DPARAKEET_BUILD_TESTS=OFF \
        -DPARAKEET_GGML_VULKAN=ON \
    && cmake --build build -j"$(nproc)" --target parakeet

# Recoge libparakeet.so y TODOS los .so de backend (ggml-base/cpu/vulkan/...).
RUN mkdir -p /install/lib \
    && cp build/libparakeet.so /install/lib/ \
    && find build -name '*.so*' -exec cp -av {} /install/lib/ \;

# ---------------------------------------------------------------------------
# runtime: Flask + sherpa-onnx + libparakeet.so + Vulkan (mesa).
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        ffmpeg \
        libgomp1 \
        libvulkan1 mesa-vulkan-drivers \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# venv (Ubuntu 24.04 marca el Python del sistema como "externally managed").
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
# Deps Python primero (capa cacheada; solo se reconstruye si cambian).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# libparakeet.so + backends ggml (incl. Vulkan); refresca el caché del linker.
COPY --from=cppbuild /install/lib/ /usr/local/lib/
RUN ldconfig

COPY server.py .

# Dentro del contenedor escucha en todas las interfaces; la exposición real al
# host se acota al arrancar con  -p 127.0.0.1:8000:8000.
# Por defecto ENGINE=sherpa (compat hacia atrás); el quadlet lo sube a
# parakeet-cpp. GGML_BACKEND_PATH ayuda a ggml a localizar el backend Vulkan.
ENV PARAKEET_HOST=0.0.0.0 \
    PARAKEET_PORT=8000 \
    PARAKEET_ENGINE=sherpa \
    PARAKEET_MODEL_DIR=/models/parakeet-tdt-0.6b-v3 \
    PARAKEET_GGUF=/models/parakeet-cpp/tdt-0.6b-v3-q8_0.gguf \
    PARAKEET_LIB=libparakeet.so \
    GGML_BACKEND_PATH=/usr/local/lib

EXPOSE 8000

# Healthcheck: el endpoint /health ya existe en server.py.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["python3", "server.py"]
