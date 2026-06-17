#!/usr/bin/env bash
# Construye y gestiona el servidor Parakeet en un contenedor (podman, rootless).
# El modelo NO va dentro de la imagen: se monta desde ./models como volumen.
#
#   ./container.sh build     -> construye la imagen
#   ./container.sh run       -> (re)arranca el contenedor en 127.0.0.1:8000
#   ./container.sh stop      -> lo para y borra
#   ./container.sh logs      -> sigue los logs
#   ./container.sh up        -> build + run (lo habitual)
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="parakeet-server:latest"
NAME="parakeet-server"
MODEL_DIR="$DIR/models"

build() {
    # --format docker: necesario para que se respete el HEALTHCHECK del Containerfile.
    podman build --format docker -t "$IMAGE" "$DIR"
}

run() {
    # Motor por defecto: parakeet-cpp (GGUF q8_0 + iGPU). "sherpa" para ONNX int8 CPU.
    local engine="${PARAKEET_ENGINE:-parakeet-cpp}"
    if [[ "$engine" == "sherpa" && ! -d "$MODEL_DIR/parakeet-tdt-0.6b-v3" ]]; then
        echo "ERROR: falta el modelo ONNX en $MODEL_DIR/parakeet-tdt-0.6b-v3" >&2
        echo "       Descárgalo con: ./download-model.sh" >&2
        exit 1
    fi
    if [[ "$engine" == "parakeet-cpp" && ! -f "$MODEL_DIR/parakeet-cpp/tdt-0.6b-v3-q8_0.gguf" ]]; then
        echo "ERROR: falta el GGUF en $MODEL_DIR/parakeet-cpp/tdt-0.6b-v3-q8_0.gguf" >&2
        echo "       Descárgalo con: ./download-model.sh" >&2
        exit 1
    fi
    # --replace: idempotente (re-crea si ya existía).
    # -p 127.0.0.1:8000:8000  exposición SOLO a localhost (no a la red).
    # :z  reetiqueta SELinux para que el contenedor pueda leer el volumen (Fedora).
    # ro  el modelo es de solo lectura.
    # --device renderD128: iGPU para el backend Vulkan (solo lo usa parakeet-cpp).
    local gpu=()
    [[ "$engine" == "parakeet-cpp" && -e /dev/dri/renderD128 ]] && gpu=(--device /dev/dri/renderD128)
    podman run -d --replace --name "$NAME" \
        -p 127.0.0.1:8000:8000 \
        -v "$MODEL_DIR:/models:z,ro" \
        -e "PARAKEET_ENGINE=$engine" \
        "${gpu[@]}" \
        --restart unless-stopped \
        "$IMAGE"
    echo "Arrancado. Comprobando salud..."
    sleep 4
    curl -fsS http://127.0.0.1:8000/health && echo " <- OK" || echo " <- aún arrancando, mira ./container.sh logs"
}

case "${1:-up}" in
    build) build ;;
    run)   run ;;
    up)    build && run ;;
    stop)  podman rm -f "$NAME" ;;
    logs)  podman logs -f "$NAME" ;;
    *) echo "uso: $0 {build|run|up|stop|logs}" >&2; exit 1 ;;
esac
