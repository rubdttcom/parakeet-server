#!/usr/bin/env bash
# Descarga el modelo Parakeet TDT 0.6b v3 (ONNX int8) desde el repo oficial de
# sherpa-onnx en HuggingFace. Independiente de OpenWhispr: con esto el sistema
# es reproducible desde cero en cualquier máquina.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="$DIR/models/parakeet-tdt-0.6b-v3"
# Tarball oficial publicado por k2-fsa/sherpa-onnx (NeMo Parakeet TDT 0.6b v3, int8).
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"

if [[ -f "$DST/encoder.int8.onnx" ]]; then
    echo "El modelo ya está en $DST — nada que hacer."
    exit 0
fi

mkdir -p "$DIR/models"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Descargando modelo (~640 MB)..."
curl -fL "$URL" -o "$TMP/model.tar.bz2"
echo "Extrayendo..."
tar xjf "$TMP/model.tar.bz2" -C "$TMP"

# El tarball extrae a una carpeta sherpa-onnx-nemo-parakeet-...; normalizamos el nombre.
SRC="$(find "$TMP" -maxdepth 1 -type d -name 'sherpa-onnx-nemo-parakeet*' | head -1)"
mkdir -p "$DST"
cp "$SRC"/encoder.int8.onnx "$SRC"/decoder.int8.onnx "$SRC"/joiner.int8.onnx "$SRC"/tokens.txt "$DST"/

echo "Modelo ONNX (sherpa) listo en $DST:"
ls -la "$DST"

# --- Modelo GGUF para el motor parakeet-cpp ---------------------------------
# q8_0 (mismo Parakeet TDT 0.6b v3) publicado por mudler para parakeet.cpp.
# Más preciso que el int8 de sherpa y, en iGPU (Vulkan), más rápido.
GGUF_DST="$DIR/models/parakeet-cpp"
GGUF="$GGUF_DST/tdt-0.6b-v3-q8_0.gguf"
GGUF_URL="https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt-0.6b-v3-q8_0.gguf"
if [[ -f "$GGUF" ]]; then
    echo "El GGUF ya está en $GGUF — nada que hacer."
else
    mkdir -p "$GGUF_DST"
    echo "Descargando GGUF q8_0 (~940 MB)..."
    curl -fL "$GGUF_URL" -o "$GGUF"
    echo "GGUF listo en $GGUF"
fi

# --- Modelo streaming (Nemotron) para el endpoint WS /v1/audio/stream -------
# Cache-aware streaming, multilingüe. Necesario para el modo dictado en vivo.
NEM="$GGUF_DST/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf"
NEM_URL="https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf"
if [[ -f "$NEM" ]]; then
    echo "El GGUF streaming ya está en $NEM — nada que hacer."
else
    echo "Descargando GGUF streaming Nemotron (~980 MB)..."
    curl -fL "$NEM_URL" -o "$NEM"
    echo "GGUF streaming listo en $NEM"
fi
ls -la "$GGUF_DST"
