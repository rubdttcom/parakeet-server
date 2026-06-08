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

echo "Modelo listo en $DST:"
ls -la "$DST"
