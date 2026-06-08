#!/usr/bin/env python3
"""Servidor de transcripción Parakeet con API compatible con OpenAI.

Expone POST /v1/audio/transcriptions (igual que faster-whisper-server) usando
el modelo Parakeet TDT 0.6b v3 (ONNX int8) vía sherpa-onnx, residente en RAM.
Diseñado como reemplazo directo para turbo-whisper en localhost:8000.

Motor ~8x más rápido que faster-whisper small en CPU (RTF ~0.04).
"""
import os
import subprocess
import time

import numpy as np
import sherpa_onnx
from flask import Flask, jsonify, request

# Modelo propio (copiado de la descarga original de OpenWhispr; ya no dependemos
# de su caché — su carpeta puede borrarse). Sobrescribible por env.
MODEL_DIR = os.environ.get(
    "PARAKEET_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "parakeet-tdt-0.6b-v3"),
)
NUM_THREADS = int(os.environ.get("PARAKEET_THREADS", "6"))
SAMPLE_RATE = 16000

app = Flask(__name__)

print(f"Cargando Parakeet desde {MODEL_DIR} ({NUM_THREADS} hilos)...", flush=True)
_t0 = time.time()
recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=f"{MODEL_DIR}/encoder.int8.onnx",
    decoder=f"{MODEL_DIR}/decoder.int8.onnx",
    joiner=f"{MODEL_DIR}/joiner.int8.onnx",
    tokens=f"{MODEL_DIR}/tokens.txt",
    num_threads=NUM_THREADS,
    model_type="nemo_transducer",
    debug=False,
)
print(f"Modelo cargado en {time.time() - _t0:.2f}s. Listo.", flush=True)


def _decode_to_samples(raw_bytes: bytes) -> np.ndarray:
    """Decodifica cualquier audio (wav/mp3/webm/...) a float32 mono 16 kHz."""
    proc = subprocess.run(
        ["ffmpeg", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
        input=raw_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32)


@app.post("/v1/audio/transcriptions")
def transcriptions():
    f = request.files.get("file")
    if f is None:
        return jsonify({"error": "missing 'file'"}), 400

    t0 = time.time()
    samples = _decode_to_samples(f.read())
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    recognizer.decode_stream(stream)
    text = stream.result.text.strip()
    dur = len(samples) / SAMPLE_RATE
    print(f"transcrito {dur:.1f}s audio en {(time.time()-t0)*1000:.0f}ms -> {text[:60]!r}", flush=True)

    # OpenAI devuelve texto plano para response_format=text, JSON si no.
    if request.form.get("response_format") == "text":
        return text, 200, {"Content-Type": "text/plain; charset=utf-8"}
    return jsonify({"text": text})


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    # 127.0.0.1 nativo (solo localhost). En contenedor se pone 0.0.0.0 para que
    # podman pueda redirigir el puerto; la exposición al host se acota con -p 127.0.0.1:8000.
    app.run(
        host=os.environ.get("PARAKEET_HOST", "127.0.0.1"),
        port=int(os.environ.get("PARAKEET_PORT", "8000")),
        threaded=False,
    )
