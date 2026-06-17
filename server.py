#!/usr/bin/env python3
"""Servidor de transcripción Parakeet con API compatible con OpenAI.

Expone POST /v1/audio/transcriptions (igual que faster-whisper-server) usando un
modelo Parakeet TDT 0.6b v3 residente en RAM. Reemplazo directo de turbo-whisper
en localhost:8000.

Dos motores seleccionables con la variable PARAKEET_ENGINE:

  - "sherpa"        (por defecto): ONNX int8 vía sherpa-onnx, en CPU.
  - "parakeet-cpp": GGUF q8_0 vía libparakeet.so (parakeet.cpp sobre ggml), con
                    aceleración por iGPU (Vulkan) si el backend está disponible.
                    Más preciso (q8_0 > int8: el int8 confunde dígitos y palabras
                    como "ahogar"/"hogar") y, en iGPU, ~2-3x más rápido.

Ambos motores cargan el modelo una sola vez al arrancar (residente) y comparten
el mismo decodificador de audio (ffmpeg -> float32 mono 16 kHz) y la misma API.
"""
import ctypes
import os
import subprocess
import time

import numpy as np
from flask import Flask, jsonify, request

ENGINE = os.environ.get("PARAKEET_ENGINE", "sherpa").lower()
NUM_THREADS = int(os.environ.get("PARAKEET_THREADS", "6"))
LANG = os.environ.get("PARAKEET_LANG", "es")
SAMPLE_RATE = 16000

app = Flask(__name__)


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


# --- Motor: sherpa-onnx (ONNX int8, CPU) ------------------------------------
class SherpaEngine:
    def __init__(self):
        import sherpa_onnx  # import perezoso: solo si se usa este motor.

        model_dir = os.environ.get(
            "PARAKEET_MODEL_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "parakeet-tdt-0.6b-v3"),
        )
        print(f"[sherpa] cargando ONNX int8 desde {model_dir} ({NUM_THREADS} hilos)...", flush=True)
        self._r = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=f"{model_dir}/encoder.int8.onnx",
            decoder=f"{model_dir}/decoder.int8.onnx",
            joiner=f"{model_dir}/joiner.int8.onnx",
            tokens=f"{model_dir}/tokens.txt",
            num_threads=NUM_THREADS,
            model_type="nemo_transducer",
            debug=False,
        )

    def transcribe(self, samples: np.ndarray) -> str:
        stream = self._r.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self._r.decode_stream(stream)
        return stream.result.text.strip()


# --- Motor: parakeet.cpp (GGUF q8_0, ggml + Vulkan/iGPU) --------------------
_FPTR = ctypes.POINTER(ctypes.c_float)


class ParakeetCppEngine:
    """Bindings ctypes sobre libparakeet.so (API C de parakeet.cpp).

    El modelo queda residente en el proceso (parakeet_capi_load una vez); cada
    request reutiliza el mismo contexto vía parakeet_capi_transcribe_pcm_lang,
    alimentándolo con el PCM que ya decodifica ffmpeg (sin ficheros temporales).
    """

    def __init__(self):
        gguf = os.environ.get("PARAKEET_GGUF", "/models/parakeet-cpp/tdt-0.6b-v3-q8_0.gguf")
        lib_path = os.environ.get("PARAKEET_LIB", "libparakeet.so")
        print(f"[parakeet-cpp] cargando {lib_path} + {gguf} (lang={LANG})...", flush=True)
        lib = ctypes.CDLL(lib_path)
        lib.parakeet_capi_load.restype = ctypes.c_void_p
        lib.parakeet_capi_load.argtypes = [ctypes.c_char_p]
        # char* transcribe_pcm_lang(ctx, const float* samples, int n, int sr, int decoder, const char* lang)
        lib.parakeet_capi_transcribe_pcm_lang.restype = ctypes.c_void_p
        lib.parakeet_capi_transcribe_pcm_lang.argtypes = [
            ctypes.c_void_p, _FPTR, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
        ]
        lib.parakeet_capi_free_string.argtypes = [ctypes.c_void_p]
        lib.parakeet_capi_last_error.restype = ctypes.c_char_p
        lib.parakeet_capi_last_error.argtypes = [ctypes.c_void_p]
        self._lib = lib
        self._ctx = lib.parakeet_capi_load(gguf.encode())
        if not self._ctx:
            raise RuntimeError(f"parakeet_capi_load falló para {gguf}")
        self._lang = LANG.encode()

    def transcribe(self, samples: np.ndarray) -> str:
        arr = np.ascontiguousarray(samples, dtype=np.float32)
        ptr = arr.ctypes.data_as(_FPTR)
        p = self._lib.parakeet_capi_transcribe_pcm_lang(
            self._ctx, ptr, len(arr), SAMPLE_RATE, 0, self._lang
        )
        if not p:
            err = self._lib.parakeet_capi_last_error(self._ctx)
            raise RuntimeError("parakeet.cpp transcribe falló: " + (err.decode() if err else "?"))
        s = ctypes.cast(p, ctypes.c_char_p).value.decode()
        self._lib.parakeet_capi_free_string(p)
        return s.strip()


_ENGINES = {"sherpa": SherpaEngine, "parakeet-cpp": ParakeetCppEngine}

print(f"Motor: PARAKEET_ENGINE={ENGINE}", flush=True)
if ENGINE not in _ENGINES:
    raise SystemExit(f"PARAKEET_ENGINE inválido: {ENGINE!r} (opciones: {', '.join(_ENGINES)})")
_t0 = time.time()
engine = _ENGINES[ENGINE]()
print(f"Modelo cargado en {time.time() - _t0:.2f}s. Listo.", flush=True)


@app.post("/v1/audio/transcriptions")
def transcriptions():
    f = request.files.get("file")
    if f is None:
        return jsonify({"error": "missing 'file'"}), 400

    t0 = time.time()
    samples = _decode_to_samples(f.read())
    text = engine.transcribe(samples)
    dur = len(samples) / SAMPLE_RATE
    print(f"[{ENGINE}] transcrito {dur:.1f}s audio en {(time.time()-t0)*1000:.0f}ms -> {text[:60]!r}", flush=True)

    # OpenAI devuelve texto plano para response_format=text, JSON si no.
    if request.form.get("response_format") == "text":
        return text, 200, {"Content-Type": "text/plain; charset=utf-8"}
    return jsonify({"text": text})


@app.get("/health")
def health():
    return jsonify({"status": "healthy", "engine": ENGINE})


if __name__ == "__main__":
    # 127.0.0.1 nativo (solo localhost). En contenedor se pone 0.0.0.0 para que
    # podman pueda redirigir el puerto; la exposición al host se acota con -p 127.0.0.1:8000.
    app.run(
        host=os.environ.get("PARAKEET_HOST", "127.0.0.1"),
        port=int(os.environ.get("PARAKEET_PORT", "8000")),
        threaded=False,
    )
