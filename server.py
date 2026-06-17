#!/usr/bin/env python3
"""Servidor de transcripción Parakeet con API compatible con OpenAI.

Expone:
  - POST /v1/audio/transcriptions  (batch, igual que faster-whisper-server)
  - WS   /v1/audio/stream          (streaming cache-aware, opcional)

Motor batch seleccionable con PARAKEET_ENGINE:
  - "sherpa"        (por defecto): ONNX int8 vía sherpa-onnx, en CPU.
  - "parakeet-cpp": GGUF q8_0 vía libparakeet.so (parakeet.cpp/ggml), con
                    aceleración por iGPU (Vulkan). Más preciso y más rápido.

El endpoint de streaming se activa solo si PARAKEET_STREAM_GGUF apunta a un
modelo cache-aware streaming (p.ej. nemotron-3.5-asr-streaming-0.6b en GGUF).
Si no, /v1/audio/stream responde 503. El batch y el streaming conviven: cada uno
carga su modelo (residente) y comparten libparakeet.so + iGPU.
"""
import ctypes
import json
import os
import re
import subprocess
import threading
import time

import numpy as np
from flask import Flask, jsonify, request

ENGINE = os.environ.get("PARAKEET_ENGINE", "sherpa").lower()
STREAM_GGUF = os.environ.get("PARAKEET_STREAM_GGUF", "")
NUM_THREADS = int(os.environ.get("PARAKEET_THREADS", "6"))
LANG = os.environ.get("PARAKEET_LANG", "es")
SAMPLE_RATE = 16000

# ggml no es seguro para cómputo concurrente sobre el mismo backend: serializamos
# toda inferencia (batch + cada feed de streaming) con un lock global. Para un
# único usuario (turbo-whisper) no hay contención apreciable.
_INFER_LOCK = threading.Lock()

# Nemotron (multilingüe prompt-conditioned) emite la etiqueta de idioma inline,
# p.ej. "<es-US>", al cerrar cada enunciado. La quitamos del texto que se
# teclearía y la usamos como señal de fin de segmento (EOU de facto).
_TAG = re.compile(r"\s*<[a-z]{2}-[A-Z]{2}>\s*")

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


# --- libparakeet.so (compartida por el motor batch y el de streaming) -------
_FPTR = ctypes.POINTER(ctypes.c_float)
_LIB = None


def _parakeet_lib():
    """Carga y enlaza libparakeet.so una sola vez (batch + streaming)."""
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = ctypes.CDLL(os.environ.get("PARAKEET_LIB", "libparakeet.so"))
    lib.parakeet_capi_load.restype = ctypes.c_void_p
    lib.parakeet_capi_load.argtypes = [ctypes.c_char_p]
    lib.parakeet_capi_transcribe_pcm_lang.restype = ctypes.c_void_p
    lib.parakeet_capi_transcribe_pcm_lang.argtypes = [
        ctypes.c_void_p, _FPTR, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
    ]
    lib.parakeet_capi_stream_begin_lang.restype = ctypes.c_void_p
    lib.parakeet_capi_stream_begin_lang.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.parakeet_capi_stream_feed_json.restype = ctypes.c_void_p
    lib.parakeet_capi_stream_feed_json.argtypes = [ctypes.c_void_p, _FPTR, ctypes.c_int]
    lib.parakeet_capi_stream_finalize_json.restype = ctypes.c_void_p
    lib.parakeet_capi_stream_finalize_json.argtypes = [ctypes.c_void_p]
    lib.parakeet_capi_stream_free.argtypes = [ctypes.c_void_p]
    lib.parakeet_capi_free_string.argtypes = [ctypes.c_void_p]
    lib.parakeet_capi_last_error.restype = ctypes.c_char_p
    lib.parakeet_capi_last_error.argtypes = [ctypes.c_void_p]
    _LIB = lib
    return lib


def _take_json(lib, ptr) -> dict:
    """Convierte el char* (malloc'd) del capi a dict y libera la memoria C."""
    if not ptr:
        return {}
    s = ctypes.cast(ptr, ctypes.c_char_p).value.decode()
    lib.parakeet_capi_free_string(ptr)
    return json.loads(s) if s else {}


# --- Motor batch: sherpa-onnx (ONNX int8, CPU) ------------------------------
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


# --- Motor batch: parakeet.cpp (GGUF q8_0, ggml + Vulkan/iGPU) --------------
class ParakeetCppEngine:
    def __init__(self):
        gguf = os.environ.get("PARAKEET_GGUF", "/models/parakeet-cpp/tdt-0.6b-v3-q8_0.gguf")
        print(f"[parakeet-cpp] cargando {gguf} (lang={LANG})...", flush=True)
        self._lib = _parakeet_lib()
        self._ctx = self._lib.parakeet_capi_load(gguf.encode())
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


# --- Motor streaming: parakeet.cpp cache-aware (Nemotron) -------------------
class StreamingModel:
    """Modelo streaming residente. Cada conexión abre su propia sesión (stream)."""

    def __init__(self, gguf: str):
        print(f"[stream] cargando modelo streaming {gguf} (lang={LANG})...", flush=True)
        self._lib = _parakeet_lib()
        self._ctx = self._lib.parakeet_capi_load(gguf.encode())
        if not self._ctx:
            raise RuntimeError(f"parakeet_capi_load (streaming) falló para {gguf}")

    def begin(self, lang: str):
        s = self._lib.parakeet_capi_stream_begin_lang(self._ctx, lang.encode())
        if not s:
            err = self._lib.parakeet_capi_last_error(self._ctx)
            raise RuntimeError("stream_begin falló: " + (err.decode() if err else "?"))
        return s

    def feed(self, stream, samples: np.ndarray) -> dict:
        arr = np.ascontiguousarray(samples, dtype=np.float32)
        ptr = arr.ctypes.data_as(_FPTR)
        with _INFER_LOCK:
            doc = _take_json(self._lib, self._lib.parakeet_capi_stream_feed_json(stream, ptr, len(arr)))
        return doc

    def finalize(self, stream) -> dict:
        with _INFER_LOCK:
            return _take_json(self._lib, self._lib.parakeet_capi_stream_finalize_json(stream))

    def free(self, stream):
        self._lib.parakeet_capi_stream_free(stream)


_ENGINES = {"sherpa": SherpaEngine, "parakeet-cpp": ParakeetCppEngine}

print(f"Motor batch: PARAKEET_ENGINE={ENGINE}", flush=True)
if ENGINE not in _ENGINES:
    raise SystemExit(f"PARAKEET_ENGINE inválido: {ENGINE!r} (opciones: {', '.join(_ENGINES)})")
_t0 = time.time()
engine = _ENGINES[ENGINE]()
print(f"Modelo batch cargado en {time.time() - _t0:.2f}s.", flush=True)

stream_model = None
if STREAM_GGUF:
    try:
        _t0 = time.time()
        stream_model = StreamingModel(STREAM_GGUF)
        print(f"Modelo streaming cargado en {time.time() - _t0:.2f}s. WS /v1/audio/stream activo.", flush=True)
    except Exception as e:
        # No fatal: el batch sigue sirviendo aunque el streaming no cargue.
        print(f"AVISO: no se pudo cargar el modelo streaming ({e}); WS deshabilitado.", flush=True)
        stream_model = None
else:
    print("Streaming desactivado (define PARAKEET_STREAM_GGUF para activarlo).", flush=True)


@app.post("/v1/audio/transcriptions")
def transcriptions():
    f = request.files.get("file")
    if f is None:
        return jsonify({"error": "missing 'file'"}), 400

    t0 = time.time()
    samples = _decode_to_samples(f.read())
    with _INFER_LOCK:
        text = engine.transcribe(samples)
    dur = len(samples) / SAMPLE_RATE
    print(f"[{ENGINE}] transcrito {dur:.1f}s audio en {(time.time()-t0)*1000:.0f}ms -> {text[:60]!r}", flush=True)

    if request.form.get("response_format") == "text":
        return text, 200, {"Content-Type": "text/plain; charset=utf-8"}
    return jsonify({"text": text})


@app.get("/health")
def health():
    return jsonify({"status": "healthy", "engine": ENGINE, "streaming": bool(stream_model)})


# --- WebSocket de streaming (flask-sock) ------------------------------------
# Registrado siempre; si no hay modelo streaming, cierra con un aviso.
try:
    from flask_sock import Sock

    sock = Sock(app)

    @sock.route("/v1/audio/stream")
    def audio_stream(ws):
        """Recibe PCM float32 16k mono (frames binarios); responde JSON por chunk.

        Mensajes del cliente:  binario = PCM;  texto "EOS" = finalizar.
        Mensajes del servidor:
          {"type":"chunk","text":"<recién confirmado>","eou":bool,"t":<seg>}
          {"type":"final","text":"<cola final>"}
        """
        if stream_model is None:
            ws.send(json.dumps({"type": "error", "msg": "streaming no habilitado en el servidor"}))
            return
        # lang por query (?lang=es)
        lang = LANG
        try:
            q = request.query_string.decode()
            for kv in q.split("&"):
                if kv.startswith("lang="):
                    lang = kv[5:] or LANG
        except Exception:
            pass

        stream = stream_model.begin(lang)
        print(f"[stream] conexión nueva (lang={lang})", flush=True)
        t_start = time.time()
        fed = 0
        try:
            while True:
                msg = ws.receive()
                if msg is None:
                    break
                if isinstance(msg, (bytes, bytearray)):
                    arr = np.frombuffer(msg, dtype=np.float32)
                    fed += len(arr)
                    doc = stream_model.feed(stream, arr)
                    text = _TAG.sub(" ", doc.get("text", ""))
                    eou = bool(doc.get("eou", 0)) or bool(_TAG.search(doc.get("text", "")))
                    if text or eou:
                        ws.send(json.dumps({
                            "type": "chunk", "text": text, "eou": eou,
                            "t": round(time.time() - t_start, 3),
                            "audio_s": round(fed / SAMPLE_RATE, 2),
                        }))
                elif msg == "EOS":
                    doc = stream_model.finalize(stream)
                    ws.send(json.dumps({"type": "final", "text": _TAG.sub(" ", doc.get("text", ""))}))
        finally:
            stream_model.free(stream)
            print(f"[stream] conexión cerrada ({fed/SAMPLE_RATE:.1f}s de audio)", flush=True)

except ImportError:
    print("flask-sock no instalado: WS /v1/audio/stream no disponible.", flush=True)


if __name__ == "__main__":
    # threaded=True: necesario para WS de larga duración + peticiones batch.
    app.run(
        host=os.environ.get("PARAKEET_HOST", "127.0.0.1"),
        port=int(os.environ.get("PARAKEET_PORT", "8000")),
        threaded=True,
    )
