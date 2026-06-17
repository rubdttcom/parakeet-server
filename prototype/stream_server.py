#!/usr/bin/env python3
"""PROTOTIPO — servidor WebSocket de transcripción en streaming (cache-aware).

NO es producción: corre en el host usando la libparakeet.so + Vulkan ya
compiladas y el modelo Nemotron streaming. Sirve para *sentir* la escritura
realtime antes de tocar turbo-whisper.

Protocolo (ws://HOST:PORT):
  - El cliente abre la conexión (query ?lang=es opcional).
  - Envía frames BINARIOS = PCM float32 little-endian, mono, 16 kHz.
  - Envía el texto "EOS" para finalizar (flush del tail).
  - El servidor responde mensajes JSON:
      {"type":"chunk","text":"...","eou":bool,"eob":bool,"t":<seg desde inicio>}
      {"type":"final","text":"..."}        (al recibir EOS)
    "text" es SOLO lo recién confirmado desde el frame anterior (commit-on-chunk;
    eou=True marca fin de enunciado -> punto natural para teclear/forzar salto).

Lanzar:
  LD_LIBRARY_PATH=... PARAKEET_GGUF=... ./.venv/bin/python stream_server.py
"""
import asyncio
import ctypes
import json
import os
import re
import time

import websockets

# Nemotron (prompt-conditioned multilingüe) emite la etiqueta de idioma inline
# en el texto, p.ej. "<es-US>", al cerrar cada enunciado. La quitamos del texto
# que se teclearía y la usamos como señal de fin de segmento (EOU de facto).
_TAG = re.compile(r"\s*<[a-z]{2}-[A-Z]{2}>\s*")

GGUF = os.environ.get("PARAKEET_GGUF", os.path.expanduser("~/parakeet.cpp/nemotron-3.5-q8_0.gguf"))
LIB = os.environ.get("PARAKEET_LIB", "libparakeet.so")
LANG = os.environ.get("PARAKEET_LANG", "es")
HOST = os.environ.get("STREAM_HOST", "127.0.0.1")
PORT = int(os.environ.get("STREAM_PORT", "8770"))

_FPTR = ctypes.POINTER(ctypes.c_float)


def _load_lib():
    lib = ctypes.CDLL(LIB)
    lib.parakeet_capi_load.restype = ctypes.c_void_p
    lib.parakeet_capi_load.argtypes = [ctypes.c_char_p]
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
    return lib


LIBH = _load_lib()
print(f"[server] cargando modelo {GGUF} ...", flush=True)
_t0 = time.time()
CTX = LIBH.parakeet_capi_load(GGUF.encode())
if not CTX:
    raise SystemExit("parakeet_capi_load falló")
print(f"[server] modelo cargado en {time.time()-_t0:.2f}s. WS en ws://{HOST}:{PORT}", flush=True)


def _take_json(ptr) -> dict:
    """Convierte el char* (malloc'd) del capi a dict y libera la memoria C."""
    if not ptr:
        return {}
    s = ctypes.cast(ptr, ctypes.c_char_p).value.decode()
    LIBH.parakeet_capi_free_string(ptr)
    return json.loads(s) if s else {}


async def handle(ws):
    lang = "es"
    # query ?lang=xx
    try:
        q = (ws.request.path.split("?", 1)[1] if "?" in ws.request.path else "")
        for kv in q.split("&"):
            if kv.startswith("lang="):
                lang = kv[5:] or "es"
    except Exception:
        pass

    stream = LIBH.parakeet_capi_stream_begin_lang(CTX, lang.encode())
    if not stream:
        err = LIBH.parakeet_capi_last_error(CTX)
        await ws.send(json.dumps({"type": "error", "msg": (err.decode() if err else "stream_begin falló")}))
        return
    print(f"[server] conexión nueva (lang={lang})", flush=True)
    t_start = time.time()
    fed_samples = 0
    try:
        async for msg in ws:
            if isinstance(msg, (bytes, bytearray)):
                arr = (ctypes.c_float * (len(msg) // 4)).from_buffer_copy(msg)
                fed_samples += len(arr)
                doc = _take_json(LIBH.parakeet_capi_stream_feed_json(stream, arr, len(arr)))
                raw_text = doc.get("text", "")
                text = _TAG.sub(" ", raw_text)
                # EOU real del modelo EOU, o la etiqueta de idioma de Nemotron.
                eou = bool(doc.get("eou", 0)) or bool(_TAG.search(raw_text))
                eob = bool(doc.get("eob", 0))
                if text or eou or eob:
                    await ws.send(json.dumps({
                        "type": "chunk", "text": text, "eou": eou, "eob": eob,
                        "t": round(time.time() - t_start, 3),
                        "audio_s": round(fed_samples / 16000, 2),
                    }))
            elif msg == "EOS":
                doc = _take_json(LIBH.parakeet_capi_stream_finalize_json(stream))
                await ws.send(json.dumps({"type": "final", "text": _TAG.sub(" ", doc.get("text", ""))}))
    finally:
        LIBH.parakeet_capi_stream_free(stream)
        print(f"[server] conexión cerrada ({fed_samples/16000:.1f}s de audio)", flush=True)


async def main():
    async with websockets.serve(handle, HOST, PORT, max_size=None):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
