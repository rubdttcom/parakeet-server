#!/usr/bin/env python3
"""PROTOTIPO — alimenta un WAV al servidor de streaming a ritmo real (1x).

Sirve para verificar el servidor y *ver la cadencia* sin micrófono. Decodifica
el wav a 16 kHz mono con ffmpeg, lo trocea en frames de 100 ms y los envía con
el espaciado real, imprimiendo el texto confirmado y los EOU conforme llegan.

  ./.venv/bin/python wav_feed.py /ruta/al.wav [ws://127.0.0.1:8770/?lang=es] [speed]
  speed: multiplicador de velocidad (1.0 = tiempo real; 4.0 = 4x más rápido)
"""
import json
import subprocess
import sys
import threading
import time

import numpy as np
import websocket

WAV = sys.argv[1]
URL = sys.argv[2] if len(sys.argv) > 2 else "ws://127.0.0.1:8770/?lang=es"
SPEED = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
SR = 16000
BLOCK = 1600  # 100 ms

GREEN = "\033[32m"; DIM = "\033[2m"; CYAN = "\033[36m"; RESET = "\033[0m"

raw = subprocess.run(
    ["ffmpeg", "-i", WAV, "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True).stdout
samples = np.frombuffer(raw, dtype=np.float32)
print(f"{CYAN}{WAV}: {len(samples)/SR:.1f}s de audio -> {URL} a {SPEED}x{RESET}\n", flush=True)

ws = websocket.create_connection(URL, max_size=None)
t_start = time.time()


def reader():
    full = []
    while True:
        try:
            raw = ws.recv()
        except Exception:
            break
        if not raw:
            break
        m = json.loads(raw)
        if m["type"] == "chunk":
            if m.get("text"):
                sys.stdout.write(GREEN + m["text"] + RESET); full.append(m["text"])
            if m.get("eou"):
                lag = (time.time() - t_start) - m.get("audio_s", 0)
                sys.stdout.write(f"  {DIM}· EOU @audio {m.get('audio_s')}s (retardo ~{lag:.1f}s){RESET}\n")
            sys.stdout.flush()
        elif m["type"] == "final":
            if m.get("text"):
                sys.stdout.write(GREEN + m["text"] + RESET); full.append(m["text"])
            print(f"\n\n{CYAN}=== completo ==={RESET}\n{''.join(full).strip()}")
            break


th = threading.Thread(target=reader, daemon=True); th.start()

frame_dt = (BLOCK / SR) / SPEED
for i in range(0, len(samples), BLOCK):
    ws.send_binary(samples[i:i + BLOCK].tobytes())
    time.sleep(frame_dt)
ws.send("EOS")
time.sleep(2.0)
ws.close()
