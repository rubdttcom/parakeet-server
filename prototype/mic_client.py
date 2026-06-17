#!/usr/bin/env python3
"""PROTOTIPO — cliente de micrófono para sentir la transcripción en streaming.

Captura el micro (16 kHz mono float32), lo envía en tiempo real al servidor
WebSocket y va imprimiendo el texto confirmado conforme llega. Cada salto de
línea es un <EOU> (fin de enunciado): el punto donde, en el sistema real,
turbo-whisper teclearía ese segmento.

Muestra a la derecha el "retardo": segundos de reloj transcurridos menos
segundos de audio ya capturados -> cuánto va por detrás del habla.

  ./.venv/bin/python mic_client.py            # ws://127.0.0.1:8770, lang es
  Habla. Ctrl+C para finalizar y ver el transcript completo.
"""
import json
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import websocket  # websocket-client (sync)

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8770/?lang=es"
SR = 16000
BLOCK = 1600  # 100 ms por frame

GREEN = "\033[32m"; DIM = "\033[2m"; CYAN = "\033[36m"; RESET = "\033[0m"


def main():
    ws = websocket.create_connection(URL, max_size=None)
    print(f"{CYAN}Conectado a {URL}. Habla… (Ctrl+C para terminar){RESET}\n", flush=True)
    t_start = time.time()
    fed_s = [0.0]
    line_open = [False]

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
                txt = m.get("text", "")
                if txt:
                    sys.stdout.write(GREEN + txt + RESET)
                    line_open[0] = True
                    full.append(txt)
                if m.get("eou"):
                    lag = (time.time() - t_start) - m.get("audio_s", 0)
                    sys.stdout.write(f"  {DIM}· EOU (retardo ~{lag:.1f}s){RESET}\n")
                    line_open[0] = False
                sys.stdout.flush()
            elif m["type"] == "final":
                if m.get("text"):
                    sys.stdout.write(GREEN + m["text"] + RESET)
                    full.append(m["text"])
                print(f"\n\n{CYAN}=== transcript completo ==={RESET}\n{''.join(full).strip()}")
                break
            elif m["type"] == "error":
                print(f"\n[error servidor] {m.get('msg')}")
                break

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    def on_audio(indata, frames, t, status):
        ws.send_binary(indata[:, 0].astype(np.float32).tobytes())
        fed_s[0] += frames / SR

    try:
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            blocksize=BLOCK, callback=on_audio):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n{DIM}(finalizando…){RESET}", flush=True)
        try:
            ws.send("EOS")
            time.sleep(1.5)  # deja llegar el 'final'
        except Exception:
            pass
    finally:
        try:
            ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
