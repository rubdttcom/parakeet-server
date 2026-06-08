# parakeet-server

Servidor de transcripción de voz **100 % local** con API compatible con OpenAI
(`POST /v1/audio/transcriptions`). Es el "motor" que usa
[turbo-whisper](https://github.com/knowall-ai/turbo-whisper) cuando se apunta a
`http://localhost:8000` — turbo-whisper solo graba y teclea; la inferencia ocurre
aquí.

Usa **Parakeet TDT 0.6b v3** (ONNX int8, vía sherpa-onnx). En CPU (AMD Ryzen AI 9
365) transcribe a **~0.22 s/clip** (RTF ~0.04), ~8× más rápido que faster-whisper
small, con español preciso. No requiere GPU.

## Arranque rápido (contenedor, recomendado)

```bash
./download-model.sh        # baja el modelo (~640 MB) de HuggingFace
./container.sh up          # construye la imagen y arranca en 127.0.0.1:8000
curl http://127.0.0.1:8000/health
```

`container.sh`: `build` | `run` | `up` | `stop` | `logs`.

### Persistencia con systemd (arranque en el boot)

Hay un quadlet en `~/.config/containers/systemd/parakeet.container` que hace que
systemd gestione el contenedor:

```bash
systemctl --user daemon-reload
systemctl --user start parakeet.service
loginctl enable-linger "$USER"   # arrancar sin iniciar sesión
```

## Arranque nativo (sin contenedor)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo dnf install -y ffmpeg        # decodifica el audio de entrada
./download-model.sh
python server.py                  # escucha en 127.0.0.1:8000
```

## Configuración (variables de entorno)

| Variable | Por defecto | Descripción |
|---|---|---|
| `PARAKEET_MODEL_DIR` | `./models/parakeet-tdt-0.6b-v3` | Carpeta del modelo. |
| `PARAKEET_HOST` | `127.0.0.1` | Interfaz (en contenedor se pone `0.0.0.0`). |
| `PARAKEET_PORT` | `8000` | Puerto. |
| `PARAKEET_THREADS` | `6` | Hilos de inferencia. |

## API

`POST /v1/audio/transcriptions` (multipart): campo `file` (audio en cualquier
formato que lea ffmpeg). Devuelve `{"text": "..."}`, o texto plano si
`response_format=text`. `GET /health` → `{"status": "healthy"}`.
