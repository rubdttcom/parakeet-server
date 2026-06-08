# parakeet-server

Servidor de transcripción de voz **100 % local** con API compatible con OpenAI
(`POST /v1/audio/transcriptions`). Es el "motor" que usa
[turbo-whisper](https://github.com/knowall-ai/turbo-whisper) cuando se apunta a
`http://localhost:8000` — turbo-whisper solo graba y teclea; la inferencia ocurre
aquí.

Usa **Parakeet TDT 0.6b v3** (ONNX int8, vía sherpa-onnx). En CPU (AMD Ryzen AI 9
365) transcribe a **~0.22 s/clip** (RTF ~0.04), ~8× más rápido que faster-whisper
small, con español preciso. No requiere GPU.

## Prerequisitos

- **Contenedor:** `podman` (o `docker`).
- **Nativo:** Python 3.10+, `ffmpeg` en el PATH.
- ~640 MB de disco (modelo) + ~1 GB de RAM en ejecución.

## Arranque rápido (contenedor, recomendado)

```bash
./download-model.sh        # baja el modelo (~640 MB) de HuggingFace
./container.sh up          # construye la imagen y arranca en 127.0.0.1:8000
curl http://127.0.0.1:8000/health
```

`container.sh`: `build` | `run` | `up` | `stop` | `logs`.

### Persistencia con systemd (arranque en el boot)

El repo incluye un quadlet (`parakeet.container`). Cópialo a la carpeta de
quadlets de usuario y deja que systemd gestione el contenedor:

```bash
mkdir -p ~/.config/containers/systemd
cp parakeet.container ~/.config/containers/systemd/
systemctl --user daemon-reload          # genera parakeet.service desde el quadlet
systemctl --user start parakeet.service
loginctl enable-linger "$USER"          # arrancar en el boot sin iniciar sesión
```

> El quadlet asume que la imagen `localhost/parakeet-server:latest` ya existe
> (`./container.sh build`) y que el modelo está en `~/parakeet-server/models`.

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

### `POST /v1/audio/transcriptions`

Multipart con el campo `file` (audio en cualquier formato que lea ffmpeg:
wav/mp3/webm/...). Devuelve JSON, o texto plano si `response_format=text`.

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=json"
# -> {"text": "el texto transcrito"}
```

| Campo (form) | Efecto |
|---|---|
| `file` | **Obligatorio.** El audio a transcribir. |
| `response_format` | `text` → devuelve texto plano; cualquier otro → JSON. |
| `model`, `language`, `prompt` | **Aceptados pero ignorados** (compatibilidad con la API de OpenAI; el modelo es multilingüe y autodetecta). |

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
# -> {"status": "healthy"}
```

## El modelo

Parakeet TDT 0.6b v3 convertido a ONNX int8 por
[k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (descargado por
`download-model.sh`). No se versiona en git (640 MB); está en `.gitignore`.
