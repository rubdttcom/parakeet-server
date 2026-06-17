# parakeet-server

Servidor de transcripción de voz **100 % local** con API compatible con OpenAI
(`POST /v1/audio/transcriptions`). Es el "motor" que usa
[turbo-whisper](https://github.com/knowall-ai/turbo-whisper) cuando se apunta a
`http://localhost:8000` — turbo-whisper solo graba y teclea; la inferencia ocurre
aquí.

Usa **Parakeet TDT 0.6b v3**. Dos motores **batch** seleccionables con
`PARAKEET_ENGINE`:

- **`parakeet-cpp`** (recomendado): GGUF q8_0 vía [parakeet.cpp](https://github.com/mudler/parakeet.cpp)
  (ggml) con aceleración **iGPU (Vulkan)**. Más preciso que el int8 (acierta dígitos y
  palabras que el int8 confunde) y ~2-3× más rápido: **~0.17 s** en un clip de 6.4 s en
  una Radeon 880M.
- **`sherpa`**: ONNX int8 vía sherpa-onnx, en **CPU** (sin GPU). ~0.22 s/clip (RTF ~0.04).

Y un modo **streaming** (dictado en vivo) opcional vía WebSocket
(`/v1/audio/stream`, modelo cache-aware Nemotron) — ver más abajo.

## Prerequisitos

- **Contenedor:** `podman` (o `docker`). Para `parakeet-cpp`/streaming, iGPU con
  Vulkan (`/dev/dri/renderD128`) y mesa ≥ 24 (la imagen ya lo incluye).
- **Nativo:** Python 3.10+, `ffmpeg` en el PATH.
- Disco: ~2.5 GB (los tres modelos). RAM: ~1 GB (sherpa) / ~1 GB (parakeet-cpp);
  con streaming activo se cargan dos modelos (~2.7 GB).

## Arranque rápido (contenedor, recomendado)

```bash
./download-model.sh        # baja los modelos (~2.5 GB) de HuggingFace
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
| `PARAKEET_ENGINE` | `sherpa` | Motor batch: `parakeet-cpp` (GGUF+iGPU) o `sherpa` (ONNX int8 CPU). El quadlet usa `parakeet-cpp`. |
| `PARAKEET_GGUF` | `/models/parakeet-cpp/tdt-0.6b-v3-q8_0.gguf` | Modelo GGUF para `parakeet-cpp`. |
| `PARAKEET_STREAM_GGUF` | *(vacío)* | Si apunta a un GGUF cache-aware (Nemotron), activa el WS `/v1/audio/stream`. Vacío = streaming off. |
| `PARAKEET_MODEL_DIR` | `./models/parakeet-tdt-0.6b-v3` | Carpeta del modelo ONNX (motor `sherpa`). |
| `PARAKEET_THREADS` | `8` | Hilos de inferencia. **Solo `sherpa`** (parakeet-cpp usa la iGPU). |
| `PARAKEET_LANG` | `es` | Idioma para parakeet-cpp/streaming (Nemotron). |
| `PARAKEET_HOST` | `127.0.0.1` | Interfaz (en contenedor se pone `0.0.0.0`). |
| `PARAKEET_PORT` | `8000` | Puerto. |

> El motor `parakeet-cpp` requiere `libparakeet.so` + backends ggml (se compilan en la
> imagen) y acceso a la iGPU (`/dev/dri/renderD128`, ya en el quadlet).

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

### `WS /v1/audio/stream` (streaming, dictado en vivo)

Activo solo si `PARAKEET_STREAM_GGUF` apunta a un modelo cache-aware (Nemotron).
El cliente envía frames **binarios** de PCM `float32` mono 16 kHz y el texto `EOS`
para finalizar; el servidor responde JSON por chunk:

```
{"type":"chunk","text":"<recién confirmado>","eou":bool,"t":<seg>}   # texto inmutable
{"type":"final","text":"<cola final>"}                               # al recibir EOS
```

El texto es **confirmado** (no se revisa) → se puede teclear según llega sin corregir.
Hay clientes de prueba en `prototype/` (`mic_client.py`, `wav_feed.py`).

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
# -> {"status":"healthy","engine":"parakeet-cpp","streaming":true}
```

## Los modelos

`download-model.sh` baja los tres (ninguno se versiona en git; van en `.gitignore`):

- **ONNX int8** (motor `sherpa`): Parakeet TDT 0.6b v3 de
  [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) → `models/parakeet-tdt-0.6b-v3/`.
- **GGUF q8_0** (motor `parakeet-cpp`): de
  [mudler/parakeet-cpp-gguf](https://huggingface.co/mudler/parakeet-cpp-gguf) →
  `models/parakeet-cpp/tdt-0.6b-v3-q8_0.gguf`.
- **GGUF Nemotron streaming** (WS): `models/parakeet-cpp/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf`.

## Licencia

[MIT](LICENSE) © Rubén Fernández
