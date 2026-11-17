# Local transcription API

A small FastAPI service for speech-to-text, with a single-page web UI to upload audio, read the transcript and download subtitles.

The service runs **Whisper** ([faster-whisper](https://github.com/SYSTRAN/faster-whisper)) with four model sizes, selectable per request:

- **small** — the default; good accuracy/speed balance.
- **turbo** — `large-v3-turbo`; the most accurate of the four and still faster than large-v3.
- **base** — a middle ground.
- **tiny** — fastest, for quick drafts.

All four support language hints, translation to English and the VAD toggle, and run with `int8` on CPU.

Optionally, **speaker diarization** (sherpa-onnx) labels each segment with its speaker.

Everything runs on your machine; uploaded audio is deleted right after transcription.

## Quick start

```bash
uv sync                # create .venv and install dependencies
uv run python app.py   # http://127.0.0.1:8000
```

Open <http://127.0.0.1:8000>, drop an audio file, press **Transcribe**.

For development, auto-reload on file changes:

```bash
uv run uvicorn app:app --reload
```

The default engine (`whisper-small`) is preloaded before the server starts serving. On the first local start its weights (~460 MB) download into the Hugging Face cache; `turbo`, `base` and `tiny` download the first time you select them. The Docker image already contains all four. The diarization models (~46 MB) download the first time you enable speaker separation.

No ffmpeg install is needed: audio is decoded through PyAV, which ships as a wheel.

## API

Interactive docs: <http://127.0.0.1:8000/docs>

### `POST /transcribe`

Multipart upload with query parameters:

| Parameter  | Default        | Description                                                       |
|------------|----------------|-------------------------------------------------------------------|
| `file`     | —              | audio or video file (mp3, wav, m4a, ogg, flac, mp4, webm, mkv, …) |
| `engine`   | `whisper-small`| `whisper-small`, `whisper-turbo`, `whisper-base` or `whisper-tiny` |
| `language` | auto-detect    | ISO code, e.g. `en`, `pt`                                         |
| `task`     | `transcribe`   | `translate` renders any language as English                       |
| `vad`      | `true`         | Silero VAD, skips silence so timestamps stay tight                |
| `diarize`  | `false`        | Tag every segment with its speaker                                 |
| `speakers` | auto           | Expected speaker count (1–20), helps when you know it              |
| `format`   | `json`         | `json`, `text` or `srt`                                            |

```bash
curl -X POST "http://127.0.0.1:8000/transcribe?engine=whisper-small&format=text" \
     -F "file=@meeting.mp3"
```

With a language hint, the tiny model and speakers:

```bash
curl -X POST "http://127.0.0.1:8000/transcribe?engine=whisper-tiny&language=en&diarize=true&speakers=2&format=srt" \
     -F "file=@interview.m4a" -o interview.srt
```

JSON response:

```json
{
  "filename": "interview.m4a",
  "size_bytes": 2411520,
  "engine": "whisper-small",
  "model": "small",
  "language": "en",
  "language_probability": 0.9971,
  "duration": 183.4,
  "elapsed": 21.8,
  "task": "transcribe",
  "diarized": true,
  "speakers": 2,
  "text": "Hello everyone, thanks for joining...",
  "srt": "1\n00:00:01,200 --> 00:00:04,800\nSpeaker 1: Hello everyone...\n",
  "segments": [
    {
      "id": 1,
      "start": 1.2,
      "end": 4.8,
      "text": "Hello everyone thanks for joining",
      "speaker": "Speaker 1"
    }
  ]
}
```

`GET /healthz` lists the enabled engines, whether each model is loaded or loading, the CPU thread count, and whether a request is running.

## Engines

| Engine          | Model             | Notes                                      |
|-----------------|-------------------|--------------------------------------------|
| `whisper-small` | `small`           | default; good speed/accuracy balance       |
| `whisper-turbo` | `large-v3-turbo`  | most accurate; still faster than large-v3  |
| `whisper-base`  | `base`            | balance between speed and accuracy         |
| `whisper-tiny`  | `tiny`            | fastest, useful for quick drafts           |

All sizes run with `int8` on CPU (`WHISPER_COMPUTE_TYPE=int8`).

`engine=whisper` is still accepted as an alias for `whisper-small`.

Models load lazily: the default size is preloaded at startup, and the others load the first time they are selected. Each size keeps its model in RAM once loaded; turbo is the largest (roughly 2 GB resident), so using all four needs around 4 GB. The Docker target has 8 GB.

To expose only one or two sizes:

```bash
ASR_ENGINES=whisper-base uv run python app.py
ASR_ENGINES=whisper-small,whisper-tiny ASR_ENGINE=whisper-tiny uv run python app.py
```

## Speaker diarization

`sherpa-onnx` runs pyannote segmentation 3.0 followed by 3D-Speaker ERes2Net embeddings and fast clustering, entirely on CPU and without a Hugging Face token.

- Models download once to `SHERPA_MODEL_DIR` (default `~/.cache/whisper-api/sherpa`) and total ~46 MB.
- Each transcript segment gets the speaker that covers most of its time. Segments are VAD/ASR units, so a speaker change in the middle of a segment is not split — short segments make this rare, and `speakers=N` helps clustering when the count is known.
- Works with every Whisper size: `diarize=true&engine=whisper-tiny` is supported.
- Disable the feature with `DIARIZATION=0`.

## Configuration

Set environment variables before starting:

| Variable                | Default          | Notes                                                     |
|-------------------------|------------------|-----------------------------------------------------------|
| `ASR_ENGINE`            | `whisper-small`  | default engine for requests without `engine`               |
| `ASR_ENGINES`           | all four sizes   | comma-separated list of sizes to expose                    |
| `ASR_PRELOAD`           | `1`              | load the default engine before the server starts serving  |
| `WHISPER_DEVICE`        | `auto`           | `cpu`, `cuda` or `auto`                                    |
| `WHISPER_COMPUTE_TYPE`  | `int8` on CPU, `float16` on GPU | e.g. `int8_float16`, `float32`               |
| `WHISPER_BEAM_SIZE`     | `5`              | lower = faster, higher = slightly more accurate            |
| `WHISPER_CPU_THREADS`   | auto             | threads for CTranslate2 / OpenMP; auto reads the cgroup CPU quota |
| `WHISPER_MAX_UPLOAD_MB` | `200`            | upload cap                                                 |
| `DIARIZATION`           | `1`              | set to `0` to hide and disable diarization                 |
| `DIARIZATION_THREADS`   | `2`              | threads for the diarization models                         |
| `HOST` / `PORT`         | `127.0.0.1:8000` | where uvicorn listens                                      |

Example with a GPU:

```bash
WHISPER_DEVICE=cuda WHISPER_COMPUTE_TYPE=float16 uv run python app.py
```

### CPU threads

By default CTranslate2 sizes its thread pool from the cores the process can see. Inside a container that is usually the host count, not the CPU limit, so the decoder ends up oversubscribed. The app reads the cgroup CPU quota (`cpu.max`, or the cgroup v1 quota) and passes that to `cpu_threads`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` before faster-whisper is imported. With `cpus: "4.0"` it uses 4 threads, even on a large host.

Override it with `WHISPER_CPU_THREADS=4`. The current value is visible at `GET /healthz` as `cpu_threads`; `0` means CTranslate2 chose its own default (used when there is no quota or affinity limit, e.g. bare metal). On a dedicated container, using the whole quota is intentional; reserving 25% (as some desktop guides suggest) only makes sense when other workloads share the machine.

## Docker

The image bakes all four Whisper models (tiny, base, small and turbo), the Silero VAD and the diarization models in, and preloads `whisper-small` before the server starts accepting requests. Selecting another size loads it from disk — nothing downloads at runtime. All sizes use `int8` on CPU.

To bake only some sizes, build with `--build-arg WHISPER_MODELS="small turbo"` and expose the same list through `ASR_ENGINES`.

```bash
docker compose up --build          # cpus: "4.0", mem_limit: 8g
```

Equivalent direct run:

```bash
docker build -t faster-whisper-api:small .
docker run --rm -p 8000:8000 --cpus 4 --memory 8g --memory-swap 8g faster-whisper-api:small
```

Notes for a 4 vCPU / 8 GB box (a Fargate task with 4 vCPU and 8 GB):

- One container = one process. Keep a single uvicorn worker: every extra worker loads its own copy of the model. Scale with replicas behind a load balancer, not with workers.
- Keep `int8` on CPU; it is roughly twice as fast as `float32` and uses about half the memory.
- Threads follow the `cpus` limit automatically; compose also sets `WHISPER_CPU_THREADS=4` explicitly to match `cpus: "4.0"`.
- Set `ASR_ENGINES` to the sizes you actually use if the container is short on memory.
- The image runs as a non-root user, exposes `8000`, and has a health check against `/healthz`.
- Uploads go to `/tmp/uploads` and are deleted after each request, so no volume is needed.

Measured earlier on Whisper small with 2 CPUs and 4 GB:

| Input | Wall time | Speed | Peak memory |
|-------|-----------|-------|-------------|
| 13 s clip | 5.3 s | 2.5× realtime | ~900 MB |
| 5 min 30 s recording | 94 s | 3.5× realtime | ~900 MB |

Two uploads arriving together are queued: the second waited for the first (5 s and 10 s) and peak memory did not grow.

## How it works

- `app.py` — API, uploads, speaker labelling and SRT generation. Inference runs in a worker thread so the server keeps answering.
- `engines.py` — the Whisper engines, thread detection, and model loading behind one inference lock.
- `diarization.py` — sherpa-onnx pipeline, model download and the segment-to-speaker assignment.
- `static/index.html` — the UI: drag & drop, engine and language pickers, speaker toggle, audio player, clickable timestamps, copy and `.txt` / `.srt` download.
- Uploads stream to `.uploads/` with a size cap and are removed in a `finally` block.

## Notes

- PyAV is pinned to `av<19` because faster-whisper 1.2 still passes the `metadata_errors` argument that PyAV 19 removed. Drop the pin once faster-whisper supports it.
- The `small` model needs ~1 GB of RAM while loaded; `base` and `tiny` need less and `turbo` needs the most (~2 GB). All run with `int8` on CPU.
- Uploaded files are kept only while a request is in flight; leftovers from a crash are cleaned at startup after 24 h.

## Layout

```
app.py              API, uploads, diarization glue
engines.py          Whisper engines
diarization.py      sherpa-onnx speaker separation
static/index.html   web UI
pyproject.toml      uv project
.uploads/           temporary audio files (git-ignored)
```
