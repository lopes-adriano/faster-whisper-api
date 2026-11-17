"""Whisper transcription API with tiny, base, small and turbo models.

Run with:  uv run python app.py
Docs at:   http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from diarization import Diarizer, assign_speakers
from engines import CPU_THREADS, INFERENCE_LOCK, Engine, ParakeetEngine, WhisperEngine, load_waveform

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = Path(os.getenv("WHISPER_UPLOAD_DIR") or BASE_DIR / ".uploads")

WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
MAX_UPLOAD_MB = int(os.getenv("WHISPER_MAX_UPLOAD_MB", "200"))
PRELOAD = os.getenv("ASR_PRELOAD", os.getenv("WHISPER_PRELOAD", "1")).lower() not in {"0", "false", "no"}
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
DIARIZATION_ENABLED = os.getenv("DIARIZATION", "1").lower() not in {"0", "false", "no"}
DIARIZATION_THREADS = int(os.getenv("DIARIZATION_THREADS", "2"))

# Exposed engine key -> faster-whisper model size. Order is the UI order.
WHISPER_SIZES = {
    "whisper-small": "small",
    "whisper-turbo": "turbo",
    "whisper-base": "base",
    "whisper-tiny": "tiny",
}
WHISPER_LABELS = {"whisper-turbo": "Whisper turbo (large-v3-turbo)"}
# Older requests used engine=whisper for the small model.
ENGINE_ALIASES = {"whisper": "whisper-small"}

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
)
log = logging.getLogger("whisper-api")


def _build_engines() -> dict[str, Engine]:
    factories: dict[str, Any] = {
        key: (lambda key=key, size=size: WhisperEngine(
            key, WHISPER_LABELS.get(key, f"Whisper {size}"), size, WHISPER_BEAM_SIZE, CPU_THREADS
        ))
        for key, size in WHISPER_SIZES.items()
    }
    # Parakeet is not exposed by default, but stays available to ASR_ENGINES.
    factories["parakeet-v2"] = lambda: ParakeetEngine(
        "parakeet-v2",
        "Parakeet 0.6B v2 (English)",
        os.getenv("PARAKEET_V2_MODEL", "nemo-parakeet-tdt-0.6b-v2"),
    )
    factories["parakeet-v3"] = lambda: ParakeetEngine(
        "parakeet-v3",
        "Parakeet 0.6B v3 (multilingual)",
        os.getenv("PARAKEET_V3_MODEL", "nemo-parakeet-tdt-0.6b-v3"),
    )
    wanted = [name.strip() for name in os.getenv("ASR_ENGINES", "").split(",") if name.strip()]
    if not wanted:
        wanted = list(WHISPER_SIZES)
    unknown = [name for name in wanted if name not in factories]
    if unknown:
        log.warning("Ignoring unknown engines in ASR_ENGINES: %s", ", ".join(unknown))
    engines = {name: factories[name]() for name in wanted if name in factories}
    if not engines:
        raise RuntimeError("ASR_ENGINES does not enable any known engine")
    return engines


ENGINES = _build_engines()
DEFAULT_ENGINE = (
    os.getenv("ASR_ENGINE", "").strip()
    or ("whisper-small" if "whisper-small" in ENGINES else next(iter(ENGINES)))
)
if DEFAULT_ENGINE not in ENGINES:
    raise RuntimeError(f"ASR_ENGINE={DEFAULT_ENGINE} is not enabled; available: {', '.join(ENGINES)}")

DIARIZER = Diarizer(num_threads=DIARIZATION_THREADS)


def _format_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _segments_to_srt(segments: list[dict[str, Any]]) -> str:
    blocks = []
    for segment in segments:
        text = segment["text"]
        if segment.get("speaker"):
            text = f"{segment['speaker']}: {text}"
        blocks.append(
            f"{segment['id']}\n{_format_timestamp(segment['start'])} --> {_format_timestamp(segment['end'])}\n{text}\n"
        )
    return "\n".join(blocks)


@asynccontextmanager
async def lifespan(_: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _clean_stale_uploads()
    if PRELOAD:
        # Load before the server accepts requests so the first upload never waits
        # for a model load (or a download).
        log.info("Whisper CPU threads: %s", CPU_THREADS or "auto")
        await asyncio.to_thread(_preload, ENGINES[DEFAULT_ENGINE])
    yield


def _clean_stale_uploads(max_age_hours: int = 24) -> None:
    """Remove files left behind by an interrupted run."""
    cutoff = time.time() - max_age_hours * 3600
    for leftover in UPLOAD_DIR.glob("*"):
        try:
            if leftover.is_file() and leftover.stat().st_mtime < cutoff:
                leftover.unlink()
        except OSError:
            pass


def _preload(engine: Engine) -> None:
    started = time.perf_counter()
    log.info("Preloading %s (%s)...", engine.key, engine.model_name)
    try:
        engine.load()
    except Exception:  # pragma: no cover - startup is best effort
        log.exception("Could not preload %s; it will be retried on the first request", engine.key)
    else:
        log.info("Preloaded %s in %.1fs", engine.key, time.perf_counter() - started)


app = FastAPI(
    title="Transcription API",
    version="1.1.0",
    description=(
        "Local speech-to-text with faster-whisper (tiny, base, small and turbo), optional "
        "speaker diarization. POST an audio file to `/transcribe` or use the web UI at `/`."
    ),
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    page = STATIC_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=500, detail="static/index.html is missing")
    return FileResponse(page)


@app.get("/healthz", summary="Service, model and engine status")
async def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "default_engine": DEFAULT_ENGINE,
        "busy": INFERENCE_LOCK.locked() or DIARIZER.busy,
        "diarization": DIARIZATION_ENABLED,
        "cpu_threads": CPU_THREADS,
        "engines": {
            key: {
                "label": engine.label,
                "model": engine.model_name,
                "supports_language": engine.supports_language,
                "ready": engine.ready,
                "loading": engine.loading,
            }
            for key, engine in ENGINES.items()
        },
    }


@app.post("/transcribe", summary="Transcribe an audio file")
async def transcribe(
    file: UploadFile = File(..., description="Audio or video file (mp3, wav, m4a, ogg, flac, mp4, webm...)"),
    engine: str | None = Query(None, description="whisper-small (default), whisper-turbo, whisper-base or whisper-tiny"),
    language: str | None = Query(None, pattern="^[a-z]{2,3}$", description="ISO code such as 'en' or 'pt'"),
    task: str = Query("transcribe", pattern="^(transcribe|translate)$", description="'translate' turns any language into English"),
    vad: bool = Query(True, description="Filter out silence with Silero VAD so timestamps stay tight"),
    diarize: bool = Query(False, description="Tag each segment with its speaker"),
    speakers: int | None = Query(None, ge=1, le=20, description="Expected number of speakers when diarizing"),
    output: str = Query("json", alias="format", pattern="^(json|text|srt)$", description="Response format"),
) -> Any:
    key = ENGINE_ALIASES.get(engine or DEFAULT_ENGINE, engine or DEFAULT_ENGINE)
    active = ENGINES.get(key)
    if active is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown engine '{key}'. Available: {', '.join(ENGINES)}",
        )
    if not active.supports_language:
        if language:
            raise HTTPException(status_code=422, detail=f"{key} detects the language itself; drop the 'language' parameter.")
        if task == "translate":
            raise HTTPException(status_code=422, detail=f"{key} cannot translate; use engine=whisper-small.")
    if diarize and not DIARIZATION_ENABLED:
        raise HTTPException(status_code=422, detail="Speaker diarization is disabled on this server.")

    started = time.perf_counter()
    path, size = await _store_upload(file)
    try:
        waveform = None
        if diarize or active.requires_waveform:
            try:
                waveform = await asyncio.to_thread(load_waveform, path)
            except Exception as exc:
                log.warning("Could not decode %s: %s", file.filename, exc)
                raise HTTPException(status_code=400, detail=f"Could not decode the audio file: {exc}") from exc
            if waveform.size == 0:
                raise HTTPException(status_code=400, detail="The file contains no audio.")

        try:
            transcript = await asyncio.to_thread(
                active.transcribe, path, waveform, language or None, task, vad
            )
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("Transcription failed")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc

        speaker_count = None
        if diarize and waveform is not None:
            try:
                turns, speaker_count = await asyncio.to_thread(DIARIZER.diarize, waveform, speakers)
            except Exception as exc:
                log.exception("Diarization failed")
                raise HTTPException(status_code=500, detail=f"Speaker diarization failed: {exc}") from exc
            assign_speakers(transcript.segments, turns)
    finally:
        path.unlink(missing_ok=True)

    text = " ".join(segment["text"] for segment in transcript.segments).strip()
    srt = _segments_to_srt(transcript.segments)
    elapsed = time.perf_counter() - started

    if output == "text":
        return PlainTextResponse(text or "", media_type="text/plain; charset=utf-8")
    if output == "srt":
        return PlainTextResponse(srt or "", media_type="text/plain; charset=utf-8")

    return {
        "filename": file.filename,
        "size_bytes": size,
        "engine": key,
        "model": active.model_name,
        "language": transcript.language,
        "language_probability": transcript.language_probability,
        "duration": transcript.duration,
        "elapsed": round(elapsed, 2),
        "task": task,
        "diarized": diarize,
        "speakers": speaker_count,
        "text": text,
        "srt": srt,
        "segments": transcript.segments,
    }


async def _store_upload(file: UploadFile) -> tuple[Path, int]:
    """Stream the upload to disk while enforcing the size limit."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "").suffix.lower()[:12]
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    limit = MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    try:
        with path.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail=f"File is larger than {MAX_UPLOAD_MB} MB")
                handle.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if size == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return path, size


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
