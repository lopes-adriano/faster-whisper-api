# syntax=docker/dockerfile:1

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

# Whisper sizes baked into the image so no request or startup downloads a model.
ARG WHISPER_MODELS="tiny base small turbo"

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/models \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    SHERPA_MODEL_DIR=/models/sherpa \
    HOST=0.0.0.0 \
    PORT=8000 \
    ASR_ENGINE=whisper-small \
    ASR_ENGINES=whisper-small,whisper-turbo,whisper-base,whisper-tiny \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE_TYPE=int8 \
    WHISPER_UPLOAD_DIR=/tmp/uploads \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# libgomp is needed by ctranslate2 / onnxruntime on Debian slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev \
    && rm -rf /root/.cache/uv

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /models /tmp/uploads \
    && chown app:app /models /tmp/uploads

COPY --chown=app:app app.py engines.py diarization.py ./
COPY --chown=app:app static ./static

USER app

# Bake the tiny, base, small and turbo Whisper models.
RUN for size in $WHISPER_MODELS; do \
        python -c "import sys; from faster_whisper import WhisperModel; WhisperModel(sys.argv[1])" "$size"; \
    done

# Speaker diarization models (~46 MB) are small enough to always ship.
RUN python -c "import diarization; diarization.ensure_models()"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

CMD ["python", "app.py"]
