"""Speech-to-text engines: Whisper, plus optional Parakeet via onnx-asr."""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("whisper-api")

SAMPLE_RATE = 16_000


def _container_cpu_limit() -> int | None:
    """CPU quota set by Docker/Kubernetes, which os.cpu_count() does not reflect."""
    try:  # cgroup v2
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0:
            return max(1, math.ceil(quota / period))
    except (OSError, ValueError):
        pass
    return None


def _resolve_cpu_threads() -> int:
    override = os.getenv("WHISPER_CPU_THREADS", "").strip()
    if override:
        try:
            value = int(override)
        except ValueError:
            log.warning("Ignoring WHISPER_CPU_THREADS=%s (not an integer)", override)
        else:
            if value > 0:
                return value
    # CTranslate2 already sizes itself from the physical cores when left alone,
    # but inside a container it sees the host, not the CPU quota. Only override
    # when a quota or affinity limit is actually detected.
    quota = _container_cpu_limit()
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    if quota:
        return quota
    if affinity and os.cpu_count() and affinity < os.cpu_count():
        return affinity
    return 0  # auto: CTranslate2 picks the number of physical cores


CPU_THREADS = _resolve_cpu_threads()
# OpenMP reads these when the runtime initializes, so set them before importing
# faster-whisper (which imports CTranslate2). An explicit value in the
# environment wins. 0 means CTranslate2's own default.
if CPU_THREADS:
    os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))

import onnx_asr  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402
from faster_whisper.audio import decode_audio  # noqa: E402

# Both engines are CPU bound and the deployment target is small, so the work is
# serialised no matter which engine a request picked.
INFERENCE_LOCK = threading.Lock()


@dataclass
class Transcript:
    segments: list[dict[str, Any]]
    duration: float
    language: str | None = None
    language_probability: float | None = None


class Engine:
    """Lazy-loaded ASR engine with a shared interface."""

    supports_language = False
    requires_waveform = False

    def __init__(self, key: str, label: str, model_name: str) -> None:
        self.key = key
        self.label = label
        self.model_name = model_name
        self._load_lock = threading.Lock()
        self._loading = False
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def loading(self) -> bool:
        return self._loading

    def load(self) -> Any:
        raise NotImplementedError

    def transcribe(
        self,
        path: Path,
        waveform: np.ndarray | None,
        language: str | None,
        task: str,
        vad: bool,
    ) -> Transcript:
        raise NotImplementedError


class WhisperEngine(Engine):
    supports_language = True

    def __init__(self, key: str, label: str, model_name: str, beam_size: int, cpu_threads: int = 0) -> None:
        super().__init__(key, label, model_name)
        self.beam_size = beam_size
        self.cpu_threads = cpu_threads
        self._model: WhisperModel | None = None

    def load(self) -> WhisperModel:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                self._loading = True
                started = time.perf_counter()
                log.info("Loading Whisper '%s'...", self.model_name)
                try:
                    self._model = WhisperModel(
                        self.model_name,
                        device=_whisper_device(),
                        compute_type=_whisper_compute_type(),
                        cpu_threads=self.cpu_threads,
                    )
                    self._ready = True
                finally:
                    self._loading = False
                log.info("Whisper ready in %.1fs", time.perf_counter() - started)
        return self._model

    def transcribe(
        self,
        path: Path,
        waveform: np.ndarray | None,
        language: str | None,
        task: str,
        vad: bool,
    ) -> Transcript:
        model = self.load()
        with INFERENCE_LOCK:
            segments, info = model.transcribe(
                str(path),
                language=language,
                task=task,
                beam_size=self.beam_size,
                vad_filter=vad,
                vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 200} if vad else None,
            )
            # The generator is lazy: decoding happens here, inside the lock.
            rows = [
                {
                    "id": index,
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip(),
                }
                for index, segment in enumerate(segments, start=1)
            ]
        return Transcript(
            segments=rows,
            duration=round(info.duration, 2),
            language=info.language,
            language_probability=round(info.language_probability, 4),
        )


class ParakeetEngine(Engine):
    """NVIDIA Parakeet TDT through onnx-asr, with Silero VAD segmentation."""

    supports_language = False
    quantization = "int8"
    requires_waveform = True

    def __init__(self, key: str, label: str, model_name: str) -> None:
        super().__init__(key, label, model_name)
        self._asr: Any = None

    def load(self) -> Any:
        if self._asr is not None:
            return self._asr
        with self._load_lock:
            if self._asr is None:
                self._loading = True
                started = time.perf_counter()
                log.info("Loading Parakeet '%s'...", self.model_name)
                try:
                    try:
                        model = onnx_asr.load_model(self.model_name, quantization=self.quantization)
                    except Exception:
                        log.warning("No %s weights for %s, using the default ones", self.quantization, self.model_name)
                        model = onnx_asr.load_model(self.model_name)
                    vad = onnx_asr.load_vad("silero")
                    self._asr = model.with_vad(vad).with_timestamps()
                    self._ready = True
                finally:
                    self._loading = False
                log.info("Parakeet ready in %.1fs", time.perf_counter() - started)
        return self._asr

    def transcribe(
        self,
        path: Path,
        waveform: np.ndarray | None,
        language: str | None,
        task: str,
        vad: bool,
    ) -> Transcript:
        if waveform is None:
            raise ValueError("Parakeet needs the decoded waveform")
        asr = self.load()
        with INFERENCE_LOCK:
            results = list(asr.recognize(waveform, sample_rate=SAMPLE_RATE))
        rows = [
            {
                "id": index,
                "start": round(result.start, 2),
                "end": round(result.end, 2),
                "text": result.text.strip(),
            }
            for index, result in enumerate(results, start=1)
        ]
        return Transcript(segments=rows, duration=round(len(waveform) / SAMPLE_RATE, 2))


def load_waveform(path: Path) -> np.ndarray:
    """Decode any supported file to 16 kHz mono float32."""
    waveform = decode_audio(str(path), sampling_rate=SAMPLE_RATE)
    if waveform.ndim > 1:  # mono safeguard
        waveform = waveform.mean(axis=1)
    return np.ascontiguousarray(waveform, dtype=np.float32)


def _whisper_device() -> str:
    return os.getenv("WHISPER_DEVICE", "auto")


def _whisper_compute_type() -> str:
    explicit = os.getenv("WHISPER_COMPUTE_TYPE", "")
    if explicit:
        return explicit
    import ctranslate2

    if _whisper_device() == "cuda" or (_whisper_device() == "auto" and ctranslate2.get_cuda_device_count() > 0):
        return "float16"
    return "int8"

