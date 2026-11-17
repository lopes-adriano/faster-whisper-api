"""Speaker diarization with sherpa-onnx.

Pipeline: pyannote segmentation 3.0 (ONNX) for speech turns, 3D-Speaker
ERes2Net embeddings for speaker identity, fast clustering to group them.
Models download once into SHERPA_MODEL_DIR (~35 MB total).
"""

from __future__ import annotations

import logging
import os
import tarfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("whisper-api")

MODEL_DIR = Path(os.getenv("SHERPA_MODEL_DIR") or Path.home() / ".cache" / "whisper-api" / "sherpa")

SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)
SEGMENTATION_NAME = "sherpa-onnx-pyannote-segmentation-3-0"
EMBEDDING_NAME = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"


@dataclass
class Turn:
    start: float
    end: float
    speaker: int


def ensure_models() -> tuple[Path, Path]:
    """Download the segmentation and embedding models if they are missing."""
    segmentation = MODEL_DIR / SEGMENTATION_NAME / "model.onnx"
    embedding = MODEL_DIR / EMBEDDING_NAME

    if not segmentation.is_file():
        archive = MODEL_DIR / f"{SEGMENTATION_NAME}.tar.bz2"
        _download(SEGMENTATION_URL, archive)
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(MODEL_DIR, filter="data")
        archive.unlink(missing_ok=True)

    if not embedding.is_file():
        _download(EMBEDDING_URL, embedding)

    if not segmentation.is_file() or not embedding.is_file():
        raise RuntimeError("Speaker diarization models are missing")
    return segmentation, embedding


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    log.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    partial.replace(target)


class Diarizer:
    def __init__(self, num_threads: int = 2, threshold: float = 0.5) -> None:
        self.num_threads = num_threads
        self.threshold = threshold
        self._lock = threading.Lock()
        self._pipelines: dict[int, object] = {}

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    def _pipeline(self, num_speakers: int | None):
        import sherpa_onnx

        key = num_speakers or 0
        pipeline = self._pipelines.get(key)
        if pipeline is not None:
            return pipeline

        segmentation, embedding = ensure_models()
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(segmentation)),
                num_threads=self.num_threads,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(embedding),
                num_threads=self.num_threads,
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_speakers or -1,
                threshold=self.threshold,
            ),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        pipeline = sherpa_onnx.OfflineSpeakerDiarization(config)
        self._pipelines[key] = pipeline
        return pipeline

    def diarize(self, waveform: np.ndarray, num_speakers: int | None = None) -> tuple[list[Turn], int]:
        started = time.perf_counter()
        with self._lock:
            pipeline = self._pipeline(num_speakers)
            result = pipeline.process(waveform)
            turns = [
                Turn(start=float(segment.start), end=float(segment.end), speaker=int(segment.speaker))
                for segment in result.sort_by_start_time()
            ]
            count = int(result.num_speakers)
        log.info(
            "Diarized %.1fs of audio in %.1fs (%d speakers)",
            len(waveform) / 16_000,
            time.perf_counter() - started,
            count,
        )
        return turns, count


def assign_speakers(segments: list[dict], turns: list[Turn], min_overlap: float = 0.1) -> None:
    """Tag each transcript segment with the speaker holding most of its time."""
    for segment in segments:
        best_speaker: int | None = None
        best_overlap = 0.0
        for turn in turns:
            overlap = min(segment["end"], turn.end) - max(segment["start"], turn.start)
            if overlap > best_overlap:
                best_speaker, best_overlap = turn.speaker, overlap
        if best_speaker is not None and best_overlap >= min_overlap:
            segment["speaker"] = f"Speaker {best_speaker + 1}"
        else:
            segment["speaker"] = None
