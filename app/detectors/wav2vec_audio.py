import os
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from app.detectors.base import BaseDetector
from app.schemas import DetectionResult

SAMPLE_RATE = 16000
MODEL_ID = "garystafford/wav2vec2-deepfake-voice-detector"


def _extract_audio(video_path: str) -> str | None:
    """Extract audio from video to a temp WAV file at 16kHz mono."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", str(SAMPLE_RATE),
                "-ac", "1",
                "-y", tmp.name,
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            os.unlink(tmp.name)
            return None
        if os.path.getsize(tmp.name) < 1000:
            os.unlink(tmp.name)
            return None
        return tmp.name
    except Exception:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        return None


class Wav2VecAudioDetector(BaseDetector):
    """Wav2Vec2-XLSR deepfake voice detector.

    Fine-tuned Wav2Vec2-XLSR (300M params) for binary real/fake audio
    classification. Trained on 6 modern TTS engines including ElevenLabs,
    Amazon Polly, and Kokoro.
    """

    MODEL_NAME = "Wav2Vec2 Voice Detector"

    def __init__(self):
        self.model_name = self.MODEL_NAME
        self.model = None
        self.feature_extractor = None

    def load(self) -> None:
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        self.model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
        self.model.eval()

    def detect(self, file_path: str) -> list[DetectionResult]:
        start = time.perf_counter()

        audio_path = _extract_audio(file_path)
        if audio_path is None:
            return []

        try:
            audio, sr = sf.read(audio_path)
        finally:
            os.unlink(audio_path)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        if len(audio) < SAMPLE_RATE * 0.5:
            return []

        inputs = self.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True,
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        # Class 0 = real, class 1 = fake
        real_score = float(probs[0])
        fake_score = float(probs[1])
        elapsed_ms = (time.perf_counter() - start) * 1000

        return [
            DetectionResult(
                label="fake",
                score=round(fake_score, 4),
                model_used=self.model_name,
                media_type="audio",
                processing_time_ms=round(elapsed_ms, 2),
            ),
            DetectionResult(
                label="real",
                score=round(real_score, 4),
                model_used=self.model_name,
                media_type="audio",
                processing_time_ms=round(elapsed_ms, 2),
            ),
        ]

    def supported_media_types(self) -> list[str]:
        return ["video"]
