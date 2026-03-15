import os
import subprocess
import tempfile
import time
import urllib.request

import numpy as np
import soundfile as sf
import torch

from app.detectors.aasist.model import AASIST
from app.detectors.base import BaseDetector
from app.schemas import DetectionResult

SAMPLE_RATE = 16000
NB_SAMP = 64600  # ~4.04 seconds at 16kHz

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "aasist")
WEIGHTS_URL = "https://github.com/clovaai/aasist/raw/main/models/weights/AASIST.pth"
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "AASIST.pth")

AASIST_CONFIG = {
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


def _download_weights() -> str:
    if os.path.isfile(WEIGHTS_PATH):
        return WEIGHTS_PATH
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    print(f"Downloading AASIST weights to {WEIGHTS_PATH}...")
    urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
    return WEIGHTS_PATH


def _extract_audio(video_path: str) -> str | None:
    """Extract audio from video to a temp WAV file at 16kHz mono. Returns None if no audio."""
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


def _load_audio(audio_path: str) -> np.ndarray:
    """Load audio and pad/truncate to NB_SAMP."""
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    # Pad or truncate to fixed length
    if len(audio) < NB_SAMP:
        # Wrap-pad (repeat audio to fill)
        repeats = NB_SAMP // len(audio) + 1
        audio = np.tile(audio, repeats)[:NB_SAMP]
    else:
        audio = audio[:NB_SAMP]

    return audio


class AasistDetector(BaseDetector):
    """AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks."""

    MODEL_NAME = "AASIST"

    def __init__(self):
        self.model_name = self.MODEL_NAME
        self.model = None

    def load(self) -> None:
        weights_path = _download_weights()
        self.model = AASIST(AASIST_CONFIG)
        self.model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        self.model.eval()

    def detect(self, file_path: str) -> list[DetectionResult]:
        start = time.perf_counter()

        audio_path = _extract_audio(file_path)
        if audio_path is None:
            return []

        try:
            audio = _load_audio(audio_path)
        finally:
            os.unlink(audio_path)

        if len(audio) < SAMPLE_RATE * 0.5:
            return []

        tensor = torch.tensor(audio).unsqueeze(0)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=-1)[0]

        # AASIST: class 0 = spoof (fake), class 1 = bonafide (real)
        fake_score = float(probs[0])
        real_score = float(probs[1])

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
