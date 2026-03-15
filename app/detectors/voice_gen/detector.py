import os
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf
import torch

from app.detectors.base import BaseDetector
from app.detectors.voice_gen.model import AudioFakeDetector
from app.schemas import DetectionResult

SAMPLE_RATE = 16000
NB_SAMP = 64000  # 4 seconds at 16kHz

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "voice_gen")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "AudioDeepFakeDetection-ckpt-28.pth")

# Google Drive file ID from the repo README
GDRIVE_FILE_ID = "1J8defEI-JJmJVMq4iVlTh825UnyZugQo"


def _download_weights() -> str:
    if os.path.isfile(WEIGHTS_PATH):
        return WEIGHTS_PATH
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    print(f"Downloading VoiceGen checkpoint to {WEIGHTS_PATH}...")
    print("NOTE: Download from Google Drive. If this fails, download manually from:")
    print(f"  https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view")
    print(f"  and place it at: {WEIGHTS_PATH}")
    # Use gdown if available, otherwise try curl
    try:
        import gdown
        gdown.download(id=GDRIVE_FILE_ID, output=WEIGHTS_PATH, quiet=False)
    except ImportError:
        # Fallback: direct curl with Google Drive confirm trick
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}&confirm=t"
        subprocess.run(
            ["curl", "-L", "-o", WEIGHTS_PATH, url],
            check=True,
            timeout=120,
        )
    return WEIGHTS_PATH


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


def _load_audio(audio_path: str) -> np.ndarray:
    """Load audio and pad/truncate to NB_SAMP."""
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    if len(audio) < NB_SAMP:
        repeats = NB_SAMP // len(audio) + 1
        audio = np.tile(audio, repeats)[:NB_SAMP]
    else:
        audio = audio[:NB_SAMP]

    return audio


def _load_checkpoint(weights_path: str, device: str) -> dict:
    """Load and extract the model state dict from the training checkpoint."""
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    # The checkpoint may be a full training state dict with 'model' key
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    return state_dict


def _remap_state_dict(state_dict: dict) -> dict:
    """Remap keys from the training checkpoint to our model structure.

    The training code wraps the model in AudioFakeDetector with the same attribute
    names we use, but may have extra keys from training-only components (con_gan,
    loss_func, etc.) that we skip.
    """
    new_sd = {}
    # Keys we need for inference
    inference_prefixes = (
        "encoder_f.",
        "block_spe.",
        "block_sha.",
        "head_spe.",
        "head_sha.",
    )
    # Also keep encoder_c if present (we don't use it at inference but load for completeness)
    keep_prefixes = inference_prefixes + ("encoder_c.",)

    for key, value in state_dict.items():
        # Strip DataParallel 'module.' prefix if present
        clean_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        if any(clean_key.startswith(p) for p in keep_prefixes):
            new_sd[clean_key] = value

    return new_sd


class VoiceGenDetector(BaseDetector):
    """AI-Synthesized Voice Generalization detector (AAAI 2025).

    Dual RawNet2 encoder with domain-agnostic feature disentanglement.
    Trained on LibriSeVoc with SAM optimization for cross-domain generalization.
    """

    MODEL_NAME = "VoiceGen (Dual-RawNet2)"

    def __init__(self):
        self.model_name = self.MODEL_NAME
        self.model = None

    def load(self) -> None:
        weights_path = _download_weights()
        self.model = AudioFakeDetector(device="cpu")

        state_dict = _load_checkpoint(weights_path, "cpu")
        remapped = _remap_state_dict(state_dict)

        # Load with strict=False to skip training-only components
        missing, unexpected = self.model.load_state_dict(remapped, strict=False)
        if missing:
            # Filter out encoder_c keys (not used at inference) from missing report
            critical_missing = [k for k in missing if not k.startswith("encoder_c.")]
            if critical_missing:
                raise RuntimeError(f"Missing critical keys: {critical_missing}")

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
            logits = self.model(tensor)
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
