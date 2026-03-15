import os
import tempfile
import time
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from app.config import settings
from app.detectors.base import BaseDetector
from app.schemas import DetectionResult

NUM_FRAMES = 20
WEIGHTS_URL = "https://github.com/TRahulsingh/DeepfakeDetector/raw/main/models/best_model-v3.pt"
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "best_model-v3.pt")

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _build_model() -> nn.Module:
    """Build EfficientNet-B0 with custom 2-class classifier head."""
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 2),
    )
    return model


def _download_weights() -> str:
    """Download weights if not cached locally."""
    if os.path.isfile(WEIGHTS_PATH):
        return WEIGHTS_PATH
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    print(f"Downloading DeepfakeDetector weights to {WEIGHTS_PATH}...")
    urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
    return WEIGHTS_PATH


def _extract_frames(video_path: str, num_frames: int = NUM_FRAMES) -> list[Image.Image]:
    """Extract evenly spaced frames from video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indexes = set(np.linspace(0, total - 1, num=num_frames, dtype=int))

    frames = []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indexes:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


class EfficientNetVideoDetector(BaseDetector):
    """EfficientNet-B0 deepfake detector (TRahulsingh/DeepfakeDetector) on sampled video frames."""

    MODEL_NAME = "TRahulsingh/DeepfakeDetector"

    def __init__(self):
        self.model_name = self.MODEL_NAME
        self.model = None

    def load(self) -> None:
        weights_path = _download_weights()
        self.model = _build_model()
        self.model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        self.model.eval()

    def detect(self, file_path: str) -> list[DetectionResult]:
        start = time.perf_counter()

        frames = _extract_frames(file_path)
        if not frames:
            raise RuntimeError("Could not extract any frames from video")

        all_probs = []
        with torch.no_grad():
            for frame in frames:
                tensor = _transform(frame).unsqueeze(0)
                output = self.model(tensor)
                prob = torch.softmax(output, dim=1)
                all_probs.append(prob)

        avg_prob = torch.mean(torch.stack(all_probs), dim=0)[0]
        real_score = float(avg_prob[0])
        fake_score = float(avg_prob[1])

        elapsed_ms = (time.perf_counter() - start) * 1000

        return [
            DetectionResult(
                label="fake",
                score=round(fake_score, 4),
                model_used=self.model_name,
                media_type="video",
                processing_time_ms=round(elapsed_ms, 2),
            ),
            DetectionResult(
                label="real",
                score=round(real_score, 4),
                model_used=self.model_name,
                media_type="video",
                processing_time_ms=round(elapsed_ms, 2),
            ),
        ]

    def supported_media_types(self) -> list[str]:
        return ["video"]
