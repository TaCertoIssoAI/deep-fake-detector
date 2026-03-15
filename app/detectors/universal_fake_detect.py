import os
import time
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.detectors.base import BaseDetector
from app.schemas import DetectionResult

CLIP_MODEL = "openai/clip-vit-large-patch14"
FEATURE_DIM = 768

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "universal_fake_detect")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "fc_weights.pth")
WEIGHTS_URL = "https://github.com/WisconsinAIVision/UniversalFakeDetect/raw/main/pretrained_weights/fc_weights.pth"

NUM_FRAMES = 20


def _extract_frames(video_path: str, num_frames: int = NUM_FRAMES) -> list[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

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


class _UFDHead(nn.Module):
    """Single linear layer classifier from UniversalFakeDetect (CVPR 2023)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(FEATURE_DIM, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _download_weights() -> str:
    if os.path.isfile(WEIGHTS_PATH):
        return WEIGHTS_PATH
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    print(f"Downloading UniversalFakeDetect weights to {WEIGHTS_PATH}...")
    urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
    return WEIGHTS_PATH


class UniversalFakeDetectDetector(BaseDetector):
    """UniversalFakeDetect (CVPR 2023) — CLIP ViT-L/14 + linear probe.

    Frozen CLIP features with a single linear layer trained on ProGAN,
    generalizes to unseen generative models (diffusion, autoregressive).
    """

    MODEL_NAME = "UniversalFakeDetect (CLIP ViT-L/14)"

    def __init__(self):
        self.model_name = self.MODEL_NAME
        self.head = None
        self.backbone = None
        self.processor = None

    def load(self) -> None:
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.backbone = CLIPModel.from_pretrained(CLIP_MODEL)
        self.backbone.eval()

        self.head = _UFDHead()
        weights_path = _download_weights()
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Checkpoint has bare "weight"/"bias" keys; remap to "fc.weight"/"fc.bias"
        mapped = {f"fc.{k}": v for k, v in state_dict.items()}
        self.head.load_state_dict(mapped)
        self.head.eval()

    def _predict_image(self, image: Image.Image) -> float:
        """Run inference on a single image, returns fake probability."""
        inputs = self.processor(images=image, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            vision_out = self.backbone.vision_model(inputs).pooler_output
            features = self.backbone.visual_projection(vision_out)
            logit = self.head(features)
            return torch.sigmoid(logit).item()

    def detect(self, file_path: str) -> list[DetectionResult]:
        start = time.perf_counter()

        frames = _extract_frames(file_path)
        if frames:
            # Video: sample frames and average
            probs = [self._predict_image(frame) for frame in frames]
            prob = sum(probs) / len(probs)
            media_type = "video"
        else:
            # Image
            image = Image.open(file_path).convert("RGB")
            prob = self._predict_image(image)
            media_type = "image"

        fake_score = round(prob, 4)
        real_score = round(1.0 - prob, 4)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return [
            DetectionResult(
                label="fake",
                score=fake_score,
                model_used=self.model_name,
                media_type=media_type,
                processing_time_ms=round(elapsed_ms, 2),
            ),
            DetectionResult(
                label="real",
                score=real_score,
                model_used=self.model_name,
                media_type=media_type,
                processing_time_ms=round(elapsed_ms, 2),
            ),
        ]

    def supported_media_types(self) -> list[str]:
        return ["image", "video"]
