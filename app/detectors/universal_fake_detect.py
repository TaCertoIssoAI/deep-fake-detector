import os
import time
import urllib.request

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

    def detect(self, file_path: str) -> list[DetectionResult]:
        start = time.perf_counter()

        image = Image.open(file_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")["pixel_values"]

        with torch.no_grad():
            vision_out = self.backbone.vision_model(inputs).pooler_output
            features = self.backbone.visual_projection(vision_out)
            logit = self.head(features)
            prob = torch.sigmoid(logit).item()

        fake_score = round(prob, 4)
        real_score = round(1.0 - prob, 4)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return [
            DetectionResult(
                label="fake",
                score=fake_score,
                model_used=self.model_name,
                media_type="image",
                processing_time_ms=round(elapsed_ms, 2),
            ),
            DetectionResult(
                label="real",
                score=real_score,
                model_used=self.model_name,
                media_type="image",
                processing_time_ms=round(elapsed_ms, 2),
            ),
        ]

    def supported_media_types(self) -> list[str]:
        return ["image"]
