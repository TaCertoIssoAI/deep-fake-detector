import os
import time
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.detectors.base import BaseDetector
from app.schemas import DetectionResult

NUM_FRAMES = 20
CLIP_MODEL = "openai/clip-vit-large-patch14"
PATCH_SIZE = 14

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "d3")
WEIGHTS_URL = "https://raw.githubusercontent.com/BigAandSmallq/D3/main/ckpt/classifier.pth"
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "classifier.pth")


class _AttentionHead(nn.Module):
    """TransformerAttention head from D3: self-attention over shuffled + original features."""

    def __init__(self, input_dim=1024, num_tokens=2, num_classes=1):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim * num_tokens, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))
        attention = self.softmax(attention)
        output = torch.matmul(attention, v)
        output = output.view(output.shape[0], -1)
        return self.fc(output)


class _D3Model(nn.Module):
    """D3: Discrepancy-Guided Deepfake Detection.

    Processes each image twice through frozen CLIP ViT-L/14:
    1. With spatially shuffled patches (destroys layout, preserves local artifacts)
    2. Original image
    Then compares features via attention to detect deepfake discrepancies.
    """

    def __init__(self):
        super().__init__()
        clip = CLIPModel.from_pretrained(CLIP_MODEL)
        self.vision_model = clip.vision_model
        # Freeze backbone
        for param in self.vision_model.parameters():
            param.requires_grad = False
        self.attention_head = _AttentionHead(
            input_dim=self.vision_model.config.hidden_size,  # 1024
            num_tokens=2,
            num_classes=1,
        )

    def _shuffle_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Spatially shuffle non-overlapping patches of the input image."""
        B, C, H, W = x.size()
        patches = F.unfold(x, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        shuffled = patches[:, :, torch.randperm(patches.size(-1))]
        return F.fold(shuffled, output_size=(H, W), kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            shuffled_features = self.vision_model(self._shuffle_patches(x)).pooler_output
            original_features = self.vision_model(x).pooler_output

        # Stack: [B, 2, 1024]
        features = torch.stack([shuffled_features, original_features], dim=1)
        return self.attention_head(features)


def _download_weights() -> str:
    if os.path.isfile(WEIGHTS_PATH):
        return WEIGHTS_PATH
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    print(f"Downloading D3 attention head weights to {WEIGHTS_PATH}...")
    urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
    return WEIGHTS_PATH


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


class D3ClipDetector(BaseDetector):
    """D3 Discrepancy-Guided Deepfake Detection on sampled video frames."""

    MODEL_NAME = "D3 (CLIP ViT-L/14)"

    def __init__(self):
        self.model_name = self.MODEL_NAME
        self.model = None
        self.processor = None

    def load(self) -> None:
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.model = _D3Model()

        weights_path = _download_weights()
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.attention_head.load_state_dict(state_dict)
        self.model.eval()

    def detect(self, file_path: str) -> list[DetectionResult]:
        start = time.perf_counter()

        frames = _extract_frames(file_path)
        if not frames:
            raise RuntimeError("Could not extract any frames from video")

        all_probs = []
        with torch.no_grad():
            for frame in frames:
                inputs = self.processor(images=frame, return_tensors="pt")["pixel_values"]
                logits = self.model(inputs)
                prob = torch.sigmoid(logits)
                all_probs.append(prob)

        avg_prob = torch.mean(torch.stack(all_probs), dim=0)[0]
        fake_score = float(avg_prob[0])
        real_score = 1.0 - fake_score

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
