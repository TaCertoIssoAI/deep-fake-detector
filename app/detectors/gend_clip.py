import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import CLIPModel, CLIPProcessor

from app.detectors.base import BaseDetector
from app.schemas import DetectionResult

NUM_FRAMES = 20
REPO_ID = "yermandy/GenD_CLIP_L_14"
CLIP_MODEL = "openai/clip-vit-large-patch14"


class _GenDModel(nn.Module):
    """CLIP ViT-L/14 vision encoder + normalized linear probe for deepfake detection."""

    def __init__(self):
        super().__init__()
        clip = CLIPModel.from_pretrained(CLIP_MODEL)
        self.vision_model = clip.vision_model
        features_dim = self.vision_model.config.hidden_size
        self.linear = nn.Linear(features_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.vision_model(x).pooler_output
        features = F.normalize(features, p=2, dim=1)
        return self.linear(features)


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


class GenDClipDetector(BaseDetector):
    """GenD CLIP ViT-L/14 deepfake detector applied to sampled video frames."""

    MODEL_NAME = "yermandy/GenD_CLIP_L_14"

    def __init__(self):
        self.model_name = self.MODEL_NAME
        self.model = None
        self.processor = None

    def load(self) -> None:
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.model = _GenDModel()

        # Load GenD head weights from HuggingFace
        weights_path = hf_hub_download(REPO_ID, "model.safetensors")
        state_dict = load_file(weights_path)

        # Map keys: safetensors has "model.linear.weight" etc, our model has "linear.weight"
        # Also has feature_extractor weights that match vision_model
        mapped = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                # head weights: model.linear.weight -> linear.weight
                mapped[key[len("model."):]] = value
            elif key.startswith("feature_extractor.vision_model."):
                # CLIP vision weights: feature_extractor.vision_model.* -> vision_model.*
                mapped[key[len("feature_extractor."):]] = value
            elif key.startswith("feature_extractor.visual_projection."):
                # skip visual_projection (not used for classification)
                continue

        self.model.load_state_dict(mapped, strict=False)
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
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs)

        avg_probs = torch.mean(torch.stack(all_probs), dim=0)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        # GenD: class 0 = real, class 1 = fake
        real_score = float(avg_probs[0])
        fake_score = float(avg_probs[1])

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
