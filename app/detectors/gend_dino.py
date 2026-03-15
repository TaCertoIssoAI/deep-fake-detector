import time

import cv2
import numpy as np
import timm
import timm.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file

from app.detectors.base import BaseDetector
from app.schemas import DetectionResult

NUM_FRAMES = 20
REPO_ID = "yermandy/GenD_PE_L"
PE_MODEL = "vit_pe_core_large_patch14_336"


class _GenDPEModel(nn.Module):
    """Perception Encoder ViT-L + L2-normalized linear probe for deepfake detection."""

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(PE_MODEL, pretrained=False, dynamic_img_size=True)
        self.backbone.head = nn.Identity()
        self.linear = nn.Linear(self.backbone.num_features, 2)  # 1024 -> 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
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


class GenDPEDetector(BaseDetector):
    """GenD Perception Encoder ViT-L deepfake detector with frame sampling.

    Apple Perception Encoder (EVA ViT-L, 300M params) + L2-normalized linear probe,
    from the same GenD framework as our CLIP detector but with a different
    vision backbone for feature diversity. (WACV 2026)
    """

    MODEL_NAME = "yermandy/GenD_PE_L"

    def __init__(self):
        self.model_name = self.MODEL_NAME
        self.model = None
        self.transform = None

    def load(self) -> None:
        self.model = _GenDPEModel()

        # Build preprocessing transform from timm model config
        data_config = timm.data.resolve_model_data_config(self.model.backbone)
        data_config["input_size"] = (3, 224, 224)
        self.transform = timm.data.create_transform(**data_config, is_training=False)

        # Load GenD weights (backbone + linear head) from HuggingFace
        weights_path = hf_hub_download(REPO_ID, "model.safetensors")
        state_dict = load_file(weights_path)

        # Remap keys: feature_extractor.backbone.* -> backbone.*, model.linear.* -> linear.*
        mapped = {}
        for key, value in state_dict.items():
            if key.startswith("model.linear."):
                mapped[key[len("model."):]] = value
            elif key.startswith("feature_extractor.backbone."):
                mapped[key[len("feature_extractor."):]] = value

        self.model.load_state_dict(mapped, strict=False)
        self.model.eval()

    def _predict_frame(self, image: Image.Image) -> torch.Tensor:
        """Run inference on a single frame, returns softmax probabilities."""
        inputs = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(inputs)
            return torch.softmax(logits, dim=-1)

    def detect(self, file_path: str) -> list[DetectionResult]:
        start = time.perf_counter()

        frames = _extract_frames(file_path)
        if frames:
            # Video: sample frames and average
            all_probs = [self._predict_frame(frame) for frame in frames]
            avg_probs = torch.mean(torch.stack(all_probs), dim=0)[0]
            media_type = "video"
        else:
            # Image
            image = Image.open(file_path).convert("RGB")
            avg_probs = self._predict_frame(image)[0]
            media_type = "image"

        elapsed_ms = (time.perf_counter() - start) * 1000

        # GenD: class 0 = real, class 1 = fake
        real_score = float(avg_probs[0])
        fake_score = float(avg_probs[1])

        return [
            DetectionResult(
                label="fake",
                score=round(fake_score, 4),
                model_used=self.model_name,
                media_type=media_type,
                processing_time_ms=round(elapsed_ms, 2),
            ),
            DetectionResult(
                label="real",
                score=round(real_score, 4),
                model_used=self.model_name,
                media_type=media_type,
                processing_time_ms=round(elapsed_ms, 2),
            ),
        ]

    def supported_media_types(self) -> list[str]:
        return ["image", "video"]
