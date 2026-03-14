import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from huggingface_hub import hf_hub_download
from PIL import Image

from app.config import settings
from app.detectors.base import BaseDetector
from app.schemas import DetectionResult

IM_SIZE = 112
SEQUENCE_LENGTH = 20
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class _ResNextModel(nn.Module):
    """ResNext50 + LSTM model matching the actual checkpoint layer names."""

    def __init__(self, num_classes=2, latent_dim=2048, hidden_dim=2048):
        super().__init__()
        resnext = models.resnext50_32x4d(weights="DEFAULT")
        self.model = nn.Sequential(*list(resnext.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers=1, bias=False)
        self.linear1 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        features = self.model(x)
        pooled = self.avgpool(features)
        pooled = pooled.view(batch_size, seq_length, -1)
        lstm_out, _ = self.lstm(pooled, None)
        final = lstm_out[:, -1, :]
        logits = self.linear1(final)
        return logits


def _preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    img = Image.fromarray(frame).resize((IM_SIZE, IM_SIZE))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - np.array(MEAN)) / np.array(STD)
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return torch.tensor(arr, dtype=torch.float32)


def _extract_frames(video_path: str) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // SEQUENCE_LENGTH)

    frames = []
    count = 0
    success = True

    while success and len(frames) < SEQUENCE_LENGTH:
        success, image = cap.read()
        if success and count % interval == 0:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Use OpenCV's Haar cascade for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                x, y, w_f, h_f = faces[0]
                padding = 40
                h, w = image.shape[:2]
                y1 = max(0, y - padding)
                y2 = min(h, y + h_f + padding)
                x1 = max(0, x - padding)
                x2 = min(w, x + w_f + padding)
                image = image[y1:y2, x1:x2]
            frames.append(_preprocess_frame(image))
        count += 1

    cap.release()

    # Pad if not enough frames
    while len(frames) < SEQUENCE_LENGTH:
        frames.append(frames[-1] if frames else torch.zeros(3, IM_SIZE, IM_SIZE))

    return torch.stack(frames).unsqueeze(0)  # (1, seq_len, 3, H, W)


class HFVideoDetector(BaseDetector):
    """Video deepfake detector using ResNext50 + LSTM (Naman712/Deep-fake-detection)."""

    def __init__(self):
        self.model_name = settings.VIDEO_MODEL
        self.model = None

    def load(self) -> None:
        weights_path = hf_hub_download(
            self.model_name,
            "model_87_acc_20_frames_final_data.pt",
            token=settings.HF_TOKEN,
        )
        self.model = _ResNextModel()
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def detect(self, file_path: str) -> list[DetectionResult]:
        start = time.perf_counter()

        frames = _extract_frames(file_path)
        with torch.no_grad():
            logits = self.model(frames)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Index 0 = fake, index 1 = real
        labels = ["fake", "real"]
        return [
            DetectionResult(
                label=labels[i],
                score=float(probs[i]),
                model_used=self.model_name,
                media_type="video",
                processing_time_ms=round(elapsed_ms, 2),
            )
            for i in range(len(labels))
        ]

    def supported_media_types(self) -> list[str]:
        return ["video"]
