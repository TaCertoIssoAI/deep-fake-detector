import os
import re
import time
import urllib.request

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.detectors.base import BaseDetector
from app.detectors.genconvit.genconvit import GenConViT
from app.schemas import DetectionResult

NUM_FRAMES = 15
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "genconvit")
ED_WEIGHTS_URL = "https://huggingface.co/Deressa/GenConViT/resolve/main/genconvit_ed_inference.pth"
VAE_WEIGHTS_URL = "https://huggingface.co/Deressa/GenConViT/resolve/main/genconvit_vae_inference.pth"
ED_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "genconvit_ed_inference.pth")
VAE_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "genconvit_vae_inference.pth")

GENCONVIT_CONFIG = {
    "backbone": "convnext_tiny",
    "embedder": "swin_tiny_patch4_window7_224",
    "latent_dims": 12544,
    "img_size": 224,
    "num_classes": 2,
}

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _download_weights(url: str, path: str) -> str:
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading GenConViT weights to {path}...")
    urllib.request.urlretrieve(url, path)
    return path


def _detect_face(frame: np.ndarray) -> np.ndarray | None:
    """Detect and crop the largest face from a frame using Haar cascade."""
    cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    # Pick largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return frame[y:y+h, x:x+w]


def _extract_face_frames(video_path: str, num_frames: int = NUM_FRAMES) -> list[Image.Image]:
    """Extract frames from video with face detection."""
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
            face = _detect_face(frame)
            if face is not None:
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(face_rgb))
            else:
                # Fall back to full frame if no face detected
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


class GenConViTDetector(BaseDetector):
    """GenConViT deepfake detector using both ED and VAE networks."""

    MODEL_NAME = "GenConViT (ED+VAE)"

    def __init__(self):
        self.model_name = self.MODEL_NAME
        self.model = None

    @staticmethod
    def _remap_state_dict(state_dict: dict) -> dict:
        """Remap keys from old timm format to current timm format.

        Old timm placed downsample at end of layer N, new timm places it
        at start of layer N+1. Also head.weight → head.fc.weight.
        """
        remapped = {}
        for key, value in state_dict.items():
            new_key = key
            # Skip buffer keys (not part of state_dict in new timm)
            if "relative_position_index" in key or "attn_mask" in key:
                continue
            # Remap downsample: layers.N.downsample → layers.(N+1).downsample
            match = re.match(r"(.*layers\.)(\d+)(\.downsample\..*)", new_key)
            if match:
                layer_idx = int(match.group(2))
                new_key = f"{match.group(1)}{layer_idx + 1}{match.group(3)}"
            # Remap head: head.weight/bias → head.fc.weight/bias
            new_key = new_key.replace(".head.weight", ".head.fc.weight")
            new_key = new_key.replace(".head.bias", ".head.fc.bias")
            remapped[new_key] = value
        return remapped

    def load(self) -> None:
        # Download weights
        ed_path = _download_weights(ED_WEIGHTS_URL, ED_WEIGHTS_PATH)
        vae_path = _download_weights(VAE_WEIGHTS_URL, VAE_WEIGHTS_PATH)

        # Build combined model
        self.model = GenConViT(GENCONVIT_CONFIG)

        # Load ED weights (remap old timm keys)
        ed_state = torch.load(ed_path, map_location="cpu", weights_only=True)
        ed_state = self._remap_state_dict(ed_state)
        self.model.ed.load_state_dict(ed_state, strict=False)

        # Load VAE weights (remap old timm keys)
        vae_state = torch.load(vae_path, map_location="cpu", weights_only=True)
        vae_state = self._remap_state_dict(vae_state)
        self.model.vae.load_state_dict(vae_state, strict=False)

        self.model.eval()

    def detect(self, file_path: str) -> list[DetectionResult]:
        start = time.perf_counter()

        frames = _extract_face_frames(file_path)
        if not frames:
            raise RuntimeError("Could not extract any frames from video")

        all_preds = []
        with torch.no_grad():
            for frame in frames:
                tensor = _transform(frame).unsqueeze(0)
                output = self.model(tensor)
                # output shape: (2, 2) — row 0 = ED [class0, class1], row 1 = VAE [class0, class1]
                # Average ED and VAE logits, then softmax for probabilities
                avg_logits = torch.mean(output, dim=0)  # shape (2,)
                probs = torch.softmax(avg_logits, dim=0)
                all_preds.append(probs)

        # Average probabilities across frames: shape (2,)
        combined = torch.mean(torch.stack(all_preds), dim=0)

        # Original repo mapping: argmax ^ 1
        # argmax=0 → FAKE, argmax=1 → REAL
        # So class 0 = fake, class 1 = real
        fake_score = float(combined[0])
        real_score = float(combined[1])

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
