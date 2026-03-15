import time
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.detectors.base import BaseDetector
from app.detectors.hf_image import HFImageDetector
from app.schemas import DetectionResult

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


class FrameSamplerDetector(BaseDetector):
    """Samples video frames and runs the ViT image detector on each, averaging scores."""

    MODEL_NAME = f"frame_sampler({settings.IMAGE_MODEL})"

    def __init__(self):
        self.model_name = self.MODEL_NAME
        self.image_detector = HFImageDetector()

    def load(self) -> None:
        self.image_detector.load()

    def detect(self, file_path: str) -> list[DetectionResult]:
        start = time.perf_counter()

        frames = _extract_frames(file_path)
        if not frames:
            raise RuntimeError("Could not extract any frames from video")

        label_map = {"deepfake": "fake", "realism": "real"}
        scores = defaultdict(list)
        for frame in frames:
            results = self.image_detector.pipe(frame)
            for r in results:
                label = r["label"].lower()
                label = label_map.get(label, label)
                scores[label].append(r["score"])

        elapsed_ms = (time.perf_counter() - start) * 1000

        return [
            DetectionResult(
                label=label,
                score=round(sum(s) / len(s), 4),
                model_used=self.model_name,
                media_type="video",
                processing_time_ms=round(elapsed_ms, 2),
            )
            for label, s in scores.items()
        ]

    def supported_media_types(self) -> list[str]:
        return ["video"]
