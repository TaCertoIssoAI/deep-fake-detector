import time

from transformers import pipeline

from app.config import settings
from app.detectors.base import BaseDetector
from app.schemas import DetectionResult


class HFImageDetector(BaseDetector):
    """HuggingFace image deepfake detector using ViT-base."""

    def __init__(self):
        self.model_name = settings.IMAGE_MODEL
        self.pipe = None

    def load(self) -> None:
        self.pipe = pipeline(
            "image-classification",
            model=self.model_name,
            device=settings.DEVICE,
            token=settings.HF_TOKEN,
        )

    LABEL_MAP = {"deepfake": "fake", "realism": "real"}

    def detect(self, file_path: str) -> list[DetectionResult]:
        start = time.perf_counter()
        outputs = self.pipe(file_path)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return [
            DetectionResult(
                label=self.LABEL_MAP.get(out["label"].lower(), out["label"].lower()),
                score=out["score"],
                model_used=self.model_name,
                media_type="image",
                processing_time_ms=round(elapsed_ms, 2),
            )
            for out in outputs
        ]

    def supported_media_types(self) -> list[str]:
        return ["image"]
