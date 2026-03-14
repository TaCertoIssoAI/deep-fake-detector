import os
import tempfile
import time
from collections import defaultdict

import cv2
from PIL import Image

from app.config import settings
from app.detectors.base import BaseDetector
from app.detectors.hf_image import HFImageDetector
from app.schemas import DetectionResult

NUM_FRAMES = 20


def _extract_frame_images(video_path: str, num_frames: int = NUM_FRAMES) -> list[str]:
    """Extract evenly spaced frames from video and save as temp images."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // num_frames)

    paths = []
    count = 0
    success = True

    while success and len(paths) < num_frames:
        success, frame = cap.read()
        if success and count % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(tmp.name)
            tmp.close()
            paths.append(tmp.name)
        count += 1

    cap.release()
    return paths


class FrameSamplerDetector(BaseDetector):
    """Samples frames from video and runs the image detector on each, then averages."""

    def __init__(self):
        self.image_detector = HFImageDetector()
        self.model_name = f"{settings.IMAGE_MODEL} (frame-sampling)"

    def load(self) -> None:
        self.image_detector.load()

    def detect(self, file_path: str) -> list[DetectionResult]:
        start = time.perf_counter()

        frame_paths = _extract_frame_images(file_path)
        if not frame_paths:
            raise RuntimeError("Could not extract any frames from video")

        # Run image detector on each frame and accumulate scores per label
        score_sums: dict[str, float] = defaultdict(float)
        num_frames = len(frame_paths)

        try:
            for path in frame_paths:
                results = self.image_detector.detect(path)
                for r in results:
                    score_sums[r.label] += r.score
        finally:
            for path in frame_paths:
                os.unlink(path)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Average scores across frames
        return [
            DetectionResult(
                label=label,
                score=round(total / num_frames, 4),
                model_used=self.model_name,
                media_type="video",
                processing_time_ms=round(elapsed_ms, 2),
            )
            for label, total in sorted(score_sums.items(), key=lambda x: -x[1])
        ]

    def supported_media_types(self) -> list[str]:
        return ["video"]
