import logging

from app.detectors.base import BaseDetector
from app.detectors.frame_sampler import FrameSamplerDetector
from app.detectors.hf_image import HFImageDetector
from app.detectors.hf_video import HFVideoDetector

logger = logging.getLogger(__name__)

DETECTOR_REGISTRY: dict[str, type[BaseDetector]] = {
    "hf_image": HFImageDetector,
    "hf_video": HFVideoDetector,
    "frame_sampler": FrameSamplerDetector,
}

MEDIA_TYPE_TO_DETECTORS: dict[str, list[str]] = {
    "image": ["hf_image"],
    "video": ["hf_video", "frame_sampler"],
}

_loaded_detectors: dict[str, BaseDetector] = {}


def load_all_detectors() -> None:
    """Instantiate and load all registered detectors. Skips models that fail to load."""
    for name, cls in DETECTOR_REGISTRY.items():
        try:
            detector = cls()
            detector.load()
            _loaded_detectors[name] = detector
            logger.info(f"Loaded detector: {name}")
        except Exception as e:
            logger.warning(f"Failed to load detector '{name}': {e}")


def get_detectors_for_media_type(media_type: str) -> list[BaseDetector]:
    """Return all loaded detectors for a given media type."""
    detector_names = MEDIA_TYPE_TO_DETECTORS.get(media_type)
    if detector_names is None:
        raise ValueError(f"No detector registered for media type: {media_type}")
    detectors = []
    for name in detector_names:
        detector = _loaded_detectors.get(name)
        if detector is not None:
            detectors.append(detector)
    if not detectors:
        raise RuntimeError(f"No detectors loaded for media type: {media_type}")
    return detectors
