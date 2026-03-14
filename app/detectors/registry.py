import logging

from app.detectors.base import BaseDetector
from app.detectors.hf_image import HFImageDetector
from app.detectors.hf_video import HFVideoDetector

logger = logging.getLogger(__name__)

DETECTOR_REGISTRY: dict[str, type[BaseDetector]] = {
    "hf_image": HFImageDetector,
    "hf_video": HFVideoDetector,
}

MEDIA_TYPE_TO_DETECTOR: dict[str, str] = {
    "image": "hf_image",
    "video": "hf_video",
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


def get_detector_for_media_type(media_type: str) -> BaseDetector:
    """Return the loaded detector for a given media type."""
    detector_name = MEDIA_TYPE_TO_DETECTOR.get(media_type)
    if detector_name is None:
        raise ValueError(f"No detector registered for media type: {media_type}")
    detector = _loaded_detectors.get(detector_name)
    if detector is None:
        raise RuntimeError(f"Detector '{detector_name}' not loaded. Call load_all_detectors() first.")
    return detector
