import logging

from app.detectors.base import BaseDetector
from app.detectors.efficientnet_video import EfficientNetVideoDetector
from app.detectors.frame_sampler import FrameSamplerDetector
from app.detectors.gend_clip import GenDClipDetector
from app.detectors.genconvit.detector import GenConViTDetector
from app.detectors.hf_image import HFImageDetector

logger = logging.getLogger(__name__)

DETECTOR_REGISTRY: dict[str, type[BaseDetector]] = {
    "hf_image": HFImageDetector,
    "frame_sampler": FrameSamplerDetector,
    "efficientnet_video": EfficientNetVideoDetector,
    "genconvit": GenConViTDetector,
    "gend_clip": GenDClipDetector,
}

MEDIA_TYPE_TO_DETECTORS: dict[str, list[str]] = {
    "image": ["hf_image"],
    "video": ["frame_sampler", "efficientnet_video", "genconvit", "gend_clip"],
}

_loaded_detectors: dict[str, BaseDetector] = {}
_loading_complete = False


def load_all_detectors() -> None:
    """Instantiate and load all registered detectors. Skips models that fail to load."""
    global _loading_complete
    for name, cls in DETECTOR_REGISTRY.items():
        try:
            detector = cls()
            detector.load()
            _loaded_detectors[name] = detector
            logger.info(f"Loaded detector: {name}")
        except Exception as e:
            logger.warning(f"Failed to load detector '{name}': {e}")
    _loading_complete = True


def is_ready() -> bool:
    return _loading_complete


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
