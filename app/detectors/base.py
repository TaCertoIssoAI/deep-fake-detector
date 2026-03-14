from abc import ABC, abstractmethod

from app.schemas import DetectionResult


class BaseDetector(ABC):
    """Base class for all deepfake detectors."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory."""
        ...

    @abstractmethod
    def detect(self, file_path: str) -> list[DetectionResult]:
        """Run detection on a file. Returns structured results."""
        ...

    @abstractmethod
    def supported_media_types(self) -> list[str]:
        """Return list of supported types: ['image'], ['video'], or both."""
        ...
