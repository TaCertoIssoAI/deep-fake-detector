from pydantic import BaseModel


class DetectionResult(BaseModel):
    label: str
    score: float
    model_used: str
    media_type: str
    processing_time_ms: float


class DetectResponse(BaseModel):
    results: list[DetectionResult]
