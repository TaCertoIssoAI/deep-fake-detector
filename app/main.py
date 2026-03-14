import mimetypes
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile

from app.detectors.registry import get_detector_for_media_type, load_all_detectors
from app.schemas import DetectResponse

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def detect_media_type(filename: str) -> str:
    """Determine media type from file extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    mime, _ = mimetypes.guess_type(filename)
    if mime:
        if mime.startswith("image/"):
            return "image"
        if mime.startswith("video/"):
            return "video"
    raise ValueError(f"Unsupported file type: {ext or filename}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_detectors()
    yield


app = FastAPI(title="Deep-Fake Detector", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    try:
        media_type = detect_media_type(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    ext = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        detector = get_detector_for_media_type(media_type)
        results = detector.detect(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    return DetectResponse(results=results)
