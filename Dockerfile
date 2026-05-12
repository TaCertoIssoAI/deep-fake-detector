FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*


# Download VoiceGen checkpoint at build time
RUN mkdir -p models/voice_gen && \
    curl -L -o models/voice_gen/AudioDeepFakeDetection-ckpt-28.pth \
    "https://drive.google.com/uc?export=download&id=1J8defEI-JJmJVMq4iVlTh825UnyZugQo&confirm=t"


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download HuggingFace ViT model at build time
RUN python -c "from transformers import pipeline; pipeline('image-classification', model='prithivMLmods/Deep-Fake-Detector-v2-Model')"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
