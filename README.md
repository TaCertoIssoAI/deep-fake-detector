# Deep-Fake Detection Service

A Python FastAPI server with a pluggable detector abstraction layer for deepfake detection. Supports multiple models running in parallel across image and video media types.

## Architecture

```
POST /detect  (multipart file upload)
     │
     ├─ detect media type (image/video)
     │
     ├─ route to all registered detectors for that type
     │
     └─ return combined results from all detectors
```

Every detector implements `BaseDetector` (load, detect, supported_media_types) and is registered in `app/detectors/registry.py`. New models plug in by implementing the interface and adding one line to the registry.

## Current Models

| Model | Media | Architecture | Source |
|-------|-------|-------------|--------|
| **ViT Deep-Fake Detector v2** | Image | ViT-base (HuggingFace pipeline) | [prithivMLmods/Deep-Fake-Detector-v2-Model](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model) |
| **Frame Sampler** | Video | Samples 20 video frames, runs ViT image detector on each, averages scores | Uses the image model above |
| **GenD CLIP L/14** | Video | CLIP ViT-L/14 vision encoder + linear probe, 20-frame averaging | [yermandy/GenD_CLIP_L_14](https://huggingface.co/yermandy/GenD_CLIP_L_14) |
| **D3** | Video | Dual-branch CLIP ViT-L/14 (shuffled + original patches) + attention head, 20-frame averaging | [BigAandSmallq/D3](https://github.com/BigAandSmallq/D3) |

## Quick Start

### Requirements

- Python 3.11+
- ~1.5GB disk for model weights (downloaded on first run)

### Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### CLI

Starts the server, sends a file, prints results, and shuts down:

```bash
python cli.py path/to/video.mp4
python cli.py path/to/image.jpg
```

### API

Start the server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Detect a file:

```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@video.mp4"
```

Health check:

```bash
curl http://localhost:8000/health
```

### Benchmark

Evaluates all models against labeled test files in `media/` (filenames prefixed with `fake-` or `real-`):

```bash
python benchmark.py
```

Outputs per-model accuracy and writes detailed results to `benchmark_results.csv`.

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace token (for gated models) |
| `DEVICE` | `cpu` | `cpu` or `cuda` |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

## Project Structure

```
deep-fake-detection/
├── app/
│   ├── main.py              # FastAPI app, /detect and /health endpoints
│   ├── config.py             # Settings from env vars
│   ├── schemas.py            # Pydantic request/response models
│   └── detectors/
│       ├── base.py           # BaseDetector abstract class
│       ├── registry.py       # Detector registry and media type routing
│       ├── hf_image.py       # ViT image detector (HuggingFace)
│       ├── frame_sampler.py  # Video → frame sampling → image detector
│       ├── gend_clip.py      # GenD CLIP ViT-L/14 video detector
│       └── d3_clip.py        # D3 dual-branch CLIP video detector
├── cli.py                    # CLI tool
├── benchmark.py              # Model evaluation script
├── media/                    # Test files (fake-*.mp4, real-*.mp4)
├── requirements.txt
└── Dockerfile
```

## Models to Evaluate in the Future

### TrueMedia ML Models

Collection of deepfake detectors from TrueMedia.org covering image, video, and audio:

- **DistilDIRE** (image) — distilled diffusion-based detector, 3.2x faster than DIRE, handles GAN and diffusion outputs
- **UniversalFakeDetectV2** (image) — CLIP-ViT feature spaces with nearest-neighbor/linear probing
- **GenConViT** (video) — ConvNeXt + Swin Transformer hybrid with CNN autoencoder and VAE
- **StyleFlow** (video) — style-latent flow anomaly detection with StyleGRU + supervised contrastive learning
- **FTCN** (video) — temporal convolution network for long-term coherence detection
- **Transcript Based Detector** (audio) — speech recognition + LLM analysis for factual coherence

Repository: https://github.com/truemediaorg/ml-models

> **Note:** TrueMedia model weights require a formal request to `aerin@truemedia.org` with affiliation and intended use.
