import os


class Settings:
    IMAGE_MODEL: str = os.getenv(
        "IMAGE_MODEL", "prithivMLmods/Deep-Fake-Detector-v2-Model"
    )
    VIDEO_MODEL: str = os.getenv(
        "VIDEO_MODEL", "Naman712/Deep-fake-detection"
    )
    EFFICIENTNET_MODEL: str = os.getenv(
        "EFFICIENTNET_MODEL", "Wvolf/ViT_Deepfake_Detection"
    )
    HF_TOKEN: str | None = os.getenv("HF_TOKEN")
    DEVICE: str = os.getenv("DEVICE", "cpu")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()
