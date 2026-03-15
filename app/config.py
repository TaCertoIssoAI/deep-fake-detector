import os


class Settings:
    IMAGE_MODEL: str = os.getenv(
        "IMAGE_MODEL", "prithivMLmods/Deep-Fake-Detector-v2-Model"
    )
    HF_TOKEN: str | None = os.getenv("HF_TOKEN")
    DEVICE: str = os.getenv("DEVICE", "cpu")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()
