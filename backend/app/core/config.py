"""
Application configuration settings.

Responsibilities:
- Load environment variables
- Define paths for models, assets, and temp storage
- Configure API settings (host, port, debug mode)
"""

import os
from pathlib import Path

class Settings:
    PROJECT_NAME: str = "Glimpse3D"
    API_V1_STR: str = "/api/v1"
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent
    MODEL_DIR: str = os.getenv("MODEL_DIR", "model_checkpoints")
    ASSET_DIR: str = os.getenv("ASSET_DIR", "assets")
    DEMO_MODE: bool = os.getenv("GLIMPSE3D_DEMO_MODE", "").strip().lower() in ("1", "true", "yes")
    PIPELINE_TIMEOUT: int = int(os.getenv("PIPELINE_TIMEOUT", "2700"))
    GSPLAT_TRAIN_TIMEOUT: int = int(os.getenv("GSPLAT_TRAIN_TIMEOUT", "1800"))
    RENDER_TIMEOUT: int = int(os.getenv("RENDER_TIMEOUT", "60"))

settings = Settings()
