"""
Application configuration settings.

Responsibilities:
- Load environment variables
- Define paths for models, assets, and temp storage
- Configure API settings (host, port, debug mode)
"""

import os

class Settings:
    PROJECT_NAME: str = "Glimpse3D"
    API_V1_STR: str = "/api/v1"
    MODEL_DIR: str = os.getenv("MODEL_DIR", "model_checkpoints")
    ASSET_DIR: str = os.getenv("ASSET_DIR", "assets")

settings = Settings()
