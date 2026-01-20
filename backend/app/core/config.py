"""
Application configuration settings.

Responsibilities:
- Load environment variables
- Define paths for models, assets, and temp storage
- Configure API settings (host, port, debug mode)
- Supabase credentials
"""

import os
from pathlib import Path

class Settings:
    PROJECT_NAME: str = "Glimpse3D"
    API_V1_STR: str = "/api/v1"
    MODEL_DIR: str = os.getenv("MODEL_DIR", "model_checkpoints")
    ASSET_DIR: str = os.getenv("ASSET_DIR", "assets")
    
    # Supabase Configuration
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")
    
    # Storage Bucket Names (must match Supabase bucket creation)
    BUCKET_UPLOADS: str = "project-uploads"
    BUCKET_PROCESSED: str = "processed-images"
    BUCKET_MULTIVIEW: str = "multiview-images"
    BUCKET_DEPTH: str = "depth-maps"
    BUCKET_ENHANCED: str = "enhanced-views"
    BUCKET_MODELS: str = "3d-models"

settings = Settings()
