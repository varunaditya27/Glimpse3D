"""
Pydantic models for API request validation.

Responsibilities:
- Define schemas for incoming JSON payloads
- Validate data types and required fields
"""

from pydantic import BaseModel

class GenerateRequest(BaseModel):
    upload_id: str
    settings: dict = {}

class RefineRequest(BaseModel):
    model_id: str
    camera_pose: dict
    prompt: str = "high quality, detailed"
