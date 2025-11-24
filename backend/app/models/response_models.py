"""
Pydantic models for API response schemas.

Responsibilities:
- Define standard response structures
- Ensure consistent API output
"""

from pydantic import BaseModel

class GenerateResponse(BaseModel):
    status: str
    model_url: str
    message: str = None

class RefineResponse(BaseModel):
    status: str
    updated_model_url: str
