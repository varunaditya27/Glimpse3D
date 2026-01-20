"""
Response models for image validation.

Responsibilities:
- Define schemas for validation responses
- Include quality metrics and metadata
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict


class ValidationMetadata(BaseModel):
    """Metadata from image validation process."""
    original_size: tuple = Field(..., description="Original image dimensions (width, height)")
    processed_size: tuple = Field(..., description="Processed image dimensions (width, height)")
    num_objects_detected: int = Field(..., description="Number of objects found in image")
    object_area_pixels: int = Field(..., description="Area of detected object in pixels")
    blur_score: float = Field(..., description="Image sharpness score (Laplacian variance)")
    validation_passed: bool = Field(True, description="Overall validation status")


class UploadResponse(BaseModel):
    """Response from successful image upload."""
    project_id: str = Field(..., description="UUID for this generation project")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Upload status")
    original_url: str = Field(..., description="URL of original uploaded image")
    processed_url: str = Field(..., description="URL of processed (background-removed) image")
    validation_metadata: ValidationMetadata = Field(..., description="Image validation metrics")


class ValidationError(BaseModel):
    """Response for validation failure."""
    detail: str = Field(..., description="Error message explaining validation failure")
    validation_metadata: Optional[Dict] = Field(None, description="Partial validation data if available")
