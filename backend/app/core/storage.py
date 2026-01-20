"""
Supabase Storage helper functions for file uploads.

Handles all interactions with Supabase Storage buckets:
- project-uploads: Original user images
- processed-images: Background-removed images
- multiview-images: SyncDreamer 16-view outputs
- depth-maps: MiDaS depth maps and heatmaps
- enhanced-views: SDXL-enhanced images
- 3d-models: Gaussian Splat models and exports
"""

import io
import os
from pathlib import Path
from typing import Union, BinaryIO
from PIL import Image
import numpy as np
from app.core.supabase_client import get_supabase
from app.core.logger import logger


class StorageManager:
    """Handles file uploads to Supabase Storage buckets."""
    
    # Bucket names
    BUCKET_UPLOADS = "project-uploads"
    BUCKET_PROCESSED = "processed-images"
    BUCKET_MULTIVIEW = "multiview-images"
    BUCKET_DEPTH = "depth-maps"
    BUCKET_ENHANCED = "enhanced-views"
    BUCKET_MODELS = "3d-models"
    
    @staticmethod
    def _get_public_url(bucket: str, file_path: str) -> str:
        """
        Get the public URL for a file in storage.
        
        Args:
            bucket: Bucket name
            file_path: File path within bucket
            
        Returns:
            Public URL to access the file
        """
        supabase = get_supabase()
        response = supabase.storage.from_(bucket).get_public_url(file_path)
        return response
    
    @staticmethod
    def upload_file(
        bucket: str,
        file_path: str,
        file_data: Union[bytes, BinaryIO, str, Path],
        content_type: str = "application/octet-stream"
    ) -> str:
        """
        Upload a file to Supabase Storage.
        
        Args:
            bucket: Bucket name
            file_path: Destination path within bucket
            file_data: File data (bytes, file object, or path to local file)
            content_type: MIME type of the file
            
        Returns:
            Public URL of the uploaded file
            
        Raises:
            Exception: If upload fails
        """
        supabase = get_supabase()
        
        try:
            # Handle different input types
            if isinstance(file_data, (str, Path)):
                with open(file_data, "rb") as f:
                    data = f.read()
            elif isinstance(file_data, bytes):
                data = file_data
            else:
                data = file_data.read()
            
            # Upload to Supabase Storage
            supabase.storage.from_(bucket).upload(
                path=file_path,
                file=data,
                file_options={"content-type": content_type, "upsert": "true"}
            )
            
            # Get public URL
            public_url = StorageManager._get_public_url(bucket, file_path)
            logger.info(f"Uploaded file to {bucket}/{file_path}")
            
            return public_url
            
        except Exception as e:
            logger.error(f"Failed to upload file to {bucket}/{file_path}: {str(e)}")
            raise
    
    @staticmethod
    def upload_image(
        bucket: str,
        file_path: str,
        image: Union[Image.Image, np.ndarray, bytes, str, Path],
        format: str = "PNG"
    ) -> str:
        """
        Upload an image to Supabase Storage.
        
        Args:
            bucket: Bucket name
            file_path: Destination path within bucket
            image: PIL Image, numpy array, bytes, or path to image file
            format: Image format (PNG, JPEG, etc.)
            
        Returns:
            Public URL of the uploaded image
        """
        # Convert image to bytes
        if isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format=format)
            data = buffer.getvalue()
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format)
            data = buffer.getvalue()
        elif isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                data = f.read()
        else:
            data = image
        
        content_type = f"image/{format.lower()}"
        return StorageManager.upload_file(bucket, file_path, data, content_type)
    
    @staticmethod
    def upload_numpy_array(
        bucket: str,
        file_path: str,
        array: np.ndarray
    ) -> str:
        """
        Upload a numpy array as .npy file to Supabase Storage.
        
        Args:
            bucket: Bucket name
            file_path: Destination path within bucket (should end with .npy)
            array: Numpy array to upload
            
        Returns:
            Public URL of the uploaded file
        """
        buffer = io.BytesIO()
        np.save(buffer, array)
        data = buffer.getvalue()
        
        return StorageManager.upload_file(
            bucket, file_path, data, "application/octet-stream"
        )
    
    @staticmethod
    def upload_original_image(project_id: str, image: Union[Image.Image, bytes, str, Path]) -> str:
        """Upload original user image."""
        file_path = f"{project_id}/original.png"
        return StorageManager.upload_image(
            StorageManager.BUCKET_UPLOADS, file_path, image, format="PNG"
        )
    
    @staticmethod
    def upload_processed_image(project_id: str, image: Union[Image.Image, bytes]) -> str:
        """Upload background-removed image."""
        file_path = f"{project_id}/processed.png"
        return StorageManager.upload_image(
            StorageManager.BUCKET_PROCESSED, file_path, image, format="PNG"
        )
    
    @staticmethod
    def upload_multiview_image(project_id: str, view_index: int, image: Union[Image.Image, np.ndarray]) -> str:
        """Upload a single multi-view image."""
        file_path = f"{project_id}/views/view_{view_index:02d}.png"
        return StorageManager.upload_image(
            StorageManager.BUCKET_MULTIVIEW, file_path, image, format="PNG"
        )
    
    @staticmethod
    def upload_depth_map(project_id: str, view_index: int, depth_array: np.ndarray) -> str:
        """Upload raw depth map (.npy)."""
        file_path = f"{project_id}/depth/view_{view_index:02d}.npy"
        return StorageManager.upload_numpy_array(
            StorageManager.BUCKET_DEPTH, file_path, depth_array
        )
    
    @staticmethod
    def upload_depth_heatmap(project_id: str, view_index: int, heatmap: Union[Image.Image, np.ndarray]) -> str:
        """Upload depth visualization heatmap."""
        file_path = f"{project_id}/depth/view_{view_index:02d}_heatmap.png"
        return StorageManager.upload_image(
            StorageManager.BUCKET_DEPTH, file_path, heatmap, format="PNG"
        )
    
    @staticmethod
    def upload_enhanced_view(
        project_id: str,
        iteration: int,
        view_index: int,
        image: Union[Image.Image, np.ndarray],
        is_rendered: bool = False
    ) -> str:
        """Upload enhanced or rendered view image."""
        prefix = "rendered" if is_rendered else "enhanced"
        file_path = f"{project_id}/enhanced/iter_{iteration}/{prefix}_{view_index:02d}.png"
        return StorageManager.upload_image(
            StorageManager.BUCKET_ENHANCED, file_path, image, format="PNG"
        )
    
    @staticmethod
    def upload_model(
        project_id: str,
        version: int,
        model_data: Union[bytes, str, Path],
        is_final: bool = False
    ) -> str:
        """Upload Gaussian Splat model."""
        if is_final:
            file_path = f"{project_id}/models/final.ply"
        else:
            file_path = f"{project_id}/models/v{version}.ply"
        
        return StorageManager.upload_file(
            StorageManager.BUCKET_MODELS, file_path, model_data, "application/octet-stream"
        )
    
    @staticmethod
    def upload_exported_model(
        project_id: str,
        format: str,
        model_data: Union[bytes, str, Path]
    ) -> str:
        """Upload exported model in specific format."""
        file_path = f"{project_id}/models/final.{format}"
        return StorageManager.upload_file(
            StorageManager.BUCKET_MODELS, file_path, model_data, "application/octet-stream"
        )
