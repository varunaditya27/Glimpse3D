"""
Image Validation Service

Responsibilities:
- Validate image quality (resolution, blur, blank check)
- Remove background using rembg
- Detect and count objects using Connected Components
- Validate single object requirement
- Crop and center object for optimal 3D reconstruction
- Return detailed validation results
"""

import cv2
import numpy as np
from PIL import Image
import io
from typing import Dict, Tuple, Optional
from rembg import remove
from app.core.logger import logger


class ImageValidator:
    """
    Validates and preprocesses images for 3D reconstruction.
    
    Uses rembg for background removal and Connected Components
    for object detection - no heavy models required!
    """
    
    # Validation thresholds
    MIN_WIDTH = 256
    MIN_HEIGHT = 256
    MAX_WIDTH = 4096
    MAX_HEIGHT = 4096
    MIN_ASPECT_RATIO = 0.33  # 1:3
    MAX_ASPECT_RATIO = 3.0   # 3:1
    MIN_BLUR_THRESHOLD = 100  # Laplacian variance
    MIN_OBJECT_AREA = 1000   # Minimum pixels for valid object
    MAX_BLANK_RATIO = 0.95   # Max % of blank pixels
    TARGET_SIZE = 512        # Output size for processing
    
    def __init__(self):
        """Initialize the validator."""
        logger.info("ImageValidator initialized")
    
    def validate_and_process(
        self, 
        image_data: bytes, 
        filename: str
    ) -> Dict:
        """
        Complete validation and preprocessing pipeline.
        
        Steps:
        1. Basic format validation
        2. Image quality checks (size, blur, blank)
        3. Background removal
        4. Object detection and counting
        5. Single object validation
        6. Crop and center object
        
        Args:
            image_data: Raw image bytes
            filename: Original filename for logging
            
        Returns:
            Dict containing:
                - valid: bool
                - processed_image: PIL Image (if valid)
                - validation_metadata: dict with quality metrics
                - error: str (if invalid)
        """
        try:
            logger.info(f"Validating image: {filename}")
            
            # Step 1: Load image
            image = Image.open(io.BytesIO(image_data))
            original_size = image.size
            
            # Step 2: Basic format validation
            format_check = self._validate_format(image)
            if not format_check["valid"]:
                return format_check
            
            # Step 3: Image quality checks
            quality_check = self._validate_quality(image)
            if not quality_check["valid"]:
                return quality_check
            
            # Step 4: Remove background
            logger.info("Removing background with rembg")
            no_bg_image = self._remove_background(image)
            
            # Step 5: Detect objects
            object_detection = self._detect_objects(no_bg_image)
            if not object_detection["valid"]:
                return object_detection
            
            # Step 6: Crop and center
            logger.info("Cropping and centering object")
            processed_image = self._crop_and_center(
                no_bg_image, 
                object_detection["bbox"]
            )
            
            # Success!
            validation_metadata = {
                "original_size": original_size,
                "processed_size": processed_image.size,
                "num_objects_detected": object_detection["num_objects"],
                "object_area_pixels": object_detection["object_area"],
                "blur_score": quality_check.get("blur_score"),
                "validation_passed": True
            }
            
            logger.info(f"✅ Validation passed: {filename}")
            
            return {
                "valid": True,
                "processed_image": processed_image,
                "validation_metadata": validation_metadata
            }
            
        except Exception as e:
            logger.error(f"Validation failed for {filename}: {str(e)}")
            return {
                "valid": False,
                "error": f"Internal validation error: {str(e)}"
            }
    
    def _validate_format(self, image: Image.Image) -> Dict:
        """
        Validate image format and basic properties.
        """
        # Check image mode
        if image.mode not in ["RGB", "RGBA", "L"]:
            return {
                "valid": False,
                "error": f"Unsupported image mode: {image.mode}. Expected RGB, RGBA, or grayscale."
            }
        
        # Check dimensions
        width, height = image.size
        
        if width < self.MIN_WIDTH or height < self.MIN_HEIGHT:
            return {
                "valid": False,
                "error": f"Image too small: {width}x{height}. Minimum size is {self.MIN_WIDTH}x{self.MIN_HEIGHT}."
            }
        
        if width > self.MAX_WIDTH or height > self.MAX_HEIGHT:
            return {
                "valid": False,
                "error": f"Image too large: {width}x{height}. Maximum size is {self.MAX_WIDTH}x{self.MAX_HEIGHT}."
            }
        
        # Check aspect ratio
        aspect_ratio = width / height
        
        if aspect_ratio < self.MIN_ASPECT_RATIO or aspect_ratio > self.MAX_ASPECT_RATIO:
            return {
                "valid": False,
                "error": f"Extreme aspect ratio: {aspect_ratio:.2f}. Image should be roughly square (between 1:3 and 3:1)."
            }
        
        return {"valid": True}
    
    def _validate_quality(self, image: Image.Image) -> Dict:
        """
        Check image quality: blur detection and blank image detection.
        """
        # Convert to numpy for OpenCV processing
        image_np = np.array(image)
        
        # Convert to grayscale for quality checks
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        # 1. Blur Detection (Laplacian Variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < self.MIN_BLUR_THRESHOLD:
            return {
                "valid": False,
                "error": f"Image is too blurry (blur score: {laplacian_var:.1f}). Please upload a sharper image.",
                "blur_score": laplacian_var
            }
        
        # 2. Blank Image Detection
        # Check if image is mostly uniform (blank white/black)
        std_dev = np.std(gray)
        
        if std_dev < 10:  # Very low variation
            return {
                "valid": False,
                "error": "Image appears to be blank or contains no visible content."
            }
        
        # 3. Check if image is mostly white or black pixels
        mean_intensity = np.mean(gray)
        
        if mean_intensity > 250:  # Mostly white
            return {
                "valid": False,
                "error": "Image is almost entirely white. Please upload an image with visible content."
            }
        
        if mean_intensity < 5:  # Mostly black
            return {
                "valid": False,
                "error": "Image is almost entirely black. Please upload an image with visible content."
            }
        
        return {
            "valid": True,
            "blur_score": laplacian_var
        }
    
    def _remove_background(self, image: Image.Image) -> Image.Image:
        """
        Remove background using rembg.
        
        Returns RGBA image with transparent background.
        """
        # Convert to RGB if needed
        if image.mode == "RGBA":
            # Create white background
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")
        
        # Convert PIL to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Remove background with rembg
        output_bytes = remove(img_byte_arr.getvalue())
        no_bg_image = Image.open(io.BytesIO(output_bytes))
        
        # Ensure RGBA mode
        if no_bg_image.mode != "RGBA":
            no_bg_image = no_bg_image.convert("RGBA")
        
        return no_bg_image
    
    def _detect_objects(self, image: Image.Image) -> Dict:
        """
        Detect objects using Connected Components analysis.
        
        This is a lightweight alternative to SAM or YOLO.
        Works by analyzing the alpha channel after background removal.
        """
        # Convert to numpy
        image_np = np.array(image)
        
        # Extract alpha channel
        if image_np.shape[2] == 4:
            alpha = image_np[:, :, 3]
        else:
            # Fallback: use grayscale
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            alpha = (gray > 10).astype(np.uint8) * 255
        
        # Check if image is completely transparent
        if np.sum(alpha > 0) < self.MIN_OBJECT_AREA:
            return {
                "valid": False,
                "error": "No visible object detected in image. Background removal may have removed everything."
            }
        
        # Find connected components
        # Binary threshold
        _, binary = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up noise
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        # Filter valid objects (ignore label 0 = background)
        valid_objects = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area >= self.MIN_OBJECT_AREA:
                valid_objects.append({
                    "label": i,
                    "area": area,
                    "bbox": (
                        stats[i, cv2.CC_STAT_LEFT],
                        stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH],
                        stats[i, cv2.CC_STAT_HEIGHT]
                    ),
                    "centroid": centroids[i]
                })
        
        num_objects = len(valid_objects)
        
        # Validation
        if num_objects == 0:
            return {
                "valid": False,
                "error": "No valid objects detected. Image may contain only noise or very small objects."
            }
        
        if num_objects > 1:
            return {
                "valid": False,
                "error": f"Multiple objects detected ({num_objects}). Please upload an image with a single object.",
                "num_objects": num_objects
            }
        
        # Success - single object found
        main_object = valid_objects[0]
        
        return {
            "valid": True,
            "num_objects": 1,
            "object_area": main_object["area"],
            "bbox": main_object["bbox"],
            "centroid": main_object["centroid"]
        }
    
    def _crop_and_center(
        self, 
        image: Image.Image, 
        bbox: Tuple[int, int, int, int]
    ) -> Image.Image:
        """
        Crop image to bounding box and center on square canvas.
        
        Args:
            image: RGBA image
            bbox: (x, y, width, height)
            
        Returns:
            Centered image on square canvas
        """
        image_np = np.array(image)
        x, y, w, h = bbox
        
        # Add padding around object (10% on each side)
        padding = int(max(w, h) * 0.1)
        
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image_np.shape[1], x + w + padding)
        y_end = min(image_np.shape[0], y + h + padding)
        
        # Crop to padded bounding box
        cropped = image_np[y_start:y_end, x_start:x_end]
        
        # Get dimensions
        crop_h, crop_w = cropped.shape[:2]
        
        # Determine canvas size (use larger dimension, make it square)
        canvas_size = max(crop_w, crop_h)
        
        # Optionally resize to target size while maintaining aspect ratio
        if canvas_size > self.TARGET_SIZE:
            scale = self.TARGET_SIZE / canvas_size
            new_w = int(crop_w * scale)
            new_h = int(crop_h * scale)
            cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
            crop_h, crop_w = cropped.shape[:2]
            canvas_size = self.TARGET_SIZE
        
        # Create square canvas (transparent)
        canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
        
        # Calculate position to center object
        y_offset = (canvas_size - crop_h) // 2
        x_offset = (canvas_size - crop_w) // 2
        
        # Place cropped image on canvas
        canvas[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w] = cropped
        
        # Convert back to PIL
        centered_image = Image.fromarray(canvas, mode='RGBA')
        
        return centered_image


# Singleton instance
_validator_instance = None

def get_validator() -> ImageValidator:
    """Get or create singleton ImageValidator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = ImageValidator()
    return _validator_instance
