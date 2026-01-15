"""
ai_modules/diffusion/image_utils.py

Image Processing Utilities for Diffusion Enhancement.

Provides pre/post processing functions for the enhancement pipeline.
Designed to work with both PIL Images and numpy arrays for seamless
integration with the midas_depth module.
"""

import os
from typing import Union, Tuple, Optional, List
import numpy as np
from PIL import Image


def preprocess_for_diffusion(
    image: Union[str, Image.Image, np.ndarray],
    target_size: int = 1024,
    maintain_aspect: bool = True
) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    Preprocess image for SDXL diffusion pipeline.
    
    SDXL works best with:
    - Size: 1024x1024 (base) or multiples of 64
    - Format: RGB PIL Image
    
    Args:
        image: Input image (path, PIL, or numpy array)
        target_size: Target dimension (1024 for SDXL)
        maintain_aspect: If True, pad to square instead of stretch
    
    Returns:
        processed_image: PIL Image ready for diffusion
        original_size: Original (H, W) for restoring later
    
    Example:
        processed, orig_size = preprocess_for_diffusion("render.png")
        enhanced = pipe(image=processed, ...)
        restored = postprocess_from_diffusion(enhanced, orig_size)
    """
    # Load image
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            # Grayscale
            img = Image.fromarray(np.stack([image]*3, axis=-1).astype(np.uint8))
        elif image.dtype != np.uint8:
            # Normalize float arrays
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
            img = Image.fromarray(image).convert("RGB")
        else:
            img = Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    original_size = img.size[::-1]  # PIL is (W, H), we return (H, W)
    
    if maintain_aspect:
        img = resize_with_aspect(img, target_size)
        # Pad to exact target size
        img = pad_to_size(img, target_size, target_size)
    else:
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    return img, original_size


def postprocess_from_diffusion(
    image: Union[Image.Image, np.ndarray],
    original_size: Tuple[int, int],
    crop_padding: bool = True
) -> Image.Image:
    """
    Post-process diffusion output back to original size.
    
    Args:
        image: Diffusion output
        original_size: Original (H, W) from preprocess_for_diffusion
        crop_padding: If True, remove padding before resizing
    
    Returns:
        Restored PIL Image at original size
    """
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image
    
    h, w = original_size
    
    if crop_padding:
        # Assume image was padded to square, crop center region
        img_w, img_h = img.size
        aspect = w / h
        
        if aspect > 1:  # Wide image
            crop_h = int(img_w / aspect)
            top = (img_h - crop_h) // 2
            img = img.crop((0, top, img_w, top + crop_h))
        elif aspect < 1:  # Tall image
            crop_w = int(img_h * aspect)
            left = (img_w - crop_w) // 2
            img = img.crop((left, 0, left + crop_w, img_h))
    
    # Resize to original dimensions
    img = img.resize((w, h), Image.Resampling.LANCZOS)
    
    return img


def resize_with_aspect(
    image: Image.Image,
    target_size: int,
    resample: int = Image.Resampling.LANCZOS
) -> Image.Image:
    """
    Resize image maintaining aspect ratio.
    
    The longest edge will be target_size.
    
    Args:
        image: PIL Image
        target_size: Target size for longest edge
        resample: PIL resampling method
    
    Returns:
        Resized PIL Image
    """
    w, h = image.size
    
    if w >= h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)
    
    # Round to multiple of 64 (required by many diffusion models)
    new_w = (new_w // 64) * 64
    new_h = (new_h // 64) * 64
    
    return image.resize((new_w, new_h), resample)


def pad_to_size(
    image: Image.Image,
    target_width: int,
    target_height: int,
    fill_color: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """
    Pad image to target size with fill color.
    
    Args:
        image: PIL Image
        target_width: Target width
        target_height: Target height
        fill_color: RGB color for padding
    
    Returns:
        Padded PIL Image
    """
    w, h = image.size
    
    if w == target_width and h == target_height:
        return image
    
    # Create new image with fill color
    result = Image.new("RGB", (target_width, target_height), fill_color)
    
    # Center the original image
    left = (target_width - w) // 2
    top = (target_height - h) // 2
    result.paste(image, (left, top))
    
    return result


def blend_images(
    original: Union[Image.Image, np.ndarray],
    enhanced: Union[Image.Image, np.ndarray],
    alpha: float = 0.7,
    mask: Optional[np.ndarray] = None
) -> Image.Image:
    """
    Blend original and enhanced images.
    
    Useful for controlling enhancement strength or blending based on
    confidence maps from midas_depth module.
    
    Args:
        original: Original image
        enhanced: Enhanced image
        alpha: Blend factor (0 = original, 1 = enhanced)
        mask: Optional per-pixel blend mask (H, W), values [0, 1]
              Can use depth confidence from midas_depth
    
    Returns:
        Blended PIL Image
    
    Example:
        from ai_modules.midas_depth import estimate_depth_confidence
        
        confidence = estimate_depth_confidence(depth_map, rgb_image)
        # Blend more enhanced in high-confidence regions
        blended = blend_images(original, enhanced, mask=confidence)
    """
    # Convert to numpy arrays
    if isinstance(original, Image.Image):
        orig_arr = np.array(original).astype(np.float32)
    else:
        orig_arr = original.astype(np.float32)
    
    if isinstance(enhanced, Image.Image):
        enh_arr = np.array(enhanced).astype(np.float32)
    else:
        enh_arr = enhanced.astype(np.float32)
    
    # Ensure same size
    if orig_arr.shape != enh_arr.shape:
        enh_pil = Image.fromarray(enh_arr.astype(np.uint8))
        enh_pil = enh_pil.resize((orig_arr.shape[1], orig_arr.shape[0]), Image.Resampling.LANCZOS)
        enh_arr = np.array(enh_pil).astype(np.float32)
    
    # Apply blending
    if mask is not None:
        # Per-pixel blending using mask
        mask = np.clip(mask, 0, 1)
        if mask.ndim == 2:
            mask = mask[:, :, np.newaxis]
        # Resize mask if needed
        if mask.shape[:2] != orig_arr.shape[:2]:
            mask_pil = Image.fromarray((mask[:, :, 0] * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((orig_arr.shape[1], orig_arr.shape[0]), Image.Resampling.LANCZOS)
            mask = np.array(mask_pil).astype(np.float32) / 255.0
            mask = mask[:, :, np.newaxis]
        
        blend_factor = mask * alpha
        result = orig_arr * (1 - blend_factor) + enh_arr * blend_factor
    else:
        # Global blending
        result = orig_arr * (1 - alpha) + enh_arr * alpha
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def match_histogram(
    source: Union[Image.Image, np.ndarray],
    reference: Union[Image.Image, np.ndarray]
) -> Image.Image:
    """
    Match histogram of source image to reference.
    
    Useful for maintaining color consistency between original render
    and enhanced version.
    
    Args:
        source: Image to modify (enhanced)
        reference: Image to match (original)
    
    Returns:
        Source image with matched histogram
    """
    if isinstance(source, Image.Image):
        source = np.array(source)
    if isinstance(reference, Image.Image):
        reference = np.array(reference)
    
    result = np.zeros_like(source)
    
    for c in range(3):  # RGB channels
        src_channel = source[:, :, c].flatten()
        ref_channel = reference[:, :, c].flatten()
        
        # Get histograms
        src_values, src_indices, src_counts = np.unique(
            src_channel, return_inverse=True, return_counts=True
        )
        ref_values, ref_counts = np.unique(ref_channel, return_counts=True)
        
        # Compute CDFs
        src_cdf = np.cumsum(src_counts).astype(np.float64)
        src_cdf /= src_cdf[-1]
        
        ref_cdf = np.cumsum(ref_counts).astype(np.float64)
        ref_cdf /= ref_cdf[-1]
        
        # Map source values to reference values
        interp_values = np.interp(src_cdf, ref_cdf, ref_values)
        result[:, :, c] = interp_values[src_indices].reshape(source.shape[:2])
    
    return Image.fromarray(result.astype(np.uint8))


def prepare_depth_image(
    depth: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    invert: bool = False,
    normalize: bool = True
) -> Image.Image:
    """
    Prepare depth map for visualization or ControlNet input.
    
    Args:
        depth: Depth map (H, W), values typically [0, 1]
        target_size: Optional (W, H) to resize
        invert: If True, invert depth (far = white, near = black)
        normalize: If True, normalize to full [0, 255] range
    
    Returns:
        Grayscale PIL Image
    """
    depth_arr = depth.copy()
    
    if normalize:
        d_min, d_max = depth_arr.min(), depth_arr.max()
        if d_max - d_min > 1e-8:
            depth_arr = (depth_arr - d_min) / (d_max - d_min)
    
    if invert:
        depth_arr = 1.0 - depth_arr
    
    depth_uint8 = (np.clip(depth_arr, 0, 1) * 255).astype(np.uint8)
    depth_img = Image.fromarray(depth_uint8, mode='L').convert('RGB')
    
    if target_size is not None:
        depth_img = depth_img.resize(target_size, Image.Resampling.LANCZOS)
    
    return depth_img


def save_comparison(
    original: Union[Image.Image, np.ndarray],
    enhanced: Union[Image.Image, np.ndarray],
    output_path: str,
    depth: Optional[np.ndarray] = None,
    titles: Optional[List[str]] = None
) -> None:
    """
    Save side-by-side comparison of original and enhanced images.
    
    Args:
        original: Original image
        enhanced: Enhanced image
        output_path: Path to save comparison
        depth: Optional depth map to include
        titles: Optional list of titles for each image
    """
    # Convert to PIL
    if isinstance(original, np.ndarray):
        original = Image.fromarray(original)
    if isinstance(enhanced, np.ndarray):
        enhanced = Image.fromarray(enhanced)
    
    images = [original, enhanced]
    default_titles = ["Original", "Enhanced"]
    
    if depth is not None:
        depth_img = prepare_depth_image(depth)
        depth_img = depth_img.resize(original.size, Image.Resampling.LANCZOS)
        images.append(depth_img)
        default_titles.append("Depth")
    
    if titles is None:
        titles = default_titles
    
    # Ensure all images same size
    target_size = images[0].size
    images = [img.resize(target_size, Image.Resampling.LANCZOS) for img in images]
    
    # Create comparison grid
    n_images = len(images)
    total_width = target_size[0] * n_images
    total_height = target_size[1]
    
    comparison = Image.new('RGB', (total_width, total_height))
    
    for i, img in enumerate(images):
        comparison.paste(img, (i * target_size[0], 0))
    
    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    comparison.save(output_path)
