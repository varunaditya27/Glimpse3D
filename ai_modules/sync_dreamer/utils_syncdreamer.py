"""
Utility functions for SyncDreamer integration with Glimpse3D
"""

import os
import math
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Optional
from pathlib import Path


def segment_foreground(
    image: Image.Image,
    method: str = "rembg"
) -> Image.Image:
    """
    Segment foreground from background to create RGBA image.
    
    SyncDreamer requires images with transparent backgrounds for best results.
    This function removes the background using various methods.
    
    Args:
        image: Input RGB or RGBA image
        method: Segmentation method:
               - "rembg": Uses rembg library (recommended, fast)
               - "carvekit": Uses carvekit (higher quality, slower)
    
    Returns:
        RGBA image with transparent background
    
    Raises:
        ImportError: If required library is not installed
        ValueError: If unknown method specified
    """
    if method == "rembg":
        try:
            from rembg import remove
            return remove(image)
        except ImportError:
            raise ImportError(
                "rembg not installed. Install with: pip install rembg\n"
                "For GPU acceleration: pip install rembg[gpu]"
            )
    
    elif method == "carvekit":
        try:
            from carvekit.api.high import HiInterface
            
            interface = HiInterface(
                object_type="object",
                batch_size_seg=5,
                batch_size_matting=1,
                device='cuda',
                seg_mask_size=640,
                matting_mask_size=2048,
                trimap_prob_threshold=231,
                trimap_dilation=30,
                trimap_erosion_iters=5,
                fp16=False
            )
            
            result = interface([image.convert('RGB')])[0]
            return result
            
        except ImportError:
            raise ImportError(
                "carvekit not installed. Install with: pip install carvekit"
            )
    
    else:
        raise ValueError(f"Unknown segmentation method: {method}")


def preprocess_for_syncdreamer(
    image: Image.Image,
    crop_size: int = 200,
    image_size: int = 256,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """
    Preprocess an image for SyncDreamer input.
    
    This function:
    1. Ensures RGBA format
    2. Crops to foreground bounding box
    3. Resizes to fit within crop_size
    4. Centers on a white background of image_size
    
    Args:
        image: Input image (RGB or RGBA)
        crop_size: Target size for the foreground object
        image_size: Final output image size (256 for SyncDreamer)
        background_color: RGB background color (default white)
    
    Returns:
        Preprocessed RGBA image ready for SyncDreamer
    """
    # Ensure RGBA
    if image.mode != "RGBA":
        # Try to segment if no alpha channel
        try:
            image = segment_foreground(image)
        except ImportError:
            # Fallback: convert to RGBA with full opacity
            image = image.convert("RGBA")
    
    # Get alpha channel for bounding box
    alpha_np = np.asarray(image)[:, :, 3]
    
    # Find foreground bounding box
    coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
    if len(coords) == 0:
        raise ValueError("No foreground found in image")
    
    min_x, min_y = np.min(coords, 0)
    max_x, max_y = np.max(coords, 0)
    
    # Crop to bounding box
    cropped = image.crop((min_x, min_y, max_x, max_y))
    h, w = cropped.height, cropped.width
    
    # Calculate scale to fit in crop_size
    scale = crop_size / max(h, w)
    new_h, new_w = int(scale * h), int(scale * w)
    
    # Resize
    cropped = cropped.resize((new_w, new_h), resample=Image.BICUBIC)
    
    # Create output with background
    result = Image.new("RGBA", (image_size, image_size), (*background_color, 255))
    
    # Center the cropped image
    paste_x = (image_size - new_w) // 2
    paste_y = (image_size - new_h) // 2
    result.paste(cropped, (paste_x, paste_y), cropped)
    
    return result


def get_camera_matrices(
    elevations: List[float],
    azimuths: List[float],
    radius: float = 1.5
) -> List[np.ndarray]:
    """
    Generate camera-to-world transformation matrices.
    
    This is useful for downstream 3D reconstruction from the generated views.
    
    Args:
        elevations: List of elevation angles in degrees
        azimuths: List of azimuth angles in degrees
        radius: Distance from camera to origin
    
    Returns:
        List of 4x4 camera transformation matrices
    """
    matrices = []
    
    for elev, azim in zip(elevations, azimuths):
        elev_rad = np.radians(elev)
        azim_rad = np.radians(azim)
        
        # Camera position in spherical coordinates
        x = radius * np.cos(elev_rad) * np.sin(azim_rad)
        y = radius * np.sin(elev_rad)
        z = radius * np.cos(elev_rad) * np.cos(azim_rad)
        
        pos = np.array([x, y, z])
        
        # Look at origin
        forward = -pos / np.linalg.norm(pos)
        
        # Use world up vector to compute right
        world_up = np.array([0, 1, 0])
        right = np.cross(world_up, forward)
        right = right / np.linalg.norm(right)
        
        # Compute actual up vector
        up = np.cross(forward, right)
        
        # Build 4x4 transformation matrix
        mat = np.eye(4)
        mat[:3, 0] = right
        mat[:3, 1] = up
        mat[:3, 2] = -forward
        mat[:3, 3] = pos
        
        matrices.append(mat)

    return matrices


def syncdreamer_cameras(
    image_size: int = 252,
    radius: float = 1.5,
    fov_deg: float = 51.98
) -> List["CameraParams"]:
    """
    Build the 16 SyncDreamer view cameras for Glimpse3D back-projection.

    Produces one CameraParams per SyncDreamer output view, using the fixed
    elevation/azimuth schedule defined on SyncDreamerService (8 azimuths at
    +30 elevation followed by 8 azimuths at -20 elevation).

    Each camera is positioned at spherical (elevation, azimuth, radius) and
    looks at the origin. Extrinsics are 4x4 world->camera matrices in the
    OpenCV convention (+X right, +Y down, +Z forward into the scene).
    Intrinsics are a 3x3 pinhole matrix derived from fov_deg and image_size.

    Args:
        image_size: Square render resolution in px. Defaults to 252 (=18*14) so
            it stays divisible by the DINOv2 patch size (14) used by the MVCRM
            feature-consistency check; SyncDreamer's native 256px views are
            resized to this internally.
        radius: Distance from each camera to the origin
        fov_deg: Field of view in degrees used for the pinhole intrinsics

    Returns:
        List of exactly 16 CameraParams objects (one per SyncDreamer view)
    """
    # Imported here to avoid a hard import dependency at module load time.
    # The refine_module package __init__ pulls in heavy optional deps (scipy,
    # etc.); if that fails in a lightweight environment, load the back_projector
    # submodule directly so CameraParams stays importable.
    try:
        from ai_modules.refine_module.back_projector import CameraParams
    except Exception:
        import importlib.util
        bp_path = Path(__file__).resolve().parents[1] / "refine_module" / "back_projector.py"
        spec = importlib.util.spec_from_file_location(
            "ai_modules.refine_module.back_projector", bp_path
        )
        bp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bp_module)
        CameraParams = bp_module.CameraParams

    # SyncDreamerService is the source of truth for the view schedule. Importing
    # it pulls in the diffusion stack, which may be unavailable in lightweight
    # (e.g. CPU-only) environments; fall back to the locally-mirrored constants,
    # which are kept identical to the class attributes.
    try:
        from ai_modules.sync_dreamer.inference import SyncDreamerService
        elevations = SyncDreamerService.ELEVATIONS
        azimuths = SyncDreamerService.AZIMUTHS
    except Exception:
        elevations = SYNCDREAMER_CAMERAS["elevations"]
        azimuths = SYNCDREAMER_CAMERAS["azimuths"]

    # Pinhole intrinsics from field of view (shared by all 16 views).
    fov_rad = math.radians(fov_deg)
    focal = 0.5 * image_size / math.tan(0.5 * fov_rad)
    center = image_size / 2.0
    intrinsics = torch.tensor([
        [focal, 0.0, center],
        [0.0, focal, center],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)

    cameras: List[CameraParams] = []

    for elev, azim in zip(elevations, azimuths):
        elev_rad = math.radians(float(elev))
        azim_rad = math.radians(float(azim))

        # Camera center in world space (spherical -> cartesian, +Y up world).
        cam_center = torch.tensor([
            radius * math.cos(elev_rad) * math.sin(azim_rad),
            radius * math.sin(elev_rad),
            radius * math.cos(elev_rad) * math.cos(azim_rad),
        ], dtype=torch.float32)

        # OpenCV look-at: +Z (forward) points from the camera toward the origin.
        forward = -cam_center / torch.norm(cam_center)

        # Right = up_world x forward; in OpenCV +Y is down, so the camera up is
        # the negative of the world up vector used to seed the basis.
        world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        right = torch.cross(world_up, forward, dim=0)
        right = right / torch.norm(right)
        down = torch.cross(forward, right, dim=0)  # +Y axis (down) in OpenCV

        # Rotation rows are the camera-space axes expressed in world coords:
        # R maps world directions into camera space (world->camera).
        R = torch.stack([right, down, forward], dim=0)  # (3, 3)
        t = -torch.matmul(R, cam_center)  # (3,)

        extrinsics = torch.eye(4, dtype=torch.float32)
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = t

        cameras.append(CameraParams(
            intrinsics=intrinsics.clone(),
            extrinsics=extrinsics,
            width=image_size,
            height=image_size,
        ))

    return cameras


def views_to_video(
    image_paths: List[str],
    output_path: str,
    fps: int = 8,
    loop: bool = True
) -> None:
    """
    Create a turntable video from multi-view images.
    
    Args:
        image_paths: Paths to view images (should be sorted by azimuth)
        output_path: Output video file path (e.g., "output.mp4")
        fps: Frames per second
        loop: Whether to add reverse frames for seamless looping
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv not installed. Install with: pip install opencv-python")
    
    # Read images
    images = [cv2.imread(p) for p in image_paths]
    
    if loop:
        # Add reverse for smooth loop
        images = images + images[-2:0:-1]
    
    h, w = images[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for img in images:
        writer.write(img)
    
    writer.release()
    print(f"[SyncDreamer] Video saved to {output_path}")


def create_comparison_grid(
    input_image: Image.Image,
    output_images: List[Image.Image],
    output_path: str
) -> None:
    """
    Create a comparison image showing input and all outputs.
    
    Args:
        input_image: Original input image
        output_images: List of 16 generated views
        output_path: Path to save comparison grid
    """
    # Resize input to match output size
    size = output_images[0].size[0]
    input_resized = input_image.resize((size, size), Image.LANCZOS)
    
    # Create grid: input on left, 4x4 outputs on right
    grid_width = size + (size * 4)
    grid_height = size * 4
    
    grid = Image.new("RGB", (grid_width, grid_height), (40, 40, 40))
    
    # Place input (centered vertically on left side)
    input_y = (grid_height - size) // 2
    if input_resized.mode == "RGBA":
        # Handle transparency
        bg = Image.new("RGB", (size, size), (255, 255, 255))
        bg.paste(input_resized, mask=input_resized.split()[3])
        input_resized = bg
    grid.paste(input_resized, (0, input_y))
    
    # Place outputs in 4x4 grid
    for i, img in enumerate(output_images[:16]):
        row = i // 4
        col = i % 4
        x = size + (col * size)
        y = row * size
        grid.paste(img, (x, y))
    
    grid.save(output_path)
    print(f"[SyncDreamer] Comparison grid saved to {output_path}")


def estimate_elevation(
    image: Image.Image,
    use_depth: bool = False
) -> float:
    """
    Estimate the elevation angle of an input image.
    
    This is a heuristic estimation. For best results, users should
    provide the elevation angle if known.
    
    Args:
        image: Input image
        use_depth: Whether to use depth estimation (requires MiDaS)
    
    Returns:
        Estimated elevation angle in degrees
    
    Note:
        This returns a default value of 30° which works well for
        most object photographs. For more accurate results, the user
        should specify the elevation manually.
    """
    # Default assumption: slightly elevated view (common in product photos)
    # Most object photographs are taken at ~20-40° elevation
    return 30.0


# Camera configuration for Glimpse3D pipeline integration
SYNCDREAMER_CAMERAS = {
    "num_views": 16,
    "elevations": [30, 30, 30, 30, 30, 30, 30, 30,
                   -20, -20, -20, -20, -20, -20, -20, -20],
    "azimuths": [0, 45, 90, 135, 180, 225, 270, 315,
                 0, 45, 90, 135, 180, 225, 270, 315],
    "radius": 1.5,
    "image_size": 256
}


def get_view_info(view_index: int) -> dict:
    """
    Get camera information for a specific view.
    
    Args:
        view_index: Index of the view (0-15)
    
    Returns:
        Dictionary with elevation, azimuth, and other camera info
    """
    if not 0 <= view_index < 16:
        raise ValueError(f"View index must be 0-15, got {view_index}")
    
    return {
        "index": view_index,
        "elevation": SYNCDREAMER_CAMERAS["elevations"][view_index],
        "azimuth": SYNCDREAMER_CAMERAS["azimuths"][view_index],
        "radius": SYNCDREAMER_CAMERAS["radius"],
        "image_size": SYNCDREAMER_CAMERAS["image_size"]
    }
