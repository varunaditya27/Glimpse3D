"""
SyncDreamer Inference Service for Glimpse3D
============================================
Generates 16 multi-view consistent images from a single input image.

This module provides a clean interface for SyncDreamer inference,
replacing Zero123 for multi-view synthesis with better consistency
and lower VRAM requirements.

Usage:
    from ai_modules.sync_dreamer import SyncDreamerService, generate_multiview
    
    # Quick inference
    output_paths = generate_multiview("input.png", "outputs/", elevation=30.0)
    
    # Or use the service class for more control
    service = SyncDreamerService()
    service.load_model()
    images = service.generate(image, elevation=30.0)
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

# Add local ldm to path (prioritize over any installed version)
SYNC_DREAMER_PATH = Path(__file__).parent
sys.path.insert(0, str(SYNC_DREAMER_PATH))

from ldm.models.diffusion.sync_dreamer import SyncMultiviewDiffusion, SyncDDIMSampler
from ldm.util import instantiate_from_config, prepare_inputs


class SyncDreamerService:
    """
    Service wrapper for SyncDreamer multi-view generation.
    
    Generates 16 consistent views from a single foreground-segmented image.
    Views are rendered at two elevation levels (30° and -20°) with 8 azimuth
    angles each (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°).
    
    Attributes:
        ELEVATIONS: Fixed elevation angles for 16 output views
        AZIMUTHS: Fixed azimuth angles for 16 output views
    """
    
    # Fixed camera configurations from SyncDreamer
    # First 8 views at 30° elevation, next 8 at -20° elevation
    ELEVATIONS = [30, 30, 30, 30, 30, 30, 30, 30,
                  -20, -20, -20, -20, -20, -20, -20, -20]
    AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315,
                0, 45, 90, 135, 180, 225, 270, 315]
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize SyncDreamer service.
        
        Args:
            config_path: Path to syncdreamer.yaml config file.
                        Defaults to configs/syncdreamer.yaml
            checkpoint_path: Path to pretrained checkpoint.
                            Defaults to ckpt/syncdreamer-pretrain.ckpt
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        
        # Set default paths relative to this file
        base_path = Path(__file__).parent
        
        if config_path is None:
            config_path = base_path / "configs" / "syncdreamer.yaml"
        if checkpoint_path is None:
            checkpoint_path = base_path / "ckpt" / "syncdreamer-pretrain.ckpt"
        
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        
        self.model: Optional[SyncMultiviewDiffusion] = None
        self._loaded = False
        
        # Validate paths exist
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                "Download from: https://drive.google.com/file/d/1ypyD5WXxAnsWjnHgAfOAGolV0Zd9kpam/view"
            )
    
    def load_model(self) -> None:
        """
        Load the SyncDreamer model into memory.
        
        This loads the model weights and moves them to the specified device.
        Call this before running inference, or it will be called automatically.
        """
        if self._loaded:
            print("[SyncDreamer] Model already loaded")
            return
        
        print(f"[SyncDreamer] Loading config from {self.config_path}")
        config = OmegaConf.load(self.config_path)
        
        # Update CLIP path to be relative to this module
        clip_path = Path(__file__).parent / "ckpt" / "ViT-L-14.pt"
        if clip_path.exists():
            config.model.params.clip_image_encoder_path = str(clip_path)
        
        print(f"[SyncDreamer] Loading checkpoint from {self.checkpoint_path}")
        self.model = instantiate_from_config(config.model)
        
        # Load weights
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        self.model.load_state_dict(state_dict, strict=True)
        
        # Move to device and set to eval mode
        self.model = self.model.to(self.device).eval()
        
        self._loaded = True
        print(f"[SyncDreamer] Model loaded successfully on {self.device}")
    
    def unload_model(self) -> None:
        """
        Unload the model to free GPU memory.
        
        Call this when you're done with inference to release VRAM.
        """
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        self._loaded = False
        print("[SyncDreamer] Model unloaded")
    
    @torch.no_grad()
    def generate(
        self,
        image: Union[Image.Image, str, Path],
        elevation: float = 30.0,
        crop_size: int = 200,
        cfg_scale: float = 2.0,
        sample_num: int = 1,
        batch_view_num: int = 1,
        sample_steps: int = 50,
        seed: int = 42
    ) -> List[Image.Image]:
        """
        Generate 16 multi-view consistent images from a single input.
        
        Args:
            image: Input image (PIL Image, path string, or Path object).
                   Should be RGBA with transparent background for best results.
            elevation: Elevation angle of input view in degrees.
                      Typically 0-40°, default 30°.
            crop_size: Size to crop foreground object to.
                      Use -1 to skip cropping. Default 200.
            cfg_scale: Classifier-free guidance scale. Default 2.0.
            sample_num: Number of sample sets to generate. Default 1.
            batch_view_num: Views to process per batch (lower = less VRAM).
                           Default 8, use 4 for GPUs with <12GB VRAM.
            sample_steps: Number of DDIM sampling steps. Default 50.
            seed: Random seed for reproducibility.
        
        Returns:
            List of 16 PIL Images at the fixed viewpoints.
            For sample_num > 1, returns 16 * sample_num images.
        """
        if not self._loaded:
            self.load_model()
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Handle different input types
        if isinstance(image, (str, Path)):
            image_path = str(image)
        else:
            # Save PIL image to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                image.save(f.name)
                image_path = f.name
        
        # Prepare input data
        data = prepare_inputs(image_path, elevation, crop_size)
        for k, v in data.items():
            data[k] = v.unsqueeze(0).to(self.device)
            data[k] = torch.repeat_interleave(data[k], sample_num, dim=0)
        
        # Create sampler and generate
        sampler = SyncDDIMSampler(self.model, sample_steps)
        x_sample = self.model.sample(sampler, data, cfg_scale, batch_view_num)
        
        # Convert output to PIL Images
        B, N, _, H, W = x_sample.shape
        x_sample = (torch.clamp(x_sample, max=1.0, min=-1.0) + 1) * 0.5
        x_sample = x_sample.permute(0, 1, 3, 4, 2).cpu().numpy() * 255
        x_sample = x_sample.astype(np.uint8)
        
        output_images = []
        for bi in range(B):
            for ni in range(N):
                output_images.append(Image.fromarray(x_sample[bi, ni]))
        
        return output_images
    
    def generate_and_save(
        self,
        image: Union[Image.Image, str, Path],
        output_dir: str,
        elevation: float = 30.0,
        save_grid: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate multi-view images and save to disk.
        
        Args:
            image: Input image (PIL Image or path)
            output_dir: Directory to save output images
            elevation: Input view elevation in degrees
            save_grid: Whether to save a 4x4 grid visualization
            **kwargs: Additional arguments passed to generate()
        
        Returns:
            List of saved file paths
        """
        views = self.generate(image, elevation=elevation, **kwargs)
        
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        for i, view in enumerate(views[:16]):  # First 16 views
            elev = self.ELEVATIONS[i]
            azim = self.AZIMUTHS[i]
            filename = f"view_{i:02d}_elev{elev}_azim{azim}.png"
            path = os.path.join(output_dir, filename)
            view.save(path)
            output_paths.append(path)
        
        if save_grid:
            grid_path = os.path.join(output_dir, "multiview_grid.png")
            self._save_grid(views[:16], grid_path)
            output_paths.append(grid_path)
        
        return output_paths
    
    def _save_grid(self, images: List[Image.Image], output_path: str) -> None:
        """Save 16 images as a 4x4 grid."""
        size = images[0].size[0]
        grid = Image.new("RGB", (size * 4, size * 4))
        
        for i, img in enumerate(images[:16]):
            row = i // 4
            col = i % 4
            grid.paste(img, (col * size, row * size))
        
        grid.save(output_path)
        print(f"[SyncDreamer] Grid saved to {output_path}")


# =============================================================================
# Convenience Functions
# =============================================================================

_service_instance: Optional[SyncDreamerService] = None


def get_service(**kwargs) -> SyncDreamerService:
    """
    Get or create a singleton SyncDreamer service instance.
    
    Args:
        **kwargs: Arguments passed to SyncDreamerService constructor
                 (only used when creating new instance)
    
    Returns:
        SyncDreamerService instance
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = SyncDreamerService(**kwargs)
    return _service_instance


def generate_multiview(
    image_path: str,
    output_dir: str,
    elevation: float = 30.0,
    seed: int = 42,
    **kwargs
) -> List[str]:
    """
    Quick function to generate and save multi-view images.
    
    This is the simplest way to use SyncDreamer for Glimpse3D pipeline.
    
    Args:
        image_path: Path to input image (RGBA with transparent background)
        output_dir: Directory to save output images
        elevation: Input view elevation angle in degrees
        seed: Random seed for reproducibility
        **kwargs: Additional arguments passed to generate()
    
    Returns:
        List of generated image file paths
    
    Example:
        >>> from ai_modules.sync_dreamer import generate_multiview
        >>> paths = generate_multiview("input.png", "outputs/", elevation=30)
        >>> print(f"Generated {len(paths)} views")
    """
    service = get_service()
    return service.generate_and_save(
        image_path,
        output_dir,
        elevation=elevation,
        seed=seed,
        **kwargs
    )


def cleanup() -> None:
    """Release GPU memory by unloading the model."""
    global _service_instance
    if _service_instance is not None:
        _service_instance.unload_model()
        _service_instance = None
