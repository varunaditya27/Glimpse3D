"""
ai_modules/diffusion/controlnet_depth.py

ControlNet Depth Adapter for SDXL.

Responsibilities:
- Prepare depth maps from midas_depth module for ControlNet input
- Load and configure ControlNet depth models for SDXL
- Provide depth conditioning for structure-preserving enhancement

Integration with midas_depth module:
    from ai_modules.midas_depth import estimate_depth
    from ai_modules.diffusion import prepare_depth_for_controlnet
    
    depth = estimate_depth("render.png")
    controlnet_image = prepare_depth_for_controlnet(depth)

Key Models:
- xinsir/controlnet-depth-sdxl-1.0 (recommended for SDXL)
- diffusers/controlnet-depth-sdxl-1.0
"""

from typing import Optional, Union, Tuple, Any
import numpy as np
from PIL import Image


# Recommended ControlNet depth models for SDXL
CONTROLNET_MODELS = {
    "xinsir_depth": {
        "repo_id": "xinsir/controlnet-depth-sdxl-1.0",
        "description": "High quality depth ControlNet for SDXL",
        "recommended": True,
    },
    "diffusers_depth": {
        "repo_id": "diffusers/controlnet-depth-sdxl-1.0",
        "description": "Official diffusers depth ControlNet",
        "recommended": False,
    },
}


def prepare_depth_for_controlnet(
    depth: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    invert: bool = False,
    normalize: bool = True,
    three_channel: bool = True
) -> Image.Image:
    """
    Prepare depth map from midas_depth module for ControlNet input.
    
    ControlNet expects:
    - RGB image (3 channels) even for depth
    - Values in [0, 255] range
    - Typically 1024x1024 for SDXL
    
    Args:
        depth: Depth map from midas_depth module (H, W), values [0, 1]
               where 1 = closest (MiDaS convention)
        target_size: Optional (W, H) for resizing
        invert: If True, invert depth (far = white, near = black)
                Set True if your ControlNet expects inverted depth
        normalize: If True, normalize to full [0, 1] range first
        three_channel: If True, output RGB (required for most pipelines)
    
    Returns:
        PIL Image ready for ControlNet conditioning
    
    Example:
        from ai_modules.midas_depth import estimate_depth
        
        depth = estimate_depth("rendered_view.png")
        controlnet_image = prepare_depth_for_controlnet(depth, target_size=(1024, 1024))
        
        # Use in SDXL pipeline
        output = pipe(
            prompt="...",
            image=input_image,
            control_image=controlnet_image,
            ...
        )
    """
    depth_arr = depth.copy().astype(np.float32)
    
    # Normalize to [0, 1] if needed
    if normalize:
        d_min, d_max = depth_arr.min(), depth_arr.max()
        if d_max - d_min > 1e-8:
            depth_arr = (depth_arr - d_min) / (d_max - d_min)
    
    # Invert if needed (some ControlNets expect inverted depth)
    if invert:
        depth_arr = 1.0 - depth_arr
    
    # Convert to uint8
    depth_uint8 = (np.clip(depth_arr, 0, 1) * 255).astype(np.uint8)
    
    # Create PIL Image
    if three_channel:
        # Stack to RGB (grayscale in all channels)
        depth_rgb = np.stack([depth_uint8] * 3, axis=-1)
        depth_img = Image.fromarray(depth_rgb, mode='RGB')
    else:
        depth_img = Image.fromarray(depth_uint8, mode='L')
    
    # Resize if target size specified
    if target_size is not None:
        depth_img = depth_img.resize(target_size, Image.Resampling.LANCZOS)
    
    return depth_img


def prepare_depth_batch(
    depths: list,
    target_size: Optional[Tuple[int, int]] = None,
    **kwargs
) -> list:
    """
    Prepare multiple depth maps for ControlNet.
    
    Args:
        depths: List of depth maps
        target_size: Target size for all
        **kwargs: Additional arguments for prepare_depth_for_controlnet
    
    Returns:
        List of prepared depth images
    """
    return [
        prepare_depth_for_controlnet(d, target_size=target_size, **kwargs)
        for d in depths
    ]


class DepthControlNetAdapter:
    """
    Adapter class for loading and using ControlNet depth models.
    
    Handles model loading, configuration, and provides convenient
    methods for depth-conditioned generation.
    
    Example:
        adapter = DepthControlNetAdapter(model_name="xinsir_depth")
        controlnet = adapter.get_controlnet()
        
        # Use with SDXL pipeline
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            ...
        )
    """
    
    def __init__(
        self,
        model_name: str = "xinsir_depth",
        device: str = "cuda",
        dtype: Optional[Any] = None
    ):
        """
        Initialize ControlNet adapter.
        
        Args:
            model_name: Name from CONTROLNET_MODELS or HuggingFace repo ID
            device: Device to load model on
            dtype: Torch dtype (None = auto, typically torch.float16)
        """
        self.device = device
        self.dtype = dtype
        self.controlnet = None
        
        # Resolve model repo ID
        if model_name in CONTROLNET_MODELS:
            self.repo_id = CONTROLNET_MODELS[model_name]["repo_id"]
            self.model_info = CONTROLNET_MODELS[model_name]
        else:
            self.repo_id = model_name
            self.model_info = {"repo_id": model_name, "description": "Custom model"}
    
    def load(self) -> "DepthControlNetAdapter":
        """
        Load the ControlNet model.
        
        Returns:
            self (for chaining)
        """
        import torch
        from diffusers import ControlNetModel
        
        if self.dtype is None:
            self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"Loading ControlNet: {self.repo_id}")
        
        self.controlnet = ControlNetModel.from_pretrained(
            self.repo_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
        )
        
        if self.device != "cpu":
            self.controlnet = self.controlnet.to(self.device)
        
        print(f"✓ ControlNet loaded successfully")
        return self
    
    def get_controlnet(self) -> Any:
        """
        Get the loaded ControlNet model.
        
        Returns:
            ControlNetModel instance
        """
        if self.controlnet is None:
            self.load()
        return self.controlnet
    
    def prepare_depth(
        self,
        depth: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """
        Prepare depth map for this ControlNet.
        
        Args:
            depth: Depth map from midas_depth module
            target_size: Optional target size
        
        Returns:
            Prepared depth image
        """
        # xinsir model expects non-inverted depth (near = white)
        # Adjust based on model requirements
        invert = "diffusers" in self.repo_id.lower()
        
        return prepare_depth_for_controlnet(
            depth,
            target_size=target_size,
            invert=invert
        )
    
    def get_recommended_scale(self) -> float:
        """
        Get recommended conditioning scale for this model.
        
        Returns:
            Recommended controlnet_conditioning_scale value
        """
        # xinsir models work well with 0.5-0.7
        # Adjust based on desired structure preservation
        return 0.5
    
    def unload(self) -> None:
        """Unload model to free memory."""
        if self.controlnet is not None:
            del self.controlnet
            self.controlnet = None
            
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def get_default_controlnet_config() -> dict:
    """
    Get default configuration for ControlNet depth.
    
    Returns:
        Dictionary with recommended settings
    """
    return {
        "model": "xinsir_depth",
        "conditioning_scale": 0.5,  # Balance between structure and creativity
        "guess_mode": False,
        "control_guidance_start": 0.0,
        "control_guidance_end": 1.0,
    }
