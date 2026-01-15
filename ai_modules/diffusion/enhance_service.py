"""
ai_modules/diffusion/enhance_service.py

Main Enhancement Service Orchestrator.

This is the primary interface for the diffusion enhancement module.
It orchestrates the full enhancement pipeline including:
- Depth estimation (via midas_depth module)
- SDXL Lightning enhancement
- ControlNet depth conditioning
- Memory management

Integration Point:
    This service integrates with midas_depth module for depth maps
    and provides the refinement capability for the Glimpse3D pipeline.

Usage:
    from ai_modules.diffusion import EnhanceService
    
    service = EnhanceService(device="cuda")
    enhanced = service.enhance(
        image="rendered_view.png",
        prompt="high quality 3D render"
    )
"""

import os
from typing import Optional, Union, List, Dict, Any, Callable
from dataclasses import dataclass, field
from PIL import Image
import numpy as np


@dataclass
class EnhanceConfig:
    """
    Configuration for enhancement service.
    
    Attributes:
        device: CUDA device or CPU
        lightning_steps: SDXL Lightning steps (1, 2, or 4)
        use_controlnet: Enable depth ControlNet
        controlnet_scale: ControlNet conditioning strength
        strength: img2img denoising strength
        prompt_template: Prompt template name
        auto_depth: Automatically estimate depth if not provided
        depth_model: MiDaS model type for auto depth
        optimize_memory: Apply T4 GPU optimizations
    """
    device: str = "cuda"
    lightning_steps: int = 4
    use_controlnet: bool = True
    controlnet_scale: float = 0.5
    strength: float = 0.75
    prompt_template: str = "photorealistic"
    auto_depth: bool = True
    depth_model: str = "MiDaS_small"  # Fast, from midas_depth module
    optimize_memory: bool = True
    seed: Optional[int] = None
    
    @classmethod
    def for_t4_gpu(cls) -> "EnhanceConfig":
        """Optimized config for T4 GPU (Google Colab)."""
        return cls(
            device="cuda",
            lightning_steps=4,
            use_controlnet=True,
            controlnet_scale=0.5,
            strength=0.75,
            optimize_memory=True,
            depth_model="MiDaS_small",  # Fastest
        )
    
    @classmethod
    def for_quality(cls) -> "EnhanceConfig":
        """Config prioritizing quality over speed."""
        return cls(
            device="cuda",
            lightning_steps=4,
            use_controlnet=True,
            controlnet_scale=0.6,
            strength=0.65,  # Lower = preserve more original
            optimize_memory=True,
            depth_model="DPT_Large",  # Best quality
        )
    
    @classmethod
    def for_speed(cls) -> "EnhanceConfig":
        """Config prioritizing speed."""
        return cls(
            device="cuda",
            lightning_steps=2,
            use_controlnet=True,
            controlnet_scale=0.5,
            strength=0.5,
            optimize_memory=True,
            depth_model="MiDaS_small",
        )


class EnhanceService:
    """
    Main enhancement service orchestrator.
    
    Integrates SDXL Lightning with midas_depth module for
    depth-guided 3D view enhancement.
    
    Example:
        from ai_modules.diffusion import EnhanceService
        
        # Initialize
        service = EnhanceService(config=EnhanceConfig.for_t4_gpu())
        
        # Enhance single view
        enhanced = service.enhance(
            image="rendered_view.png",
            prompt="photorealistic 3D model"
        )
        
        # Enhance multiple views
        enhanced_views = service.enhance_batch(
            images=rendered_views,
            prompt="detailed texture, studio lighting"
        )
        
        # With pre-computed depth from midas_depth
        from ai_modules.midas_depth import estimate_depth
        depth = estimate_depth("view.png")
        enhanced = service.enhance("view.png", depth_map=depth)
    """
    
    def __init__(
        self,
        config: Optional[EnhanceConfig] = None,
        device: Optional[str] = None,
        optimize_memory: Optional[bool] = None
    ):
        """
        Initialize enhancement service.
        
        Args:
            config: Enhancement configuration
            device: Override device (uses config.device if None)
            optimize_memory: Override memory optimization setting
        """
        self.config = config or EnhanceConfig.for_t4_gpu()
        
        # Allow overrides
        if device is not None:
            self.config.device = device
        if optimize_memory is not None:
            self.config.optimize_memory = optimize_memory
        
        self.pipeline = None
        self.depth_estimator = None
        self.prompt_builder = None
        self._loaded = False
    
    def load(self) -> "EnhanceService":
        """
        Load all required models.
        
        Returns:
            self (for chaining)
        """
        if self._loaded:
            return self
        
        print("=" * 60)
        print("Loading EnhanceService")
        print("=" * 60)
        
        # Load SDXL Lightning pipeline
        from .sdxl_lightning import SDXLLightningPipeline
        
        self.pipeline = SDXLLightningPipeline(
            device=self.config.device,
            steps=self.config.lightning_steps,
            use_controlnet=self.config.use_controlnet,
            optimize_memory=self.config.optimize_memory
        )
        self.pipeline.load()
        
        # Load depth estimator if auto_depth enabled
        if self.config.auto_depth:
            try:
                from ai_modules.midas_depth import get_estimator
                self.depth_estimator = get_estimator(
                    model_type=self.config.depth_model,
                    device=self.config.device
                )
                print(f"✓ Depth estimator loaded ({self.config.depth_model})")
            except ImportError:
                print("⚠ midas_depth module not available, auto_depth disabled")
                self.config.auto_depth = False
        
        # Load prompt builder
        from .prompt_builder import PromptBuilder
        self.prompt_builder = PromptBuilder(template=self.config.prompt_template)
        
        self._loaded = True
        print("=" * 60)
        print("EnhanceService ready!")
        print("=" * 60)
        
        return self
    
    def enhance(
        self,
        image: Union[str, Image.Image, np.ndarray],
        prompt: Optional[str] = None,
        depth_map: Optional[np.ndarray] = None,
        negative_prompt: Optional[str] = None,
        subject: Optional[str] = None,
        strength: Optional[float] = None,
        controlnet_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Enhance a single image.
        
        Args:
            image: Input image (path, PIL, or numpy)
            prompt: Custom prompt (uses template if None)
            depth_map: Pre-computed depth from midas_depth (auto-computed if None)
            negative_prompt: Custom negative prompt
            subject: Subject description for prompt building
            strength: Override denoising strength
            controlnet_scale: Override ControlNet scale
            seed: Random seed for reproducibility
        
        Returns:
            Enhanced PIL Image
        
        Example:
            # Simple enhancement
            enhanced = service.enhance("render.png")
            
            # With custom prompt
            enhanced = service.enhance(
                "render.png",
                prompt="photorealistic car, metallic paint"
            )
            
            # With pre-computed depth
            from ai_modules.midas_depth import estimate_depth
            depth = estimate_depth("render.png")
            enhanced = service.enhance("render.png", depth_map=depth)
        """
        if not self._loaded:
            self.load()
        
        # Build prompt if not provided
        if prompt is None:
            prompt, neg = self.prompt_builder.build(subject=subject)
            if negative_prompt is None:
                negative_prompt = neg
        
        if negative_prompt is None:
            negative_prompt = "blurry, low quality, artifacts, distorted"
        
        # Auto-compute depth if needed and enabled
        if depth_map is None and self.config.auto_depth and self.config.use_controlnet:
            if self.depth_estimator is not None:
                print("  Computing depth map...")
                depth_map = self.depth_estimator.estimate(image)
        
        # Use config defaults if not overridden
        actual_strength = strength if strength is not None else self.config.strength
        actual_cn_scale = controlnet_scale if controlnet_scale is not None else self.config.controlnet_scale
        actual_seed = seed if seed is not None else self.config.seed
        
        # Enhance using pipeline
        enhanced = self.pipeline.enhance(
            image=image,
            prompt=prompt,
            depth_map=depth_map,
            negative_prompt=negative_prompt,
            strength=actual_strength,
            controlnet_scale=actual_cn_scale,
            seed=actual_seed,
        )
        
        return enhanced
    
    def enhance_batch(
        self,
        images: List[Union[str, Image.Image, np.ndarray]],
        prompt: Optional[str] = None,
        depth_maps: Optional[List[np.ndarray]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> List[Image.Image]:
        """
        Enhance multiple images.
        
        Args:
            images: List of input images
            prompt: Prompt for all images (or use template)
            depth_maps: Optional list of pre-computed depth maps
            progress_callback: Called with (current, total) for progress
            **kwargs: Additional arguments for enhance()
        
        Returns:
            List of enhanced images
        
        Example:
            enhanced_views = service.enhance_batch(
                images=rendered_views,
                prompt="detailed 3D render",
                progress_callback=lambda i, n: print(f"{i}/{n}")
            )
        """
        if not self._loaded:
            self.load()
        
        n = len(images)
        results = []
        
        # Prepare depth maps
        if depth_maps is None:
            depth_maps = [None] * n
        
        for i, (img, depth) in enumerate(zip(images, depth_maps)):
            if progress_callback:
                progress_callback(i + 1, n)
            else:
                print(f"Enhancing {i + 1}/{n}...")
            
            result = self.enhance(
                image=img,
                prompt=prompt,
                depth_map=depth,
                **kwargs
            )
            results.append(result)
        
        return results
    
    def enhance_with_depth_confidence(
        self,
        image: Union[str, Image.Image, np.ndarray],
        prompt: Optional[str] = None,
        blend_with_original: bool = True,
        confidence_threshold: float = 0.5,
        **kwargs
    ) -> Image.Image:
        """
        Enhance with confidence-weighted blending.
        
        Uses depth confidence from midas_depth to blend enhanced
        and original images - preserving original in low-confidence regions.
        
        Args:
            image: Input image
            prompt: Enhancement prompt
            blend_with_original: If True, blend based on confidence
            confidence_threshold: Minimum confidence for full enhancement
            **kwargs: Additional enhancement arguments
        
        Returns:
            Enhanced and optionally blended image
        
        Example:
            # Preserves original in uncertain depth regions
            enhanced = service.enhance_with_depth_confidence(
                "render.png",
                blend_with_original=True,
                confidence_threshold=0.6
            )
        """
        if not self._loaded:
            self.load()
        
        # Load original image
        if isinstance(image, str):
            original = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            original = Image.fromarray(image)
        else:
            original = image
        
        # Compute depth and confidence
        depth_map = None
        confidence = None
        
        if self.config.auto_depth and self.depth_estimator is not None:
            depth_map = self.depth_estimator.estimate(image)
            
            # Try to get confidence from midas_depth
            try:
                from ai_modules.midas_depth import estimate_depth_confidence
                confidence = estimate_depth_confidence(depth_map, np.array(original))
            except ImportError:
                pass
        
        # Enhance
        enhanced = self.enhance(
            image=image,
            prompt=prompt,
            depth_map=depth_map,
            **kwargs
        )
        
        # Blend based on confidence if available
        if blend_with_original and confidence is not None:
            from .image_utils import blend_images
            
            # Create blend mask (high confidence = use enhanced)
            mask = np.clip(confidence / confidence_threshold, 0, 1)
            
            enhanced = blend_images(
                original=original,
                enhanced=enhanced,
                alpha=0.8,
                mask=mask
            )
        
        return enhanced
    
    def get_depth_for_image(
        self,
        image: Union[str, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """
        Get depth map for an image using midas_depth module.
        
        Args:
            image: Input image
        
        Returns:
            Depth map (H, W) with values [0, 1]
        """
        if self.depth_estimator is None:
            if self.config.auto_depth:
                from ai_modules.midas_depth import get_estimator
                self.depth_estimator = get_estimator(
                    model_type=self.config.depth_model,
                    device=self.config.device
                )
            else:
                raise RuntimeError("Depth estimator not available")
        
        return self.depth_estimator.estimate(image)
    
    def unload(self) -> None:
        """Unload all models to free memory."""
        if self.pipeline is not None:
            self.pipeline.unload()
            self.pipeline = None
        
        self.depth_estimator = None
        self._loaded = False
        
        from .memory_utils import clear_gpu_memory
        clear_gpu_memory()
        
        print("EnhanceService unloaded")
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status and configuration."""
        from .memory_utils import get_memory_status
        
        return {
            "loaded": self._loaded,
            "config": {
                "device": self.config.device,
                "lightning_steps": self.config.lightning_steps,
                "use_controlnet": self.config.use_controlnet,
                "controlnet_scale": self.config.controlnet_scale,
                "strength": self.config.strength,
                "auto_depth": self.config.auto_depth,
                "depth_model": self.config.depth_model,
            },
            "memory": get_memory_status(),
        }


# Convenience functions for simple usage

def enhance_view(
    image: Union[str, Image.Image, np.ndarray],
    prompt: str = "high quality 3D render, detailed texture",
    depth_map: Optional[np.ndarray] = None,
    device: str = "cuda",
    **kwargs
) -> Image.Image:
    """
    Convenience function to enhance a single view.
    
    Creates a temporary EnhanceService for one-off enhancement.
    For multiple images, use EnhanceService directly.
    
    Args:
        image: Input image
        prompt: Enhancement prompt
        depth_map: Optional pre-computed depth
        device: Device to use
        **kwargs: Additional enhancement arguments
    
    Returns:
        Enhanced PIL Image
    
    Example:
        from ai_modules.diffusion import enhance_view
        enhanced = enhance_view("render.png")
    """
    config = EnhanceConfig.for_t4_gpu()
    config.device = device
    
    service = EnhanceService(config=config)
    result = service.enhance(image, prompt=prompt, depth_map=depth_map, **kwargs)
    service.unload()
    
    return result


def enhance_views_batch(
    images: List[Union[str, Image.Image, np.ndarray]],
    prompt: str = "high quality 3D render, detailed texture",
    depth_maps: Optional[List[np.ndarray]] = None,
    device: str = "cuda",
    **kwargs
) -> List[Image.Image]:
    """
    Convenience function to enhance multiple views.
    
    Args:
        images: List of input images
        prompt: Enhancement prompt
        depth_maps: Optional list of depth maps
        device: Device to use
        **kwargs: Additional enhancement arguments
    
    Returns:
        List of enhanced PIL Images
    
    Example:
        from ai_modules.diffusion import enhance_views_batch
        enhanced = enhance_views_batch(rendered_views)
    """
    config = EnhanceConfig.for_t4_gpu()
    config.device = device
    
    service = EnhanceService(config=config)
    results = service.enhance_batch(images, prompt=prompt, depth_maps=depth_maps, **kwargs)
    service.unload()
    
    return results
