"""
ai_modules/diffusion/sdxl_lightning.py

SDXL Lightning Pipeline Wrapper.

Provides fast inference using SDXL Lightning (2-4 steps vs 30-50 for base SDXL).
Optimized for T4 GPU with ControlNet depth support.

Key Benefits:
- 4-10x faster than base SDXL
- Same VRAM footprint
- Compatible with ControlNet
- Works with img2img for enhancement

Research Sources:
- ByteDance SDXL-Lightning paper
- HuggingFace Diffusers documentation
"""

from typing import Optional, Union, Any, Dict, List
from dataclasses import dataclass
import numpy as np
from PIL import Image


@dataclass
class LightningConfig:
    """
    Configuration for SDXL Lightning pipeline.
    
    Presets for different step counts:
    - 1-step: Fastest, lower quality
    - 2-step: Good balance (recommended)
    - 4-step: Higher quality, still fast
    """
    num_inference_steps: int = 4
    guidance_scale: float = 0.0  # Lightning uses CFG=0
    strength: float = 0.75  # For img2img (0.25/0.5/0.75 for 1/2/4 steps)
    
    @classmethod
    def for_steps(cls, steps: int) -> "LightningConfig":
        """Create config optimized for specific step count."""
        if steps == 1:
            return cls(num_inference_steps=1, guidance_scale=0.0, strength=0.25)
        elif steps == 2:
            return cls(num_inference_steps=2, guidance_scale=0.0, strength=0.5)
        elif steps == 4:
            return cls(num_inference_steps=4, guidance_scale=0.0, strength=0.75)
        else:
            return cls(num_inference_steps=steps, guidance_scale=0.0, strength=0.75)


# SDXL Lightning model variants
LIGHTNING_MODELS = {
    "lightning_4step": {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "lora_repo": "ByteDance/SDXL-Lightning",
        "lora_file": "sdxl_lightning_4step_lora.safetensors",
        "steps": 4,
        "description": "4-step Lightning (recommended)",
    },
    "lightning_2step": {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "lora_repo": "ByteDance/SDXL-Lightning",
        "lora_file": "sdxl_lightning_2step_lora.safetensors",
        "steps": 2,
        "description": "2-step Lightning (fastest with good quality)",
    },
    "lightning_1step": {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "lora_repo": "ByteDance/SDXL-Lightning",
        "lora_file": "sdxl_lightning_1step_lora.safetensors",
        "steps": 1,
        "description": "1-step Lightning (fastest, lower quality)",
    },
}


class SDXLLightningPipeline:
    """
    SDXL Lightning Pipeline with ControlNet support.
    
    Provides fast, high-quality image enhancement using SDXL Lightning
    with optional depth-guided ControlNet conditioning.
    
    Example:
        from ai_modules.midas_depth import estimate_depth
        from ai_modules.diffusion import SDXLLightningPipeline
        
        # Initialize
        pipe = SDXLLightningPipeline(device="cuda", steps=4)
        pipe.load()
        
        # Enhance with depth guidance
        depth = estimate_depth("render.png")
        enhanced = pipe.enhance(
            image="render.png",
            depth_map=depth,
            prompt="high quality 3D render"
        )
    """
    
    def __init__(
        self,
        device: str = "cuda",
        steps: int = 4,
        use_controlnet: bool = True,
        controlnet_model: str = "xinsir_depth",
        optimize_memory: bool = True
    ):
        """
        Initialize SDXL Lightning pipeline.
        
        Args:
            device: Device to run on ("cuda" or "cpu")
            steps: Number of inference steps (1, 2, or 4)
            use_controlnet: Whether to load ControlNet for depth guidance
            controlnet_model: ControlNet model name from CONTROLNET_MODELS
            optimize_memory: Apply memory optimizations for T4 GPU
        """
        self.device = device
        self.steps = steps
        self.use_controlnet = use_controlnet
        self.controlnet_model_name = controlnet_model
        self.optimize_memory = optimize_memory
        
        self.pipe = None
        self.controlnet = None
        self.config = LightningConfig.for_steps(steps)
        
        # Get model info
        if steps == 1:
            self.model_info = LIGHTNING_MODELS["lightning_1step"]
        elif steps == 2:
            self.model_info = LIGHTNING_MODELS["lightning_2step"]
        else:
            self.model_info = LIGHTNING_MODELS["lightning_4step"]
    
    def load(self) -> "SDXLLightningPipeline":
        """
        Load the pipeline and models.
        
        Returns:
            self (for chaining)
        """
        import torch
        from diffusers import (
            StableDiffusionXLControlNetImg2ImgPipeline,
            StableDiffusionXLImg2ImgPipeline,
            ControlNetModel,
            EulerDiscreteScheduler,
        )
        from huggingface_hub import hf_hub_download
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"Loading SDXL Lightning ({self.steps}-step)...")
        print(f"  Device: {self.device}")
        print(f"  ControlNet: {self.use_controlnet}")
        
        # Load ControlNet if enabled
        if self.use_controlnet:
            from .controlnet_depth import CONTROLNET_MODELS
            
            if self.controlnet_model_name in CONTROLNET_MODELS:
                controlnet_repo = CONTROLNET_MODELS[self.controlnet_model_name]["repo_id"]
            else:
                controlnet_repo = self.controlnet_model_name
            
            print(f"  Loading ControlNet: {controlnet_repo}")
            self.controlnet = ControlNetModel.from_pretrained(
                controlnet_repo,
                torch_dtype=dtype,
                use_safetensors=True,
            )
        
        # Load base SDXL with or without ControlNet
        base_model = self.model_info["base_model"]
        
        if self.use_controlnet and self.controlnet is not None:
            self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                base_model,
                controlnet=self.controlnet,
                torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None,
                use_safetensors=True,
            )
        else:
            self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                base_model,
                torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None,
                use_safetensors=True,
            )
        
        # Set up Euler scheduler for Lightning
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config,
            timestep_spacing="trailing",
            prediction_type="epsilon",
        )
        
        # Load Lightning LoRA
        print(f"  Loading Lightning LoRA ({self.steps}-step)...")
        lora_path = hf_hub_download(
            self.model_info["lora_repo"],
            self.model_info["lora_file"]
        )
        self.pipe.load_lora_weights(lora_path)
        self.pipe.fuse_lora()
        
        # Apply memory optimizations
        if self.optimize_memory:
            self._apply_optimizations()
        else:
            self.pipe = self.pipe.to(self.device)
        
        print("✓ SDXL Lightning loaded successfully!")
        return self
    
    def _apply_optimizations(self) -> None:
        """Apply memory optimizations for T4 GPU."""
        from .memory_utils import setup_memory_optimization, MemoryConfig
        
        config = MemoryConfig.from_preset("t4_balanced")
        self.pipe = setup_memory_optimization(self.pipe, config, self.device)
    
    def enhance(
        self,
        image: Union[str, Image.Image, np.ndarray],
        prompt: str,
        depth_map: Optional[np.ndarray] = None,
        negative_prompt: str = "blurry, low quality, artifacts",
        strength: Optional[float] = None,
        controlnet_scale: float = 0.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Enhance an image using SDXL Lightning.
        
        Args:
            image: Input image to enhance
            prompt: Text prompt describing desired output
            depth_map: Optional depth map from midas_depth for structure guidance
            negative_prompt: What to avoid
            strength: Override default strength (0-1, higher = more change)
            controlnet_scale: ControlNet conditioning scale (0-1)
            seed: Random seed for reproducibility
        
        Returns:
            Enhanced PIL Image
        
        Example:
            from ai_modules.midas_depth import estimate_depth
            
            depth = estimate_depth("render.png")
            enhanced = pipe.enhance(
                image="render.png",
                depth_map=depth,
                prompt="photorealistic 3D render, 8k quality"
            )
        """
        import torch
        from .image_utils import preprocess_for_diffusion, postprocess_from_diffusion
        from .controlnet_depth import prepare_depth_for_controlnet
        
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")
        
        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Preprocess input image
        processed_image, original_size = preprocess_for_diffusion(image, target_size=1024)
        
        # Prepare depth map for ControlNet
        control_image = None
        if depth_map is not None and self.use_controlnet:
            control_image = prepare_depth_for_controlnet(
                depth_map,
                target_size=(1024, 1024)
            )
        
        # Use configured strength or override
        img_strength = strength if strength is not None else self.config.strength
        
        # Build generation kwargs
        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": processed_image,
            "strength": img_strength,
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "generator": generator,
        }
        
        # Add ControlNet parameters if available
        if control_image is not None and self.use_controlnet:
            gen_kwargs["control_image"] = control_image
            gen_kwargs["controlnet_conditioning_scale"] = controlnet_scale
        
        # Generate
        with torch.no_grad():
            result = self.pipe(**gen_kwargs).images[0]
        
        # Restore to original size
        result = postprocess_from_diffusion(result, original_size)
        
        return result
    
    def enhance_batch(
        self,
        images: List[Union[str, Image.Image, np.ndarray]],
        prompts: Union[str, List[str]],
        depth_maps: Optional[List[np.ndarray]] = None,
        **kwargs
    ) -> List[Image.Image]:
        """
        Enhance multiple images.
        
        Args:
            images: List of input images
            prompts: Single prompt or list of prompts
            depth_maps: Optional list of depth maps
            **kwargs: Additional arguments for enhance()
        
        Returns:
            List of enhanced images
        """
        # Handle single prompt for all images
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        
        # Handle missing depth maps
        if depth_maps is None:
            depth_maps = [None] * len(images)
        
        results = []
        for i, (img, prompt, depth) in enumerate(zip(images, prompts, depth_maps)):
            print(f"Enhancing image {i + 1}/{len(images)}...")
            result = self.enhance(img, prompt, depth_map=depth, **kwargs)
            results.append(result)
        
        return results
    
    def unload(self) -> None:
        """Unload pipeline to free memory."""
        from .memory_utils import clear_gpu_memory
        
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        
        if self.controlnet is not None:
            del self.controlnet
            self.controlnet = None
        
        clear_gpu_memory()
        print("Pipeline unloaded")


def load_sdxl_pipeline(
    steps: int = 4,
    use_controlnet: bool = True,
    device: str = "cuda",
    optimize_memory: bool = True
) -> SDXLLightningPipeline:
    """
    Convenience function to load SDXL Lightning pipeline.
    
    Args:
        steps: Number of inference steps (1, 2, or 4)
        use_controlnet: Enable depth ControlNet
        device: Device to use
        optimize_memory: Apply T4 optimizations
    
    Returns:
        Loaded SDXLLightningPipeline
    
    Example:
        pipe = load_sdxl_pipeline(steps=4, use_controlnet=True)
        enhanced = pipe.enhance(image, prompt, depth_map=depth)
    """
    pipeline = SDXLLightningPipeline(
        device=device,
        steps=steps,
        use_controlnet=use_controlnet,
        optimize_memory=optimize_memory
    )
    pipeline.load()
    return pipeline


def get_recommended_settings(gpu_type: str = "t4") -> Dict[str, Any]:
    """
    Get recommended settings for specific GPU type.
    
    Args:
        gpu_type: "t4", "a100", "3080", "3090", "4090", or "cpu"
    
    Returns:
        Dictionary with recommended settings
    """
    settings = {
        "t4": {  # 15GB VRAM
            "steps": 4,
            "use_controlnet": True,
            "optimize_memory": True,
            "batch_size": 1,
            "image_size": 1024,
            "notes": "Use model CPU offload, expect ~8s per image"
        },
        "a100": {  # 40-80GB VRAM
            "steps": 4,
            "use_controlnet": True,
            "optimize_memory": False,
            "batch_size": 4,
            "image_size": 1024,
            "notes": "Full speed, can batch multiple images"
        },
        "3080": {  # 10GB VRAM
            "steps": 4,
            "use_controlnet": True,
            "optimize_memory": True,
            "batch_size": 1,
            "image_size": 1024,
            "notes": "Use sequential CPU offload"
        },
        "3090": {  # 24GB VRAM
            "steps": 4,
            "use_controlnet": True,
            "optimize_memory": True,
            "batch_size": 2,
            "image_size": 1024,
            "notes": "Minimal offloading needed"
        },
        "4090": {  # 24GB VRAM
            "steps": 4,
            "use_controlnet": True,
            "optimize_memory": False,
            "batch_size": 2,
            "image_size": 1024,
            "notes": "Near-full speed"
        },
        "cpu": {
            "steps": 4,
            "use_controlnet": False,  # Too slow with ControlNet
            "optimize_memory": True,
            "batch_size": 1,
            "image_size": 512,
            "notes": "Very slow, consider 2-step for faster results"
        },
    }
    
    if gpu_type.lower() not in settings:
        return settings["t4"]  # Default to T4 settings
    
    return settings[gpu_type.lower()]
