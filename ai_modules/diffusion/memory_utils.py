"""
ai_modules/diffusion/memory_utils.py

GPU Memory Optimization Utilities for T4 GPU (15GB VRAM).

This module provides memory management functions optimized for running
SDXL + ControlNet on limited VRAM. Critical for Google Colab T4 environments.

Key Optimizations:
- FP16 precision (~50% VRAM reduction)
- CPU offloading (fits in 8GB with trade-off in speed)
- VAE slicing (~200MB savings)
- xFormers memory-efficient attention (~20-30% savings)
- Sequential CPU offload (most aggressive, slowest)

Research Sources:
- HuggingFace Diffusers optimization guide
- ByteDance SDXL-Lightning best practices
"""

import gc
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal
import torch


@dataclass
class MemoryConfig:
    """
    Configuration for memory optimization.
    
    Presets:
        - "t4_balanced": Good balance for T4 (recommended)
        - "t4_speed": Faster but uses more VRAM
        - "t4_minimal": Fits in <10GB VRAM but slower
        - "auto": Auto-detect based on available VRAM
    
    Example:
        config = MemoryConfig.from_preset("t4_balanced")
        setup_memory_optimization(pipe, config)
    """
    use_fp16: bool = True
    enable_xformers: bool = True
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = False
    cpu_offload_mode: Literal["none", "model", "sequential"] = "model"
    attention_slicing: Optional[int] = None  # None = auto, int = slice size
    
    @classmethod
    def from_preset(cls, preset: str) -> "MemoryConfig":
        """Create config from preset name."""
        presets = {
            "t4_balanced": cls(
                use_fp16=True,
                enable_xformers=True,
                enable_vae_slicing=True,
                enable_vae_tiling=False,
                cpu_offload_mode="model",
                attention_slicing=None,
            ),
            "t4_speed": cls(
                use_fp16=True,
                enable_xformers=True,
                enable_vae_slicing=False,
                enable_vae_tiling=False,
                cpu_offload_mode="none",
                attention_slicing=None,
            ),
            "t4_minimal": cls(
                use_fp16=True,
                enable_xformers=True,
                enable_vae_slicing=True,
                enable_vae_tiling=True,
                cpu_offload_mode="sequential",
                attention_slicing=1,
            ),
            "auto": cls._auto_config(),
        }
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Choose from: {list(presets.keys())}")
        return presets[preset]
    
    @classmethod
    def _auto_config(cls) -> "MemoryConfig":
        """Auto-configure based on available VRAM."""
        if not torch.cuda.is_available():
            # CPU-only: aggressive offloading
            return cls(
                use_fp16=False,
                enable_xformers=False,
                enable_vae_slicing=True,
                enable_vae_tiling=True,
                cpu_offload_mode="none",
                attention_slicing=1,
            )
        
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if vram_gb >= 24:  # A100, 4090
            return cls.from_preset("t4_speed")
        elif vram_gb >= 12:  # T4, 3080
            return cls.from_preset("t4_balanced")
        else:  # <12GB
            return cls.from_preset("t4_minimal")


def setup_memory_optimization(
    pipe: Any,
    config: Optional[MemoryConfig] = None,
    device: str = "cuda"
) -> Any:
    """
    Apply memory optimizations to a diffusion pipeline.
    
    Args:
        pipe: Diffusers pipeline (SDXL, ControlNet, etc.)
        config: Memory configuration (uses "t4_balanced" if None)
        device: Target device ("cuda" or "cpu")
    
    Returns:
        Optimized pipeline
    
    Example:
        from diffusers import StableDiffusionXLPipeline
        
        pipe = StableDiffusionXLPipeline.from_pretrained(...)
        config = MemoryConfig.from_preset("t4_balanced")
        pipe = setup_memory_optimization(pipe, config)
    """
    if config is None:
        config = MemoryConfig.from_preset("t4_balanced")
    
    print(f"Setting up memory optimization...")
    print(f"  FP16: {config.use_fp16}")
    print(f"  xFormers: {config.enable_xformers}")
    print(f"  VAE slicing: {config.enable_vae_slicing}")
    print(f"  CPU offload: {config.cpu_offload_mode}")
    
    # 1. Enable xFormers memory-efficient attention
    if config.enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("  ✓ xFormers enabled")
        except Exception as e:
            print(f"  ✗ xFormers not available: {e}")
    
    # 2. Enable VAE slicing (process in chunks)
    if config.enable_vae_slicing:
        try:
            pipe.enable_vae_slicing()
            print("  ✓ VAE slicing enabled")
        except AttributeError:
            pass
    
    # 3. Enable VAE tiling (for very large images)
    if config.enable_vae_tiling:
        try:
            pipe.enable_vae_tiling()
            print("  ✓ VAE tiling enabled")
        except AttributeError:
            pass
    
    # 4. Enable attention slicing
    if config.attention_slicing is not None:
        try:
            pipe.enable_attention_slicing(config.attention_slicing)
            print(f"  ✓ Attention slicing enabled (size={config.attention_slicing})")
        except AttributeError:
            pass
    
    # 5. CPU offloading (mutually exclusive modes)
    # NOTE: CPU offload must be applied BEFORE moving to device manually
    # The offload functions handle device placement internally
    if config.cpu_offload_mode == "sequential":
        # Most aggressive - slowest but fits in <8GB
        try:
            pipe.enable_sequential_cpu_offload()
            print("  ✓ Sequential CPU offload enabled")
        except AttributeError:
            print("  ✗ Sequential CPU offload not available for this pipeline")
    elif config.cpu_offload_mode == "model":
        # Balanced - offloads entire model between steps
        try:
            pipe.enable_model_cpu_offload()
            print("  ✓ Model CPU offload enabled")
        except AttributeError:
            print("  ✗ Model CPU offload not available for this pipeline")
    else:
        # No offloading - fastest, requires most VRAM
        if device == "cuda":
            pipe = pipe.to(device)
            print(f"  ✓ Pipeline moved to {device}")
    
    # Note: FP16 conversion is handled during model loading (torch_dtype=torch.float16)
    # Do NOT call pipe.to(torch.float16) after CPU offload as it breaks the offload mechanism

    return pipe


def get_memory_status() -> Dict[str, float]:
    """
    Get current GPU memory status.
    
    Returns:
        Dictionary with memory info in GB:
        - total: Total VRAM
        - allocated: Currently allocated
        - reserved: Reserved by PyTorch
        - free: Available VRAM
    
    Example:
        status = get_memory_status()
        print(f"Free VRAM: {status['free']:.1f}GB")
    """
    if not torch.cuda.is_available():
        return {"total": 0, "allocated": 0, "reserved": 0, "free": 0}
    
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    free = total - allocated
    
    return {
        "total": round(total, 2),
        "allocated": round(allocated, 2),
        "reserved": round(reserved, 2),
        "free": round(free, 2),
    }


def clear_gpu_memory() -> None:
    """
    Aggressively clear GPU memory.
    
    Call this between major operations or if you run into OOM errors.
    
    Example:
        # After finishing enhancement batch
        clear_gpu_memory()
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def estimate_vram_usage(
    model_name: str = "sdxl_lightning",
    config: Optional[MemoryConfig] = None,
    batch_size: int = 1,
    image_size: int = 1024
) -> float:
    """
    Estimate VRAM usage for a given configuration.
    
    Args:
        model_name: "sdxl_lightning", "sdxl_base", "controlnet"
        config: Memory configuration
        batch_size: Number of images
        image_size: Image dimension (assumed square)
    
    Returns:
        Estimated VRAM usage in GB
    
    Example:
        vram = estimate_vram_usage("sdxl_lightning", batch_size=1)
        print(f"Estimated VRAM: {vram:.1f}GB")
    """
    if config is None:
        config = MemoryConfig.from_preset("t4_balanced")
    
    # Base model sizes (approximate, FP16)
    base_sizes = {
        "sdxl_lightning": 5.5,  # Smaller due to optimized weights
        "sdxl_base": 6.5,
        "controlnet": 2.5,
        "vae": 0.4,
    }
    
    # Start with base model
    usage = base_sizes.get(model_name, 6.0)
    
    # Add VAE
    usage += base_sizes["vae"]
    
    # Add ControlNet if using SDXL
    if model_name.startswith("sdxl"):
        usage += base_sizes["controlnet"]
    
    # FP32 doubles the size
    if not config.use_fp16:
        usage *= 2
    
    # Batch size multiplier (not linear due to shared weights)
    usage += (batch_size - 1) * 0.5
    
    # Image size factor (larger images need more activation memory)
    size_factor = (image_size / 1024) ** 2
    usage += size_factor * 1.5
    
    # CPU offloading reduces effective VRAM
    if config.cpu_offload_mode == "sequential":
        usage *= 0.4
    elif config.cpu_offload_mode == "model":
        usage *= 0.7
    
    return round(usage, 1)


def print_memory_report() -> None:
    """Print a formatted memory report."""
    status = get_memory_status()
    
    print("\n" + "=" * 50)
    print("GPU Memory Report")
    print("=" * 50)
    
    if status["total"] == 0:
        print("No GPU available")
        return
    
    print(f"Total VRAM:     {status['total']:.1f} GB")
    print(f"Allocated:      {status['allocated']:.1f} GB")
    print(f"Reserved:       {status['reserved']:.1f} GB")
    print(f"Free:           {status['free']:.1f} GB")
    print("-" * 50)
    
    usage_pct = (status['allocated'] / status['total']) * 100
    bar_len = 30
    filled = int(bar_len * usage_pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"Usage: [{bar}] {usage_pct:.1f}%")
    print("=" * 50 + "\n")
