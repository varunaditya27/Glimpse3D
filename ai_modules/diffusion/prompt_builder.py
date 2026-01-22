"""
ai_modules/diffusion/prompt_builder.py

Dynamic Prompt Builder for SDXL Enhancement.

Provides intelligent prompt construction for 3D view enhancement.
Templates are optimized for rendered 3D content enhancement.
"""

import os
from typing import Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class PromptConfig:
    """Configuration for prompt building."""
    base_prompt: str = ""
    style_modifiers: List[str] = None
    quality_modifiers: List[str] = None
    negative_prompt: str = ""
    
    def __post_init__(self):
        if self.style_modifiers is None:
            self.style_modifiers = []
        if self.quality_modifiers is None:
            self.quality_modifiers = []


# Default prompt templates for different use cases
DEFAULT_TEMPLATES: Dict[str, PromptConfig] = {
    "default": PromptConfig(
        base_prompt="high quality render",
        quality_modifiers=["8k resolution", "detailed texture", "sharp focus"],
        negative_prompt="blurry, low quality, distorted, artifacts, noise"
    ),
    "photorealistic": PromptConfig(
        base_prompt="photorealistic 3D render",
        quality_modifiers=["8k uhd", "hyperrealistic", "detailed texture", "ray tracing"],
        style_modifiers=["professional lighting", "accurate shadows"],
        negative_prompt="cartoon, anime, painting, blur, noise, artifacts, deformed"
    ),
    "product": PromptConfig(
        base_prompt="product photography, studio render",
        quality_modifiers=["8k resolution", "clean", "sharp"],
        style_modifiers=["studio lighting", "neutral background", "commercial quality"],
        negative_prompt="blurry, dark, noisy, cluttered background, unprofessional"
    ),
    "character": PromptConfig(
        base_prompt="detailed 3D character render",
        quality_modifiers=["8k resolution", "detailed skin texture"],
        style_modifiers=["subsurface scattering", "realistic eyes", "natural pose"],
        negative_prompt="uncanny valley, plastic skin, dead eyes, deformed, mutation"
    ),
    "environment": PromptConfig(
        base_prompt="detailed environment render",
        quality_modifiers=["8k resolution", "detailed textures"],
        style_modifiers=["atmospheric lighting", "natural colors", "depth of field"],
        negative_prompt="flat lighting, low detail, artificial, noisy, artifacts"
    ),
    "object": PromptConfig(
        base_prompt="3D object render",
        quality_modifiers=["8k resolution", "clean geometry", "detailed surface"],
        style_modifiers=["studio lighting", "accurate materials", "physically based rendering"],
        negative_prompt="blurry, noisy, incorrect shadows, floating artifacts"
    ),
    "anime_style": PromptConfig(
        base_prompt="anime style 3D render",
        quality_modifiers=["high resolution", "clean lines", "vibrant colors"],
        style_modifiers=["cel shading", "stylized lighting"],
        negative_prompt="photorealistic, blurry, muddy colors, inconsistent style"
    ),
}


class PromptBuilder:
    """
    Intelligent prompt builder for SDXL enhancement.
    
    Constructs optimized prompts based on template + user customization.
    
    Example:
        builder = PromptBuilder(template="photorealistic")
        prompt, negative = builder.build(
            subject="a ceramic vase with flowers",
            extra_modifiers=["soft shadows"]
        )
    """
    
    def __init__(
        self,
        template: str = "default",
        templates_file: Optional[str] = None
    ):
        """
        Initialize prompt builder.
        
        Args:
            template: Template name from DEFAULT_TEMPLATES or custom templates
            templates_file: Optional path to custom templates file
        """
        self.templates = DEFAULT_TEMPLATES.copy()
        
        if templates_file and os.path.exists(templates_file):
            self._load_custom_templates(templates_file)
        
        if template not in self.templates:
            available = list(self.templates.keys())
            raise ValueError(f"Unknown template: {template}. Available: {available}")
        
        self.current_template = template
        self.config = self.templates[template]
    
    def _load_custom_templates(self, filepath: str) -> None:
        """Load templates from text file (prompt_templates.txt format)."""
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if ':' in line:
                        name, prompt = line.split(':', 1)
                        name = name.strip()
                        prompt = prompt.strip().strip('"')
                        
                        self.templates[name] = PromptConfig(
                            base_prompt=prompt,
                            quality_modifiers=["high quality", "detailed"],
                            negative_prompt="blurry, low quality, artifacts"
                        )
        except Exception as e:
            print(f"Warning: Could not load templates from {filepath}: {e}")
    
    def build(
        self,
        subject: Optional[str] = None,
        extra_modifiers: Optional[List[str]] = None,
        override_negative: Optional[str] = None,
        include_quality: bool = True,
        include_style: bool = True
    ) -> tuple:
        """
        Build complete prompt and negative prompt.
        
        Args:
            subject: Optional subject description to prepend
            extra_modifiers: Additional modifiers to include
            override_negative: Override default negative prompt
            include_quality: Include quality modifiers
            include_style: Include style modifiers
        
        Returns:
            (prompt, negative_prompt) tuple
        """
        parts = []
        
        # Subject first (if provided)
        if subject:
            parts.append(subject)
        
        # Base prompt
        parts.append(self.config.base_prompt)
        
        # Quality modifiers
        if include_quality and self.config.quality_modifiers:
            parts.extend(self.config.quality_modifiers)
        
        # Style modifiers
        if include_style and self.config.style_modifiers:
            parts.extend(self.config.style_modifiers)
        
        # Extra modifiers
        if extra_modifiers:
            parts.extend(extra_modifiers)
        
        prompt = ", ".join(parts)
        negative = override_negative or self.config.negative_prompt
        
        return prompt, negative
    
    def set_template(self, template: str) -> None:
        """Switch to a different template."""
        if template not in self.templates:
            raise ValueError(f"Unknown template: {template}")
        self.current_template = template
        self.config = self.templates[template]
    
    def list_templates(self) -> List[str]:
        """Get list of available template names."""
        return list(self.templates.keys())
    
    def get_template_info(self, template: str) -> Dict:
        """Get details about a specific template."""
        if template not in self.templates:
            raise ValueError(f"Unknown template: {template}")
        
        config = self.templates[template]
        return {
            "name": template,
            "base_prompt": config.base_prompt,
            "quality_modifiers": config.quality_modifiers,
            "style_modifiers": config.style_modifiers,
            "negative_prompt": config.negative_prompt,
        }


def build_prompt(
    template: str = "default",
    subject: Optional[str] = None,
    extra_modifiers: Optional[List[str]] = None,
    templates_file: Optional[str] = None
) -> tuple:
    """
    Convenience function to build a prompt.
    
    Args:
        template: Template name
        subject: Optional subject description
        extra_modifiers: Additional modifiers
        templates_file: Optional custom templates file
    
    Returns:
        (prompt, negative_prompt) tuple
    
    Example:
        prompt, negative = build_prompt(
            template="photorealistic",
            subject="a 3D rendered car",
            extra_modifiers=["metallic paint", "reflections"]
        )
    """
    builder = PromptBuilder(template=template, templates_file=templates_file)
    return builder.build(subject=subject, extra_modifiers=extra_modifiers)


def load_prompt_templates(filepath: str) -> Dict[str, PromptConfig]:
    """
    Load templates from a file.
    
    Args:
        filepath: Path to templates file
    
    Returns:
        Dictionary of template name -> PromptConfig
    """
    builder = PromptBuilder(templates_file=filepath)
    return builder.templates


# Pre-built prompts for common 3D enhancement scenarios
COMMON_PROMPTS = {
    "enhance_render": (
        "high quality 3D render, detailed texture, accurate lighting, sharp focus, 8k resolution",
        "blurry, noisy, artifacts, low quality, distorted geometry"
    ),
    "fix_artifacts": (
        "clean 3D render, smooth surfaces, accurate geometry, professional quality",
        "artifacts, floating geometry, holes, missing textures, noise, blur"
    ),
    "add_detail": (
        "highly detailed 3D render, fine texture details, intricate surface, 8k uhd",
        "flat, low detail, blurry, simple, basic textures"
    ),
    "improve_lighting": (
        "professionally lit 3D render, accurate shadows, natural lighting, global illumination",
        "flat lighting, harsh shadows, overexposed, underexposed, artificial"
    ),
}


def get_common_prompt(name: str) -> tuple:
    """
    Get a pre-built common prompt.
    
    Args:
        name: One of "enhance_render", "fix_artifacts", "add_detail", "improve_lighting"
    
    Returns:
        (prompt, negative_prompt) tuple
    """
    if name not in COMMON_PROMPTS:
        available = list(COMMON_PROMPTS.keys())
        raise ValueError(f"Unknown common prompt: {name}. Available: {available}")
    return COMMON_PROMPTS[name]
