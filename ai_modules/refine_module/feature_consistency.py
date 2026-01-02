"""
ai_modules/refine_module/feature_consistency.py

Feature Consistency Validation for MVCRM.

★ SEMANTIC VALIDATION ★
Ensures that refined views maintain semantic consistency with the original
content using deep feature matching (DINOv2, CLIP, LPIPS).

Responsibilities:
- Extract deep features from rendered and enhanced views
- Compute semantic similarity scores
- Detect semantic drift and hallucinations
- Provide confidence weights for fusion

Author: Glimpse3D Team
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class FeatureConsistencyResult:
    """Results from feature consistency check."""
    consistency_mask: torch.Tensor  # (H, W) valid regions
    similarity_score: float  # Overall similarity [0, 1]
    feature_distance: torch.Tensor  # (H, W) per-pixel distance
    semantic_drift: float  # Measure of hallucination risk


class FeatureConsistencyChecker:
    """
    Validates semantic consistency between rendered and enhanced views.
    
    Uses pre-trained deep networks to detect semantic inconsistencies that
    might arise from aggressive SDXL enhancement (hallucinations, style drift).
    
    Supports multiple feature extractors:
    - DINOv2: Self-supervised vision transformer (recommended)
    - CLIP: Contrastive language-image pretraining
    - LPIPS: Learned perceptual image patch similarity
    
    Usage:
        checker = FeatureConsistencyChecker(model_type="dinov2")
        result = checker.check(rendered_image, enhanced_image)
        if result.similarity_score > 0.7:
            # Safe to apply updates
    """
    
    def __init__(
        self,
        model_type: str = "dinov2",
        similarity_threshold: float = 0.7,
        patch_size: int = 14,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize feature consistency checker.
        
        Args:
            model_type: "dinov2", "clip", or "lpips"
            similarity_threshold: Minimum similarity to accept (0.6-0.8)
            patch_size: Patch size for feature extraction
            device: Computing device
        """
        self.model_type = model_type
        self.threshold = similarity_threshold
        self.patch_size = patch_size
        self.device = torch.device(device)
        
        # Lazy load models
        self._model = None
        self._preprocess = None
    
    @property
    def model(self):
        """Lazy-load feature extractor."""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self):
        """Load the selected feature extraction model."""
        print(f"Loading {self.model_type} feature extractor...")
        
        if self.model_type == "dinov2":
            self._load_dinov2()
        elif self.model_type == "clip":
            self._load_clip()
        elif self.model_type == "lpips":
            self._load_lpips()
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        
        self._model.eval()
        self._model.to(self.device)
        
        # Freeze parameters
        for param in self._model.parameters():
            param.requires_grad = False
        
        print(f"{self.model_type} loaded successfully!")
    
    def _load_dinov2(self):
        """Load DINOv2 model (recommended)."""
        try:
            # Try to load from transformers
            from transformers import AutoModel, AutoImageProcessor
            
            model_name = "facebook/dinov2-base"
            self._model = AutoModel.from_pretrained(model_name)
            self._preprocess = AutoImageProcessor.from_pretrained(model_name)
            
        except ImportError:
            # Fallback: load from torch hub
            self._model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self._preprocess = self._default_preprocess
    
    def _load_clip(self):
        """Load CLIP model."""
        try:
            import clip
            
            self._model, self._preprocess = clip.load("ViT-B/32", device=self.device)
            
        except ImportError:
            print("WARNING: CLIP not installed. Using fallback features.")
            self._model = SimpleCNN().to(self.device)
            self._preprocess = self._default_preprocess
    
    def _load_lpips(self):
        """Load LPIPS perceptual loss model."""
        try:
            import lpips
            
            self._model = lpips.LPIPS(net='alex')  # or 'vgg'
            self._preprocess = self._default_preprocess
            
        except ImportError:
            print("WARNING: LPIPS not installed. Using fallback features.")
            self._model = SimpleCNN().to(self.device)
            self._preprocess = self._default_preprocess
    
    def _default_preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """Default preprocessing: normalize to [-1, 1]."""
        return (image - 0.5) / 0.5
    
    def check(
        self,
        rendered_image: torch.Tensor,
        enhanced_image: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> FeatureConsistencyResult:
        """
        Check feature consistency between rendered and enhanced views.
        
        Args:
            rendered_image: (H, W, 3) or (3, H, W) rendered view
            enhanced_image: (H, W, 3) or (3, H, W) enhanced view
            mask: (H, W) optional region mask
        
        Returns:
            FeatureConsistencyResult with similarity metrics
        """
        # Ensure correct format: (1, 3, H, W)
        rendered = self._prepare_image(rendered_image)
        enhanced = self._prepare_image(enhanced_image)
        
        # Extract features
        with torch.no_grad():
            feat_rendered = self._extract_features(rendered)
            feat_enhanced = self._extract_features(enhanced)
        
        # Compute similarity
        if self.model_type == "lpips":
            # LPIPS returns distance (lower is better)
            distance = self._model(rendered, enhanced).squeeze()
            similarity = 1.0 - distance.mean().item()
            feature_distance = distance
        else:
            # Cosine similarity (higher is better)
            similarity, feature_distance = self._compute_similarity(
                feat_rendered, feat_enhanced
            )
        
        # Generate consistency mask
        H, W = rendered.shape[2:]
        
        if len(feature_distance.shape) == 0:  # Scalar
            # Global similarity only
            consistency_mask = torch.ones(H, W, device=self.device, dtype=torch.bool)
            feature_distance = torch.full((H, W), feature_distance.item(), device=self.device)
        else:
            # Spatial feature distance
            if feature_distance.shape[-2:] != (H, W):
                # Resize to match image size
                feature_distance = F.interpolate(
                    feature_distance.unsqueeze(0).unsqueeze(0),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
            
            # Threshold to create mask
            threshold = 1.0 - self.threshold
            consistency_mask = feature_distance < threshold
        
        # Apply user mask if provided
        if mask is not None:
            mask = self._to_device(mask)
            consistency_mask = consistency_mask & mask
        
        # Compute semantic drift (variance in feature space)
        semantic_drift = self._compute_drift(feat_rendered, feat_enhanced)
        
        return FeatureConsistencyResult(
            consistency_mask=consistency_mask,
            similarity_score=similarity,
            feature_distance=feature_distance,
            semantic_drift=semantic_drift
        )
    
    def _extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract deep features from image.
        
        Args:
            image: (1, 3, H, W) input image
        
        Returns:
            features: Feature tensor (shape depends on model)
        """
        if self.model_type == "dinov2":
            # DINOv2 returns patch tokens
            outputs = self.model(image)
            features = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs
            
        elif self.model_type == "clip":
            # CLIP image encoder
            features = self.model.encode_image(image)
            
        elif self.model_type == "lpips":
            # LPIPS extracts features internally
            features = image  # Will be processed in distance computation
            
        else:
            # Fallback CNN
            features = self.model(image)
        
        return features
    
    def _compute_similarity(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor
    ) -> Tuple[float, torch.Tensor]:
        """
        Compute cosine similarity between feature tensors.
        
        Returns:
            global_similarity: Average similarity score
            spatial_distance: Per-location distance map
        """
        # Flatten spatial dimensions if present
        if len(feat1.shape) == 4:  # (B, C, H, W)
            B, C, H, W = feat1.shape
            feat1_flat = feat1.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
            feat2_flat = feat2.view(B, C, -1).permute(0, 2, 1)
            
            # Compute per-patch similarity
            feat1_norm = F.normalize(feat1_flat, dim=-1)
            feat2_norm = F.normalize(feat2_flat, dim=-1)
            
            similarity = (feat1_norm * feat2_norm).sum(dim=-1)  # (B, H*W)
            similarity = similarity.view(B, H, W)
            
            global_sim = similarity.mean().item()
            spatial_distance = 1.0 - similarity.squeeze(0)
            
        elif len(feat1.shape) == 3:  # (B, N, C) - sequence of patches
            feat1_norm = F.normalize(feat1, dim=-1)
            feat2_norm = F.normalize(feat2, dim=-1)
            
            similarity = (feat1_norm * feat2_norm).sum(dim=-1)  # (B, N)
            global_sim = similarity.mean().item()
            spatial_distance = 1.0 - similarity.mean(dim=1)
            
        else:  # (B, C) - global features
            feat1_norm = F.normalize(feat1, dim=-1)
            feat2_norm = F.normalize(feat2, dim=-1)
            
            global_sim = (feat1_norm * feat2_norm).sum().item()
            spatial_distance = torch.tensor(1.0 - global_sim, device=self.device)
        
        return global_sim, spatial_distance
    
    def _compute_drift(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor
    ) -> float:
        """
        Compute semantic drift (how much content has changed).
        
        Uses variance in feature space as proxy for drift.
        """
        # Compute L2 distance in feature space
        dist = torch.norm(feat1 - feat2, dim=-1 if len(feat1.shape) > 2 else -1)
        drift = dist.mean().item()
        
        # Normalize to [0, 1]
        drift = min(1.0, drift / 2.0)
        
        return drift
    
    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        """Prepare image for feature extraction."""
        image = self._to_device(image)
        
        # Convert to (1, 3, H, W) if needed
        if len(image.shape) == 3:
            if image.shape[0] == 3:  # (3, H, W)
                image = image.unsqueeze(0)
            else:  # (H, W, 3)
                image = image.permute(2, 0, 1).unsqueeze(0)
        elif len(image.shape) == 4:
            pass  # Already correct
        
        # Ensure values in [0, 1]
        if image.max() > 1.1:
            image = image / 255.0
        
        return image
    
    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on correct device."""
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.from_numpy(tensor).float()
        return tensor.to(self.device)
    
    def is_valid_update(self, result: FeatureConsistencyResult) -> bool:
        """
        Check if features are consistent enough to proceed with update.
        
        Args:
            result: FeatureConsistencyResult from check()
        
        Returns:
            True if semantically consistent, False otherwise
        """
        return (
            result.similarity_score >= self.threshold and
            result.semantic_drift < 0.5
        )


class SimpleCNN(nn.Module):
    """Fallback CNN for feature extraction if other models unavailable."""
    
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
    def forward(self, x):
        return self.features(x).squeeze(-1).squeeze(-1)


def compute_clip_similarity(
    image1: torch.Tensor,
    image2: torch.Tensor,
    device: str = "cuda"
) -> float:
    """
    Quick utility to compute CLIP similarity between two images.
    
    Args:
        image1, image2: (H, W, 3) or (3, H, W) images
        device: Computing device
    
    Returns:
        similarity: Cosine similarity [0, 1]
    """
    checker = FeatureConsistencyChecker(model_type="clip", device=device)
    result = checker.check(image1, image2)
    return result.similarity_score
