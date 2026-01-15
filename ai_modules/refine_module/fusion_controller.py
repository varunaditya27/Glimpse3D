"""
ai_modules/refine_module/fusion_controller.py

★ MVCRM ORCHESTRATOR ★
Controller for the Multi-View Consistency Refinement Module.

This is the main entry point for the refinement system. It coordinates all
components (back-projection, depth consistency, feature consistency, smoothing)
to iteratively improve a 3D Gaussian Splat model from enhanced 2D views.

Responsibilities:
- Orchestrate the fusion of enhanced 2D views back into the 3D model
- Balance contributions from different views
- Manage the iterative update loop with convergence detection
- Quality control and rollback capabilities

Author: Glimpse3D Team
Date: January 2026
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import logging

from .back_projector import BackProjector, CameraParams
from .depth_consistency import DepthConsistencyChecker
from .feature_consistency import FeatureConsistencyChecker
from .normal_smoothing import NormalSmoother


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ViewData:
    """Data for a single view to be fused."""
    enhanced_image: torch.Tensor  # (H, W, 3) enhanced RGB
    rendered_image: torch.Tensor  # (H, W, 3) current render
    depth_map: torch.Tensor  # (H, W) depth
    camera: CameraParams  # Camera parameters
    confidence: float = 1.0  # View quality weight


@dataclass
class RefinementConfig:
    """Configuration for refinement process."""
    max_iterations: int = 5
    learning_rate: float = 0.05
    lr_decay: float = 0.8
    min_improvement: float = 0.01
    depth_consistency_threshold: float = 0.1
    feature_similarity_threshold: float = 0.7
    smoothing_strength: float = 0.3
    enable_depth_check: bool = True
    enable_feature_check: bool = True
    enable_smoothing: bool = True
    early_stopping: bool = True
    max_decline_steps: int = 2


@dataclass
class RefinementResult:
    """Results from refinement process."""
    refined_positions: torch.Tensor
    refined_colors: torch.Tensor
    refined_opacities: torch.Tensor
    refined_scales: torch.Tensor
    iterations_run: int
    converged: bool
    quality_metrics: Dict[str, float]
    history: List[Dict[str, float]]


class FusionController:
    """
    Main controller for the MVCRM refinement system.
    
    Coordinates iterative refinement of a 3D Gaussian Splat model by:
    1. Rendering views from current model state
    2. Comparing with enhanced views (from SDXL)
    3. Back-projecting differences with consistency checks
    4. Smoothing geometry to prevent artifacts
    5. Repeating until convergence
    
    Usage:
        controller = FusionController(config)
        result = controller.refine(
            splat_model=gs_model,
            views=[view1, view2, ...],
            render_fn=my_renderer
        )
    """
    
    def __init__(
        self,
        config: Optional[RefinementConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize fusion controller.
        
        Args:
            config: Refinement configuration (uses defaults if None)
            device: Computing device
        """
        self.config = config or RefinementConfig()
        self.device = torch.device(device)
        
        # Initialize components
        self.back_projector = BackProjector(
            learning_rate=self.config.learning_rate,
            device=device
        )
        
        if self.config.enable_depth_check:
            self.depth_checker = DepthConsistencyChecker(
                base_threshold=self.config.depth_consistency_threshold,
                device=device
            )
        
        if self.config.enable_feature_check:
            self.feature_checker = FeatureConsistencyChecker(
                similarity_threshold=self.config.feature_similarity_threshold,
                device=device
            )
        
        if self.config.enable_smoothing:
            self.smoother = NormalSmoother(
                smoothing_strength=self.config.smoothing_strength,
                device=device
            )
        
        logger.info("FusionController initialized with config:")
        logger.info(f"  Max iterations: {self.config.max_iterations}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  Depth check: {self.config.enable_depth_check}")
        logger.info(f"  Feature check: {self.config.enable_feature_check}")
    
    def refine(
        self,
        splat_positions: torch.Tensor,
        splat_colors: torch.Tensor,
        splat_opacities: torch.Tensor,
        splat_scales: torch.Tensor,
        views: List[ViewData],
        render_fn: Callable,
        splat_contributions_fn: Optional[Callable] = None
    ) -> RefinementResult:
        """
        Refine 3D Gaussian Splat model using enhanced views.
        
        Args:
            splat_positions: (N, 3) splat 3D positions
            splat_colors: (N, 3) splat RGB colors
            splat_opacities: (N,) splat opacities
            splat_scales: (N, 3) splat scales
            views: List of ViewData with enhanced views to fuse
            render_fn: Function that renders a view given camera params
                      Signature: render_fn(positions, colors, opacities, scales, camera)
                      Returns: (image, depth, alpha_map, contributions)
            splat_contributions_fn: Optional function to get splat contributions
        
        Returns:
            RefinementResult with refined parameters and metrics
        """
        # Move to device
        positions = self._to_device(splat_positions).clone()
        colors = self._to_device(splat_colors).clone()
        opacities = self._to_device(splat_opacities).clone()
        scales = self._to_device(splat_scales).clone()
        
        # Store initial state for rollback
        best_state = {
            'positions': positions.clone(),
            'colors': colors.clone(),
            'opacities': opacities.clone(),
            'scales': scales.clone(),
            'quality': 0.0
        }
        
        history = []
        decline_count = 0
        
        # Iterative refinement loop
        for iteration in range(self.config.max_iterations):
            logger.info(f"\n--- Iteration {iteration + 1}/{self.config.max_iterations} ---")
            
            # Update learning rate (annealing)
            current_lr = self.config.learning_rate * (self.config.lr_decay ** iteration)
            self.back_projector.lr = current_lr
            
            iteration_metrics = {
                'iteration': iteration,
                'learning_rate': current_lr,
                'views_processed': 0,
                'avg_depth_consistency': 0.0,
                'avg_feature_similarity': 0.0,
                'total_updates': 0
            }
            
            # Process each view
            for view_idx, view in enumerate(views):
                logger.info(f"  Processing view {view_idx + 1}/{len(views)}")
                
                # Render current state
                try:
                    rendered, depth_rendered, alpha_map, contributions = render_fn(
                        positions, colors, opacities, scales, view.camera
                    )
                except Exception as e:
                    logger.warning(f"  Render failed: {e}, skipping view")
                    continue
                
                # 1. Check depth consistency
                depth_mask = None
                if self.config.enable_depth_check:
                    depth_result = self.depth_checker.check(
                        depth_rendered, view.depth_map
                    )
                    depth_mask = depth_result.consistency_mask
                    iteration_metrics['avg_depth_consistency'] += depth_result.consistency_score
                    
                    if not self.depth_checker.is_valid_update(depth_result):
                        logger.warning(f"  View {view_idx} failed depth check, skipping")
                        continue
                
                # 2. Check feature consistency
                feature_mask = None
                if self.config.enable_feature_check:
                    feature_result = self.feature_checker.check(
                        rendered, view.enhanced_image
                    )
                    feature_mask = feature_result.consistency_mask
                    iteration_metrics['avg_feature_similarity'] += feature_result.similarity_score
                    
                    if not self.feature_checker.is_valid_update(feature_result):
                        logger.warning(f"  View {view_idx} failed feature check, skipping")
                        continue
                
                # 3. Combine masks
                consistency_mask = self._combine_masks(depth_mask, feature_mask)
                
                # 4. Back-project updates
                try:
                    bp_result = self.back_projector.project(
                        rendered_image=rendered,
                        enhanced_image=view.enhanced_image,
                        depth_map=depth_rendered,
                        splat_means=positions,
                        splat_colors=colors,
                        splat_opacities=opacities,
                        alpha_map=alpha_map,
                        splat_contributions=contributions,
                        camera=view.camera,
                        consistency_mask=consistency_mask
                    )
                    
                    # 5. Apply updates with view confidence weighting
                    colors, opacities, scales = self.back_projector.apply_updates(
                        colors, opacities, scales, bp_result, damping=view.confidence
                    )
                    
                    iteration_metrics['views_processed'] += 1
                    iteration_metrics['total_updates'] += len(bp_result.updated_splat_indices)
                    
                except Exception as e:
                    logger.warning(f"  Back-projection failed: {e}, skipping view")
                    continue
            
            # 6. Apply geometry smoothing
            if self.config.enable_smoothing and iteration_metrics['views_processed'] > 0:
                logger.info("  Applying geometry smoothing...")
                positions = self.smoother.smooth_positions(positions, colors)
                scales = self.smoother.regularize_scales(scales, positions)
            
            # 7. Compute iteration quality
            if iteration_metrics['views_processed'] > 0:
                iteration_metrics['avg_depth_consistency'] /= iteration_metrics['views_processed']
                iteration_metrics['avg_feature_similarity'] /= iteration_metrics['views_processed']
            
            overall_quality = (
                0.5 * iteration_metrics['avg_depth_consistency'] +
                0.5 * iteration_metrics['avg_feature_similarity']
            )
            iteration_metrics['overall_quality'] = overall_quality
            
            history.append(iteration_metrics)
            
            logger.info(f"  Quality: {overall_quality:.3f}")
            logger.info(f"  Updates: {iteration_metrics['total_updates']}")
            
            # 8. Check for improvement
            if overall_quality > best_state['quality']:
                best_state = {
                    'positions': positions.clone(),
                    'colors': colors.clone(),
                    'opacities': opacities.clone(),
                    'scales': scales.clone(),
                    'quality': overall_quality
                }
                decline_count = 0
            else:
                decline_count += 1
            
            # 9. Early stopping
            if self.config.early_stopping:
                if decline_count >= self.config.max_decline_steps:
                    logger.info(f"  Early stopping: quality declined for {decline_count} steps")
                    break
                
                if iteration > 0 and abs(overall_quality - history[-2]['overall_quality']) < self.config.min_improvement:
                    logger.info(f"  Converged: improvement < {self.config.min_improvement}")
                    break
        
        # Use best state
        logger.info(f"\nRefinement complete. Best quality: {best_state['quality']:.3f}")
        
        return RefinementResult(
            refined_positions=best_state['positions'],
            refined_colors=best_state['colors'],
            refined_opacities=best_state['opacities'],
            refined_scales=best_state['scales'],
            iterations_run=len(history),
            converged=(decline_count == 0),
            quality_metrics={
                'final_quality': best_state['quality'],
                'final_depth_consistency': history[-1]['avg_depth_consistency'] if history else 0.0,
                'final_feature_similarity': history[-1]['avg_feature_similarity'] if history else 0.0
            },
            history=history
        )
    
    def fuse_views(
        self,
        model: Dict[str, torch.Tensor],
        views: List[ViewData],
        render_fn: Callable
    ) -> Dict[str, torch.Tensor]:
        """
        Simplified interface for view fusion.
        
        Args:
            model: Dictionary with keys ['positions', 'colors', 'opacities', 'scales']
            views: List of enhanced views
            render_fn: Rendering function
        
        Returns:
            refined_model: Dictionary with refined parameters
        """
        result = self.refine(
            splat_positions=model['positions'],
            splat_colors=model['colors'],
            splat_opacities=model['opacities'],
            splat_scales=model['scales'],
            views=views,
            render_fn=render_fn
        )
        
        return {
            'positions': result.refined_positions,
            'colors': result.refined_colors,
            'opacities': result.refined_opacities,
            'scales': result.refined_scales,
            'metrics': result.quality_metrics
        }
    
    def _combine_masks(
        self,
        mask1: Optional[torch.Tensor],
        mask2: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Combine multiple consistency masks with logical AND."""
        if mask1 is None and mask2 is None:
            return None
        if mask1 is None:
            return mask2
        if mask2 is None:
            return mask1
        return mask1 & mask2
    
    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on correct device."""
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.from_numpy(tensor).float()
        return tensor.to(self.device)


def create_simple_refinement_config(
    quality: str = "balanced"
) -> RefinementConfig:
    """
    Create preset refinement configurations.
    
    Args:
        quality: "fast", "balanced", or "high_quality"
    
    Returns:
        RefinementConfig with preset parameters
    """
    if quality == "fast":
        return RefinementConfig(
            max_iterations=3,
            learning_rate=0.08,
            depth_consistency_threshold=0.15,
            feature_similarity_threshold=0.6,
            smoothing_strength=0.2,
            enable_feature_check=False
        )
    elif quality == "high_quality":
        return RefinementConfig(
            max_iterations=8,
            learning_rate=0.03,
            depth_consistency_threshold=0.05,
            feature_similarity_threshold=0.8,
            smoothing_strength=0.4,
            enable_depth_check=True,
            enable_feature_check=True,
            enable_smoothing=True
        )
    else:  # balanced
        return RefinementConfig()
