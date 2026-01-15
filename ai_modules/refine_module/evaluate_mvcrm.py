"""
ai_modules/refine_module/evaluate_mvcrm.py

Evaluation and Metrics for MVCRM Refinement Module.

★ RESEARCH & VALIDATION ★
Provides comprehensive metrics to evaluate the quality of refined 3D models
for research papers, ablation studies, and quality assurance.

Responsibilities:
- Visual quality metrics (PSNR, SSIM, LPIPS)
- Multi-view consistency metrics
- Geometric quality metrics
- Comparison against baseline models
- Generate evaluation reports

Author: Glimpse3D Team
Date: January 2026
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    # Visual Quality
    psnr: float
    ssim: float
    lpips: float
    
    # Multi-View Consistency
    depth_variance: float
    cross_view_similarity: float
    
    # Geometric Quality
    normal_smoothness: float
    scale_consistency: float
    
    # Comparison (if baseline provided)
    psnr_improvement: Optional[float] = None
    ssim_improvement: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, path: str):
        """Save metrics to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class MVCRMEvaluator:
    """
    Comprehensive evaluator for refined 3D models.
    
    Computes visual quality, consistency, and geometric metrics to assess
    the effectiveness of the MVCRM refinement module.
    
    Usage:
        evaluator = MVCRMEvaluator()
        metrics = evaluator.evaluate(
            refined_model, ground_truth_images, camera_poses
        )
        print(f"PSNR: {metrics.psnr:.2f} dB")
    """
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize evaluator."""
        self.device = torch.device(device)
        self._lpips_model = None
    
    @property
    def lpips_model(self):
        """Lazy-load LPIPS model."""
        if self._lpips_model is None:
            try:
                import lpips
                self._lpips_model = lpips.LPIPS(net='alex').to(self.device)
                self._lpips_model.eval()
            except ImportError:
                print("WARNING: LPIPS not installed, will use simplified metric")
                self._lpips_model = "unavailable"
        return self._lpips_model
    
    def evaluate(
        self,
        model: Dict[str, torch.Tensor],
        render_fn: callable,
        ground_truth_images: List[torch.Tensor],
        camera_poses: List,
        baseline_model: Optional[Dict[str, torch.Tensor]] = None
    ) -> EvaluationMetrics:
        """
        Comprehensive evaluation of refined model.
        
        Args:
            model: Dictionary with refined splat parameters
            render_fn: Function to render views
            ground_truth_images: List of target images
            camera_poses: List of camera parameters
            baseline_model: Optional baseline for comparison
        
        Returns:
            EvaluationMetrics with all computed metrics
        """
        # Render views from refined model
        rendered_images = self._render_views(model, render_fn, camera_poses)
        
        # Visual quality metrics
        psnr = self.compute_psnr(rendered_images, ground_truth_images)
        ssim = self.compute_ssim(rendered_images, ground_truth_images)
        lpips_score = self.compute_lpips(rendered_images, ground_truth_images)
        
        # Multi-view consistency
        depth_var = self.compute_depth_variance(model, render_fn, camera_poses)
        cross_sim = self.compute_cross_view_similarity(rendered_images)
        
        # Geometric quality
        normal_smooth = self.compute_normal_smoothness(model['positions'])
        scale_cons = self.compute_scale_consistency(model['scales'])
        
        # Comparison with baseline
        psnr_imp = None
        ssim_imp = None
        if baseline_model is not None:
            baseline_rendered = self._render_views(baseline_model, render_fn, camera_poses)
            baseline_psnr = self.compute_psnr(baseline_rendered, ground_truth_images)
            baseline_ssim = self.compute_ssim(baseline_rendered, ground_truth_images)
            psnr_imp = psnr - baseline_psnr
            ssim_imp = ssim - baseline_ssim
        
        return EvaluationMetrics(
            psnr=psnr,
            ssim=ssim,
            lpips=lpips_score,
            depth_variance=depth_var,
            cross_view_similarity=cross_sim,
            normal_smoothness=normal_smooth,
            scale_consistency=scale_cons,
            psnr_improvement=psnr_imp,
            ssim_improvement=ssim_imp
        )
    
    def compute_psnr(
        self,
        pred_images: List[torch.Tensor],
        gt_images: List[torch.Tensor]
    ) -> float:
        """
        Compute Peak Signal-to-Noise Ratio.
        
        Args:
            pred_images: List of predicted images (H, W, 3)
            gt_images: List of ground truth images (H, W, 3)
        
        Returns:
            Average PSNR in dB
        """
        psnrs = []
        
        for pred, gt in zip(pred_images, gt_images):
            pred = self._to_device(pred)
            gt = self._to_device(gt)
            
            # Ensure range [0, 1]
            pred = torch.clamp(pred, 0, 1)
            gt = torch.clamp(gt, 0, 1)
            
            mse = torch.mean((pred - gt) ** 2)
            
            if mse < 1e-10:
                psnr = 100.0  # Perfect match
            else:
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            
            psnrs.append(psnr.item())
        
        return np.mean(psnrs)
    
    def compute_ssim(
        self,
        pred_images: List[torch.Tensor],
        gt_images: List[torch.Tensor],
        window_size: int = 11
    ) -> float:
        """
        Compute Structural Similarity Index.
        
        Args:
            pred_images: List of predicted images
            gt_images: List of ground truth images
            window_size: SSIM window size
        
        Returns:
            Average SSIM [0, 1]
        """
        ssims = []
        
        for pred, gt in zip(pred_images, gt_images):
            pred = self._prepare_for_ssim(pred)
            gt = self._prepare_for_ssim(gt)
            
            ssim_val = self._ssim(pred, gt, window_size)
            ssims.append(ssim_val.item())
        
        return np.mean(ssims)
    
    def _ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int = 11
    ) -> torch.Tensor:
        """Compute SSIM between two images."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Create Gaussian window
        sigma = 1.5
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2 / (2*sigma**2))
            for x in range(window_size)
        ]).to(self.device)
        gauss = gauss / gauss.sum()
        
        window = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)
        
        # Compute means
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=1)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=1)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Compute variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()
    
    def compute_lpips(
        self,
        pred_images: List[torch.Tensor],
        gt_images: List[torch.Tensor]
    ) -> float:
        """
        Compute Learned Perceptual Image Patch Similarity.
        
        Lower is better (0 = identical).
        """
        if self.lpips_model == "unavailable":
            return 0.0
        
        lpips_scores = []
        
        with torch.no_grad():
            for pred, gt in zip(pred_images, gt_images):
                pred = self._prepare_for_lpips(pred)
                gt = self._prepare_for_lpips(gt)
                
                score = self.lpips_model(pred, gt)
                lpips_scores.append(score.item())
        
        return np.mean(lpips_scores)
    
    def compute_depth_variance(
        self,
        model: Dict[str, torch.Tensor],
        render_fn: callable,
        camera_poses: List,
        num_samples: int = 100
    ) -> float:
        """
        Compute depth variance across multiple views.
        
        Lower variance indicates better multi-view consistency.
        """
        # Sample random points in 3D
        positions = model['positions']
        sample_idx = torch.randperm(len(positions))[:num_samples]
        sample_points = positions[sample_idx]
        
        # Project to each view and check depth consistency
        depth_values = []
        
        for camera in camera_poses:
            # Render depth map
            _, depth_map, _, _ = render_fn(
                model['positions'], model['colors'],
                model['opacities'], model['scales'], camera
            )
            
            # Sample depth at projected locations
            # (simplified - in production, use proper projection)
            depth_sample = depth_map.flatten()[::10][:num_samples]
            depth_values.append(depth_sample)
        
        # Compute variance
        depth_tensor = torch.stack(depth_values, dim=0)  # (n_views, n_samples)
        variance = torch.var(depth_tensor, dim=0).mean()
        
        return variance.item()
    
    def compute_cross_view_similarity(
        self,
        images: List[torch.Tensor]
    ) -> float:
        """
        Compute average similarity between overlapping views.
        
        Higher is better (more consistent).
        """
        if len(images) < 2:
            return 1.0
        
        similarities = []
        
        # Compare consecutive views
        for i in range(len(images) - 1):
            img1 = self._to_device(images[i])
            img2 = self._to_device(images[i + 1])
            
            # Simple MSE-based similarity
            mse = torch.mean((img1 - img2) ** 2)
            sim = 1.0 / (1.0 + mse)
            
            similarities.append(sim.item())
        
        return np.mean(similarities)
    
    def compute_normal_smoothness(
        self,
        positions: torch.Tensor,
        k: int = 8
    ) -> float:
        """
        Compute smoothness of surface normals.
        
        Lower values indicate smoother surfaces.
        """
        from .normal_smoother import NormalSmoother
        
        smoother = NormalSmoother(k_neighbors=k, device=self.device)
        normals = smoother.estimate_normals(positions)
        
        # Compute average normal variation
        neighbors_idx = smoother._find_knn(positions, k)
        
        variations = []
        for i in range(len(normals)):
            neighbor_normals = normals[neighbors_idx[i]]
            variation = torch.mean(
                torch.norm(neighbor_normals - normals[i], dim=1)
            )
            variations.append(variation.item())
        
        return np.mean(variations)
    
    def compute_scale_consistency(
        self,
        scales: torch.Tensor
    ) -> float:
        """
        Compute consistency of splat scales.
        
        Lower coefficient of variation indicates more uniform scales.
        """
        scales = self._to_device(scales)
        
        # Compute per-axis statistics
        mean_scale = scales.mean(dim=0)
        std_scale = scales.std(dim=0)
        
        # Coefficient of variation
        cv = (std_scale / (mean_scale + 1e-6)).mean()
        
        return cv.item()
    
    def _render_views(
        self,
        model: Dict[str, torch.Tensor],
        render_fn: callable,
        camera_poses: List
    ) -> List[torch.Tensor]:
        """Render all views from model."""
        rendered = []
        
        for camera in camera_poses:
            image, _, _, _ = render_fn(
                model['positions'], model['colors'],
                model['opacities'], model['scales'], camera
            )
            rendered.append(image)
        
        return rendered
    
    def _prepare_for_ssim(self, image: torch.Tensor) -> torch.Tensor:
        """Prepare image for SSIM computation: (1, 1, H, W)."""
        image = self._to_device(image)
        
        if len(image.shape) == 3:
            # Convert to grayscale
            if image.shape[0] == 3:  # (3, H, W)
                gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            else:  # (H, W, 3)
                gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
            image = gray.unsqueeze(0).unsqueeze(0)
        
        return image
    
    def _prepare_for_lpips(self, image: torch.Tensor) -> torch.Tensor:
        """Prepare image for LPIPS: (1, 3, H, W) in [-1, 1]."""
        image = self._to_device(image)
        
        if len(image.shape) == 3:
            if image.shape[-1] == 3:  # (H, W, 3)
                image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)
        
        # Scale to [-1, 1]
        image = (image - 0.5) / 0.5
        
        return image
    
    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on correct device."""
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.from_numpy(tensor).float()
        return tensor.to(self.device)


def compare_models(
    refined_model: Dict[str, torch.Tensor],
    baseline_model: Dict[str, torch.Tensor],
    render_fn: callable,
    test_views: List,
    camera_poses: List
) -> Dict[str, float]:
    """
    Compare refined model against baseline.
    
    Args:
        refined_model: Refined splat model
        baseline_model: Baseline splat model
        render_fn: Rendering function
        test_views: Ground truth test images
        camera_poses: Camera parameters
    
    Returns:
        Dictionary with improvement metrics
    """
    evaluator = MVCRMEvaluator()
    
    # Evaluate both models
    refined_metrics = evaluator.evaluate(
        refined_model, render_fn, test_views, camera_poses
    )
    
    baseline_metrics = evaluator.evaluate(
        baseline_model, render_fn, test_views, camera_poses
    )
    
    # Compute improvements
    improvements = {
        'psnr_improvement': refined_metrics.psnr - baseline_metrics.psnr,
        'ssim_improvement': refined_metrics.ssim - baseline_metrics.ssim,
        'lpips_improvement': baseline_metrics.lpips - refined_metrics.lpips,  # Lower is better
        'depth_var_reduction': baseline_metrics.depth_variance - refined_metrics.depth_variance,
        'consistency_improvement': refined_metrics.cross_view_similarity - baseline_metrics.cross_view_similarity
    }
    
    return improvements


def generate_evaluation_report(
    metrics: EvaluationMetrics,
    output_path: str
):
    """
    Generate markdown report of evaluation metrics.
    
    Args:
        metrics: Evaluation metrics
        output_path: Path to save report
    """
    report = f"""# MVCRM Evaluation Report

## Visual Quality Metrics
- **PSNR**: {metrics.psnr:.2f} dB
- **SSIM**: {metrics.ssim:.4f}
- **LPIPS**: {metrics.lpips:.4f}

## Multi-View Consistency
- **Depth Variance**: {metrics.depth_variance:.4f}
- **Cross-View Similarity**: {metrics.cross_view_similarity:.4f}

## Geometric Quality
- **Normal Smoothness**: {metrics.normal_smoothness:.4f}
- **Scale Consistency**: {metrics.scale_consistency:.4f}

"""
    
    if metrics.psnr_improvement is not None:
        report += f"""## Improvements Over Baseline
- **PSNR Gain**: {metrics.psnr_improvement:+.2f} dB
- **SSIM Gain**: {metrics.ssim_improvement:+.4f}
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to {output_path}")
