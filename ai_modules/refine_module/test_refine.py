"""
ai_modules/refine_module/test_refine.py

Quick test script to verify MVCRM installation and functionality.

This script creates synthetic data and runs a minimal refinement loop
to ensure all components are working correctly.

Usage:
    python test_refine.py
"""

import torch
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai_modules.refine_module import (
    FusionController,
    RefinementConfig,
    ViewData,
    CameraParams,
    info
)


def create_synthetic_data():
    """Create synthetic test data."""
    print("Creating synthetic test data...")
    
    # Synthetic splat model (100 splats)
    N = 100
    positions = torch.randn(N, 3) * 0.5
    colors = torch.rand(N, 3)
    opacities = torch.rand(N) * 0.8 + 0.2
    scales = torch.rand(N, 3) * 0.1 + 0.05
    
    # Synthetic camera
    intrinsics = torch.tensor([
        [500, 0, 256],
        [0, 500, 256],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    extrinsics = torch.eye(4)
    extrinsics[:3, 3] = torch.tensor([0, 0, -2])
    
    camera = CameraParams(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        width=512,
        height=512
    )
    
    # Synthetic images (512x512x3)
    rendered = torch.rand(512, 512, 3) * 0.5
    enhanced = rendered + torch.randn_like(rendered) * 0.1  # Slightly different
    enhanced = torch.clamp(enhanced, 0, 1)
    depth = torch.rand(512, 512) * 5 + 1
    
    view = ViewData(
        enhanced_image=enhanced,
        rendered_image=rendered,
        depth_map=depth,
        camera=camera,
        confidence=1.0
    )
    
    return positions, colors, opacities, scales, [view]


def mock_render_fn(positions, colors, opacities, scales, camera):
    """Mock rendering function for testing."""
    H, W = camera.height, camera.width
    
    # Simple mock outputs
    image = torch.rand(H, W, 3)
    depth = torch.rand(H, W) * 5 + 1
    alpha = torch.rand(H, W)
    contributions = torch.zeros(H, W, 10, 2)  # Mock contribution map
    
    return image, depth, alpha, contributions


def test_components():
    """Test individual components."""
    print("\n" + "="*60)
    print("Testing Individual Components")
    print("="*60)
    
    # Test BackProjector
    print("\n[1/5] Testing BackProjector...")
    from ai_modules.refine_module import BackProjector
    projector = BackProjector(learning_rate=0.05)
    print("✓ BackProjector initialized")
    
    # Test DepthConsistencyChecker
    print("\n[2/5] Testing DepthConsistencyChecker...")
    from ai_modules.refine_module import DepthConsistencyChecker
    depth_checker = DepthConsistencyChecker(base_threshold=0.1)
    
    # Test with synthetic data
    depth1 = torch.rand(64, 64)
    depth2 = depth1 + torch.randn(64, 64) * 0.05
    result = depth_checker.check(depth1, depth2)
    print(f"✓ Depth consistency score: {result.consistency_score:.3f}")
    
    # Test FeatureConsistencyChecker
    print("\n[3/5] Testing FeatureConsistencyChecker...")
    from ai_modules.refine_module import FeatureConsistencyChecker
    feature_checker = FeatureConsistencyChecker(model_type="dinov2")
    print("✓ FeatureConsistencyChecker initialized (model will load on first use)")
    
    # Test NormalSmoother
    print("\n[4/5] Testing NormalSmoother...")
    from ai_modules.refine_module import NormalSmoother
    smoother = NormalSmoother(smoothing_strength=0.3)
    
    positions = torch.randn(50, 3)
    smoothed = smoother.smooth_positions(positions)
    print(f"✓ Smoothed {len(positions)} positions")
    
    # Test MVCRMEvaluator
    print("\n[5/5] Testing MVCRMEvaluator...")
    from ai_modules.refine_module import MVCRMEvaluator
    evaluator = MVCRMEvaluator()
    
    # Test PSNR computation
    img1 = torch.rand(64, 64, 3)
    img2 = img1 + torch.randn_like(img1) * 0.05
    psnr = evaluator.compute_psnr([img1], [img2])
    print(f"✓ Computed PSNR: {psnr:.2f} dB")
    
    print("\n✅ All components tested successfully!")


def test_full_pipeline():
    """Test complete refinement pipeline."""
    print("\n" + "="*60)
    print("Testing Full Refinement Pipeline")
    print("="*60)
    
    # Create synthetic data
    positions, colors, opacities, scales, views = create_synthetic_data()
    
    # Create configuration
    config = RefinementConfig(
        max_iterations=2,  # Quick test
        learning_rate=0.05,
        enable_depth_check=True,
        enable_feature_check=False,  # Skip to avoid loading heavy models
        enable_smoothing=True
    )
    
    print(f"\nConfiguration:")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Depth check: {config.enable_depth_check}")
    print(f"  Feature check: {config.enable_feature_check}")
    print(f"  Smoothing: {config.enable_smoothing}")
    
    # Create controller
    print("\nInitializing FusionController...")
    controller = FusionController(config)
    
    # Run refinement
    print("\nRunning refinement...")
    try:
        result = controller.refine(
            splat_positions=positions,
            splat_colors=colors,
            splat_opacities=opacities,
            splat_scales=scales,
            views=views,
            render_fn=mock_render_fn
        )
        
        print("\n✅ Refinement completed successfully!")
        print(f"\nResults:")
        print(f"  Iterations run: {result.iterations_run}")
        print(f"  Converged: {result.converged}")
        print(f"  Final quality: {result.quality_metrics['final_quality']:.3f}")
        print(f"  Refined positions shape: {result.refined_positions.shape}")
        print(f"  Refined colors shape: {result.refined_colors.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Refinement failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("MVCRM - Refine Module Test Suite")
    print("="*60)
    
    # Show package info
    info()
    
    # Test components
    try:
        test_components()
    except Exception as e:
        print(f"\n❌ Component tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test full pipeline
    try:
        success = test_full_pipeline()
        if not success:
            return False
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("🎉 All tests passed! MVCRM is ready to use.")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
