"""
ai_modules/midas_depth/test_depth.py

Unit tests for the MiDaS depth estimation module.

Run with:
    cd Glimpse3D/ai_modules/midas_depth
    pytest test_depth.py -v

Or run directly:
    python test_depth.py
"""

import os
import sys
import tempfile
import numpy as np
from PIL import Image
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midas_depth.run_depth import (
    DepthEstimator,
    estimate_depth,
    save_depth_visualization,
    save_depth_grayscale,
    save_depth_raw,
    load_depth_raw,
)

from midas_depth.depth_alignment import (
    align_depth_scales,
    align_depth_scales_global,
    estimate_overlap_simple,
    compute_alignment_quality,
)

from midas_depth.depth_confidence import (
    estimate_depth_confidence,
    apply_confidence_mask,
    weighted_depth_fusion,
    get_reliable_depth_mask,
)


def create_test_image(size=(256, 256)) -> Image.Image:
    """Create a simple test image with gradient."""
    # Create a gradient image (simulates depth-like content)
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    # Horizontal gradient (left=dark, right=bright)
    for x in range(size[0]):
        val = int(255 * x / size[0])
        arr[:, x, :] = val
    
    # Add a bright circle in center (simulates object)
    center = (size[0] // 2, size[1] // 2)
    radius = min(size) // 4
    y, x = np.ogrid[:size[1], :size[0]]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    arr[mask] = [255, 200, 150]
    
    return Image.fromarray(arr)


class TestDepthEstimator:
    """Tests for the DepthEstimator class."""
    
    @pytest.fixture(scope="class")
    def estimator(self):
        """Create a shared estimator instance (model loading is expensive)."""
        return DepthEstimator(model_type="MiDaS_small", device="cpu")
    
    @pytest.fixture
    def test_image_path(self):
        """Create a temporary test image file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = create_test_image()
            img.save(f.name)
            yield f.name
        os.unlink(f.name)
    
    def test_estimate_from_path(self, estimator, test_image_path):
        """Test depth estimation from file path."""
        depth = estimator.estimate(test_image_path)
        
        assert isinstance(depth, np.ndarray)
        assert depth.ndim == 2
        assert depth.dtype == np.float32
        assert depth.min() >= 0.0
        assert depth.max() <= 1.0
    
    def test_estimate_from_pil(self, estimator):
        """Test depth estimation from PIL Image."""
        img = create_test_image()
        depth = estimator.estimate(img)
        
        assert isinstance(depth, np.ndarray)
        assert depth.shape == (256, 256)
    
    def test_estimate_from_numpy(self, estimator):
        """Test depth estimation from numpy array."""
        img = create_test_image()
        img_np = np.array(img)
        depth = estimator.estimate(img_np)
        
        assert isinstance(depth, np.ndarray)
        assert depth.shape == (256, 256)
    
    def test_output_size_matches_input(self, estimator):
        """Test that output size matches input size."""
        sizes = [(128, 128), (256, 256), (512, 384), (640, 480)]
        
        for size in sizes:
            img = create_test_image(size)
            depth = estimator.estimate(img)
            assert depth.shape == (size[1], size[0]), f"Failed for size {size}"
    
    def test_normalize_option(self, estimator):
        """Test normalized vs raw output."""
        img = create_test_image()
        
        depth_norm = estimator.estimate(img, normalize=True)
        depth_raw = estimator.estimate(img, normalize=False)
        
        # Normalized should be in [0, 1]
        assert depth_norm.min() >= 0.0
        assert depth_norm.max() <= 1.0
        
        # Raw may have different range
        assert depth_raw.shape == depth_norm.shape


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_estimate_depth_function(self):
        """Test the estimate_depth convenience function."""
        img = create_test_image()
        depth = estimate_depth(img, model_type="MiDaS_small", device="cpu")
        
        assert isinstance(depth, np.ndarray)
        assert depth.min() >= 0.0
        assert depth.max() <= 1.0
    
    def test_save_and_load_raw(self):
        """Test saving and loading raw depth maps."""
        depth = np.random.rand(256, 256).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            save_depth_raw(depth, f.name)
            loaded = load_depth_raw(f.name)
            
            assert np.allclose(depth, loaded)
            os.unlink(f.name)
    
    def test_save_grayscale(self):
        """Test saving grayscale visualization."""
        depth = np.random.rand(256, 256).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_depth_grayscale(depth, f.name)
            
            # Verify file was created and is valid image
            img = Image.open(f.name)
            assert img.size == (256, 256)
            os.unlink(f.name)
    
    def test_save_visualization(self):
        """Test saving colored visualization."""
        depth = np.random.rand(256, 256).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_depth_visualization(depth, f.name, colormap="magma")
            
            # Verify file was created and is valid RGB image
            img = Image.open(f.name)
            assert img.size == (256, 256)
            assert img.mode == "RGB"
            os.unlink(f.name)


class TestDepthConfidence:
    """Tests for the depth confidence module."""
    
    def test_confidence_output_shape(self):
        """Test that confidence has same shape as depth."""
        depth = np.random.rand(100, 100).astype(np.float32)
        confidence = estimate_depth_confidence(depth)
        
        assert confidence.shape == depth.shape
        assert confidence.dtype == np.float32
    
    def test_confidence_range(self):
        """Test that confidence is in [0, 1]."""
        depth = np.random.rand(100, 100).astype(np.float32)
        confidence = estimate_depth_confidence(depth)
        
        assert confidence.min() >= 0.0
        assert confidence.max() <= 1.0
    
    def test_confidence_with_rgb(self):
        """Test confidence with RGB image."""
        depth = np.random.rand(100, 100).astype(np.float32)
        rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        confidence = estimate_depth_confidence(depth, rgb)
        
        assert confidence.shape == depth.shape
        assert confidence.min() >= 0.0
        assert confidence.max() <= 1.0
    
    def test_edges_have_lower_confidence(self):
        """Test that depth edges have lower confidence."""
        # Create depth with sharp edge
        depth = np.zeros((100, 100), dtype=np.float32)
        depth[:, 50:] = 1.0  # Sharp edge at x=50
        
        confidence = estimate_depth_confidence(depth)
        
        # Edge region should have lower confidence than flat regions
        edge_conf = confidence[:, 48:52].mean()
        flat_conf = confidence[:, 10:40].mean()
        
        assert edge_conf < flat_conf
    
    def test_apply_confidence_mask(self):
        """Test applying confidence mask."""
        depth = np.random.rand(100, 100).astype(np.float32)
        confidence = np.random.rand(100, 100).astype(np.float32)
        
        masked = apply_confidence_mask(depth, confidence, threshold=0.5)
        
        # Low confidence pixels should be masked
        assert masked[confidence < 0.5].sum() == 0
    
    def test_weighted_fusion(self):
        """Test weighted depth fusion."""
        depth1 = np.random.rand(100, 100).astype(np.float32)
        depth2 = np.random.rand(100, 100).astype(np.float32)
        conf1 = np.ones_like(depth1) * 0.8
        conf2 = np.ones_like(depth2) * 0.2
        
        fused, fused_conf = weighted_depth_fusion([depth1, depth2], [conf1, conf2])
        
        assert fused.shape == depth1.shape
        assert fused_conf.shape == depth1.shape
        # Fused should be closer to depth1 (higher confidence)
        diff1 = np.abs(fused - depth1).mean()
        diff2 = np.abs(fused - depth2).mean()
        assert diff1 < diff2
    
    def test_reliable_depth_mask(self):
        """Test reliable depth mask extraction."""
        confidence = np.random.rand(100, 100).astype(np.float32)
        mask = get_reliable_depth_mask(confidence, threshold=0.5)
        
        assert mask.dtype == bool
        assert mask.shape == confidence.shape


class TestDepthAlignment:
    """Tests for the depth alignment module."""
    
    def test_align_single_depth(self):
        """Test alignment with single depth map (should return copy)."""
        depth = np.random.rand(100, 100).astype(np.float32)
        aligned = align_depth_scales([depth])
        
        assert len(aligned) == 1
        assert np.allclose(aligned[0], depth)
    
    def test_align_two_depths_different_scales(self):
        """Test alignment corrects scale differences."""
        base = np.random.rand(100, 100).astype(np.float32) * 0.5 + 0.25
        
        depth1 = base.copy()
        depth2 = (base * 1.5).clip(0, 1)  # Different scale
        
        aligned = align_depth_scales([depth1, depth2])
        
        # After alignment, medians should be closer
        median_diff_before = abs(np.median(depth1) - np.median(depth2))
        median_diff_after = abs(np.median(aligned[0]) - np.median(aligned[1]))
        
        assert median_diff_after < median_diff_before
    
    def test_align_three_depths(self):
        """Test alignment with three depth maps."""
        base = np.random.rand(100, 100).astype(np.float32) * 0.5 + 0.25
        
        depths = [
            base.copy(),
            (base * 2.0).clip(0, 1),
            base * 0.5,
        ]
        
        aligned = align_depth_scales(depths)
        
        assert len(aligned) == 3
        for d in aligned:
            assert d.dtype == np.float32
            assert d.min() >= 0.0
            assert d.max() <= 1.0
    
    def test_global_alignment(self):
        """Test global alignment optimization."""
        base = np.random.rand(100, 100).astype(np.float32) * 0.5 + 0.25
        
        depths = [
            base.copy(),
            (base * 1.8).clip(0, 1),
            base * 0.6,
        ]
        
        aligned = align_depth_scales_global(depths)
        
        assert len(aligned) == 3
        for d in aligned:
            assert d.dtype == np.float32
    
    def test_overlap_estimation(self):
        """Test simple overlap mask estimation."""
        depth1 = np.random.rand(100, 100).astype(np.float32)
        depth2 = np.random.rand(100, 100).astype(np.float32)
        
        mask1, mask2 = estimate_overlap_simple(depth1, depth2)
        
        assert mask1.shape == (100, 100)
        assert mask2.shape == (100, 100)
        assert mask1.dtype == bool
        assert mask2.dtype == bool
    
    def test_alignment_quality_metrics(self):
        """Test quality metrics computation."""
        base = np.random.rand(100, 100).astype(np.float32) * 0.5 + 0.25
        depths = [base.copy(), base.copy()]  # Same depth = perfect alignment
        
        quality = compute_alignment_quality(depths)
        
        assert "n_views" in quality
        assert "mean_error" in quality
        assert "max_error" in quality
        assert quality["n_views"] == 2
    
    def test_empty_input(self):
        """Test handling of empty input."""
        aligned = align_depth_scales([])
        assert aligned == []


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    @pytest.fixture(scope="class")
    def estimator(self):
        return DepthEstimator(model_type="MiDaS_small", device="cpu")
    
    def test_grayscale_input(self, estimator):
        """Test handling of grayscale images."""
        gray = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        depth = estimator.estimate(gray)
        
        assert depth.shape == (256, 256)
    
    def test_rgba_input(self, estimator):
        """Test handling of RGBA images."""
        rgba = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
        depth = estimator.estimate(rgba)
        
        assert depth.shape == (256, 256)
    
    def test_file_not_found(self, estimator):
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError):
            estimator.estimate("nonexistent_file.jpg")
    
    def test_invalid_model_type(self):
        """Test error on invalid model type."""
        with pytest.raises(ValueError):
            DepthEstimator(model_type="InvalidModel")


# ============================================================================
# Quick manual test
# ============================================================================

def run_alignment_test():
    """Test depth alignment functionality."""
    print("\n" + "=" * 60)
    print("Testing Depth Alignment (Novel Feature)")
    print("=" * 60)
    
    # Create test depths with different scales
    np.random.seed(42)
    base = np.random.rand(128, 128).astype(np.float32) * 0.5 + 0.25
    
    depths = [
        base.copy(),              # Reference
        (base * 2.0).clip(0, 1),  # 2x scale
        base * 0.5,               # 0.5x scale
    ]
    
    print("\n1. Before alignment (medians):")
    for i, d in enumerate(depths):
        print(f"   View {i}: {np.median(d):.4f}")
    
    # Align
    print("\n2. Aligning depth scales...")
    aligned = align_depth_scales(depths)
    
    print("\n3. After alignment (medians):")
    for i, d in enumerate(aligned):
        print(f"   View {i}: {np.median(d):.4f}")
    
    # Quality
    quality = compute_alignment_quality(aligned)
    print(f"\n4. Quality metrics:")
    print(f"   Mean error: {quality['mean_error']:.4f}")
    print(f"   Max error:  {quality['max_error']:.4f}")
    
    # Verify improvement
    before_spread = max(np.median(d) for d in depths) - min(np.median(d) for d in depths)
    after_spread = max(np.median(d) for d in aligned) - min(np.median(d) for d in aligned)
    
    print(f"\n5. Median spread: {before_spread:.4f} → {after_spread:.4f}")
    
    if after_spread < before_spread:
        print("   ✅ Alignment improved consistency!")
    else:
        print("   ⚠️ Alignment may need tuning")
    
    return True


def run_quick_test():
    """Run a quick manual test without pytest."""
    print("=" * 60)
    print("Running quick manual test...")
    print("=" * 60)
    
    # Create test image
    print("\n1. Creating test image...")
    img = create_test_image((256, 256))
    print(f"   Image size: {img.size}")
    
    # Run estimation
    print("\n2. Running depth estimation (this may take a moment on first run)...")
    depth = estimate_depth(img, model_type="MiDaS_small", device="cpu")
    print(f"   Depth shape: {depth.shape}")
    print(f"   Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"   Depth dtype: {depth.dtype}")
    
    # Save outputs
    print("\n3. Saving outputs...")
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    img.save(os.path.join(output_dir, "test_input.png"))
    save_depth_grayscale(depth, os.path.join(output_dir, "test_depth_gray.png"))
    save_depth_visualization(depth, os.path.join(output_dir, "test_depth_color.png"))
    save_depth_raw(depth, os.path.join(output_dir, "test_depth.npy"))
    
    print(f"   Saved to: {output_dir}/")
    
    # Verify
    print("\n4. Verification checks...")
    assert depth.ndim == 2, "Depth should be 2D"
    assert depth.dtype == np.float32, "Depth should be float32"
    assert 0 <= depth.min() <= depth.max() <= 1, "Depth should be normalized"
    print("   ✅ All checks passed!")
    
    print("\n" + "=" * 60)
    print("✅ Quick test completed successfully!")
    print("=" * 60)


def run_confidence_test():
    """Test depth confidence estimation."""
    print("\n" + "=" * 60)
    print("Testing Depth Confidence (Novel Feature)")
    print("=" * 60)
    
    # Create test depth with known features
    print("\n1. Creating test depth with edge...")
    depth = np.zeros((128, 128), dtype=np.float32)
    depth[:, 64:] = 0.8  # Sharp edge at x=64
    depth[:, :30] = 0.3  # Flat region
    
    # Create test RGB with texture variation
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.linspace(0, 255, 128).astype(np.uint8)  # Gradient
    rgb[40:80, 40:80] = [100, 100, 100]  # Textureless patch
    
    print(f"   Depth shape: {depth.shape}")
    print(f"   RGB shape: {rgb.shape}")
    
    # Compute confidence
    print("\n2. Computing confidence...")
    confidence = estimate_depth_confidence(depth, rgb)
    print(f"   Confidence shape: {confidence.shape}")
    print(f"   Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    print(f"   Mean confidence: {confidence.mean():.3f}")
    
    # Check edge has lower confidence
    print("\n3. Checking edge detection...")
    edge_region_conf = confidence[:, 62:66].mean()
    flat_region_conf = confidence[:, 10:25].mean()
    print(f"   Edge region confidence: {edge_region_conf:.3f}")
    print(f"   Flat region confidence: {flat_region_conf:.3f}")
    
    if edge_region_conf < flat_region_conf:
        print("   ✅ Edge correctly identified as lower confidence!")
    else:
        print("   ⚠️ Edge detection may need tuning")
    
    # Test masking
    print("\n4. Testing confidence masking...")
    masked = apply_confidence_mask(depth, confidence, threshold=0.3)
    masked_pct = 100 * np.sum(masked == 0) / masked.size
    print(f"   Masked pixels: {masked_pct:.1f}%")
    
    # Test fusion
    print("\n5. Testing weighted fusion...")
    depth2 = depth + np.random.rand(128, 128).astype(np.float32) * 0.1
    conf1 = confidence
    conf2 = np.ones_like(depth) * 0.3  # Lower confidence
    
    fused, fused_conf = weighted_depth_fusion([depth, depth2], [conf1, conf2])
    print(f"   Fused depth range: [{fused.min():.3f}, {fused.max():.3f}]")
    print(f"   Fused confidence range: [{fused_conf.min():.3f}, {fused_conf.max():.3f}]")
    
    # Verify fusion is closer to higher confidence input
    diff1 = np.abs(fused - depth).mean()
    diff2 = np.abs(fused - depth2).mean()
    if diff1 < diff2:
        print("   ✅ Fusion correctly weighted toward higher confidence!")
    else:
        print("   ⚠️ Fusion weighting may need review")
    
    print("\n" + "=" * 60)
    print("✅ Confidence test completed!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    run_quick_test()
    run_alignment_test()
    run_confidence_test()
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
