"""
Test script for image validation module.

Tests various validation scenarios:
- Valid single object image
- Multiple objects
- Blurry image
- Blank image
- Invalid dimensions
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.services.image_validator import ImageValidator
from PIL import Image
import numpy as np
import io


def create_test_image(scenario: str) -> bytes:
    """Create test images for different scenarios."""
    
    if scenario == "valid":
        # Create a simple object on white background
        img = Image.new('RGB', (512, 512), color='white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        # Draw a circle (single object)
        draw.ellipse([150, 150, 350, 350], fill='blue', outline='darkblue', width=3)
        
    elif scenario == "multiple_objects":
        # Create multiple distinct objects
        img = Image.new('RGB', (512, 512), color='white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        # Draw 3 circles
        draw.ellipse([50, 50, 150, 150], fill='red')
        draw.ellipse([200, 200, 300, 300], fill='blue')
        draw.ellipse([350, 350, 450, 450], fill='green')
        
    elif scenario == "blurry":
        # Create an intentionally blurry image
        img = Image.new('RGB', (512, 512), color='white')
        from PIL import ImageDraw, ImageFilter
        draw = ImageDraw.Draw(img)
        draw.ellipse([150, 150, 350, 350], fill='blue')
        # Apply heavy blur
        img = img.filter(ImageFilter.GaussianBlur(radius=20))
        
    elif scenario == "blank":
        # Solid white image
        img = Image.new('RGB', (512, 512), color='white')
        
    elif scenario == "too_small":
        # Image below minimum size
        img = Image.new('RGB', (100, 100), color='white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.ellipse([10, 10, 90, 90], fill='blue')
        
    elif scenario == "extreme_aspect":
        # Very wide image
        img = Image.new('RGB', (2000, 300), color='white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.ellipse([800, 50, 1200, 250], fill='blue')
        
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()


def run_tests():
    """Run all validation tests."""
    
    print("=" * 60)
    print("IMAGE VALIDATION MODULE TESTS")
    print("=" * 60)
    
    validator = ImageValidator()
    
    test_cases = [
        ("valid", True, "Valid single object image"),
        ("multiple_objects", False, "Multiple objects in image"),
        ("blurry", False, "Blurry image (low Laplacian variance)"),
        ("blank", False, "Blank white image"),
        ("too_small", False, "Image dimensions too small"),
        ("extreme_aspect", False, "Extreme aspect ratio"),
    ]
    
    passed = 0
    failed = 0
    
    for scenario, should_pass, description in test_cases:
        print(f"\n{'─' * 60}")
        print(f"Test: {description}")
        print(f"Scenario: {scenario}")
        print(f"Expected: {'PASS' if should_pass else 'FAIL'}")
        
        try:
            # Create test image
            image_bytes = create_test_image(scenario)
            
            # Run validation
            result = validator.validate_and_process(image_bytes, f"{scenario}.png")
            
            # Check result
            actual_pass = result["valid"]
            
            if actual_pass == should_pass:
                print(f"✅ Result: CORRECT")
                if result["valid"]:
                    metadata = result["validation_metadata"]
                    print(f"   - Blur score: {metadata['blur_score']:.1f}")
                    print(f"   - Objects detected: {metadata['num_objects_detected']}")
                    print(f"   - Object area: {metadata['object_area_pixels']} pixels")
                else:
                    print(f"   - Error: {result['error']}")
                passed += 1
            else:
                print(f"❌ Result: INCORRECT")
                print(f"   Expected {'PASS' if should_pass else 'FAIL'}, got {'PASS' if actual_pass else 'FAIL'}")
                if not result["valid"]:
                    print(f"   Error: {result['error']}")
                failed += 1
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
            failed += 1
    
    # Summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"Passed: {passed}/{len(test_cases)}")
    print(f"Failed: {failed}/{len(test_cases)}")
    
    if failed == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ {failed} tests failed")
    
    return failed == 0


if __name__ == "__main__":
    # Note: This test requires rembg to be installed
    # If rembg is not available, tests will fail at background removal
    
    try:
        import rembg
        success = run_tests()
        sys.exit(0 if success else 1)
    except ImportError:
        print("❌ Error: 'rembg' package not installed")
        print("Install with: pip install rembg")
        sys.exit(1)
