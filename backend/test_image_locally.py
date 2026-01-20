"""
Interactive Image Validation Tester

This script allows you to:
1. Upload an image from your computer
2. See background removal in action (rembg)
3. Validate if there's a single object
4. View quality metrics (blur score, object count, etc.)
5. See the final processed image

Usage:
    python test_image_locally.py path/to/your/image.jpg
    
    Or just run it and it will prompt you:
    python test_image_locally.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.services.image_validator import ImageValidator
from PIL import Image
import numpy as np


def save_debug_images(original, no_bg, processed, output_dir="validation_output"):
    """Save intermediate images for visual inspection."""
    os.makedirs(output_dir, exist_ok=True)
    
    original.save(f"{output_dir}/1_original.png")
    print(f"   ✓ Saved: {output_dir}/1_original.png")
    
    no_bg.save(f"{output_dir}/2_background_removed.png")
    print(f"   ✓ Saved: {output_dir}/2_background_removed.png")
    
    processed.save(f"{output_dir}/3_processed_final.png")
    print(f"   ✓ Saved: {output_dir}/3_processed_final.png")
    
    return output_dir


def test_image(image_path: str):
    """Test image validation on a single image."""
    
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found: {image_path}")
        return False
    
    print("=" * 70)
    print(f"🔍 TESTING IMAGE VALIDATION")
    print("=" * 70)
    print(f"📁 Input file: {image_path}")
    print()
    
    # Read image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # Create validator
    validator = ImageValidator()
    
    # Run validation
    print("🔄 Running validation pipeline...")
    print()
    
    try:
        result = validator.validate_and_process(
            image_bytes, 
            os.path.basename(image_path)
        )
        
        print("─" * 70)
        print("📊 VALIDATION RESULTS")
        print("─" * 70)
        
        if result["valid"]:
            # Success!
            print("✅ VALIDATION PASSED!")
            print()
            
            metadata = result["validation_metadata"]
            
            print("📏 Image Dimensions:")
            print(f"   Original size:  {metadata['original_size'][0]} x {metadata['original_size'][1]} pixels")
            print(f"   Processed size: {metadata['processed_size'][0]} x {metadata['processed_size'][1]} pixels")
            print()
            
            print("🎯 Object Detection:")
            print(f"   Objects detected: {metadata['num_objects_detected']}")
            print(f"   Object area:      {metadata['object_area_pixels']:,} pixels")
            print()
            
            print("✨ Quality Metrics:")
            print(f"   Blur score:       {metadata['blur_score']:.1f}")
            print(f"                     (higher is sharper, threshold: 100)")
            print()
            
            # Save intermediate images for visual inspection
            print("💾 Saving debug images...")
            
            # Load original
            original = Image.open(image_path)
            
            # Get processed image from result
            processed = result["processed_image"]
            
            # We need to recreate the no_bg image for visualization
            # Let's just show the processed one
            from rembg import remove
            import io
            
            # Remove background for visualization
            original_pil = Image.open(image_path)
            if original_pil.mode != "RGB":
                if original_pil.mode == "RGBA":
                    background = Image.new("RGB", original_pil.size, (255, 255, 255))
                    background.paste(original_pil, mask=original_pil.split()[3])
                    original_pil = background
                else:
                    original_pil = original_pil.convert("RGB")
            
            img_byte_arr = io.BytesIO()
            original_pil.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            output_bytes = remove(img_byte_arr.getvalue())
            no_bg = Image.open(io.BytesIO(output_bytes))
            
            # Save all stages
            output_dir = save_debug_images(original, no_bg, processed)
            print()
            
            print(f"📂 Output directory: {output_dir}/")
            print()
            print("🎉 Validation complete! Check the output folder to see:")
            print("   1. Original image")
            print("   2. Background removed")
            print("   3. Final processed (cropped & centered)")
            print()
            
            return True
            
        else:
            # Validation failed
            print("❌ VALIDATION FAILED")
            print()
            print("🚫 Reason:")
            print(f"   {result['error']}")
            print()
            
            print("💡 Suggestions:")
            if "blurry" in result['error'].lower():
                print("   - Use a sharper, higher quality image")
                print("   - Ensure proper focus when taking the photo")
            elif "multiple objects" in result['error'].lower():
                print("   - Upload an image with only one object")
                print("   - Crop the image to focus on a single subject")
            elif "blank" in result['error'].lower() or "white" in result['error'].lower():
                print("   - Upload an image with visible content")
                print("   - Check if the image file is corrupted")
            elif "small" in result['error'].lower():
                print("   - Use a higher resolution image (at least 256x256)")
            elif "aspect ratio" in result['error'].lower():
                print("   - Crop the image to be more square-shaped")
                print("   - Acceptable range: 1:3 to 3:1")
            print()
            
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function - interactive or command line."""
    
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "🖼️  IMAGE VALIDATION TESTER  🖼️" + " " * 15 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Check if image path provided as argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Interactive mode
        print("Please enter the path to your image file:")
        print("(You can drag and drop the file here)")
        print()
        image_path = input("📁 Image path: ").strip().strip('"').strip("'")
        print()
    
    if not image_path:
        print("❌ No image path provided")
        return
    
    # Run test
    success = test_image(image_path)
    
    print("=" * 70)
    if success:
        print("✅ TEST PASSED - Image is ready for 3D reconstruction!")
    else:
        print("❌ TEST FAILED - Image does not meet validation requirements")
    print("=" * 70)
    print()


if __name__ == "__main__":
    try:
        import rembg
        import cv2
        main()
    except ImportError as e:
        print("❌ Error: Missing dependency")
        print(f"   {str(e)}")
        print()
        print("Install required packages:")
        print("   pip install rembg opencv-python")
        sys.exit(1)
